# Maximum Fence Area

A GPU search harness for Erich Friedman's
[Maximum Fence Area](https://erich-friedman.github.io/packing/fence/) problem.

## The problem

Place **n fences**, each a straight segment of length **exactly 1**.

- **Every fence endpoint must lie on another fence.** "On" includes the other
  fence's interior as well as its endpoints. T-junctions are legal and essential
  to good configurations, so this is not an endpoint-to-endpoint graph problem.
- **Fences may not cross.** They may touch only at endpoints or T-junctions.
  Collinear overlap of positive length is treated as illegal. That is the
  conservative reading: a record under this rule is also valid under a looser one.
- The fences induce a planar subdivision, and **every bounded face ("field") must
  have area ≤ 1**.
- **Objective:** maximize the total area of all bounded faces.

`baselines.json` holds the best-known areas for n = 3..50, scraped from the page by
`scrape_baselines.py`. A trailing `+` on the page means the value is truncated,
so the true record lies in `[A, A + 10^-decimals)`. The "Trivial" entries
(n = 3, 4, 5, 7, 12, 17, 24, 31, 40, 49) are unit-square grids or simple polygons.
The page does not claim they are optimal.

## Layout

```
fence/geometry.py   topology model, DCEL face traversal, batched torch residuals (optimizer path)
fence/referee.py    verify(config, tol): independent 50-digit mpmath referee (stdlib + mpmath only)
fence/builder.py    sketch DSL used by evolved proposer programs (seeds, cell growth, symmetry, edits)
fence/optimize.py   Stage 3: batched GPU Adam (augmented Lagrangian) -> CPU SLSQP polish -> referee
fence/evolve.py     Stage 4: island MAP-Elites over propose_topology programs; per-GPU worker
tests/              validation gate (must pass before any search starts)
baselines.json      best-known areas scraped from the page
run_overnight.py    supervisor: preflight tests, GPU sharding, restarts, checkpoints, final report
report.py           table n | best_known | ours | delta | verified, plus SVG renderings
runs/<timestamp>/   checkpoints, logs, best configs (gitignored)
```

## Setup (Python 3.11)

```bash
cd maximum-fence-area
uv venv --python 3.11 venv          # or: python3.11 -m venv venv
uv pip install --python venv/bin/python -r requirements.txt
./venv/bin/python -m pytest          # the validation gate
```

`requirements.txt` pins the top-level packages. `requirements.lock.txt` is the
full frozen environment the harness was built with: torch 2.14 with CUDA 13.0
wheels, on 4× RTX 3090 with driver 595.

## Running

```bash
# 60-second end-to-end smoke test at small n (n = 6..9), all 4 GPUs
./venv/bin/python run_overnight.py --smoke

# the overnight search (detached so it survives the SSH session)
setsid nohup ./venv/bin/python run_overnight.py --budget 6h --gpus 4 \
    --archive-dir ~/fence-results > runs/launch.log 2>&1 < /dev/null &

# progress / results at any time
tail -f runs/launch.log
./venv/bin/python report.py runs/<timestamp>              # table + SVGs of ties/beats
./venv/bin/python report.py runs/<timestamp> --reverify   # re-run the referee on every best

# continue after a crash (remaining budget, from the last checkpoints)
./venv/bin/python run_overnight.py --resume runs/<timestamp>
```

Main `run_overnight.py` flags:

| Flag | Meaning |
|---|---|
| `--budget` | Wall-clock budget: `6h`, `90m`, `60s`. The run exits hard at the budget. |
| `--gpus` | Number of GPUs; one worker process per GPU via `CUDA_VISIBLE_DEVICES`. |
| `--n-range` | n values to search, e.g. `12-50` (the default) or `13,15,20-30`. |
| `--archive-dir` | Where final results are copied, outside the repo (default `~/fence-results/`). Also synced every 30 min. |
| `--batch` | GPU restarts per topology. Default depends on n: 16384 / 12288 / 10240. |
| `--steps` | Adam steps per topology (default 1000). |
| `--mutator` | `random` (default, no API) or `llm`. |
| `--resume RUN_DIR` | Restart workers from their checkpoints. |

## How it works

### Stage 1: geometry and the referee

A configuration is stored as vertices plus a topology:
- **Fences** are pairs of endpoint vertices.
- **Pins** mark a vertex that lies in the interior of a host fence, at parameter
  t ∈ (0, 1) along it.
- **Chains** list the vertices along each fence in order.

Faces come from a half-edge (DCEL) traversal. At each vertex, the outgoing
half-edges are sorted counter-clockwise. The successor of `u→w` is the outgoing
half-edge at `w` immediately clockwise of `w→u`. The orbits of that successor map
are the faces: counter-clockwise cycles are bounded faces, and clockwise cycles
are component boundaries. `fence/geometry.py` explains this in detail. Face areas
use the shoelace formula.

The optimizer works with batched differentiable residuals:
- `|len − 1|`
- proper-crossing depth of non-incident fence pairs
- `relu(area − 1)`
- pin-on-host distance
- plus safety terms: pin order along each host, no folded corners, faces
  positively oriented

`fence/referee.py` is the only code that certifies anything. It reads raw segment
coordinates as exact decimals and works at 50 digits. From those alone it
re-derives:
- incidences, clustering endpoints within `tol` and detecting T-junctions
- crossings and collinear overlaps
- the arrangement, rotation system, faces, and holes (nested components are
  subtracted from their enclosing face)
- unit lengths
- face areas ≤ 1 + `tol`

It imports only the standard library and mpmath, and a test enforces that.

### Stage 2: validation gate

`tests/` checks exact areas to at least 1e-12 (the referee matches to 1e-40) for:
- the unit triangle
- the unit square
- a T-junction split square
- a domino
- 2×2 and 2×3 grids

The referee must reject:
- crossing fences
- oversized faces (including a T-junction case)
- dangling ends, collinear overlap, and wrong lengths

Further tests cover hole subtraction, baseline-table integrity, Jacobians
against finite differences, tiny end-to-end optimizations, the builder and
mutation operators, and the status classification. `run_overnight.py` runs the
suite first. On failure it writes `~/erich-friedman-BLOCKED.txt` and does not
search.

### Stage 3: batched local optimizer

For a fixed topology, the free variables are vertex coordinates plus pin
parameters.
- **GPU stage:** B restarts run as one `[B, P]` batch, stored transposed so
  gathers are coalesced. They are perturbations and rescalings of the sketch.
  Adam minimizes an augmented Lagrangian: equalities for lengths and pins, and
  a PHR inequality term for face area ≤ 1. The penalty parameter and the safety
  weights anneal from soft to stiff, and multipliers are updated periodically.
- **CUDA graph:** the whole step is captured once and replayed. This removed a
  roughly 15× Python and launch overhead.
- **Memory:** batch size is capped by a per-restart memory estimate
  (`--mem-cap-gb`), and the worker halves the batch on OOM.
- **CPU polish:** the top-k distinct restarts go to SLSQP in a process pool. It
  enforces exact equalities, runs in chunks inside a trust-region box, and checks
  the embedding after every chunk (reverting if it breaks). A Gauss-Newton
  projection then brings residuals to ~1e-15.
- **Referee:** the best polished candidates go through `verify()` at 50 digits.
  Only a verified area counts.

### Stage 4: evolutionary outer loop

The evolved artifact is Python source defining `propose_topology(n, rng)`, which
returns a planar sketch built with `fence/builder.py`.
- **Fitness** is the best verified area the optimizer reaches on a proposed
  topology, divided by the best-known area.
- **Population:** each n has 3 islands. Each island is a MAP-Elites archive keyed
  by (face count, T-junction-count bin, rotational symmetry order), and islands
  migrate their best elite periodically.
- **Cross-n transfer:** a mutation can re-seed from the current elite of n−1, n,
  or n+1, even one found by another GPU.

Mutation backends share one interface (`fence/evolve.py:Mutator`).

**`random`** is used tonight and needs no network. Its structural mutations
operate on the program's GENOME:
- `add_face` / `remove_face`
- `split_edge` (move a fence end onto another edge, creating a new T-junction)
- `toggle_junction` (T ↔ joint)
- `move_fence`
- apply a symmetry group (C_m orbits)
- re-seed from an elite
- change the seed family (ring, grid, pinwheel)
- jitter growth-policy parameters

**`llm`** is wired but off by default. `--mutator llm` sends the program source,
its score history, and the builder API to Claude (`claude-opus-5`, streaming,
adaptive thinking). It enables server-side refusal fallbacks
(`fallbacks="default"`), and the model returns a rewritten program. LLM-written
programs run in a sandboxed subprocess with a timeout. Credentials come from the
Anthropic SDK's standard resolution: `ANTHROPIC_API_KEY` or an `ant auth login`
profile. Nothing is stored.

### Overnight runner

Each GPU worker handles a disjoint slice of n, balanced by priority. Best-known
("+") entries are weighted above the trivial grids, and small n are excluded by
default. Within a worker, time is shared fairly by priority, and n within 5% of
the record get up to 2× share.

Operational behavior:
- **Logging:** each worker writes JSONL to `runs/<ts>/logs/worker<i>.jsonl`, with
  events `eval`, `new_best`, `IMPROVEMENT`, `checkpoint`, `gpu_error`, and more.
- **Checkpoints:** every 5 minutes to `runs/<ts>/worker<i>/checkpoint.pkl`,
  covering populations, bests, and RNG states.
- **Best configs:** written to `runs/<ts>/best/n<NNN>.json` whenever they improve.
- **Crash or hang:** the supervisor logs it loudly and restarts that worker from
  its checkpoint while the others keep running. Crash recovery was tested by
  SIGKILLing a worker mid-run.
- **Wedged GPU:** after 5 consecutive GPU errors a worker exits deliberately so
  it is restarted with a fresh CUDA context.

### What counts as a result

`report.py` classifies each verified area against the page:
- **BEATS:** exceeds the largest value consistent with the printed figure (A plus
  one unit in the last printed decimal when the page shows `+`), plus a 1e-7
  safety margin.
- **TIES:** at least the printed figure.
- **NEAR:** within 1e-3 below.
- **BELOW:** further below.

Numbers that did not pass the referee are never reported as results.
