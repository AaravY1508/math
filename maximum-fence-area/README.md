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
fence/minima.py     Stage 5: per-proposer library of distinct minima kept across runs; topology signatures
fence/rules.py      the 14 hard rules, each a separately toggleable check, plus the rejection ledger
fence/heuristics.py the 8 ranking heuristics, each separately toggleable and weighted
fence/nets.py       Net stage: the counting envelope, the net generators, filter and rank
fence/netevolve.py  Geometry stage: population over one net's continuous parameters + local topology moves
fence/digitize.py   record picture (<n>.gif from the page) -> skeleton -> planar graph -> unit fences -> Topology
pipeline.py         driver for the two stages: net enumeration -> per-net evolutionary search
digitize_records.py read the page's record pictures, optimize each read, referee, write the library
tests/              validation gate (must pass before any search starts)
baselines.json      best-known areas scraped from the page
run_overnight.py    supervisor: preflight tests, GPU sharding, restarts, checkpoints, final report
report.py           table n | best_known | ours | delta | verified, plus SVG renderings
runs/<timestamp>/   checkpoints, logs, best configs (gitignored)
minima/             the minima library shared by all runs (gitignored; snapshot archived per run)
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

# several hosts sharing this repo over NFS: give each a disjoint --n-range and a --tag;
# they share one minima library safely because every n has a single writer
setsid nohup ./venv/bin/python run_overnight.py --budget 6h --gpu-ids 1,2,3 --n-range 13,15,20-30 \
    --tag ynez > runs/launch-ynez.log 2>&1 < /dev/null &

# progress / results at any time
tail -f runs/launch.log
./venv/bin/python report.py runs/<timestamp>              # table + SVGs of ties/beats
./venv/bin/python report.py runs/<timestamp> --reverify   # re-run the referee on every best

# continue after a crash (remaining budget, from the last checkpoints)
./venv/bin/python run_overnight.py --resume runs/<timestamp>

# minima library: seed it from a run made before it existed, and inspect it
./venv/bin/python -m fence.minima import-run runs/<timestamp>
./venv/bin/python -m fence.minima summary

# read the page's record pictures (<n>.gif) into topologies and optimize them, one GPU per process;
# --v2 is the wider grid that also read n = 44, 46, 49 (EXPERIMENTS.md change 11)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ./venv/bin/python digitize_records.py --fetch --ns 17,24,50 \
    --out runs/digitize-<ts>
```

Pool workers inherit the parent's thread settings: without `OMP_NUM_THREADS=1` five processes with
eight polish workers each put serrano at load 146 on 48 cores with every GPU idle (2026-09-24).

Main `run_overnight.py` flags:

| Flag | Meaning |
|---|---|
| `--budget` | Wall-clock budget: `6h`, `90m`, `60s`. The run exits hard at the budget. |
| `--gpus` | Number of GPUs; one worker process per GPU via `CUDA_VISIBLE_DEVICES`. |
| `--gpu-ids` | Physical GPUs to use instead, e.g. `1,2,3` to skip a GPU someone else is using. |
| `--tag` | Suffix for the run directory, e.g. the host name when several hosts share this repo over NFS. |
| `--n-range` | n values to search, e.g. `12-50` (the default) or `13,15,20-30`. |
| `--archive-dir` | Where final results are copied, outside the repo (default `~/fence-results/`). Also synced every 30 min. |
| `--batch` | GPU restarts per topology (default 2048; 16384 reached the same areas at ~4x the GPU time, see `EXPERIMENTS.md`). |
| `--steps` | Adam steps per topology (default 1000). |
| `--mutator` | `random` (default, no API) or `llm`. |
| `--resume RUN_DIR` | Restart workers from their checkpoints. |
| `--minima-dir` | Minima library shared by all runs (default `minima/`; `--smoke` uses a private one in its run dir). |
| `--no-minima` | Neither record nor avoid minima. |
| `--stable-after` | Evaluations of a proposer at n without improvement before its best new result counts as a stabilized minimum (default 6). |
| `--topology-cap` | Re-propose a topology once its proposer has this many minima on it (default 3; 0 = never). |
| `--face-weight` | Share of the face-aware mutation operator (default 0.25; 0 = off: the previous operator mix). |
| `--seed-elites` | Start each n's archive from its k best library minima (default 6; 0 = off: an empty archive). |
| `--focus` | Search only the k n of `--n-range` closest to their records (0 = all). |
| `--verify-policy` | `worthy` (default: only polished candidates that would change something go to the referee) or `top` (the best ones always). |
| `--drop-exhausted` | `off` (default). `on`: an n where every lineage is blocked leaves its worker's schedule for the run, instead of being re-picked every ~15 s for nothing (n=50 got 0 evaluations in 1.37 h that way). Not A/B'd yet. |
| `--trivial-block` | `grid` (default): block a lineage at n while n's best is any grid answer. `ties`: only while that grid answer ties the record -- since 2026-09 the page beats the grid at every n >= 17. Not A/B'd yet. |
| `--no-cuda-probe` | Skip the preflight check that CUDA initializes on each selected GPU. The check exists because a host whose driver wants a reboot lists idle GPUs in `nvidia-smi` and still refuses `cuInit` on all of them. |

The preflight takes GPUs by the index `nvidia-smi` prints (a GPU that fell off the bus leaves a gap,
e.g. `0, 2, 3`), leaves out any GPU holding more than 1 GB of someone else's memory rather than
sharing it, and leaves out any GPU where a fresh process cannot run a CUDA matmul.

## The two-stage net pipeline (`pipeline.py`)

`run_overnight.py` evolves *programs* that draw layouts. `pipeline.py` is the other
way round: it enumerates the combinatorial structures first, throws away the ones
that cannot work, and only then spends optimizer time on what is left.

**Terminology.** A *fence* is one unit-length segment, and there are n of them. A
*graph-edge* is a piece of a fence between consecutive points where something touches
it, so a fence with k T-junctions in its interior is k + 1 graph-edges. `T` counts the
distinct T-junction points, `V` the vertices, `F` the bounded faces and `b` the
graph-edges on the outer boundary. With E = n + T graph-edges, Euler reads
V − (n + T) + F = 1.

### Stage 1: nets

`fence/nets.py` produces candidate nets, rejects them with `fence/rules.py` and ranks
the survivors with `fence/heuristics.py`.

Two different things are called enumeration here, and the distinction matters:

- `feasible_counts(n)` is **exhaustive over the counting envelope**: every (F, T, V, b)
  the counting rules allow for this n. Cheap and complete.
- the net **generators are constructive, not exhaustive**. They build concrete nets from
  minimum-perimeter polyomino clusters, clusters plus an appendage, clusters with
  boundary T-junction edits, nets driven up to rule 14's face bound by dissolving
  corners, the search's own seed families, and the topologies already in `minima/`
  (plus `onion`, a core polygon inside rings of cells -- off by default, it failed its
  A/B, EXPERIMENTS.md change 13). Enumerating all planar embeddings is out of reach well before n = 12, so no
  completeness is claimed. `coverage()` reports which parts of the envelope the
  constructed nets actually reached.

### Stage 2: geometry

`fence/netevolve.py` runs a population over one net's continuous parameters — vertex
positions and each T-junction's position along its host — refining every candidate with
the existing local optimizer (`fence/optimize.py`) and certifying with the referee.
Mutations come in two kinds:

- *geometry*: `jitter`, `jitter_boundary` (only the outer boundary moves, so the grid
  part stays snapped), `stretch`, `slide_t`, `symmetrize`.
- *topology*: `join_to_t` and `t_to_join` (an endpoint join ↔ a T-junction),
  `slide_t_past` (a T-junction moves past its neighbour on the same host, which
  re-orders the host's chain and so re-cuts the faces), `repin`, `add_face`,
  `remove_face`, `face_fill`.

Every mutant passes the same hard rules before it is allowed to cost optimizer time.

### Everything is a separate switch

```bash
./venv/bin/python pipeline.py --list          # every rule, heuristic, generator and move
./venv/bin/python pipeline.py --calibrate     # heuristics 1 and 2 against the verified configs
./venv/bin/python pipeline.py --n-range 12-16 --stage1-only
./venv/bin/python pipeline.py --n-range 22 --no-rule record_faces   # what does rule 14 cost?
./venv/bin/python pipeline.py --n-range 13,16 --device cuda --nets-per-n 3 --evals-per-net 12
./venv/bin/python pipeline.py --ablate rules --n-range 12,16,22,30,38   # what is each rule worth?

# one process per GPU on disjoint n, then one table
for k in 0 1 2 3; do
  NS=$(python3 -c "print(','.join(str(n) for n in range(3,51) if n%4==$k))")
  CUDA_VISIBLE_DEVICES=$k setsid nohup ./venv/bin/python pipeline.py --n-range "$NS" \
      --device cuda --budget 50m --mem-cap-gb 1.5 --out runs/pipe-shard$k \
      > runs/pipe-shard$k.log 2>&1 < /dev/null &
done
./venv/bin/python pipeline.py --combine runs/pipe-shard*
```

| Flag | Meaning |
|---|---|
| `--no-rule ID` | Switch off one hard rule (repeatable). `--only-rules a,b` enables exactly those. |
| `--no-heuristic ID` / `--weight ID=W` | Zero or re-weight one ranking heuristic. |
| `--no-move ID` / `--move-weight ID=W` / `--move-kinds` | Zero, re-weight or restrict the mutation moves. |
| `--generators a,b` | Which net generators to run. |
| `--nets-per-n`, `--evals-per-net`, `--generations`, `--pop`, `--elite` | Stage-2 size. |
| `--mem-cap-gb` | Cap on one batch's GPU working set; lower it to share a GPU. |
| `--torch-threads` | CPU threads for torch (default 4; unbounded on a 64-core box it is *slower*). |
| `--library-writes` | `on` (default): each n's verified best becomes the library record if larger, and every net's best is indexed as a minimum of the `pipeline` lineage. `off`: read the library only. |
| `--per-n-budget` | Time box per n: a duration, or `auto` = the remaining `--budget` split over the remaining n. |
| `--grace` | Seconds past `--budget` before the watchdog stops the run (then 90 s more to a hard exit; default 120). |
| `--ablate rules\|heuristics` | Switch each check off in turn over `--n-range` and print what it was worth. |
| `--combine RUN_DIR...` | Merge the per-n results of several runs (one per GPU shard) into one table. |

**Resuming.** Re-run the same command with the same `--out`: every n whose `n<NNN>.json` is already
there is skipped (each n's rng depends only on `--seed` and n, so a finished n is exactly what a
re-run would produce). A SIGTERM or SIGINT, or the budget watchdog, stops the run through the same
path, so `report.txt` and `summary.json` are written for every finished n whatever ended it.

An unknown id in any of the toggle flags is an error, not a silent no-op: a typo would
otherwise look like the check costing nothing.

Rejections are counted per rule and printed with an example, so a run says *why*
candidates died:

```
accepted 89/181 candidates; rejections by rule:
  record_faces  (rule 14)      86   e.g. polyomino_appendage/c=6 + 7 fences: 9 faces, need >= 10 ...
  bridgeless    (rule 10)       6   e.g. polyomino_tjunction/c=1 + 3 boundary T-edit(s): ... is a bridge
```

### The two model-based heuristics disagree with the records

Heuristics 1 and 2 come from a square-grid model and were flagged as unverified. Measured
on the 39 referee-verified configurations in `minima/records` (`pipeline.py --calibrate`):

| | model | measured |
|---|---|---|
| boundary size b | 2√(2n) = 2.83√n | **3.31√n** (spread ±17%; 2.59√n for the T-free grid layouts, 3.65√n for the T-rich ones) |
| face count F | n/2 | **0.402n**, and exactly ⌈total area⌉ on 37 of 39 |

So the targets follow the records: the boundary target interpolates on the layout's own T
count, from 2.59√n at T = 0 to 3.65√n at T ≥ 8, and the face target is the record-based bound of
hard rule 14 itself (⌈A_record⌉) rather than n/2, which overshoots by about 25%. The model values
are no longer selectable.

### A heuristic subsumed by the hard rules was removed

`one_cluster` (heuristic 5) scored every net that survives rules 6 and 13 as 1.00, so it ranked
nothing and was deleted. `face_count` is nearly constant under rule 14 (4 distinct values) but is
kept: with `--no-rule record_faces` it is the only thing steering the face count.

## Removed after failing their A/B (2026-09-23)

Kept in `EXPERIMENTS.md` so they are not re-run; the code is gone (git ref
`refs/backup/pre-cleanup-20260923` holds the last tree that had it):

| removed | experiment | why |
|---|---|---|
| `--op-mix tuned` | 5 | weights landed as designed, area -0.205, wins/1000 unchanged |
| `inflate_face` move | 9 | 0 of 5 collapsed-face records improved; those faces are true local optima |
| rule 15 `area_cap`, heuristic 10 `area_capacity`, generator `library_face_target` | 10 | -0.058 summed over 5 stuck n, 2 of 5 |
| `--m3-growth off`, `--boundary-model`, `--face-model`, `--rule` | 4, calibration | only restored a known bug or model values the records contradict |

Still switchable, off by default and not yet proven either way: `--remove-edit-fix` (6, looked like
noise around a real effect), `--target-faces` (7, resolved 2 of 11 face-short n, exploratory),
`--drop-exhausted`, `--trivial-block ties` (not A/B'd). Off after failing its own bar (2026-09-24):
the `onion` net generator (13: legal nets at 90% of n with the records' (F, T, joints), but stage 2
fills them 5-15% below the library).

## Status (2026-09-24)

- Library records exist for every n = 3..50, each verified by the referee at 1e-9 and 1e-12; the
  2026-09-24 ones are indexed under the `digitized` lineage so `MinimaLibrary.best()` seeds from them.
- Ties with the page (at its printed precision): n = 3-7, 10-14, 16-18, 36, 37. No n beats its record.
- Shortfall against the page (re-scraped 2026-09-24, unchanged since 2026-09-21): **0.151** over
  n = 12..50 (4.939 before the 2026-09-24 run), **0.000** over n = 3..11 (was 0.489).
- What closed it: the page's own record pictures, read into topologies by `fence/digitize.py`
  (`digitize_records.py`) and optimized, then move-neighbourhood search around those topologies
  (EXPERIMENTS.md changes 11, 12). The records are rounded ring/onion layouts none of the seed
  families builds; a regular parametric onion family did not reproduce them (change 13).
- Open problem: the remaining gaps (largest 35 and 44 at 0.035, then 41 at 0.014) sit on a digitized topology the optimizer has
  already filled -- 4x restarts and 3x steps change nothing (change 14) -- so each is a local misread
  (a joint that should be a T, a fence on the wrong edge) or a genuinely different record topology.
- Throughput: `pipeline.py` keeps a GPU only 21-28% busy (synchronous polish); `run_overnight.py`
  89-100%. Launch every GPU job with `OMP_NUM_THREADS=1` (see Running).

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
  Only a verified area counts. With `--verify-policy worthy`, a candidate is sent only if
  passing would change something: raise n's best, take over the archive cell it would land
  in (predicted from its topology and coordinates, exactly as the referee would count it),
  or beat its proposer's pending minimum.

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
- jitter growth-policy parameters (growth steps can add cells of 1-3 fences; a 3-fence cell on a
  unit-long piece of boundary is a unit square)
- face-aware edit (share `--face-weight`), which edits the layout of one of n's own
  verified elites where it loses area instead of regrowing from the program. A connected
  layout with n fences and J joints (vertices that are not T-pins) has exactly
  F = n + 1 − J bounded faces, so only an edit that removes a joint can gain a face at
  fixed n. The edit is one of:
  - *relocate*: a face below area 0.9 loses the boundary fence whose removal costs the
    least area, and that fence comes back as a one-fence cell (F unchanged).
  - *dissolve*: when the layout has fewer faces than floor(record) + 1, a corner where
    exactly two fences meet goes away. One fence is removed, the other's loose end is
    pinned on the nearest edge, and the removed fence comes back as a one-fence cell
    (F + 1).

**`llm`** is wired but off by default. `--mutator llm` sends the program source,
its score history, and the builder API to Claude (`claude-opus-5`, streaming,
adaptive thinking). It enables server-side refusal fallbacks
(`fallbacks="default"`), and the model returns a rewritten program. LLM-written
programs run in a sandboxed subprocess with a timeout. Credentials come from the
Anthropic SDK's standard resolution: `ANTHROPIC_API_KEY` or an `ant auth login`
profile. Nothing is stored.

### Stage 5: minima library

Every read-modify-write at n (a record, a minimum's index, a trivial marker's count) runs under
`minima/locks/N<n>.lock`, created with `O_EXCL` so it also excludes writers on other hosts sharing
the library over NFS; a lock older than 2 minutes belongs to a dead writer and is taken over.

Every program belongs to a **proposer**: the lineage of the hand-written seed
program it descends from. The proposer id is a hash of that seed program's source,
so a lineage keeps its id across runs. A child inherits its parent's proposer, and
a child re-seeded from an elite joins the elite's lineage.

- **Stabilized minima.** A proposer's search at n has stabilized when its best new
  result is not improved on by its next `--stable-after` evaluations at n. That
  result is saved as `minima/proposers/<proposer>/N<n>.<k>.json`, where k counts
  that proposer's distinct minima at n from 0. Two minima are the same when their
  layout topology matches and their areas agree within 1e-6. A duplicate is never
  indexed.
- **Topology signature.** A canonical code of the plane graph together with its
  split into fences, so a T-junction differs from a joint. It does not change under
  relabeling, rigid motion, or reflection. It is read off the face cycles, so a
  sketch and the optimized configuration built on it always share one signature.
- **Steering.** Before any GPU time is spent, a job looks up its proposer's minima
  on the proposed topology:
  - An exhausted topology (`--topology-cap` minima on it, or two steered attempts
    that found nothing new) is re-proposed, with a new seed plus one more
    structural edit.
  - Otherwise the Adam restarts are repelled from those minima. Each minimum, in
    every relabeling the topology's symmetries allow, gets a Gaussian bump in the
    RMS vertex distance after the best rigid alignment (closed form in 2D).
  - A restart can still polish back into a known basin, so polished candidates
    whose area matches a known minimum of the topology never reach the referee.

  Tuned on the first run's minima: on topologies with several basins (n = 26, 30,
  44) the repulsion raised the number of novel minima per job from 0–2 to 1–8. On
  single-basin topologies (n = 13, 18) there is nothing new to find, and the
  re-proposal handles those.
- **Grid-trivial answers.** An integer area no larger than `lattice_max(n)` is what
  the grid proposer gets. `lattice_max(n)` is the most unit squares n fences can
  enclose: c cells need 2c + ⌈2√c⌉ fences. A proposer whose search at n stabilizes
  on such an answer gets `N<n>.trivial.json`, and is not mutated at n again while
  n's best is itself grid-trivial.
- **Records.** `minima/records/N<n>.json` holds the best verified configuration for
  n over all runs, so scheduling knows each n's standing from the first minute.
  `import-run` builds records from a run made before the library existed.
- **Archive seeding** (`--seed-elites k`). A fresh worker starts each n's islands from
  the k best library minima with distinct topologies (the record first) instead of an
  empty archive, and skips the initial sweep of hand-written seed programs, so progress
  carries over between runs. No child may just re-optimize a seeded minimum. Its elite
  children get at least one edit, and any child that still rebuilds the minimum's
  topology is re-proposed (twice at most) or dropped before GPU time is spent.

### Overnight runner

Each GPU worker handles a disjoint slice of n, balanced by the same weights it
schedules with:
- Best-known ("+") entries are weighted above the trivial grids, and small n are
  excluded by default.
- Best-known n ≥ 32 get 1.5× (the gaps to the records are largest there).
- An n whose best (this run or the library record) is a grid-trivial tie with the
  record keeps 10% of its share. Only a whole new structure could improve it.

Within a worker, time is shared fairly by these weights, and n within 5% of the
record get up to 2× share.

`--focus k` narrows the run to the k n of `--n-range` closest to their records (record
minus the library's best). It leaves out n whose record is a grid (likely optimal) and n
that already tie or beat their record, so the budget goes to a few n instead of being
spread thin.

Operational behavior:
- **Logging:** each worker writes JSONL to `runs/<ts>/logs/worker<i>.jsonl`, with
  events `eval`, `new_best`, `IMPROVEMENT`, `checkpoint`, `gpu_error`, and more.
- **Checkpoints:** every 5 minutes to `runs/<ts>/worker<i>/checkpoint.pkl`,
  covering populations, bests, and RNG states.
- **Best configs:** written to `runs/<ts>/best/n<NNN>.json` whenever they improve.
  The saved file is re-read right before each write. A worker restarted from an
  older checkpoint therefore never replaces a better result found after that
  checkpoint.
- **Crash or hang:** the supervisor logs it loudly and restarts that worker from
  its checkpoint while the others keep running. Crash recovery was tested by
  SIGKILLing a worker mid-run. Heartbeats are written atomically. In the first
  run, the supervisor read half-written (empty) heartbeat files and killed 5
  healthy workers.
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
