"""Stage 4: evolutionary outer loop over topology-proposer PROGRAMS.

Modelled on OpenEvolve / ShinkaEvolve (LLM-guided program evolution):

* The evolved artifact is Python source defining ``propose_topology(n, rng)``,
  which returns a fence.builder.Sketch (a legal planar fence embedding with n
  fences). Programs made by the ``random`` backend embed a GENOME literal that
  fence.builder.build_from_genome interprets; the ``llm`` backend may rewrite
  the function freely against the same builder API.
* Fitness of a program = best referee-verified total area that Stage 3
  (fence.optimize) reaches on a topology it proposes, divided by the best known
  area from baselines.json.
* Population: per n, an island model of MAP-Elites archives. Cells are keyed by
  (face count, T-junction-count bin, rotational symmetry order) of the verified
  configuration, so structurally different solutions survive side by side.
  Islands exchange their best elite periodically.
* Mutation operators behind one interface (`Mutator.mutate`):
    random -- structural genome mutations, no API: add/remove face, split edge,
              change a junction type, apply a symmetry group, re-seed from an elite
              (incl. elites of n-1 / n+1 found by other GPUs), parameter jitter; and
              (share --face-weight) a face-aware edit of one of n's verified elites
              (fence.builder.face_fill): a fence moves off an under-filled face, or a
              corner joint is dissolved while n has fewer faces than the record needs.
    llm    -- asks Claude to rewrite the program given its source and score history.
  The overnight runner uses `random` (zero API dependency).
* Proposers: every program belongs to the lineage of the hand-written seed program it
  descends from (`Program.proposer`, a content hash of the seed's source, so a lineage
  keeps its id across runs). A child inherits its parent's proposer; a child re-seeded
  from an elite joins the elite's lineage.
* Minima library (fence.minima), shared by all runs:
    - a proposer's search at n has *stabilized* when its best new result has not been
      improved on by its next `--stable-after` evaluations at n. That result is indexed
      as N<n>.<k> for the proposer, unless it already has that minimum (same topology,
      area within 1e-6).
    - each job is steered away from its proposer's known minima: a proposed topology the
      proposer has exhausted (`--topology-cap` minima on it, or two steered attempts that
      found nothing new) is re-proposed before any GPU time is spent; otherwise the Adam
      restarts are repelled from the known minima of that topology, and polished results
      that land on one anyway are not sent to the referee.
    - a proposer whose search at n stabilizes on the grid-trivial answer (unit squares,
      `is_grid_trivial`) is not mutated at n again, for as long as n's best is itself
      grid-trivial; an n whose best ties a grid-trivial record keeps only a small share of
      its worker's time, and below-record n >= LARGE_N get more.
    - with --seed-elites k a fresh worker starts each n's islands from its k best library
      minima; no child may re-create a seeded minimum's topology (`_propose`).

This module also contains the per-GPU worker (`python -m fence.evolve ...`),
which the supervisor in run_overnight.py launches once per GPU.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import pprint
import random
import re
import signal
import subprocess
import sys
import textwrap
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fence import builder as Bd
from fence import geometry as G
from fence import minima as Mn
from fence import optimize as O

HERE = Path(__file__).resolve().parent
PROBLEM_DIR = HERE.parent

# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def load_baselines(path=None) -> dict:
    path = Path(path or PROBLEM_DIR / "baselines.json")
    doc = json.loads(path.read_text())
    if "TODO" in doc or not doc.get("baselines"):
        raise RuntimeError(f"{path} is a placeholder ({doc.get('reason')}); refusing to run against fake numbers")
    out = {}
    for k, v in doc["baselines"].items():
        upper = v["A"] + (10.0 ** (-v["decimals"]) if v["truncated"] else 0.0)
        if v.get("exact_expr"):
            upper = v["A"]
        out[int(k)] = {"A": float(v["A"]), "upper": float(upper), "status": v["status"], "display": v["display"]}
    return out


IMPROVE_MARGIN = 1e-7   # numerical safety margin on top of the baseline's upper bound


def classify(area: float | None, base: dict | None) -> str:
    """BEATS: exceeds even the largest value consistent with the page's truncated '+'
    figure; TIES: matches the printed figure; NEAR: within 1e-3 below; else BELOW."""
    if area is None:
        return "NONE"
    if base is None:
        return "NO_BASELINE"
    if area > base["upper"] + IMPROVE_MARGIN:
        return "BEATS"
    if area >= base["A"] - 1e-9:
        return "TIES"
    if area >= base["A"] - 1e-3:
        return "NEAR"
    return "BELOW"


def lattice_max(n: int) -> int:
    """Most unit squares that n unit fences enclose on the square lattice. A polyomino of
    c cells has perimeter >= 2 ceil(2 sqrt(c)) (Harary-Harborth), so it needs at least
    2c + ceil(2 sqrt(c)) fences. Equals the page's "trivial" records at 12, 17, 24, ..."""
    c = 0
    while 2 * (c + 1) + math.ceil(2 * math.sqrt(c + 1)) <= n:
        c += 1
    return c


def is_grid_trivial(area: float | None, n: int) -> bool:
    """A grid-of-unit-squares answer: an integer area no larger than lattice_max(n), which
    is what the grid proposer gets (e.g. n=24 -> 9, n=50 -> 20)."""
    if area is None:
        return False
    k = round(area)
    return abs(area - k) <= 1e-9 and k <= lattice_max(n)


# ---------------------------------------------------------------------------
# programs
# ---------------------------------------------------------------------------

PROGRAM_TEMPLATE = '''"""Evolved topology proposer for Erich Friedman's Maximum Fence Area problem.

propose_topology(n, rng) must return a fence.builder.Sketch with exactly n fences:
a legal planar fence embedding (every fence end on another fence, no crossings).
Fence lengths need not be 1 -- the optimizer enforces that. Names available:
B (the fence.builder module), build_from_genome, Sketch, np, math.
"""
GENOME = {genome}


def propose_topology(n, rng):
    return build_from_genome(n, rng, GENOME)
'''


def render_program(genome: dict) -> str:
    return PROGRAM_TEMPLATE.format(genome=pprint.pformat(genome, width=110, compact=True, sort_dicts=True))


def _program_namespace():
    return {"B": Bd, "build_from_genome": Bd.build_from_genome, "Sketch": Bd.Sketch, "np": np, "math": math,
            "__name__": "fence_program"}


def compile_program(source: str):
    ns = _program_namespace()
    exec(compile(source, "<fence-program>", "exec"), ns)  # noqa: S102 -- trusted, template-generated
    fn = ns.get("propose_topology")
    if not callable(fn):
        raise ValueError("program does not define propose_topology(n, rng)")
    return fn


SANDBOX_RUNNER = r"""
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from fence import evolve as E
src = open(sys.argv[2]).read()
fn = E.compile_program(src)
sk = fn(int(sys.argv[3]), np.random.default_rng(int(sys.argv[4])))
print(json.dumps(sk.to_json()))
"""


def run_program_sandboxed(source: str, n: int, seed: int, timeout: float = 60.0) -> Bd.Sketch:
    """Execute an untrusted (LLM-written) program in a separate process with a timeout."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(source)
        path = fh.name
    try:
        out = subprocess.run([sys.executable, "-c", SANDBOX_RUNNER, str(PROBLEM_DIR), path, str(n), str(seed)],
                             capture_output=True, text=True, timeout=timeout,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"})
    finally:
        os.unlink(path)
    if out.returncode != 0:
        raise RuntimeError(f"sandboxed program failed: {out.stderr[-500:]}")
    return Bd.Sketch.from_json(json.loads(out.stdout.strip().splitlines()[-1]))


def proposer_id(source: str) -> str:
    """Id of the lineage rooted at the program with this source."""
    return hashlib.sha1(source.encode()).hexdigest()[:10]


@dataclass
class Program:
    pid: str
    source: str
    genome: dict | None
    parent: str | None = None
    origin: str = "seed"
    generation: int = 0
    trusted: bool = True           # False for LLM-written code -> run sandboxed
    proposer: str = ""             # lineage (see module docstring); "" only in old checkpoints

    @staticmethod
    def from_genome(genome, parent=None, origin="seed", generation=0, proposer=None) -> "Program":
        """proposer=None makes the program the root of its own lineage."""
        source = render_program(genome)
        return Program(pid=uuid.uuid4().hex[:10], source=source, genome=copy.deepcopy(genome),
                       parent=parent, origin=origin, generation=generation, proposer=proposer or proposer_id(source))

    def propose(self, n: int, seed: int) -> Bd.Sketch:
        if self.trusted:
            return compile_program(self.source)(n, np.random.default_rng(seed))
        return run_program_sandboxed(self.source, n, seed)


def grid_fit(n):
    best = (1, 1)
    for r in range(1, 12):
        for c in range(r, r + 3):
            if 2 * r * c + r + c <= n and r * c > best[0] * best[1]:
                best = (r, c)
    return best


def seed_genomes(n: int) -> list[dict]:
    """Hand-written starting programs covering different construction families
    (evaluated in this order, so the strongest structural priors come first)."""
    gs = []
    if n >= 12:
        # unit-square grids are the published records at n = 12, 17, 24, 31, 40, 49: start from the
        # largest grid that fits (grown by cells) and from the next larger grid pruned down to n
        r, c = grid_fit(n)
        gs.append({"seed": {"kind": "grid", "rows": r, "cols": c}, "grow": {"inside_prob": 0.1, "snap_prob": 0.6}})
        r2, c2 = (r, c + 1) if c <= r else (r + 1, c)
        if 2 * r2 * c2 + r2 + c2 - n <= 4:
            gs.append({"seed": {"kind": "literal_grid", "rows": r2, "cols": c2}, "grow": {"inside_prob": 0.2}})
    gs += [
        {"seed": {"kind": "ring", "ring_scale": 1.0}, "grow": {"inside_prob": 1.0}},
        {"seed": {"kind": "ring", "ring_scale": 0.8}, "grow": {"inside_prob": 0.6}},
        {"seed": {"kind": "ring", "k": 4}, "grow": {"inside_prob": 0.15, "m_weights": [0.25, 0.5, 0.25]}},
        {"seed": {"kind": "ring", "k": 5}, "grow": {"inside_prob": 0.5, "snap_prob": 0.2}},
        {"seed": {"kind": "ring", "k": 3}, "grow": {"inside_prob": 0.3, "snap_prob": 0.6}},
        {"seed": {"kind": "pinwheel", "k": 3}, "grow": {"inside_prob": 0.6}},
        {"seed": {"kind": "ring"}, "sym": 2, "grow": {"inside_prob": 0.6}},
        {"seed": {"kind": "ring"}, "sym": 3, "grow": {"inside_prob": 0.6}},
        {"seed": {"kind": "ring", "ring_scale": 0.9}, "grow": {"inside_prob": 0.7}, "edits": ["split_edge", "toggle_junction"]},
        {"seed": {"kind": "ring", "k": 4}, "grow": {"inside_prob": 0.3, "target_area": 1.0}, "edits": ["move_fence", "split_edge"]},
    ]
    if n >= 12:
        r, c = grid_fit(n)
        gs.append({"seed": {"kind": "grid", "rows": r, "cols": c}, "grow": {"inside_prob": 0.2},
                   "edits": ["split_edge", "toggle_junction", "move_fence"]})
    if n >= 16:
        gs.append({"seed": {"kind": "pinwheel", "k": 4}, "grow": {"inside_prob": 0.7}})
        gs.append({"seed": {"kind": "ring"}, "sym": 4, "grow": {"inside_prob": 0.6}})
    return gs


# ---------------------------------------------------------------------------
# MAP-Elites archive
# ---------------------------------------------------------------------------

TJ_BINS = [0, 2, 5, 9, 14, 20, 30]


def feature_key(features: dict) -> tuple:
    tj = features.get("tjunctions", 0)
    tjb = sum(1 for b in TJ_BINS if tj >= b) - 1
    sym = features.get("sym", 1)
    return (int(features.get("faces", 0)), tjb, min(sym, 3))


@dataclass
class Elite:
    program: Program
    fitness: float
    area: float
    features: dict
    config: dict                   # verified configuration JSON (incl. verification block)
    evals: int = 1
    created: float = field(default_factory=time.time)


class Archive:
    def __init__(self):
        self.cells: dict[tuple, Elite] = {}

    def __len__(self):
        return len(self.cells)

    def try_insert(self, e: Elite) -> bool:
        k = feature_key(e.features)
        cur = self.cells.get(k)
        if cur is None or e.fitness > cur.fitness + 1e-12:
            self.cells[k] = e
            return True
        return False

    def best(self) -> Elite | None:
        return max(self.cells.values(), key=lambda e: e.fitness) if self.cells else None

    def sample(self, rng: random.Random, exclude=frozenset(), keep=None) -> Elite | None:
        """exclude: proposers whose elites must not be picked; keep(elite): only those it accepts."""
        el = [e for e in self.cells.values() if e.program.proposer not in exclude and (keep is None or keep(e))]
        if not el:
            return None
        if rng.random() < 0.5:
            # fitness-biased tournament
            cand = rng.sample(el, min(3, len(el)))
            return max(cand, key=lambda e: e.fitness)
        return rng.choice(el)          # uniform over cells: diversity


# ---------------------------------------------------------------------------
# mutation backends
# ---------------------------------------------------------------------------

EDIT_OPS = ["add_face", "remove_face", "split_edge", "toggle_junction", "move_fence"]


class MutationError(RuntimeError):
    pass


class Mutator:
    """Interface: produce a child Program from a parent (plus context)."""

    name = "base"

    def mutate(self, parent: Program, ctx: dict) -> Program:  # pragma: no cover - interface
        raise NotImplementedError


# defaults at the settings validated in EXPERIMENTS.md (each flag set to 0 / "top" turns its change off)
FACE_WEIGHT = 0.25         # share of the face-aware operator (worker flag --face-weight)
SEED_ELITES = 6            # library minima each n's archive starts from (--seed-elites)
VERIFY_POLICY = "worthy"   # which polished candidates go to the referee (--verify-policy)
REMOVE_EDIT_FIX = False    # remove_edit on an edit-free genome jitters instead of re-seeding (--remove-edit-fix)
TARGET_FACES = False       # face-short n seed and breed only from layouts with the target face count
                           # (--target-faces); see Worker._face_short
DROP_EXHAUSTED = False     # an n with every lineage blocked leaves the worker's schedule (--drop-exhausted)
TRIVIAL_BLOCK = "grid"     # "grid": block lineages while n's best is any grid answer; "ties": only when it
                           # ties the record (--trivial-block). Neither is A/B'd yet, so both keep old behaviour.

# Probability that a job injects a fresh hand-written seed program once the initial sweep is done.
# Measured over the 6 h run of 2026-09-12: seed-program evaluations won 2.2 run-bests per 1000,
# against 11.7 for `elite` and 14.2 for the face operator's relocate.
SEED_INJECT = 0.04


def target_faces(base: dict | None) -> int | None:
    """Fewest faces that could beat the record: every face holds at most area 1."""
    return int(math.floor(base["A"] + 1e-9)) + 1 if base else None


class RandomMutator(Mutator):
    """Structural genome mutations; no network, no LLM.

    The face-aware operator (share `face_weight`) edits the layout of one of n's own verified
    elites where it loses area (fence.builder.face_fill); its share is taken proportionally from
    the operators that rebuild a layout from its program (all but `elite`).
    """

    name = "random"
    BASE_OPS = [("jitter", 0.25), ("add_edit", 0.12), ("remove_edit", 0.05), ("seed", 0.08), ("symmetry", 0.08),
                ("elite", 0.36), ("fresh", 0.06)]
    # A mix re-weighted against measured run-best win rates failed its A/B (EXPERIMENTS.md change 5)
    # and was removed.

    def __init__(self, rng: random.Random, face_weight: float = FACE_WEIGHT,
                 remove_edit_fix: bool = REMOVE_EDIT_FIX):
        self.rng = rng
        self.face_weight = face_weight
        self.remove_edit_fix = remove_edit_fix
        self.ops = self.BASE_OPS

    def op_weights(self, has_elites: bool, face_ok: bool) -> list[tuple[str, float]]:
        ops = [(k, 0.0 if k == "elite" and not has_elites else w) for k, w in self.ops]
        fw = self.face_weight if face_ok else 0.0
        if fw <= 0:
            return ops
        regrow = sum(w for k, w in ops if k != "elite")
        s = max(0.0, regrow - fw) / regrow
        return [(k, w if k == "elite" else w * s) for k, w in ops] + [("face", fw)]

    def _face_edit(self, ctx: dict):
        """(elite, edited sketch, what face_fill did) for one of n's own elites (ctx["face_sources"]):
        the fittest with probability 1/2, else a random one, then the rest; None if none takes a
        face edit. An edit that reproduces the elite's topology is vetoed (it would only
        re-converge)."""
        r = self.rng
        own = list(ctx.get("face_sources", []))
        if r.random() < 0.5:
            own.sort(key=lambda e: -e.get("area", 0.0))
        else:
            r.shuffle(own)
        for src in own[:4]:
            sk = Bd.Sketch.from_json(src["sketch"])
            try:
                sig0 = Mn.canonical_form(sk.topology()).signature
                info = Bd.face_fill(sk, np.random.default_rng(r.randrange(2 ** 32)), ctx.get("target_faces"),
                                    reject=lambda t: Mn.canonical_form(t.topology()).signature == sig0)
            except (Bd.BuildError, G.InvalidTopology):
                continue
            if info is not None:
                return src, sk.to_json(), info
        return None

    def _jitter(self, g: dict):
        r = self.rng
        grow = g.setdefault("grow", {})
        key = r.choice(["target_area", "inside_prob", "snap_prob", "chord_tol", "m_weights", "greedy",
                        "len_jitter", "split_min_area", "ring_scale", "squash"])
        d = Bd.DEFAULT_GROW
        if key == "m_weights":
            w = np.array(grow.get("m_weights", d["m_weights"]), dtype=float)
            w = np.clip(w * np.exp(0.5 * np.array([r.gauss(0, 1) for _ in range(3)])), 0.02, None)
            grow["m_weights"] = [round(float(x), 3) for x in w / w.sum()]
        elif key in ("ring_scale", "squash"):
            seed = g.setdefault("seed", {"kind": "ring"})
            if seed.get("kind") == "ring":
                base = seed.get(key, 1.0)
                seed[key] = round(float(np.clip(base * math.exp(0.15 * r.gauss(0, 1)), 0.5, 1.6)), 3)
                if key == "ring_scale":
                    seed.pop("k", None)
        else:
            lo, hi = {"target_area": (0.4, 1.3), "inside_prob": (0.0, 1.0), "snap_prob": (0.0, 1.0),
                      "chord_tol": (0.1, 0.6), "greedy": (0.2, 1.0), "len_jitter": (0.0, 0.2),
                      "split_min_area": (0.8, 3.0)}[key]
            v = grow.get(key, d[key])
            grow[key] = round(float(np.clip(v + 0.15 * (hi - lo) * r.gauss(0, 1), lo, hi)), 3)
        return f"param:{key}"

    def mutate(self, parent: Program, ctx: dict) -> Program:
        r = self.rng
        n = ctx["n"]
        seeds = ctx.get("seeds") or seed_genomes(n)      # seed programs whose lineage may still be mutated
        g = copy.deepcopy(parent.genome) if parent.genome else copy.deepcopy(r.choice(seeds))
        proposer = parent.proposer
        elites = ctx.get("elite_sketches", [])
        face_ok = ctx.get("target_faces") is not None and bool(ctx.get("face_sources"))
        names, w = zip(*self.op_weights(bool(elites), face_ok))
        op = r.choices(names, weights=w)[0]
        elite_tag = "elite"
        if op == "face":
            got = self._face_edit(ctx)
            if got is not None:
                src, sketch, info = got
                g = {"seed": {"kind": "literal", "sketch": sketch},
                     "grow": copy.deepcopy((parent.genome or {}).get("grow", {})), "edits": [],
                     "origin_note": src["note"], "face_edit": info}
                if src.get("avoid_signature"):
                    g["avoid_signature"] = src["avoid_signature"]
                # a child of a lineage blocked at n starts its own lineage, so it can be evolved on
                return Program.from_genome(g, parent=parent.pid, generation=parent.generation + 1,
                                           origin=f"random:face:{info['move']}-{info['where']}:{src['note']}",
                                           proposer=None if src.get("blocked") else src.get("proposer") or parent.proposer)
            # no elite of n takes a face edit: re-seed from an elite, else another operator by weight
            rest = [(k, x) for k, x in zip(names, w) if k not in ("face", "elite") and x > 0]
            if elites:
                op, elite_tag = "elite", "elite-fb"
            else:
                op = r.choices(*zip(*rest))[0] if rest else "fresh"
        if op == "remove_edit" and not g.get("edits") and self.remove_edit_fix:
            # Nothing to remove. Without this the `elif op == "remove_edit" and g.get("edits")`
            # below misses, the chain falls through to its `else` branch and the draw silently
            # becomes `fresh` -- the worst-scoring operator, and not what any weight asked for.
            # Jitter the genome instead.
            op = "jitter"
        if op == "jitter":
            tag = self._jitter(g)
        elif op == "add_edit":
            e = r.choice(EDIT_OPS)
            g.setdefault("edits", []).append(e)
            g["edits"] = g["edits"][-6:]
            tag = f"edit+{e}"
        elif op == "remove_edit" and g.get("edits"):
            g["edits"].pop(r.randrange(len(g["edits"])))
            tag = "edit-"
        elif op == "seed":
            kind = r.choice(["ring", "ring", "grid", "pinwheel"] if n >= 12 else ["ring", "ring", "pinwheel"])
            if kind == "ring":
                g["seed"] = {"kind": "ring", "ring_scale": round(r.uniform(0.6, 1.3), 3)} if r.random() < 0.6 \
                    else {"kind": "ring", "k": r.randint(3, max(4, int(2 * math.sqrt(math.pi * Bd.a_est(n)))))}
            elif kind == "grid":
                rr, cc = grid_fit(n)
                g["seed"] = {"kind": "grid", "rows": rr, "cols": cc}
            else:
                g["seed"] = {"kind": "pinwheel", "k": r.choice([3, 3, 4, 5])}
            tag = f"seed:{kind}"
        elif op == "symmetry":
            # apply a symmetry group (C_m); m = 1 removes the constraint
            g["sym"] = r.choice([1, 1, 2, 3, 4, 5, 6])
            tag = f"sym:{g['sym']}"
        elif op == "elite":
            src = r.choice(elites)
            avoid = src.get("avoid_signature")          # a library minimum: never just re-optimize it
            g = {"seed": {"kind": "literal", "sketch": src["sketch"]},
                 "grow": copy.deepcopy((parent.genome or {}).get("grow", {})),
                 "edits": [r.choice(EDIT_OPS) for _ in range(r.choice([1, 1, 2, 2, 3] if avoid else [0, 1, 1, 2, 2, 3]))],
                 "origin_note": src["note"]}
            if avoid:
                g["avoid_signature"] = avoid
            tag = f"{elite_tag}:{src['note']}:{'+'.join(g['edits']) or 'reopt'}"
            proposer = src.get("proposer") or parent.proposer    # the child is built from the elite's layout
        else:
            g = copy.deepcopy(r.choice(seeds))
            tag = "fresh"
            proposer = None                                       # a seed program again: its own lineage
        return Program.from_genome(g, parent=parent.pid, origin=f"random:{tag}", generation=parent.generation + 1,
                                   proposer=proposer)


LLM_SYSTEM = """You are an expert in computational geometry and evolutionary program search.
You improve Python programs that propose fence topologies for Erich Friedman's
"Maximum Fence Area" problem: n straight fences of length exactly 1; every fence
endpoint must lie on another fence (its endpoint or its INTERIOR -- T-junctions are
legal and essential); fences may not cross; every bounded face must have area <= 1;
maximize the total bounded area. A downstream GPU optimizer fixes exact lengths and
geometry; your program only has to propose a good planar incidence structure."""

LLM_API_DOC = textwrap.dedent('''
    Builder API (module `B` = fence.builder), all names already in scope:
      build_from_genome(n, rng, genome) -> Sketch   # interpreter for genome dicts, see GENOME in the program
      B.ring(k, side=1.0, squash=1.0, rot=0.0) -> Sketch        # closed k-gon of fences
      B.grid(rows, cols) -> Sketch                               # unit-square grid
      B.pinwheel_in_ring(k, K, s=0.35) -> Sketch                  # k-fence pinwheel inside a K-ring
      B.grow_once(sk, rng, params=None, m_max=3, sym=1, face=None) -> bool
            # add a path of 1..3 fences between two boundary points of a face (outer face grows
            # the region, a bounded face gets split); params keys: see B.DEFAULT_GROW
      B.fit_count(sk, n, rng, grow_params, sym=1)                 # prune/grow to exactly n fences
      B.remove_fence(sk, rng) / B.toggle_junction(sk, rng) / B.reattach(sk, rng) -> bool
      sk.add_vertex(x, y) / sk.add_fence(a, b) / sk.add_pin(host_fence, t) / sk.topology()
      Sketch(P, fences, pins): P list of [x, y]; fences list of [a, b]; pins {vertex: host_fence}
    Rules: return a Sketch with exactly n fences that passes sk.topology() (connected, planar,
    every endpoint supported). Must be deterministic given rng and finish in < 20 s.
''')


class LLMMutator(Mutator):
    """Rewrite a proposer program with Claude, given its source and score history.

    Reads credentials the standard SDK way (ANTHROPIC_API_KEY or an `ant auth login`
    profile); this module never stores or logs credentials. Enabled with
    `--mutator llm`; NOT used by the default overnight run.
    """

    name = "llm"

    def __init__(self, rng: random.Random, model: str = "claude-opus-5", effort: str = "high",
                 max_tokens: int = 32000, fallback: Mutator | None = None):
        import anthropic  # imported lazily so the random backend has zero API dependency

        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=3, timeout=900.0)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.rng = rng
        self.fallback = fallback or RandomMutator(rng)

    def _prompt(self, parent: Program, ctx: dict) -> str:
        hist = ctx.get("history", [])[-25:]
        lines = [f"  gen={h['generation']} origin={h['origin'][:40]} area={h['area']} fitness={h['fitness']}"
                 for h in hist]
        base = ctx.get("baseline")
        return textwrap.dedent(f"""
            Target: n = {ctx['n']} fences. Best known area (Friedman's page): {base}.
            Our best verified area so far for this n: {ctx.get('best_area')}.

            Current program (fitness {ctx.get('parent_fitness')}):
            ```python
            {{src}}
            ```

            Recent evaluations of programs on this island (area = referee-verified total area):
            {{hist}}

            {LLM_API_DOC}
            Write an improved version of the program that is likely to reach a LARGER verified area.
            Think about which planar structures (cell layouts, T-junction patterns, pinwheels,
            symmetry) let unit fences enclose many faces of area exactly 1. Reply with ONE complete
            Python program in a single ```python fenced block defining propose_topology(n, rng).
        """).replace("{src}", parent.source).replace("{hist}", "\n".join(lines) or "  (none yet)")

    def mutate(self, parent: Program, ctx: dict) -> Program:
        an = self.anthropic
        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                system=LLM_SYSTEM,
                messages=[{"role": "user", "content": self._prompt(parent, ctx)}],
            ) as stream:
                msg = stream.get_final_message()
        except an.RateLimitError as exc:
            raise MutationError(f"rate limited: {exc}") from exc
        except an.APIStatusError as exc:
            raise MutationError(f"API error {exc.status_code}: {exc.message}") from exc
        except an.APIConnectionError as exc:
            raise MutationError(f"connection error: {exc}") from exc
        if msg.stop_reason == "refusal":
            raise MutationError("model declined the request")
        text = "".join(b.text for b in msg.content if b.type == "text")
        m = re.search(r"```python\s*\n(.*?)```", text, re.S)
        if not m:
            raise MutationError("no python code block in response")
        source = m.group(1)
        compile(source, "<llm-program>", "exec")   # syntax check only; execution is sandboxed
        return Program(pid=uuid.uuid4().hex[:10], source=source, genome=None, parent=parent.pid,
                       origin=f"llm:{self.model}", generation=parent.generation + 1, trusted=False,
                       proposer=parent.proposer)


def make_mutator(kind: str, rng: random.Random, face_weight: float = FACE_WEIGHT,
                 remove_edit_fix: bool = REMOVE_EDIT_FIX, **kw) -> Mutator:
    if kind == "random":
        return RandomMutator(rng, face_weight, remove_edit_fix)
    if kind == "llm":
        return LLMMutator(rng, **kw)
    raise ValueError(f"unknown mutator {kind!r}")


# ---------------------------------------------------------------------------
# per-n state
# ---------------------------------------------------------------------------

@dataclass
class NState:
    n: int
    baseline: dict | None
    islands: list[Archive]
    best: dict | None = None               # {"area", "config", "program", "status", "time"}
    evals: int = 0
    time_spent: float = 0.0
    fails: int = 0
    history: list = field(default_factory=list)
    seeded: int = 0
    # proposer -> {"best": best new result since its last stabilization, "stale": evaluations since}
    epochs: dict = field(default_factory=dict)
    # (proposer, topology signature) -> steered attempts on that topology that found nothing new
    barren: dict = field(default_factory=dict)
    # library minima this run's archive was seeded with (slim, as in Worker.known): known to every lineage
    seeded_minima: list = field(default_factory=list)

    @property
    def base_A(self) -> float:
        return self.baseline["A"] if self.baseline else max(Bd.a_est(self.n), 0.5)


LARGE_N = 32                # below-record n from here up get more time: the gaps to the records are largest there
LARGE_N_WEIGHT = 1.5
TRIVIAL_TIE_WEIGHT = 0.1    # remaining share of an n whose best already ties a grid-trivial record


def priority(n: int, base: dict | None) -> float:
    if base is None:
        return 0.5
    if n < 10:
        return 0.25
    if base["status"] != "best_known":
        return 0.4
    return LARGE_N_WEIGHT if n >= LARGE_N else 1.0


def trivial_tie(n: int, base: dict | None, best_area: float | None) -> bool:
    """n's best is a grid-trivial answer that already ties the record (n = 12, 17, 24, ...)."""
    return base is not None and is_grid_trivial(best_area, n) and classify(best_area, base) == "TIES"


def focus_ns(ns, baselines: dict, records: dict, k: int) -> dict[int, float]:
    """The k n of `ns` closest to their published record, as {n: record - best area} (best from
    the minima library). Left out: n whose record is a grid or other trivial answer (likely
    optimal), n already tying or beating their record, and n with no verified area yet."""
    gaps = {}
    for n in ns:
        b, a = baselines.get(n), records.get(n)
        if b is None or b["status"] == "trivial" or a is None or classify(a, b) in ("TIES", "BEATS"):
            continue
        gaps[n] = b["A"] - a
    return dict(sorted(sorted(gaps.items(), key=lambda kv: kv[1])[:k]))


def schedule_weight(n: int, base: dict | None, best_area: float | None) -> float:
    """Share of GPU time for n given its best area so far (this run or any earlier one)."""
    w = priority(n, base)
    return w * TRIVIAL_TIE_WEIGHT if trivial_tie(n, base, best_area) else w


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self, args):
        self.args = args
        self.gpu = args.gpu
        self.run_dir = Path(args.run_dir)
        self.wdir = self.run_dir / f"worker{self.gpu}"
        self.wdir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "best").mkdir(exist_ok=True)
        (self.run_dir / "improvements").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        self.log_path = self.run_dir / "logs" / f"worker{self.gpu}.jsonl"
        self.hb_path = self.run_dir / "logs" / f"worker{self.gpu}.heartbeat"
        self.ckpt_path = self.wdir / "checkpoint.pkl"
        self.deadline = float(args.deadline)
        self.baselines = load_baselines()
        self.ns = [int(x) for x in args.ns.split(",") if x]
        seed = int(args.seed) * 1000 + self.gpu
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)
        self.state: dict[int, NState] = {}
        self.pending: list[dict] = []
        self.cpu_wait = 0.0                                  # seconds the GPU loop waited for the CPU pool
        self.last_ckpt = time.time()
        self.stop = False
        self._neighbor_cache = {}
        self.lib = Mn.MinimaLibrary(args.minima_dir) if args.minima_dir else None
        self.known: dict[int, dict[str, list[dict]]] = {}   # n -> proposer -> its indexed minima N<n>.<k>
        self.trivial: dict[int, set[str]] = {}               # n -> proposers stabilized on the grid-trivial answer
        self.records: dict[int, float | None] = {}           # n -> best verified area over all runs
        self.roots: dict[str, tuple] = {}                    # proposer -> (genome, source) of its root program
        resumed = False
        if args.resume and self.ckpt_path.exists():
            resumed = self._load_checkpoint()
        for n in self.ns:
            if n not in self.state:
                self.state[n] = NState(n=n, baseline=self.baselines.get(n),
                                       islands=[Archive() for _ in range(args.islands)])
        self._load_library()
        self.face_short = self._face_short() if getattr(args, "target_faces", "off") == "on" else set()
        self.dropped: set[int] = set()      # n this worker stopped scheduling (--drop-exhausted)
        if not resumed and self.lib and args.seed_elites > 0:
            for n in self.ns:
                self._seed_archive(n, args.seed_elites)
        self.mutator = make_mutator(args.mutator, self.rng, face_weight=args.face_weight,
                                    remove_edit_fix=args.remove_edit_fix == "on")
        self.log({"event": "worker_start", "ns": self.ns, "resumed": resumed, "pid": os.getpid(),
                  "deadline": self.deadline, "mutator": args.mutator, "minima_dir": args.minima_dir})

    # -- logging / persistence ---------------------------------------------
    def log(self, rec: dict):
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "ts": round(time.time(), 3), "gpu": self.gpu, **rec}
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def heartbeat(self):
        # atomic: the supervisor must never read a truncated (empty) heartbeat file, which it
        # would take for a worker that has been silent since it was spawned
        self._atomic_write(self.hb_path, str(time.time()).encode())

    def _atomic_write(self, path: Path, data: bytes):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def checkpoint(self, reason="periodic"):
        import torch

        blob = {"state": self.state, "rng": self.rng.getstate(), "nprng": self.nprng.bit_generator.state,
                "torch_rng": torch.get_rng_state(), "ns": self.ns, "saved": time.time()}
        self._atomic_write(self.ckpt_path, pickle.dumps(blob))
        self.last_ckpt = time.time()
        self.log({"event": "checkpoint", "reason": reason, "cpu_wait_s": round(self.cpu_wait, 1),
                  "evals": {n: s.evals for n, s in self.state.items()},
                  "best": {n: (round(s.best["area"], 6) if s.best else None) for n, s in self.state.items()}})

    def _load_checkpoint(self) -> bool:
        import torch

        try:
            blob = pickle.loads(self.ckpt_path.read_bytes())
        except Exception as exc:  # noqa: BLE001
            self.log({"event": "checkpoint_load_failed", "error": repr(exc)})
            return False
        self.state = {n: s for n, s in blob["state"].items() if n in self.ns}
        for s in self.state.values():
            # checkpoints written before lineages and minima were tracked
            for name in ("epochs", "barren"):
                if name not in s.__dict__:
                    setattr(s, name, {})
            if "seeded_minima" not in s.__dict__:
                s.seeded_minima = []
            for isl in s.islands:
                for e in isl.cells.values():
                    if not e.program.proposer:
                        e.program.proposer = proposer_id(e.program.source)
        self.rng.setstate(blob["rng"])
        self.nprng.bit_generator.state = blob["nprng"]
        torch.set_rng_state(blob["torch_rng"])
        return True

    def _load_library(self):
        """Known minima, grid-trivial lineages and all-time records for this worker's n
        (re-read from disk on every start, so a resumed worker sees what it indexed)."""
        for n in self.ns:
            rec = self.lib.record(n) if self.lib else None
            self.known[n] = {p: [_slim(m) for m in ms] for p, ms in self.lib.minima(n).items()} if self.lib else {}
            self.trivial[n] = self.lib.trivial_proposers(n) if self.lib else set()
            self.records[n] = float(rec["area"]) if rec else None
        if self.lib:
            self.log({"event": "library_loaded", "dir": str(self.lib.root),
                      "minima": {n: sum(map(len, self.known[n].values())) for n in self.ns},
                      "trivial_lineages": {n: len(self.trivial[n]) for n in self.ns},
                      "records": self.records})

    def _face_short(self) -> set[int]:
        """n whose all-time record has fewer faces than target_faces: floor(record) + 1, the fewest
        that could beat the published record. Their best layouts are almost all the integer grid
        answer at F - 1, so seeding by area fills the archive with the grid and nothing else --
        on the 6 h run of 2026-09-13 every one of the 11 such n was seeded with six grid layouts
        and not one at the target face count, although each had 20-159 of them in the library.
        Fixed for the run: if a target-F layout becomes n's best, breeding from target-F is right."""
        out = set()
        for n in self.ns:
            rec = self.lib.record(n) if self.lib else None
            t = target_faces(self.state[n].baseline)
            if rec is not None and t is not None and rec["config"]["verification"]["n_faces"] < t:
                out.add(n)
        self.log({"event": "face_short", "ns": sorted(out)})
        return out

    def _faces_ok(self, n: int):
        """Predicate on an Elite: may it seed or parent a child at n? (Always, unless n is face-short
        under --target-faces, where only layouts with at least the target face count may -- and not a
        grid-trivial area either: a grid plus a near-zero sliver face has the target face count and
        the grid's integer area, and is the grid answer in all but name.) Such a layout is never the
        grid-trivial answer, which is what lets make_job exempt it from the lineage block."""
        if n not in self.face_short:
            return None
        t = target_faces(self.state[n].baseline)
        return lambda e: int(e.features.get("faces", 0)) >= t and not is_grid_trivial(e.area, n)

    def _seed_archive(self, n: int, k: int):
        """Start n's islands from its k best distinct library minima (the all-time record first),
        spread over the islands, so the run continues where earlier ones stopped. Their programs
        replay the minimum's layout and carry its topology signature, which no child may re-create:
        their elite children get at least one edit and _propose re-rolls or drops the rest. The
        seeded minima are known minima of every lineage at n (_known_minima), so no child is
        steered back onto one of them either."""
        st = self.state[n]
        placed = []
        docs = self.lib.best(n, 4 * k)
        if n in self.face_short:
            t = target_faces(st.baseline)
            docs = [d for d in self.lib.best(n, 10 ** 6) if d["config"]["verification"]["n_faces"] >= t
                    and not is_grid_trivial(float(d["area"]), n)][: 4 * k]
        for doc in docs:
            if len(placed) >= k:
                break
            cfg = doc["config"]
            ver = cfg["verification"]
            segs = np.array([[float(v) for v in s] for s in cfg["segments"]])
            feats = {"faces": ver["n_faces"], "tjunctions": ver["n_tjunctions"],
                     "sym": G.rotational_symmetry_order(segs, tol=1e-6)}
            genome = {"seed": {"kind": "literal", "sketch": _config_to_sketch(cfg)}, "avoid_signature": doc["signature"]}
            prog = Program.from_genome(genome, origin=f"library:{doc['name']}", proposer=doc["proposer"])
            area = float(doc["area"])
            if st.islands[len(placed) % len(st.islands)].try_insert(Elite(prog, area / st.base_A, area, feats, cfg)):
                placed.append([doc["name"], round(area, 6)])
                topo, X = G.config_from_json(cfg)
                form = Mn.canonical_form(topo)
                st.seeded_minima.append({"name": f"seeded:{doc['name']}", "k": -1, "area": area,
                                         "signature": form.signature, "canon_coords": X[form.orders[0]].tolist()})
        if placed:
            st.seeded = len(seed_genomes(n))    # skip the initial sweep of the hand-written seed programs
        self.log({"event": "archive_seeded", "n": n, "elites": placed})

    # -- scheduling ----------------------------------------------------------
    def _n_best_area(self, n: int) -> float | None:
        """Best verified area for n known to this worker: this run's, or the all-time record."""
        st = self.state[n]
        vals = [a for a in (st.best["area"] if st.best else None, self.records.get(n)) if a is not None]
        return max(vals) if vals else None

    def _blocked(self, n: int) -> set[str]:
        """Lineages not to mutate at n: while n's best is the grid-trivial answer, a proposer
        whose search already stabilized on it would only re-derive it.

        Under --trivial-block ties, only while that grid answer also ties the record. When the
        record is above the grid (n = 22, 27, 29, 36, 38, 41, 45, 47, 50 on 2026-09-20, and now
        17, 24, 31, 40, 49 too), blocking every lineage that reached the grid shuts the search
        down exactly where a new structure is needed: n=41 had 30 blocked lineages and n=50 27.
        """
        best = self._n_best_area(n)
        if not is_grid_trivial(best, n):
            return set()
        if getattr(self.args, "trivial_block", TRIVIAL_BLOCK) == "ties" and \
                not trivial_tie(n, self.state[n].baseline, best):
            return set()
        return self.trivial.get(n, set())

    def pick_n(self) -> int:
        """Fair share of wall time weighted by schedule_weight, with a little randomness. An n
        whose best is within 5% of the record gets up to 2x share: a small gain could beat it
        there (not for a grid-trivial tie, which only a whole new structure can improve on)."""
        def score(n):
            s = self.state[n]
            best = self._n_best_area(n)
            w = schedule_weight(n, s.baseline, best)
            if best is not None and s.baseline and not trivial_tie(n, s.baseline, best):
                gap = max(0.0, (s.baseline["A"] - best) / s.baseline["A"])
                w *= 1.0 + max(0.0, 1.0 - gap / 0.05)
            return (s.time_spent + 5.0) / w * (0.85 + 0.3 * self.rng.random())
        live = [n for n in self.ns if n not in self.dropped] or self.ns
        return min(live, key=score)

    def _elite_sketches(self, n: int, with_blocked: bool = False) -> list[dict]:
        """Verified configurations usable as literal seeds: own archive + best of n-1, n, n+1
        (the latter possibly found by another GPU; read from run_dir/best). Elites of
        lineages blocked at n are left out, or with_blocked: kept and flagged "blocked"."""
        out = []
        st = self.state[n]
        blocked = self._blocked(n)
        ok = self._faces_ok(n)
        for isl in st.islands:
            for e in sorted((c for c in isl.cells.values() if ok is None or ok(c)), key=lambda e: -e.fitness)[:4]:
                if with_blocked or e.program.proposer not in blocked:
                    out.append({"sketch": _config_to_sketch(e.config), "note": f"n{n}:{e.area:.4f}",
                                "proposer": e.program.proposer, "n": n, "area": e.area,
                                "blocked": e.program.proposer in blocked,
                                "avoid_signature": (e.program.genome or {}).get("avoid_signature")})
        # other n's bests, and n's own saved best, can have any face count: a face-short n under
        # --target-faces takes its sources from its own target-F elites only
        for m in (() if ok is not None else (n - 1, n + 1, n)):
            p = self.run_dir / "best" / f"n{m:03d}.json"
            now = time.time()
            cached = self._neighbor_cache.get(m)
            if not (cached and now - cached[0] < 60):
                item = None
                if p.exists():
                    try:
                        cfg = json.loads(p.read_text())
                        a = float(cfg["verification"]["total_area"])
                        item = {"sketch": _config_to_sketch(cfg), "proposer": cfg.get("proposer"),
                                "note": f"n{m}:{a:.4f}", "n": m, "area": a}
                    except Exception:  # noqa: BLE001
                        item = None
                cached = self._neighbor_cache[m] = (now, item)
            if cached[1] and (with_blocked or cached[1]["proposer"] not in blocked):
                out.append({**cached[1], "blocked": cached[1]["proposer"] in blocked})
        return out

    # -- one evaluation, split into GPU part + asynchronous CPU parts --------------
    def make_job(self, n: int) -> dict | None:
        st = self.state[n]
        isl_idx = (st.evals + len(self.pending)) % len(st.islands)
        isl = st.islands[isl_idx]
        blocked = self._blocked(n)
        seeds = [Program.from_genome(g, origin=f"seed:{k}") for k, g in enumerate(seed_genomes(n))]
        open_seeds = [p for p in seeds if p.proposer not in blocked]
        child = None
        # every island first evaluates the hand-written seed programs (spread over islands),
        # later a fresh seed program is injected now and then to keep diversity; seeds of
        # blocked lineages are skipped
        if open_seeds and (st.seeded < len(seeds) or len(isl) == 0 or self.rng.random() < SEED_INJECT):
            while child is None:
                cand = seeds[st.seeded % len(seeds)]
                st.seeded += 1
                if cand.proposer not in blocked:
                    child = cand
        if child is None:
            ok = self._faces_ok(n)
            # The block stops a lineage that stabilized on the grid-trivial answer from re-deriving it.
            # It is per lineage, so it also bars that lineage's other minima -- at a face-short n, the
            # very target-F layouts --target-faces breeds from (n=38 got no evaluation at all in 5 h
            # because of it). Those layouts are never the grid answer (see _faces_ok), so under
            # --target-faces they are exempt; every other parent is still blocked as before.
            excl = frozenset() if ok is not None else blocked
            parent_e = isl.sample(self.rng, exclude=excl, keep=ok)
            for other in st.islands:           # this island may hold only blocked lineages
                if parent_e is not None:
                    break
                parent_e = other.sample(self.rng, exclude=excl, keep=ok)
            if parent_e is None and not open_seeds:
                st.time_spent += 30.0          # nothing left to evolve at n: let the scheduler move on
                self.log({"event": "n_exhausted", "n": n, "blocked": sorted(blocked)})
                if getattr(self.args, "drop_exhausted", "off") == "on":
                    # nothing at n can unblock while n is not evaluated, so the scheduler would come
                    # back every ~15 s for nothing (266 such events for n=50, 0 evals in 1.37 h)
                    self.dropped.add(n)
                    self.log({"event": "n_dropped", "n": n})
                return None
            parent = parent_e.program if parent_e else self.rng.choice(open_seeds)
            # the face-aware operator may also edit elites of lineages blocked at n: it changes their
            # layout where it loses area, so it cannot re-derive the grid-trivial answer
            sketches = self._elite_sketches(n, with_blocked=self.args.face_weight > 0 or ok is not None)
            ctx = {"n": n, "elite_sketches": [e for e in sketches if ok is not None or not e["blocked"]],
                   "face_sources": [e for e in sketches if e["n"] == n], "history": st.history,
                   "seeds": [p.genome for p in open_seeds], "target_faces": target_faces(st.baseline),
                   "baseline": st.baseline["display"] if st.baseline else None,
                   "best_area": st.best["area"] if st.best else None,
                   "parent_fitness": parent_e.fitness if parent_e else None}
            try:
                child = self.mutator.mutate(parent, ctx)
            except MutationError as exc:
                self.log({"event": "mutation_error", "n": n, "error": str(exc)[:300]})
                child = RandomMutator(self.rng, self.args.face_weight,
                                      self.args.remove_edit_fix == "on").mutate(parent, ctx)
        if child.proposer == proposer_id(child.source):
            self.roots.setdefault(child.proposer, (child.genome, child.source))
        seed = int(self.nprng.integers(2 ** 31))
        t0 = time.time()
        try:
            child, sk, topo, form, known, rerolls = self._propose(n, child, seed)
        except Exception as exc:  # noqa: BLE001 -- a broken program is just an unfit program
            st.fails += 1
            st.time_spent += time.time() - t0
            self.log({"event": "propose_failed", "n": n, "program": child.pid, "origin": child.origin,
                      "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            return None
        return {"n": n, "island": isl_idx, "program": child, "sketch": sk, "topo": topo, "seed": seed,
                "sig": form.signature if form else None, "known": known, "rerolls": rerolls,
                "avoid": Mn.avoid_images(form, known) if known else None,
                "t0": t0, "t_build": time.time() - t0}

    # -- steering away from known minima ---------------------------------------------
    @staticmethod
    def _build(n: int, prog: Program, seed: int):
        sk = prog.propose(n, seed)
        topo = sk.topology()
        if topo.n != n:
            raise Bd.BuildError(f"program returned {topo.n} fences")
        return sk, topo

    def _known_minima(self, n: int, proposer: str, sig: str | None) -> list[dict]:
        """Minima a result on topology `sig` must not be: the proposer's own indexed minima and the
        library minima n's archive was seeded with (of any lineage), each once."""
        if sig is None:
            return []
        own = [m for m in self.known[n].get(proposer, []) if m["signature"] == sig]
        return own + [m for m in self.state[n].seeded_minima if m["signature"] == sig
                      and not any(Mn.same_minimum(o, sig, m["area"]) for o in own)]

    def _known_on(self, n: int, proposer: str, topo: G.Topology):
        """(canonical form of topo, the known minima on that same topology: _known_minima)."""
        try:
            form = Mn.canonical_form(topo)
        except G.InvalidTopology:
            return None, []
        return form, self._known_minima(n, proposer, form.signature)

    def _exhausted(self, n: int, proposer: str, sig: str, known: list) -> bool:
        cap = self.args.topology_cap
        return cap > 0 and (len(known) >= cap or self.state[n].barren.get((proposer, sig), 0) >= 2)

    def _propose(self, n: int, child: Program, seed: int):
        """Build the child's sketch and find its proposer's known minima on that topology.

        A topology the proposer has exhausted, or the library minimum a child of a seeded elite
        must not re-create (genome["avoid_signature"]), gets up to two re-proposals (a new rng
        seed, plus one more structural edit for genome programs) before any GPU time is spent on
        it; a child still re-creating that minimum is dropped (BuildError). Otherwise the GPU
        stage is steered away from the known minima (job["avoid"]).
        """
        sk, topo = self._build(n, child, seed)
        form, known = self._known_on(n, child.proposer, topo)
        avoid = (child.genome or {}).get("avoid_signature")
        rerolls = 0
        while form is not None and rerolls < 2 and (
                form.signature == avoid or self._exhausted(n, child.proposer, form.signature, known)):
            rerolls += 1
            cand = child
            if child.genome is not None:
                g = copy.deepcopy(child.genome)
                g.setdefault("edits", []).append(self.rng.choice(EDIT_OPS))
                cand = Program.from_genome(g, parent=child.parent, origin=f"{child.origin}+novel",
                                           generation=child.generation, proposer=child.proposer)
            try:
                sk2, topo2 = self._build(n, cand, int(self.nprng.integers(2 ** 31)))
            except Exception:  # noqa: BLE001 -- keep the previous proposal
                continue
            child, sk, topo = cand, sk2, topo2
            form, known = self._known_on(n, child.proposer, topo)
        if avoid is not None and form is not None and form.signature == avoid:
            raise Bd.BuildError("child re-creates the library minimum it was seeded from")
        return child, sk, topo, form, known, rerolls

    def run_gpu(self, job: dict, cfg: O.OptConfig):
        t = time.time()
        Zs, As, stats = O.adam_stage(job["topo"], job["sketch"].coords(), cfg, device="cuda",
                                     seed=job["seed"], log=self.log, avoid=job.get("avoid"))
        job["adam"] = stats
        job["t_gpu"] = time.time() - t
        job["cfg"] = cfg
        tj = job["topo"].to_json()
        job["topo_json"] = tj
        job["polish"] = [self.pool.apply_async(O.polish, (tj, z, cfg.polish_iters)) for z in Zs]
        job["stage"] = "polish"
        job["t_submit"] = time.time()

    def drain(self, block=False, max_wait=None):
        """Advance pending jobs whose CPU work finished. Returns number finalized.
        block=False: one non-blocking pass; "one": wait until one job finishes; True: wait for all."""
        done = 0
        t_start = time.time()
        while True:
            progressed = False
            for job in list(self.pending):
                if job["stage"] == "polish" and all(f.ready() for f in job["polish"]):
                    res = []
                    for f in job["polish"]:
                        try:
                            res.append(f.get(0))
                        except Exception as exc:  # noqa: BLE001
                            res.append({"ok": False, "error": repr(exc)})
                    good = [p for p in res if p.get("ok")]
                    # results that polished back into a known minimum of this topology are not new
                    # (looked up now, not at proposal time: the proposer may have stabilized since)
                    known = [m["area"] for m in self._known_minima(job["n"], job["program"].proposer, job.get("sig"))]
                    uniq, job["excluded_known"] = O.distinct_candidates(res, known, Mn.MINIMUM_EPS)
                    job["polished"] = [round(p["area"], 8) for p in uniq[:5]]
                    job["n_polish_ok"] = len(good)
                    cfg = job["cfg"]
                    # the best distinct polished candidates go to the (expensive) 50-digit referee;
                    # nothing counts until it has passed there. --verify-policy worthy sends only
                    # those that would change something if they passed
                    tops = uniq[: cfg.verify_top]
                    if self.args.verify_policy == "worthy":
                        keep = [p for p in tops if self._worthy(job, p)]
                        job["verify_skipped"] = len(tops) - len(keep)
                        tops = keep
                    job["verify"] = [self.pool.apply_async(O.verify_candidate, (job["topo_json"], p["z"], job["n"], cfg.tol))
                                     for p in tops]
                    job["stage"] = "verify"
                    progressed = True
                elif job["stage"] == "verify" and all(f.ready() for f in job["verify"]):
                    vers = []
                    for f in job["verify"]:
                        try:
                            vers.append(f.get(0))
                        except Exception as exc:  # noqa: BLE001
                            vers.append({"valid": False, "summary": repr(exc)})
                    self.finalize(job, vers)
                    self.pending.remove(job)
                    done += 1
                    progressed = True
                elif time.time() - job["t_submit"] > self.args.job_timeout:
                    self.log({"event": "job_timeout", "n": job["n"], "program": job["program"].pid,
                              "stage": job["stage"]})
                    self.finalize(job, [])
                    self.pending.remove(job)
                    done += 1
                    progressed = True
                    self.timeouts += 1
                    if self.timeouts >= 8:
                        # hung polish/verify tasks hold pool slots forever: replace the pool
                        self.log({"event": "pool_recreate", "timeouts": self.timeouts})
                        self._new_pool(terminate_old=True)
                        for j in self.pending:
                            self.finalize(j, [])
                        self.pending.clear()
                        return done
            if not block or not self.pending:
                return done
            if max_wait is not None and time.time() - t_start > max_wait:
                return done
            if done and block == "one":
                return done
            if not progressed:
                time.sleep(0.05)

    def _worthy(self, job: dict, p: dict) -> bool:
        """Would this polished candidate change anything if the referee passed it? It must beat
        n's best of this run, the elite of the archive cell it would land in (faces and
        T-junctions are the topology's, symmetry is read off the polished coordinates: exactly
        what finalize files it under), or its proposer's best new result since the last
        stabilization (the minimum that would be indexed)."""
        n = job["n"]
        st = self.state[n]
        a = p["area"]
        if st.best is None or a > st.best["area"] + 1e-12:
            return True
        topo = job["topo"]
        X = np.asarray(p["z"][: 2 * topo.n_vertices]).reshape(-1, 2)
        feats = {"faces": len(topo.faces), "tjunctions": len(topo.pins),
                 "sym": G.rotational_symmetry_order(G.segments_of(X, topo.fences), tol=1e-6)}
        cell = st.islands[job["island"]].cells.get(feature_key(feats))
        if cell is None or a / st.base_A > cell.fitness + 1e-12:
            return True
        if self.lib:
            ep = st.epochs.get(job["program"].proposer)
            return ep is None or ep["best"] is None or a > ep["best"]["area"] + Mn.MINIMUM_EPS
        return False

    def finalize(self, job: dict, vers: list[dict]):
        n = job["n"]
        st = self.state[n]
        prog: Program = job["program"]
        valid = [v for v in vers if v.get("valid")]
        best = max(valid, key=lambda v: v["area"]) if valid else None
        elapsed = time.time() - job["t0"]
        st.evals += 1
        st.time_spent += elapsed
        area = best["area"] if best else None
        fitness = (area / st.base_A) if area else 0.0
        sig = job.get("sig")
        known = job.get("known", [])
        dup = self._dup_of(n, prog.proposer, sig, area)
        rec = {"event": "eval", "n": n, "island": job["island"], "program": prog.pid, "parent": prog.parent,
               "origin": prog.origin, "generation": prog.generation, "proposer": prog.proposer, "area": area,
               "fitness": round(fitness, 6), "verified": bool(best), "status": classify(area, st.baseline),
               "faces": len(job["topo"].faces), "tjunctions": len(job["topo"].pins), "signature": sig,
               "known_on_topology": len(known), "rerolls": job.get("rerolls", 0),
               "excluded_known": job.get("excluded_known", 0), "dup_of": dup,
               "from_library": bool((prog.genome or {}).get("avoid_signature")),
               "verify_skipped": job.get("verify_skipped", 0),
               "polished": job.get("polished"), "polish_ok": job.get("n_polish_ok"), "adam": job.get("adam"),
               "t_build": round(job["t_build"], 2), "t_gpu": round(job.get("t_gpu", 0), 2), "t_total": round(elapsed, 2),
               "rejected": [v.get("summary", "")[:160] for v in vers if not v.get("valid")][:2]}
        self.log(rec)
        st.history.append({"generation": prog.generation, "origin": prog.origin, "area": area,
                           "fitness": round(fitness, 5)})
        st.history = st.history[-300:]
        if known and (best is None or dup is not None):
            # a steered attempt on a known topology that found nothing new
            st.barren[(prog.proposer, sig)] = st.barren.get((prog.proposer, sig), 0) + 1
        self._track_epoch(n, prog, area, best["config"] if best else None, sig, dup)
        if not best:
            return
        cfg = best["config"]
        ver = cfg["verification"]
        segs = np.array([[float(v) for v in s] for s in cfg["segments"]])
        feats = {"faces": ver["n_faces"], "tjunctions": ver["n_tjunctions"],
                 "sym": G.rotational_symmetry_order(segs, tol=1e-6)}
        elite = Elite(program=prog, fitness=fitness, area=area, features=feats, config=cfg)
        inserted = st.islands[job["island"]].try_insert(elite)
        if inserted:
            self.log({"event": "archive_insert", "n": n, "island": job["island"], "cell": list(feature_key(feats)),
                      "area": area, "cells": len(st.islands[job["island"]])})
        if st.evals % self.args.migrate_every == 0 and len(st.islands) > 1:
            for i, isl in enumerate(st.islands):
                b = isl.best()
                if b:
                    st.islands[(i + 1) % len(st.islands)].try_insert(copy.copy(b))
            self.log({"event": "migration", "n": n})
        if st.best is None or area > st.best["area"] + 1e-12:
            self._new_best(n, prog, area, cfg, feats, sig)

    def _saved_best(self, n: int) -> dict | None:
        """This run's saved best configuration for n, re-read from disk (None if absent)."""
        try:
            cfg = json.loads((self.run_dir / "best" / f"n{n:03d}.json").read_text())
            float(cfg["verification"]["total_area"])
            return cfg
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _new_best(self, n: int, prog: Program, area: float, cfg: dict, feats: dict, sig: str | None):
        st = self.state[n]
        saved = self._saved_best(n)
        if saved is not None and area <= float(saved["verification"]["total_area"]) + 1e-12:
            # st.best came from a checkpoint (up to checkpoint_every old) when this worker was
            # restarted, and the saved file holds a better result found after it. Keep the
            # file and adopt its value, so the stale one is not compared against again.
            a = float(saved["verification"]["total_area"])
            st.best = {"area": a, "config": saved, "program": saved.get("found_by_program"), "source": None,
                       "origin": saved.get("program_origin"), "status": classify(a, st.baseline),
                       "time": time.time(), "features": None}
            self.log({"event": "best_kept", "n": n, "area": area, "saved_area": a})
            return
        status = classify(area, st.baseline)
        st.best = {"area": area, "config": cfg, "program": prog.pid, "source": prog.source, "origin": prog.origin,
                   "status": status, "time": time.time(), "features": feats}
        cfg_out = {**cfg, "status": status, "baseline": st.baseline, "found_by_program": prog.pid,
                   "program_origin": prog.origin, "proposer": prog.proposer, "signature": sig,
                   "found_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self._atomic_write(self.run_dir / "best" / f"n{n:03d}.json", json.dumps(cfg_out, indent=1).encode())
        (self.wdir / "programs").mkdir(exist_ok=True)
        (self.wdir / "programs" / f"n{n:03d}_best.py").write_text(prog.source)
        self.log({"event": "new_best", "n": n, "area": area, "status": status,
                  "baseline": st.baseline["A"] if st.baseline else None, "program": prog.pid})
        try:
            if self.lib and self.lib.update_record(n, cfg_out, {"run": self.run_dir.name, "proposer": prog.proposer}):
                self.records[n] = area
                self.log({"event": "new_record", "n": n, "area": area})
        except OSError as exc:
            self.log({"event": "library_error", "n": n, "error": repr(exc)[:300]})
        if status == "BEATS":
            name = f"n{n:03d}_{area:.8f}.json"
            self._atomic_write(self.run_dir / "improvements" / name, json.dumps(cfg_out, indent=1).encode())
            self.log({"event": "IMPROVEMENT", "n": n, "area": area, "baseline": st.baseline["A"],
                      "delta": area - st.baseline["A"], "verified": True,
                      "referee_dps": cfg["verification"]["dps"], "file": f"improvements/{name}"})

    # -- minima library -------------------------------------------------------------
    def _dup_of(self, n: int, proposer: str, sig: str | None, area: float | None) -> str | None:
        """Name of the known minimum (_known_minima) this result re-converged to, if any."""
        if sig is None or area is None:
            return None
        return next((m["name"] for m in self._known_minima(n, proposer, sig) if Mn.same_minimum(m, sig, area)), None)

    def _track_epoch(self, n: int, prog: Program, area, cfg, sig, dup):
        """Detect stabilization: a proposer's best new result at n that its next stable_after
        evaluations at n did not improve on (by more than MINIMUM_EPS). Results that are
        already indexed minima never count as new; failed evaluations count as not improving."""
        if not self.lib:
            return
        st = self.state[n]
        ep = st.epochs.setdefault(prog.proposer, {"best": None, "stale": 0})
        if area is not None and dup is None and (ep["best"] is None or area > ep["best"]["area"] + Mn.MINIMUM_EPS):
            ep["best"] = {"area": area, "config": cfg, "program": prog.pid, "origin": prog.origin,
                          "found_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            ep["stale"] = 0
            return
        ep["stale"] += 1
        if ep["best"] is not None and ep["stale"] >= self.args.stable_after:
            st.epochs[prog.proposer] = {"best": None, "stale": 0}
            try:
                self._stabilized(n, prog.proposer, ep["best"])
            except Exception as exc:  # noqa: BLE001 -- the library must never stop the search
                self.log({"event": "library_error", "n": n, "error": repr(exc)[:300],
                          "trace": traceback.format_exc()[-1500:]})

    def _stabilized(self, n: int, proposer: str, best: dict):
        meta = {"run": self.run_dir.name, "program": best["program"], "origin": best["origin"],
                "found_at": best["found_at"], "stable_after": self.args.stable_after}
        if is_grid_trivial(best["area"], n):
            self.lib.mark_trivial(proposer, n, best["config"], meta)
            self.trivial[n].add(proposer)
            self.log({"event": "minimum_trivial", "n": n, "proposer": proposer, "area": best["area"]})
        else:
            topo, coords = G.config_from_json(best["config"])
            form = Mn.canonical_form(topo)
            new, doc = self.lib.add_minimum(proposer, n, best["config"], form.signature, coords[form.orders[0]], meta)
            if new:
                self.known[n].setdefault(proposer, []).append(_slim(doc))
                self.log({"event": "minimum_indexed", "n": n, "proposer": proposer, "name": doc["name"],
                          "area": doc["area"], "signature": form.signature})
            else:
                self.log({"event": "minimum_duplicate", "n": n, "proposer": proposer, "of": doc["name"],
                          "area": best["area"]})
        self._register_proposer(n, proposer)

    def _register_proposer(self, n: int, proposer: str):
        """Save the lineage's root program next to its minima (roots are seed programs)."""
        root = self.roots.get(proposer)
        if root is None:           # e.g. created before a restart: look among the seed programs
            root = (None, None)
            for m in (n, n - 1, n + 1):
                for g in seed_genomes(m):
                    source = render_program(g)
                    if proposer_id(source) == proposer:
                        root = (g, source)
        self.lib.register_proposer(proposer, *root)

    # -- main loop -----------------------------------------------------------
    def _new_pool(self, terminate_old=False):
        if terminate_old and getattr(self, "pool", None) is not None:
            self.pool.terminate()
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(self.args.cpu_procs, initializer=O.pool_init, maxtasksperchild=200)
        self.timeouts = 0

    def run(self):
        import torch

        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.92)
        self._new_pool()
        signal.signal(signal.SIGTERM, self._on_term)
        signal.signal(signal.SIGINT, self._on_term)
        soft = self.deadline - self.args.drain_seconds
        gpu_errors_in_a_row = 0
        jobs_run = 0
        try:
            while not self.stop and time.time() < soft:
                self.heartbeat()
                self.drain(block=False)
                if len(self.pending) >= self.args.max_pending:
                    t_wait = time.time()
                    self.drain(block="one", max_wait=30)
                    self.cpu_wait += time.time() - t_wait      # the GPU idles, waiting for polish/referee
                    continue
                n = self.pick_n()
                cfg = self._cfg_for(n)
                job = self.make_job(n)
                if job is not None:
                    try:
                        self.run_gpu(job, cfg)
                        self.pending.append(job)
                        gpu_errors_in_a_row = 0
                        jobs_run += 1
                        if jobs_run % 50 == 0:
                            torch.cuda.empty_cache()     # keep the caching allocator from fragmenting
                    except Exception as exc:  # noqa: BLE001
                        self.state[n].fails += 1
                        gpu_errors_in_a_row += 1
                        self.log({"event": "gpu_error", "n": n, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                  "trace": traceback.format_exc()[-1500:], "in_a_row": gpu_errors_in_a_row})
                        torch.cuda.empty_cache()
                        if gpu_errors_in_a_row >= 5:
                            # the CUDA context may be wedged: exit so the supervisor restarts this
                            # worker from its checkpoint with a fresh context
                            self.log({"event": "gpu_wedged_exit"})
                            self.checkpoint(reason="gpu_wedged")
                            self.pool.terminate()
                            os._exit(17)
                if time.time() - self.last_ckpt > self.args.checkpoint_every:
                    self.checkpoint()
            # budget reached: finish outstanding CPU work, then save
            self.log({"event": "draining", "pending": len(self.pending)})
            self.heartbeat()
            self.drain(block=True, max_wait=max(5.0, self.deadline - time.time() - 10))
        finally:
            self.checkpoint(reason="final")
            self.pool.terminate()
            self.log({"event": "worker_exit", "evals": {n: s.evals for n, s in self.state.items()},
                      "best": {n: (s.best["area"] if s.best else None) for n, s in self.state.items()}})

    def _cfg_for(self, n: int) -> O.OptConfig:
        a = self.args
        return O.OptConfig(batch=a.batch or O.default_batch(n), steps=a.steps, topk=a.topk,
                           polish_iters=a.polish_iters, verify_top=a.verify_top, tol=a.tol,
                           mem_cap_gb=a.mem_cap_gb)

    def _on_term(self, signum, frame):
        self.stop = True
        self.log({"event": "signal", "signum": signum})


def _slim(minimum: dict) -> dict:
    """What steering needs of an indexed minimum (its full configuration stays on disk)."""
    return {k: minimum[k] for k in ("name", "k", "area", "signature", "canon_coords")}


def _config_to_sketch(cfg: dict) -> dict:
    topo, coords = G.config_from_json(cfg)
    return Bd.Sketch.from_config(topo, coords).to_json()


def worker_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="one GPU worker of the overnight search")
    ap.add_argument("--gpu", type=int, required=True, help="logical id (CUDA_VISIBLE_DEVICES is set by the supervisor)")
    ap.add_argument("--ns", required=True, help="comma-separated n values for this worker")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--deadline", type=float, required=True, help="unix time at which the worker must be done")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mutator", choices=["random", "llm"], default="random")
    ap.add_argument("--islands", type=int, default=3)
    ap.add_argument("--migrate-every", type=int, default=30)
    ap.add_argument("--batch", type=int, default=0, help="restarts per topology (0 = default by n)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--polish-iters", type=int, default=1200)
    ap.add_argument("--verify-top", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--mem-cap-gb", type=float, default=14.0)
    ap.add_argument("--cpu-procs", type=int, default=14)
    ap.add_argument("--max-pending", type=int, default=4)
    ap.add_argument("--job-timeout", type=float, default=600.0)
    ap.add_argument("--checkpoint-every", type=float, default=300.0)
    ap.add_argument("--drain-seconds", type=float, default=90.0)
    ap.add_argument("--minima-dir", default="", help="minima library shared across runs (empty: off)")
    ap.add_argument("--stable-after", type=int, default=6,
                    help="a proposer's best new result at n is a stabilized minimum after this many "
                         "evaluations of that proposer at n without improvement")
    ap.add_argument("--topology-cap", type=int, default=3,
                    help="re-propose a topology once its proposer has this many minima on it (0: never)")
    ap.add_argument("--face-weight", type=float, default=FACE_WEIGHT,
                    help="share of the face-aware mutation operator (0: off, the previous operator mix)")
    ap.add_argument("--verify-policy", choices=["top", "worthy"], default=VERIFY_POLICY,
                    help="top: the verify-top best polished candidates go to the referee; worthy: only those "
                         "that would raise n's best, their archive cell or their proposer's pending minimum")
    ap.add_argument("--seed-elites", type=int, default=SEED_ELITES,
                    help="start each n's archive from its best k distinct library minima (fresh starts; 0: empty)")
    ap.add_argument("--drop-exhausted", choices=["on", "off"], default="on" if DROP_EXHAUSTED else "off",
                    help="on: an n where every lineage is blocked leaves this worker's schedule for the run")
    ap.add_argument("--trivial-block", choices=["grid", "ties"], default=TRIVIAL_BLOCK,
                    help="grid: block lineages while n's best is a grid answer; ties: only when it ties the record")
    ap.add_argument("--target-faces", choices=["on", "off"], default="on" if TARGET_FACES else "off",
                    help="on: an n whose record has fewer faces than floor(record) + 1 seeds its archive and "
                         "picks parents only from layouts with at least that many faces; off: by area, as before")
    ap.add_argument("--remove-edit-fix", choices=["on", "off"], default="on" if REMOVE_EDIT_FIX else "off",
                    help="on: a remove_edit draw on a genome with no edits jitters it; off: the previous "
                         "behaviour, where that draw fell through the operator chain and became `fresh`")
    return ap


def worker_main(argv=None):
    args = worker_parser().parse_args(argv)
    Worker(args).run()


if __name__ == "__main__":
    worker_main()
