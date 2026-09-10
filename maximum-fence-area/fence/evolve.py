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
              (incl. elites of n-1 / n+1 found by other GPUs), parameter jitter.
    llm    -- asks Claude to rewrite the program given its source and score history.
  The overnight runner uses `random` (zero API dependency).

This module also contains the per-GPU worker (`python -m fence.evolve ...`),
which the supervisor in run_overnight.py launches once per GPU.
"""
from __future__ import annotations

import argparse
import copy
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


@dataclass
class Program:
    pid: str
    source: str
    genome: dict | None
    parent: str | None = None
    origin: str = "seed"
    generation: int = 0
    trusted: bool = True           # False for LLM-written code -> run sandboxed

    @staticmethod
    def from_genome(genome, parent=None, origin="seed", generation=0) -> "Program":
        return Program(pid=uuid.uuid4().hex[:10], source=render_program(genome), genome=copy.deepcopy(genome),
                       parent=parent, origin=origin, generation=generation)

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

    def sample(self, rng: random.Random) -> Elite | None:
        if not self.cells:
            return None
        el = list(self.cells.values())
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


class RandomMutator(Mutator):
    """Structural genome mutations; no network, no LLM."""

    name = "random"

    def __init__(self, rng: random.Random):
        self.rng = rng

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
        g = copy.deepcopy(parent.genome) if parent.genome else copy.deepcopy(r.choice(seed_genomes(n)))
        elites = ctx.get("elite_sketches", [])
        ops = [("jitter", 0.25), ("add_edit", 0.12), ("remove_edit", 0.05), ("seed", 0.08), ("symmetry", 0.08),
               ("elite", 0.36 if elites else 0.0), ("fresh", 0.06)]
        names, w = zip(*ops)
        op = r.choices(names, weights=w)[0]
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
            g = {"seed": {"kind": "literal", "sketch": src["sketch"]},
                 "grow": copy.deepcopy((parent.genome or {}).get("grow", {})),
                 "edits": [r.choice(EDIT_OPS) for _ in range(r.choice([0, 1, 1, 2, 2, 3]))],
                 "origin_note": src["note"]}
            tag = f"elite:{src['note']}:{'+'.join(g['edits']) or 'reopt'}"
        else:
            g = copy.deepcopy(r.choice(seed_genomes(n)))
            tag = "fresh"
        return Program.from_genome(g, parent=parent.pid, origin=f"random:{tag}", generation=parent.generation + 1)


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
                       origin=f"llm:{self.model}", generation=parent.generation + 1, trusted=False)


def make_mutator(kind: str, rng: random.Random, **kw) -> Mutator:
    if kind == "random":
        return RandomMutator(rng)
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

    @property
    def base_A(self) -> float:
        return self.baseline["A"] if self.baseline else max(Bd.a_est(self.n), 0.5)


def priority(n: int, base: dict | None) -> float:
    if base is None:
        return 0.5
    if n < 10:
        return 0.25
    return 1.0 if base["status"] == "best_known" else 0.4


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
        self.last_ckpt = time.time()
        self.stop = False
        self._neighbor_cache = {}
        resumed = False
        if args.resume and self.ckpt_path.exists():
            resumed = self._load_checkpoint()
        for n in self.ns:
            if n not in self.state:
                self.state[n] = NState(n=n, baseline=self.baselines.get(n),
                                       islands=[Archive() for _ in range(args.islands)])
        self.mutator = make_mutator(args.mutator, self.rng)
        self.log({"event": "worker_start", "ns": self.ns, "resumed": resumed, "pid": os.getpid(),
                  "deadline": self.deadline, "mutator": args.mutator})

    # -- logging / persistence ---------------------------------------------
    def log(self, rec: dict):
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "ts": round(time.time(), 3), "gpu": self.gpu, **rec}
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def heartbeat(self):
        self.hb_path.write_text(str(time.time()))

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
        self.log({"event": "checkpoint", "reason": reason,
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
        self.rng.setstate(blob["rng"])
        self.nprng.bit_generator.state = blob["nprng"]
        torch.set_rng_state(blob["torch_rng"])
        return True

    # -- scheduling ----------------------------------------------------------
    def pick_n(self) -> int:
        """Fair share of wall time weighted by priority, with a little randomness. An n whose
        best is within 5% of the record gets up to 2x share: a small gain could beat it there."""
        def score(n):
            s = self.state[n]
            w = priority(n, s.baseline)
            if s.best and s.baseline:
                gap = max(0.0, (s.baseline["A"] - s.best["area"]) / s.baseline["A"])
                w *= 1.0 + max(0.0, 1.0 - gap / 0.05)
            return (s.time_spent + 5.0) / w * (0.85 + 0.3 * self.rng.random())
        return min(self.ns, key=score)

    def _elite_sketches(self, n: int) -> list[dict]:
        """Verified configurations usable as literal seeds: own archive + best of n-1, n, n+1
        (the latter possibly found by another GPU; read from run_dir/best)."""
        out = []
        st = self.state[n]
        for isl in st.islands:
            for e in sorted(isl.cells.values(), key=lambda e: -e.fitness)[:4]:
                out.append({"sketch": _config_to_sketch(e.config), "note": f"n{n}:{e.area:.4f}"})
        for m in (n - 1, n + 1, n):
            p = self.run_dir / "best" / f"n{m:03d}.json"
            now = time.time()
            cached = self._neighbor_cache.get(m)
            if cached and now - cached[0] < 60:
                if cached[1]:
                    out.append(cached[1])
                continue
            item = None
            if p.exists():
                try:
                    cfg = json.loads(p.read_text())
                    item = {"sketch": _config_to_sketch(cfg), "note": f"n{m}:{float(cfg['verification']['total_area']):.4f}"}
                except Exception:  # noqa: BLE001
                    item = None
            self._neighbor_cache[m] = (now, item)
            if item:
                out.append(item)
        return out

    # -- one evaluation, split into GPU part + asynchronous CPU parts --------------
    def make_job(self, n: int) -> dict | None:
        st = self.state[n]
        isl_idx = (st.evals + len(self.pending)) % len(st.islands)
        isl = st.islands[isl_idx]
        seeds = seed_genomes(n)
        # every island first evaluates the hand-written seed programs (spread over islands),
        # later a fresh seed program is injected now and then to keep diversity
        if st.seeded < len(seeds) or len(isl) == 0 or self.rng.random() < 0.04:
            k = st.seeded % len(seeds)
            child = Program.from_genome(seeds[k], origin=f"seed:{k}")
            st.seeded += 1
        else:
            parent_e = isl.sample(self.rng)
            parent = parent_e.program if parent_e else Program.from_genome(self.rng.choice(seeds))
            ctx = {"n": n, "elite_sketches": self._elite_sketches(n), "history": st.history,
                   "baseline": st.baseline["display"] if st.baseline else None,
                   "best_area": st.best["area"] if st.best else None,
                   "parent_fitness": parent_e.fitness if parent_e else None}
            try:
                child = self.mutator.mutate(parent, ctx)
            except MutationError as exc:
                self.log({"event": "mutation_error", "n": n, "error": str(exc)[:300]})
                child = RandomMutator(self.rng).mutate(parent, ctx)
        seed = int(self.nprng.integers(2 ** 31))
        t0 = time.time()
        try:
            sk = child.propose(n, seed)
            topo = sk.topology()
            if topo.n != n:
                raise Bd.BuildError(f"program returned {topo.n} fences")
        except Exception as exc:  # noqa: BLE001 -- a broken program is just an unfit program
            st.fails += 1
            st.time_spent += time.time() - t0
            self.log({"event": "propose_failed", "n": n, "program": child.pid, "origin": child.origin,
                      "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            return None
        return {"n": n, "island": isl_idx, "program": child, "sketch": sk, "topo": topo, "seed": seed,
                "t0": t0, "t_build": time.time() - t0}

    def run_gpu(self, job: dict, cfg: O.OptConfig):
        t = time.time()
        Zs, As, stats = O.adam_stage(job["topo"], job["sketch"].coords(), cfg, device="cuda",
                                     seed=job["seed"], log=self.log)
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
                    good = sorted([p for p in res if p.get("ok")], key=lambda p: -p["area"])
                    uniq = []
                    for p in good:
                        if all(abs(p["area"] - q["area"]) > 1e-9 for q in uniq):
                            uniq.append(p)
                    job["polished"] = [round(p["area"], 8) for p in uniq[:5]]
                    job["n_polish_ok"] = len(good)
                    cfg = job["cfg"]
                    # the best distinct polished candidates go to the (expensive) 50-digit referee;
                    # nothing counts until it has passed there
                    tops = uniq[: cfg.verify_top]
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
        rec = {"event": "eval", "n": n, "island": job["island"], "program": prog.pid, "parent": prog.parent,
               "origin": prog.origin, "generation": prog.generation, "area": area, "fitness": round(fitness, 6),
               "verified": bool(best), "status": classify(area, st.baseline),
               "faces": len(job["topo"].faces), "tjunctions": len(job["topo"].pins),
               "polished": job.get("polished"), "polish_ok": job.get("n_polish_ok"), "adam": job.get("adam"),
               "t_build": round(job["t_build"], 2), "t_gpu": round(job.get("t_gpu", 0), 2), "t_total": round(elapsed, 2),
               "rejected": [v.get("summary", "")[:160] for v in vers if not v.get("valid")][:2]}
        self.log(rec)
        st.history.append({"generation": prog.generation, "origin": prog.origin, "area": area,
                           "fitness": round(fitness, 5)})
        st.history = st.history[-300:]
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
            status = classify(area, st.baseline)
            st.best = {"area": area, "config": cfg, "program": prog.pid, "source": prog.source, "origin": prog.origin,
                       "status": status, "time": time.time(), "features": feats}
            cfg_out = {**cfg, "status": status, "baseline": st.baseline, "found_by_program": prog.pid,
                       "program_origin": prog.origin, "found_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self._atomic_write(self.run_dir / "best" / f"n{n:03d}.json", json.dumps(cfg_out, indent=1).encode())
            (self.wdir / "programs").mkdir(exist_ok=True)
            (self.wdir / "programs" / f"n{n:03d}_best.py").write_text(prog.source)
            self.log({"event": "new_best", "n": n, "area": area, "status": status,
                      "baseline": st.baseline["A"] if st.baseline else None, "program": prog.pid})
            if status == "BEATS":
                name = f"n{n:03d}_{area:.8f}.json"
                self._atomic_write(self.run_dir / "improvements" / name, json.dumps(cfg_out, indent=1).encode())
                self.log({"event": "IMPROVEMENT", "n": n, "area": area, "baseline": st.baseline["A"],
                          "delta": area - st.baseline["A"], "verified": True, "referee_dps": ver["dps"],
                          "file": f"improvements/{name}"})

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
                    self.drain(block="one", max_wait=30)
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


def _config_to_sketch(cfg: dict) -> dict:
    topo, coords = G.config_from_json(cfg)
    return Bd.Sketch.from_config(topo, coords).to_json()


def worker_main(argv=None):
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
    args = ap.parse_args(argv)
    Worker(args).run()


if __name__ == "__main__":
    worker_main()
