"""Stage 1: enumerate candidate nets for a given n, reject, rank.

A *net* is the combinatorial half of a configuration (see `fence.rules`). This
module produces candidate nets, passes each through the enabled hard rules, and
ranks the survivors with the enabled heuristics. Every rejection goes into a
`Ledger` naming the rule that fired.

What "enumerate" means here
---------------------------
Two different things, and it is worth being precise about which is which:

* `feasible_counts` is a genuine exhaustive enumeration, but of the *counting
  envelope* only: every (F, T, V, b) that the counting rules (6, 7, 8, 9, 10, 11,
  14) allow for this n. It is cheap and complete, and it bounds what any net for
  this n can look like.

* The net generators below are a *constructive* enumeration: they build concrete
  nets from the families the heuristics say are worth looking at (minimum-perimeter
  polyomino clusters, those clusters plus an appendage, boundary T-junction
  variants, grids, rings, pinwheels, and the topologies the search already found).
  They are not exhaustive over all planar embeddings -- the number of those is
  astronomical by n = 12, let alone n = 50 -- and no claim of completeness is made
  or implied by this module. `feasible_counts` is what says which parts of the
  envelope the constructed nets actually cover, and `coverage()` reports the gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fence import builder as Bd
from fence import geometry as G
from fence import heuristics as H
from fence import minima as Mn
from fence import rules as R


# ---------------------------------------------------------------------------
# the counting envelope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CountSpec:
    """One (F, T, V, b) the counting rules allow for this n."""
    n: int
    F: int
    T: int
    V: int
    b: int

    @property
    def E(self) -> int:
        return self.n + self.T

    @property
    def J(self) -> int:
        return self.V - self.T

    def as_tuple(self) -> tuple:
        return (self.F, self.T, self.V, self.b)


def feasible_counts(n: int, min_faces: int | None = None, record_area: float | None = None,
                    enabled: set[str] | None = None) -> list[CountSpec]:
    """Every (F, T, V, b) consistent with the enabled counting rules.

    The constraints used, and which rule each comes from:
      6   V = 1 + (n + T) - F                     (Euler, connected)
      7   sum of face sizes = 2(n + T) - b        (the incidence identity)
      8   T <= 2n
      9   degrees >= 2, and a T-junction point has degree >= 3 (two host sides plus
          at least one stem), which turns the degree sum 2(n+T) >= 3T + 2(V-T) into
          T <= 2F - 2
      10  bridgeless, so the outer boundary is a simple cycle: b >= 3
      11  every bounded face has >= 3 graph-edges: 2(n + T) - b >= 3F; and the outer
          boundary, whose graph-edges are each at most 1 long, encloses the whole
          bounded area, so record_area <= b^2 / (4 pi)
      14  F >= ceil(record_area)
    """
    enabled = set(R.RULES) if enabled is None else set(enabled)
    out = []
    lo_F = 1
    if "record_faces" in enabled and min_faces:
        lo_F = max(lo_F, min_faces)
    hi_T = 2 * n if "t_bound" in enabled else 4 * n
    for F in range(lo_F, n + 2):
        for T in range(0, hi_T + 1):
            if "min_degree" in enabled and T > 2 * F - 2:
                break
            V = 1 + (n + T) - F
            if V < 3:
                continue
            if "min_degree" in enabled and 2 * (n + T) < T + 2 * V:
                continue
            for b in range(3, 2 * (n + T) + 1):
                if "face_size" in enabled and 2 * (n + T) - b < 3 * F:
                    break
                if "face_size" in enabled and record_area is not None and \
                        b * b / (4 * math.pi) < record_area - 1e-12:
                    continue
                if b > V:      # the outer boundary is a simple cycle of distinct vertices
                    continue
                out.append(CountSpec(n, F, T, V, b))
    return out


def envelope_summary(specs: list[CountSpec]) -> dict:
    if not specs:
        return {"count": 0}
    F = [s.F for s in specs]
    T = [s.T for s in specs]
    b = [s.b for s in specs]
    return {"count": len(specs), "F": (min(F), max(F)), "T": (min(T), max(T)), "b": (min(b), max(b))}


# ---------------------------------------------------------------------------
# polyomino cluster seeds (heuristic 3)
# ---------------------------------------------------------------------------

def min_perimeter(c: int) -> int:
    """Smallest perimeter of a polyomino with c cells (Harary-Harborth)."""
    return 2 * math.ceil(2 * math.sqrt(c))


def polyomino_shapes(c: int, limit: int = 8) -> list[list[tuple[int, int]]]:
    """Near-square clusters of c unit cells, minimum perimeter first.

    A p x q rectangle plus the leftover cells laid along one side is the standard
    minimum-perimeter family; the variants differ in which side takes the leftovers
    and whether they start at the near or far end, which changes where a later
    appendage can attach even when the perimeter is the same.
    """
    shapes, seen = [], set()
    cands = []
    for p in range(1, c + 1):
        q, r = divmod(c, p)
        q_full = q if r == 0 else q
        if q_full == 0:
            continue
        if abs(p - q_full) > max(3, int(math.sqrt(c))):
            continue
        cands.append((p, q_full, c - p * q_full))
    cands.sort(key=lambda t: (abs(t[0] - t[1]), t[2]))
    for p, q, extra in cands:
        for side in ("top", "right"):
            for flip in (False, True):
                cells = [(i, j) for i in range(p) for j in range(q)]
                for k in range(extra):
                    kk = (q - 1 - k) if flip else k
                    cells.append((p, kk) if side == "top" else (kk % p, q))
                if len(set(cells)) != c:
                    continue
                key = _canon_cells(cells)
                if key in seen:
                    continue
                seen.add(key)
                shapes.append(sorted(set(cells)))
                if len(shapes) >= limit:
                    return shapes
    return shapes


def _canon_cells(cells) -> tuple:
    """Shape key invariant under translation, the 4 rotations and reflection."""
    best = None
    s = set(cells)
    for k in range(4):
        s = {(j, -i) for i, j in s}
        for m in (s, {(-i, j) for i, j in s}):
            i0 = min(i for i, _ in m)
            j0 = min(j for _, j in m)
            key = tuple(sorted(((i - i0, j - j0) for i, j in m)))
            if best is None or key < best:
                best = key
    return best


def polyomino_sketch(cells) -> Bd.Sketch:
    """Unit-square cluster as a sketch: every unit edge bounding a cell is one fence."""
    cells = list(cells)
    edges = set()
    for i, j in cells:
        edges |= {((i, j), (i, j + 1)), ((i, j + 1), (i + 1, j + 1)),
                  ((i + 1, j), (i + 1, j + 1)), ((i, j), (i + 1, j))}
    pts = sorted({p for e in edges for p in e})
    vid = {p: k for k, p in enumerate(pts)}
    P = [[float(j), float(i)] for i, j in pts]       # (row, col) -> (x, y)
    F = [[vid[u], vid[v]] for u, v in sorted(edges)]
    sk = Bd.Sketch(P, F, {})
    X = sk.coords()
    sk.center = X.mean(axis=0)
    sk.P = (X - sk.center).tolist()
    sk.center = np.zeros(2)
    sk._dirty()
    return sk


def polyomino_fences(cells) -> int:
    """Number of unit edges a c-cell cluster uses: 2c + perimeter/2."""
    s = set(cells)
    shared = sum(1 for i, j in s if (i + 1, j) in s) + sum(1 for i, j in s if (i, j + 1) in s)
    return 4 * len(s) - shared


# ---------------------------------------------------------------------------
# candidate generators
# ---------------------------------------------------------------------------

@dataclass
class NetGen:
    id: str
    summary: str
    fn: object
    default: bool = True


GENERATORS: dict[str, NetGen] = {}


def generator(id: str, summary: str, default: bool = True):
    def deco(fn):
        GENERATORS[id] = NetGen(id, summary, fn, default)
        return fn
    return deco


@generator("polyomino", "minimum-perimeter clusters of unit squares that use exactly n fences")
def _gen_polyomino(n: int, rng, ctx: dict):
    for c in range(1, n):
        if 2 * c + math.ceil(2 * math.sqrt(c)) > n:
            break
        for cells in polyomino_shapes(c, ctx.get("shapes_per_c", 8)):
            if polyomino_fences(cells) != n:
                continue
            sk = polyomino_sketch(cells)
            yield sk, "polyomino", f"c={c} cells {len(cells)}"


@generator("polyomino_appendage",
           "a cluster plus cells of 1-3 fences that use up the leftover fences")
def _gen_polyomino_appendage(n: int, rng, ctx: dict):
    tries = ctx.get("appendage_tries", 3)
    for c in range(1, n):
        base = 2 * c + math.ceil(2 * math.sqrt(c))
        if base > n:
            break
        for cells in polyomino_shapes(c, ctx.get("shapes_per_c", 8)):
            used = polyomino_fences(cells)
            left = n - used
            if not (0 < left <= ctx.get("max_appendage", 8)):
                continue
            for _ in range(tries):
                sk = polyomino_sketch(cells)
                grow = {**Bd.DEFAULT_GROW, "inside_prob": 0.0, "snap_prob": 0.55,
                        "target_area": 0.95}
                try:
                    Bd.fit_count(sk, n, rng, grow, 1)
                except (Bd.BuildError, G.InvalidTopology):
                    continue
                yield sk, "polyomino+appendage", f"c={c} + {left} fences"


@generator("polyomino_tjunction",
           "a cluster with T-junction edits, weighted to the boundary (heuristic 4)")
def _gen_polyomino_t(n: int, rng, ctx: dict):
    k_max = ctx.get("t_edits", 3)
    for c in range(1, n):
        if 2 * c + math.ceil(2 * math.sqrt(c)) > n:
            break
        for cells in polyomino_shapes(c, max(3, ctx.get("shapes_per_c", 8) // 2)):
            for k in range(1, k_max + 1):
                sk = polyomino_sketch(cells)
                grow = {**Bd.DEFAULT_GROW, "inside_prob": 0.0, "snap_prob": 0.5}
                try:
                    if sk.n != n:
                        Bd.fit_count(sk, n, rng, grow, 1)
                    ok = True
                    for _ in range(k):
                        ok = _boundary_edit(sk, rng) and ok
                    if not ok:
                        continue
                    if sk.n != n:
                        Bd.fit_count(sk, n, rng, grow, 1)
                except (Bd.BuildError, G.InvalidTopology):
                    continue
                yield sk, "polyomino+t", f"c={c} + {k} boundary T-edit(s)"


def _boundary_edit(sk: Bd.Sketch, rng) -> bool:
    """One T-junction edit, retried until it lands on the outer boundary if it can.

    Heuristic 4 says T-junctions pay off at the boundary and when fitting leftover
    fences, not in the bulk, so an edit that only touches interior vertices is
    retried a few times before it is accepted anyway.
    """
    for attempt in range(6):
        snap = sk.copy()
        before = set(sk.pins)
        op = Bd.reattach if rng.random() < 0.6 else Bd.toggle_junction
        try:
            if not op(sk, rng):
                continue
        except (Bd.BuildError, G.InvalidTopology):
            sk.P, sk.fences, sk.pins, sk._topo = snap.P, snap.fences, snap.pins, snap._topo
            continue
        try:
            outer = set(sk.topology().outer)
        except G.InvalidTopology:
            sk.P, sk.fences, sk.pins, sk._topo = snap.P, snap.fences, snap.pins, snap._topo
            continue
        new = set(sk.pins) - before
        if not new or any(v in outer for v in new) or attempt >= 4:
            return True
        sk.P, sk.fences, sk.pins, sk._topo = snap.P, snap.fences, snap.pins, snap._topo
    return False


@generator("face_target",
           "drive the face count up to rule 14's bound by dissolving corner joints")
def _gen_face_target(n: int, rng, ctx: dict):
    """Nets that actually reach F >= ceil(A_record).

    F = n + 1 - J for a connected layout with J joints, so at fixed n the only way
    to gain a face is to lose a joint: `builder.face_fill` dissolves a two-fence
    corner, re-pins the surviving fence as a T-junction and puts the freed fence
    back as a one-fence cell. Starting from a cluster that is short of the target
    and dissolving until it is reached is the one generator here that aims at the
    face count directly, which is what rule 14 rejects nearly everything else for.
    """
    target = ctx.get("min_faces")
    if not target:
        return
    for c in range(1, n):
        if 2 * c + math.ceil(2 * math.sqrt(c)) > n:
            break
        for cells in polyomino_shapes(c, max(2, ctx.get("shapes_per_c", 8) // 2)):
            for _ in range(ctx.get("face_target_tries", 2)):
                sk = polyomino_sketch(cells)
                grow = {**Bd.DEFAULT_GROW, "inside_prob": 0.0, "snap_prob": 0.5}
                try:
                    if sk.n != n:
                        Bd.fit_count(sk, n, rng, grow, 1)
                except (Bd.BuildError, G.InvalidTopology):
                    continue
                got = 0
                for _ in range(2 * n):
                    try:
                        F = len(sk.topology().faces)
                    except G.InvalidTopology:
                        break
                    if F >= target:
                        break
                    try:
                        if Bd.face_fill(sk, rng, target_faces=target, dead=0.0) is None:
                            break
                    except (Bd.BuildError, G.InvalidTopology):
                        break
                    got += 1
                if got:
                    yield sk, "polyomino+t", f"c={c} + {got} dissolve(s) -> F>={target}"


@generator("builder_seeds", "the search's own seed families: grids, rings, pinwheels")
def _gen_builder_seeds(n: int, rng, ctx: dict):
    from fence.evolve import seed_genomes

    for k, g in enumerate(seed_genomes(n)):
        kind = g.get("seed", {}).get("kind", "ring")
        fam = {"grid": "grid", "literal_grid": "grid", "ring": "ring", "pinwheel": "pinwheel"}.get(kind, "?")
        for _ in range(ctx.get("seed_tries", 2)):
            try:
                sk = Bd.build_from_genome(n, rng, g)
            except (Bd.BuildError, G.InvalidTopology):
                continue
            yield sk, fam, f"seed:{k} {kind}"


@generator("library", "topologies the search already found, from the minima library")
def _gen_library(n: int, rng, ctx: dict):
    lib = ctx.get("library")
    if lib is None:
        return
    for m in lib.best(n, ctx.get("library_k", 12)):
        cfg = m.get("config")
        if not cfg:
            continue
        try:
            topo, X = G.config_from_json(cfg)
        except Exception:  # noqa: BLE001 -- a malformed library entry is just not a candidate
            continue
        sk = Bd.Sketch.from_config(topo, X)
        yield sk, "library", f"library area={m.get('area', float('nan')):.5f}"


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------

@dataclass
class RankedNet:
    cand: R.NetCandidate
    score: H.Score
    signature: str | None = None
    sketch: Bd.Sketch | None = None
    counts: CountSpec | None = None

    @property
    def n(self) -> int:
        return self.cand.n

    def row(self) -> str:
        s = self.cand.stats
        return (f"{self.score.total:.3f}  F={s.F:<3} T={s.T:<3} V={s.V:<3} b={s.b:<3} "
                f"{self.cand.family:<20} {self.cand.origin}")


@dataclass
class NetStageResult:
    n: int
    ranked: list[RankedNet]
    ledger: R.Ledger
    envelope: list[CountSpec] = field(default_factory=list)
    generated: dict = field(default_factory=dict)
    duplicates: int = 0                  # passed the rules but repeated a net already built
    distinct: int = 0                    # distinct nets before `keep` truncated the list
    reached: set = field(default_factory=set)   # (F, T, V, b) of every distinct net

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "kept": len(self.ranked),
            "distinct": self.distinct,
            "duplicates": self.duplicates,
            "envelope": envelope_summary(self.envelope),
            "generated_by_family": self.generated,
            "ledger": self.ledger.to_json(),
            "top": [{"score": round(r.score.total, 4), "family": r.cand.family,
                     "origin": r.cand.origin, "F": r.cand.stats.F, "T": r.cand.stats.T,
                     "V": r.cand.stats.V, "b": r.cand.stats.b,
                     "parts": {k: round(v, 3) for k, v in r.score.parts.items()}}
                    for r in self.ranked[:20]],
        }


def enumerate_nets(n: int, rng, ruleset: R.RuleSet | None = None,
                   hset: H.HeuristicSet | None = None, ctx: dict | None = None,
                   generators: list[str] | None = None, max_candidates: int = 4000,
                   keep: int = 50, dedup: bool = True) -> NetStageResult:
    """Generate, filter and rank candidate nets for one n."""
    ruleset = ruleset or R.RuleSet()
    hset = hset or H.HeuristicSet()
    ctx = dict(ctx or {})
    ctx.setdefault("min_faces", R.min_faces_for(ctx.get("record_area")))
    names = [g for g in (generators or [k for k, v in GENERATORS.items() if v.default])]
    ledger = R.Ledger()
    seen_sig: dict[str, RankedNet] = {}
    ranked: list[RankedNet] = []
    generated: dict[str, int] = {}
    duplicates = 0
    budget = max_candidates

    for name in names:
        gen = GENERATORS.get(name)
        if gen is None:
            raise ValueError(f"unknown net generator {name!r}")
        made = 0
        per = max(1, budget // max(1, len(names)))
        try:
            for sk, family, origin in gen.fn(n, rng, ctx):
                if made >= per:
                    break
                made += 1
                tag = f"{name}/{origin}"
                try:
                    topo = sk.topology()
                except (G.InvalidTopology, Bd.BuildError) as exc:
                    ledger.note_error(exc, tag)
                    continue
                cand = R.NetCandidate(n=n, topo=topo, coords=sk.coords(), family=family,
                                      origin=origin)
                rej = ruleset.check(cand, ctx)
                ledger.note(rej, tag)
                if rej is not None:
                    continue
                sig = None
                if dedup:
                    try:
                        sig = Mn.canonical_form(topo).signature
                    except Exception:  # noqa: BLE001 -- a signature is for dedup, not correctness
                        sig = None
                    if sig is not None and sig in seen_sig:
                        duplicates += 1       # legal, but a net an earlier generator already built
                        continue
                sc = hset.score(cand, ctx)
                rn = RankedNet(cand, sc, sig, sk,
                               CountSpec(n, cand.stats.F, cand.stats.T, cand.stats.V, cand.stats.b))
                if sig is not None:
                    seen_sig[sig] = rn
                ranked.append(rn)
        except (Bd.BuildError, G.InvalidTopology) as exc:
            ledger.note_error(exc, name)
        generated[name] = made

    ranked.sort(key=lambda r: -r.score.total)
    env = feasible_counts(n, ctx.get("min_faces"), ctx.get("record_area"), ruleset.enabled)
    reached = {(r.cand.stats.F, r.cand.stats.T, r.cand.stats.V, r.cand.stats.b) for r in ranked}
    return NetStageResult(n=n, ranked=ranked[:keep], ledger=ledger, envelope=env,
                          generated=generated, duplicates=duplicates, distinct=len(ranked),
                          reached=reached)


def coverage(res: NetStageResult) -> dict:
    """How much of the counting envelope the constructed nets reached.

    `nets` counts the distinct (F, T, V, b) of every net that passed the rules, before
    `keep` truncated the ranked list. Every one of them must lie inside the envelope --
    the envelope is exhaustive over the counting rules, so a net outside it would mean a
    rule and a generator disagree.
    """
    env = {s.as_tuple() for s in res.envelope}
    got = set(res.reached) or {(r.cand.stats.F, r.cand.stats.T, r.cand.stats.V, r.cand.stats.b)
                               for r in res.ranked}
    return {"envelope": len(env), "nets": len(got), "inside_envelope": len(got & env),
            "outside_envelope": sorted(got - env)[:5]}


def describe_generators() -> str:
    return "\n".join(f"  {g.id:<22} {g.summary}" + ("" if g.default else "  [off by default]")
                     for g in GENERATORS.values())
