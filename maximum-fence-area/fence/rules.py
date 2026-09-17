"""Hard rules for a candidate net, each a separate toggleable check.

A *net* here is the combinatorial half of a fence configuration: the planar
embedding (which fence ends meet where, which ends are pinned in the interior of
another fence) together with a sketch of coordinates that realizes it. Stage 1 of
the pipeline (`fence.nets`) enumerates nets, rejects the ones that break a rule in
this module, and ranks the survivors with `fence.heuristics`.

Every rule is registered separately with an id, the number it carries in the
problem statement, and a default on/off state, so a run can switch one off and
measure what it costs (`RuleSet.from_flags`). Every rejection is recorded in a
`Ledger` with the id of the rule that fired, so a run can report *why* candidates
died rather than only how many.

Terminology (as in the problem statement, not the optimizer's):
  fence       one unit-length segment; there are exactly n
  graph-edge  a piece of a fence between consecutive points where something
              touches it; a fence cut by k interior pins is k + 1 graph-edges
  T-junction  a fence end lying in the interior of another fence
  T           number of distinct T-junction *points* (pin vertices), so that the
              graph-edge count is E = n + T and Euler reads V - (n + T) + F = 1.
              `t_touches` counts fence *ends* sitting on interiors, which is what
              the bound T <= 2n is about; the two differ only when two ends land
              on the same interior point.
  V, F, b     vertices, bounded faces, graph-edges on the outer boundary

Rules that need coordinates (3, 4, 11's area half, 12) are skipped with an
`n/a` result when the candidate carries no geometry, and rules that only make
sense once the lengths are exactly 1 (1's length half, 4) are skipped unless the
candidate is flagged `exact`. Nothing here certifies a result: that is
`fence.referee`, which shares no code with this module.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from fence import geometry as G

# Tolerance for the float checks in this module. Sketch-level sanity, not
# certification: the referee works at 50 digits with its own tolerance.
RULE_EPS = 1e-6


# ---------------------------------------------------------------------------
# the candidate
# ---------------------------------------------------------------------------

@dataclass
class NetCandidate:
    """One candidate net: a topology, an optional sketch of coordinates, and provenance.

    `exact` marks a candidate whose fence lengths are already 1 (a verified or
    optimized configuration) rather than a sketch waiting for the optimizer.
    """
    n: int
    topo: G.Topology
    coords: np.ndarray | None = None
    family: str = "?"                 # generating family, used by heuristic 3
    origin: str = ""                  # human-readable provenance
    exact: bool = False
    meta: dict = field(default_factory=dict)
    _stats: "NetStats | None" = field(default=None, repr=False, compare=False)

    @property
    def stats(self) -> "NetStats":
        if self._stats is None:
            self._stats = NetStats.of(self)
        return self._stats


@dataclass
class NetStats:
    """Counters read off a candidate once, and shared by every rule and heuristic."""
    n: int
    V: int
    T: int                 # distinct T-junction points (pin vertices)
    t_touches: int         # fence ends sitting on another fence's interior
    E: int                 # graph-edges
    F: int                 # bounded faces
    b: int                 # graph-edges on the outer boundary
    degrees: dict[int, int]
    face_sizes: list[int]  # graph-edges around each bounded face
    areas: np.ndarray | None
    outer_area: float | None
    bridges: list[tuple[int, int]]
    n_components: int

    @staticmethod
    def of(cand: NetCandidate) -> "NetStats":
        topo = cand.topo
        edges = topo.edges()
        deg = Counter()
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        ends = Counter()
        for a, b in topo.fences:
            ends[int(a)] += 1
            ends[int(b)] += 1
        touches = sum(ends[v] for v in topo.pins)
        areas = outer_area = None
        if cand.coords is not None:
            X = np.asarray(cand.coords, dtype=float)
            areas = G.face_areas(X, topo.faces)
            outer_area = abs(G.polygon_signed_area(X[topo.outer])) if topo.outer else None
        return NetStats(
            n=cand.n, V=topo.n_vertices, T=len(topo.pins), t_touches=touches,
            E=len(edges), F=len(topo.faces), b=len(topo.outer),
            degrees=dict(deg), face_sizes=[len(f) for f in topo.faces],
            areas=areas, outer_area=outer_area,
            bridges=find_bridges(topo.n_vertices, edges),
            n_components=count_components(topo.n_vertices, edges),
        )


def count_components(V: int, edges) -> int:
    parent = list(range(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return len({find(u) for e in edges for u in e}) if edges else V


def find_bridges(V: int, edges) -> list[tuple[int, int]]:
    """Cut edges of the multigraph, by iterative DFS low-link.

    Parallel edges (two graph-edges between the same pair of vertices, which a
    2-gon face produces) are handled by tracking edge ids, not endpoints.
    """
    adj: dict[int, list[tuple[int, int]]] = {}
    for i, (u, v) in enumerate(edges):
        adj.setdefault(u, []).append((v, i))
        adj.setdefault(v, []).append((u, i))
    disc = {}
    low = {}
    out = []
    timer = 0
    for root in list(adj):
        if root in disc:
            continue
        stack = [(root, -1, iter(adj[root]))]
        disc[root] = low[root] = timer
        timer += 1
        while stack:
            u, pe, it = stack[-1]
            advanced = False
            for v, eid in it:
                if eid == pe:
                    continue
                if v in disc:
                    low[u] = min(low[u], disc[v])
                else:
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, eid, iter(adj[v])))
                    advanced = True
                    break
            if advanced:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                low[p] = min(low[p], low[u])
                if low[u] > disc[p]:
                    out.append((p, u))
    return out


# ---------------------------------------------------------------------------
# rule registry
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    id: str
    number: int             # the rule's number in the problem statement
    summary: str
    fn: object
    default: bool = True
    needs_coords: bool = False
    needs_exact: bool = False


RULES: dict[str, Rule] = {}


def rule(id: str, number: int, summary: str, default: bool = True,
         needs_coords: bool = False, needs_exact: bool = False):
    def deco(fn):
        RULES[id] = Rule(id, number, summary, fn, default, needs_coords, needs_exact)
        return fn
    return deco


# -- 1 -----------------------------------------------------------------------
@rule("unit_fences", 1, "exactly n fences, each a straight segment of length 1")
def _unit_fences(c: NetCandidate, ctx: dict) -> str | None:
    if c.topo.n != c.n:
        return f"{c.topo.n} fences, wanted {c.n}"
    if c.exact and c.coords is not None:
        X = np.asarray(c.coords, dtype=float)
        L = np.linalg.norm(X[c.topo.fences[:, 1]] - X[c.topo.fences[:, 0]], axis=1)
        k = int(np.argmax(np.abs(L - 1.0)))
        if abs(L[k] - 1.0) > ctx.get("eps", RULE_EPS):
            return f"fence {k} has length {L[k]:.9f}"
    return None


# -- 2 -----------------------------------------------------------------------
@rule("no_free_ends", 2, "every fence end lies on another fence (joint or T-junction)")
def _no_free_ends(c: NetCandidate, ctx: dict) -> str | None:
    ends = Counter()
    for a, b in c.topo.fences:
        ends[int(a)] += 1
        ends[int(b)] += 1
    for v, k in ends.items():
        if k < 2 and v not in c.topo.pins:
            return f"vertex {v} is a free end"
    for v, h in c.topo.pins.items():
        a, b = c.topo.fences[h]
        if v in (int(a), int(b)):
            return f"vertex {v} is pinned on its own fence {h}"
    return None


# -- 3 -----------------------------------------------------------------------
@rule("no_crossings", 3, "fences never cross and never overlap collinearly", needs_coords=True)
def _no_crossings(c: NetCandidate, ctx: dict) -> str | None:
    problems = G.check_embedding(np.asarray(c.coords, dtype=float), c.topo.fences,
                                 c.topo.pins, c.topo.chains)
    hits = [p for p in problems if any(k in p for k in ("touch/cross", "overlap", "duplicate edge"))]
    return hits[0] if hits else None


# -- 4 -----------------------------------------------------------------------
@rule("face_area_max", 4, "every bounded face has area <= 1", needs_coords=True, needs_exact=True)
def _face_area_max(c: NetCandidate, ctx: dict) -> str | None:
    a = c.stats.areas
    if a is None or not len(a):
        return None
    k = int(np.argmax(a))
    if a[k] > 1.0 + ctx.get("eps", RULE_EPS):
        return f"face {k} has area {a[k]:.9f} > 1"
    return None


# -- 5 -----------------------------------------------------------------------
@rule("objective", 5, "the objective (total bounded area) is defined: at least one bounded face")
def _objective(c: NetCandidate, ctx: dict) -> str | None:
    return None if c.stats.F >= 1 else "no bounded face, so the objective is 0"


# -- 6 -----------------------------------------------------------------------
@rule("euler", 6, "Euler: V - (n + T) + F = 1 for the connected planar graph")
def _euler(c: NetCandidate, ctx: dict) -> str | None:
    s = c.stats
    if s.n_components != 1:
        return f"{s.n_components} connected components"
    if s.E != s.n + s.T:
        return f"{s.E} graph-edges but n + T = {s.n + s.T}"
    lhs = s.V - (s.n + s.T) + s.F
    if lhs != 1:
        return f"V - (n+T) + F = {lhs}, not 1 (V={s.V}, n={s.n}, T={s.T}, F={s.F})"
    return None


# -- 7 -----------------------------------------------------------------------
@rule("incidence", 7, "sum of face sizes over bounded faces = 2(n + T) - b")
def _incidence(c: NetCandidate, ctx: dict) -> str | None:
    s = c.stats
    lhs = sum(s.face_sizes)
    rhs = 2 * (s.n + s.T) - s.b
    if lhs != rhs:
        return f"face-edge incidences {lhs} != 2(n+T) - b = {rhs}"
    return None


# -- 8 -----------------------------------------------------------------------
@rule("t_bound", 8, "T <= 2n: each fence has only two ends to give")
def _t_bound(c: NetCandidate, ctx: dict) -> str | None:
    s = c.stats
    if s.t_touches > 2 * s.n:
        return f"{s.t_touches} T-junction touches > 2n = {2 * s.n}"
    if s.T > 2 * s.n:
        return f"T = {s.T} > 2n = {2 * s.n}"
    return None


# -- 9 -----------------------------------------------------------------------
@rule("min_degree", 9, "every vertex has degree >= 2")
def _min_degree(c: NetCandidate, ctx: dict) -> str | None:
    for v, d in sorted(c.stats.degrees.items()):
        if d < 2:
            return f"vertex {v} has degree {d}"
    missing = [v for v in range(c.topo.n_vertices) if v not in c.stats.degrees]
    return f"vertex {missing[0]} is isolated" if missing else None


# -- 10 ----------------------------------------------------------------------
@rule("bridgeless", 10, "2-edge-connected: no cut edges, every fence borders a bounded face")
def _bridgeless(c: NetCandidate, ctx: dict) -> str | None:
    s = c.stats
    if s.bridges:
        u, v = s.bridges[0]
        return f"graph-edge ({u}, {v}) is a bridge ({len(s.bridges)} in all)"
    on_face = set()
    for cyc in c.topo.faces:
        for i in range(len(cyc)):
            u, w = cyc[i], cyc[(i + 1) % len(cyc)]
            on_face.add((u, w))
            on_face.add((w, u))
    for f, chain in enumerate(c.topo.chains):
        if not any((chain[k], chain[k + 1]) in on_face for k in range(len(chain) - 1)):
            return f"fence {f} borders no bounded face"
    return None


# -- 11 ----------------------------------------------------------------------
@rule("face_size", 11, "every bounded face has >= 3 graph-edges; area <= the isoperimetric cap")
def _face_size(c: NetCandidate, ctx: dict) -> str | None:
    s = c.stats
    for k, m in enumerate(s.face_sizes):
        if m < 3:
            return f"face {k} has {m} graph-edges"
    if c.exact and s.areas is not None:
        for k, (m, a) in enumerate(zip(s.face_sizes, s.areas)):
            cap = isoperimetric_cap(m)
            if a > cap + ctx.get("eps", RULE_EPS):
                return (f"face {k} has area {a:.9f} above the cap {cap:.9f} "
                        f"for {m} graph-edges")
    return None


def isoperimetric_cap(m: int) -> float:
    """Largest area a bounded face with m graph-edges can hold.

    Each graph-edge is at most 1 long, so the perimeter is at most m and the
    isoperimetric inequality P >= 2 sqrt(pi A) gives A <= m^2 / (4 pi). For small
    m the polygon bound is tighter: the largest m-gon with unit sides is the
    regular one, m / (4 tan(pi/m)). The face rule caps everything at 1.
    """
    if m < 3:
        return 0.0
    regular = m / (4.0 * math.tan(math.pi / m))
    return min(1.0, regular, m * m / (4.0 * math.pi))


# -- 12 ----------------------------------------------------------------------
@rule("t_straight", 12, "a touched fence stays straight through its T-junction (angles sum to 180)",
      needs_coords=True)
def _t_straight(c: NetCandidate, ctx: dict) -> str | None:
    X = np.asarray(c.coords, dtype=float)
    # a sketch is approximate by construction (its fence lengths are not 1 either), so the
    # straightness of a host is only held to sketch accuracy until the candidate is exact
    tol = ctx.get("angle_eps", 1e-6 if c.exact else 1e-4)
    for v, h in c.topo.pins.items():
        chain = c.topo.chains[h]
        try:
            i = chain.index(v)
        except ValueError:
            return f"pin {v} is not on the chain of its host fence {h}"
        if i == 0 or i == len(chain) - 1:
            return f"pin {v} sits at an end of its host fence {h}"
        d1 = X[v] - X[chain[i - 1]]
        d2 = X[chain[i + 1]] - X[v]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-12 or n2 < 1e-12:
            return f"host fence {h} has a zero-length piece at pin {v}"
        bend = abs(float(G.cross2(d1, d2))) / (n1 * n2)
        if bend > tol or float(np.dot(d1, d2)) <= 0:
            return f"host fence {h} bends by {math.degrees(math.asin(min(1.0, bend))):.3g} deg at pin {v}"
    return None


# -- 13 ----------------------------------------------------------------------
@rule("embedding", 13, "the rotation system traces a consistent planar face structure")
def _embedding(c: NetCandidate, ctx: dict) -> str | None:
    s = c.stats
    if not c.topo.outer:
        return "no outer boundary cycle"
    if s.n_components != 1:
        return f"{s.n_components} connected components"
    # every half-edge used exactly once across bounded faces + the outer cycle
    seen = Counter()
    for cyc in list(c.topo.faces) + [c.topo.outer]:
        for i in range(len(cyc)):
            seen[(cyc[i], cyc[(i + 1) % len(cyc)])] += 1
    dup = [h for h, k in seen.items() if k != 1]
    if dup:
        return f"half-edge {dup[0]} appears {seen[dup[0]]} times in the face structure"
    if len(seen) != 2 * s.E:
        return f"face structure covers {len(seen)} of {2 * s.E} half-edges"
    if c.coords is not None:
        X = np.asarray(c.coords, dtype=float)
        if any(G.polygon_signed_area(X[f]) <= 0 for f in c.topo.faces):
            return "a bounded face is not counter-clockwise"
        if G.polygon_signed_area(X[c.topo.outer]) >= 0:
            return "the outer cycle is not clockwise"
    return None


# -- 14 ----------------------------------------------------------------------
@rule("record_faces", 14, "F >= ceil(A_record): every face holds at most area 1")
def _record_faces(c: NetCandidate, ctx: dict) -> str | None:
    need = ctx.get("min_faces")
    if need is None:
        return None
    if c.stats.F < need:
        return f"{c.stats.F} faces, need >= {need} to reach the record"
    return None


# ---------------------------------------------------------------------------
# running the rules
# ---------------------------------------------------------------------------

@dataclass
class Rejection:
    rule: str
    number: int
    reason: str

    def __str__(self) -> str:
        return f"rule {self.number} ({self.rule}): {self.reason}"


class Ledger:
    """Counts of what rejected candidates, and a sample of the reasons."""

    def __init__(self, keep: int = 3):
        self.counts: Counter = Counter()
        self.samples: dict[str, list[str]] = {}
        self.accepted = 0
        self.errors = 0
        self.keep = keep

    def note(self, rej: Rejection | None, origin: str = "") -> None:
        if rej is None:
            self.accepted += 1
            return
        self.counts[rej.rule] += 1
        s = self.samples.setdefault(rej.rule, [])
        if len(s) < self.keep:
            s.append(f"{origin}: {rej.reason}" if origin else rej.reason)

    def note_error(self, exc: BaseException, origin: str = "") -> None:
        self.errors += 1
        self.counts["build_error"] += 1
        s = self.samples.setdefault("build_error", [])
        if len(s) < self.keep:
            s.append(f"{origin}: {type(exc).__name__}: {str(exc)[:160]}")

    @property
    def seen(self) -> int:
        return self.accepted + sum(self.counts.values())

    def merge(self, other: "Ledger") -> None:
        self.counts.update(other.counts)
        self.accepted += other.accepted
        self.errors += other.errors
        for k, v in other.samples.items():
            s = self.samples.setdefault(k, [])
            s.extend(v[: max(0, self.keep - len(s))])

    def to_json(self) -> dict:
        return {"seen": self.seen, "accepted": self.accepted,
                "rejected": dict(self.counts), "samples": self.samples}

    def table(self) -> str:
        if not self.counts:
            return f"accepted {self.accepted}/{self.seen}, nothing rejected"
        w = max(len(k) for k in self.counts)
        lines = [f"accepted {self.accepted}/{self.seen} candidates; rejections by rule:"]
        for k, v in self.counts.most_common():
            num = RULES[k].number if k in RULES else "-"
            lines.append(f"  {k:<{w}}  (rule {num:>2})  {v:>6}   e.g. {self.samples.get(k, [''])[0][:90]}")
        return "\n".join(lines)


class RuleSet:
    """The enabled hard rules, applied in the order they are numbered."""

    def __init__(self, enabled: set[str] | None = None, ctx: dict | None = None):
        self.enabled = set(RULES) if enabled is None else set(enabled)
        unknown = self.enabled - set(RULES)
        if unknown:
            raise ValueError(f"unknown rules: {sorted(unknown)}")
        self.ctx = dict(ctx or {})

    @staticmethod
    def from_flags(off: list[str] | None = None, only: list[str] | None = None,
                   ctx: dict | None = None) -> "RuleSet":
        """`off`: ids to disable. `only`: the exact set to enable (off is then applied to it).

        An unknown id is an error rather than a silent no-op: a typo in a flag that
        is meant to switch a check off would otherwise look like the check costing
        nothing.
        """
        unknown = (set(off or []) | set(only or [])) - set(RULES)
        if unknown:
            raise ValueError(f"unknown rules: {sorted(unknown)} (known: {sorted(RULES)})")
        base = set(RULES) if not only else set(only)
        return RuleSet(base - set(off or []), ctx)

    def disabled(self) -> list[str]:
        return sorted(set(RULES) - self.enabled, key=lambda k: RULES[k].number)

    def check(self, cand: NetCandidate, ctx: dict | None = None) -> Rejection | None:
        """First rule that rejects the candidate, or None. Skipped rules never reject."""
        c = {**self.ctx, **(ctx or {})}
        for r in sorted(RULES.values(), key=lambda r: r.number):
            if r.id not in self.enabled:
                continue
            if r.needs_coords and cand.coords is None:
                continue
            if r.needs_exact and not cand.exact:
                continue
            why = r.fn(cand, c)
            if why:
                return Rejection(r.id, r.number, why)
        return None

    def check_all(self, cand: NetCandidate, ctx: dict | None = None) -> list[Rejection]:
        """Every rule that rejects, not just the first (for diagnosing a candidate)."""
        c = {**self.ctx, **(ctx or {})}
        out = []
        for r in sorted(RULES.values(), key=lambda r: r.number):
            if r.id not in self.enabled:
                continue
            if (r.needs_coords and cand.coords is None) or (r.needs_exact and not cand.exact):
                continue
            why = r.fn(cand, c)
            if why:
                out.append(Rejection(r.id, r.number, why))
        return out


def min_faces_for(area: float | None) -> int | None:
    """Rule 14's bound: matching total area A needs at least ceil(A) faces of area <= 1."""
    if area is None:
        return None
    k = math.ceil(area - 1e-12)
    return max(1, int(k))


def describe_rules() -> str:
    lines = []
    for r in sorted(RULES.values(), key=lambda r: r.number):
        tag = []
        if r.needs_coords:
            tag.append("needs geometry")
        if r.needs_exact:
            tag.append("needs unit lengths")
        lines.append(f"{r.number:>2}. {r.id:<14} {r.summary}" + (f"  [{', '.join(tag)}]" if tag else ""))
    return "\n".join(lines)
