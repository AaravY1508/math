"""Geometry core for the Maximum Fence Area problem -- the *optimization path*.

Problem (Erich Friedman): place n straight fences of length exactly 1 so that
  * every fence endpoint lies ON another fence (at its endpoint or in its interior
    -- T-junctions are legal and essential),
  * fences never cross (they may touch only at endpoints / T-junctions),
  * every bounded face ("field") of the induced planar subdivision has area <= 1,
and maximize the total area of the bounded faces.

Representation used by the optimizer
------------------------------------
A *topology* is a combinatorial planar embedding:
  * vertices 0..V-1 (points in R^2),
  * fences i = (a_i, b_i): vertex ids of the two endpoints,
  * pins: vertex v -> host fence h, meaning v lies in the INTERIOR of fence h at a
    scalar parameter t_v in (0, 1) along h  (v = (1-t_v) * X[a_h] + t_v * X[b_h]).
    A vertex that is not pinned is a "joint": >= 2 fence endpoints meet there.
  * chains: for each fence, the ordered vertex list along it (a, pins by t, b);
    consecutive chain vertices are the edges of the planar graph,
  * faces: bounded faces as CCW vertex cycles, obtained by a half-edge (DCEL)
    traversal of the embedding (see `trace_faces`).

For a fixed topology the free parameters are the vertex coordinates plus the pin
parameters t; `TorchTopology` evaluates the objective and all constraint residuals
for a batch [B, P] of parameter vectors as differentiable tensors.

Nothing in here is used to *certify* a result: `fence.referee.verify` re-derives
everything from raw segment coordinates at 50 digits and shares no code with this
module.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

# Numerical slack used when *building/validating* topologies in float64. This is a
# sanity threshold for sketches, not a certification tolerance.
GEOM_EPS = 1e-9


class InvalidTopology(ValueError):
    """Raised when a sketch/configuration does not describe a legal fence embedding."""


# ---------------------------------------------------------------------------
# Small float helpers
# ---------------------------------------------------------------------------

def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """z-component of the 2D cross product, broadcasting over leading dims."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def polygon_signed_area(pts: np.ndarray) -> float:
    """Shoelace formula; positive for counter-clockwise vertex order."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def fence_param(coords: np.ndarray, a: int, b: int, v: int) -> float:
    """Parameter t of the orthogonal projection of vertex v onto the line a->b."""
    d = coords[b] - coords[a]
    return float(np.dot(coords[v] - coords[a], d) / np.dot(d, d))


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized distance from points p to segments [a, b] (all [..., 2])."""
    d = b - a
    dd = np.maximum(np.sum(d * d, axis=-1), 1e-300)
    t = np.clip(np.sum((p - a) * d, axis=-1) / dd, 0.0, 1.0)
    proj = a + t[..., None] * d
    return np.linalg.norm(p - proj, axis=-1)


def segment_distance(p1, p2, q1, q2) -> np.ndarray:
    """Vectorized minimum distance between segments [p1,p2] and [q1,q2] (0 if they intersect)."""
    o1 = cross2(p2 - p1, q1 - p1)
    o2 = cross2(p2 - p1, q2 - p1)
    o3 = cross2(q2 - q1, p1 - q1)
    o4 = cross2(q2 - q1, p2 - q1)
    proper = (o1 * o2 < 0) & (o3 * o4 < 0)
    dist = np.minimum.reduce([
        point_segment_distance(q1, p1, p2),
        point_segment_distance(q2, p1, p2),
        point_segment_distance(p1, q1, q2),
        point_segment_distance(p2, q1, q2),
    ])
    return np.where(proper, 0.0, dist)


# ---------------------------------------------------------------------------
# Face extraction: half-edge (DCEL) traversal
# ---------------------------------------------------------------------------

@dataclass
class FaceTrace:
    """Result of walking every face of a planar straight-line graph."""
    cycles: list[list[int]]      # vertex cycles, each with its face on the LEFT
    halfedges: list[list[int]]   # the same cycles as half-edge id lists
    areas: np.ndarray            # signed shoelace area of each cycle
    n_components: int


def trace_faces(coords: np.ndarray, edges: list[tuple[int, int]]) -> FaceTrace:
    """Enumerate the faces of a planar straight-line graph by half-edge traversal.

    Construction (the standard DCEL face walk):

    1. Every undirected edge {u, v} becomes two half-edges: h = 2k runs u->v and
       h ^ 1 = 2k+1 runs v->u. `h ^ 1` is therefore the *twin* of h.

    2. Rotation system. At every vertex we sort its outgoing half-edges by the angle
       of their direction vector (atan2), i.e. counter-clockwise. In a planar
       straight-line embedding this cyclic order is exactly the combinatorial
       embedding, so no heuristic is needed.

    3. Face-on-the-left successor. Suppose we walk half-edge h = (u -> w) and keep
       the face on our LEFT. At w we must leave along the half-edge that makes the
       sharpest possible *left-hand* turn that still hugs that face, which is the
       outgoing half-edge immediately CLOCKWISE of twin(h) = (w -> u) in the
       rotation at w. With the CCW-sorted rotation list that is the entry just
       *before* twin(h), cyclically:
            next(h) = rot_w[(pos(twin(h)) - 1) mod deg(w)].
       Worked example: unit square, walking (0,0)->(1,0). At w=(1,0) the outgoing
       directions are pi/2 (to (1,1)) and pi (back to (0,0) = twin). The entry
       before pi in CCW order is pi/2, so we continue up the right side, keeping the
       square's interior on our left. At a T-junction the same rule turns into the
       stem, which is what splits the host's two sides into different faces.

    4. `next` is a permutation of the half-edges; its orbits are the face
       boundaries. Every half-edge lies on exactly one orbit, and the face of an
       orbit is the region to the left of all of its half-edges.

    5. Classification by signed (shoelace) area. A bounded face is walked
       counter-clockwise (area > 0). The outer boundary of each connected component
       is walked clockwise (area < 0); the face to its left is the unbounded face
       (or, for nested components, the enclosing bounded face -- callers that allow
       nesting must assign such "hole" cycles; the optimizer path requires a
       connected graph, where exactly one cycle is negative).

    Sanity check available to callers: for a connected planar graph Euler's formula
    gives #cycles = E - V + 2.
    """
    coords = np.asarray(coords, dtype=float)
    E = len(edges)
    src = np.empty(2 * E, dtype=np.int64)
    dst = np.empty(2 * E, dtype=np.int64)
    for k, (u, v) in enumerate(edges):
        src[2 * k], dst[2 * k] = u, v
        src[2 * k + 1], dst[2 * k + 1] = v, u
    V = len(coords)
    vec = coords[dst] - coords[src]
    angle = np.arctan2(vec[:, 1], vec[:, 0])

    # Rotation system in CSR form: rot[start[v]:start[v+1]] are v's outgoing
    # half-edges sorted counter-clockwise by angle.
    rot = np.lexsort((angle, src))
    deg = np.bincount(src, minlength=V)
    start = np.zeros(V + 1, dtype=np.int64)
    start[1:] = np.cumsum(deg)
    pos = np.empty(2 * E, dtype=np.int64)
    pos[rot] = np.arange(2 * E) - start[src[rot]]

    # next(h) = outgoing half-edge at dst(h) immediately clockwise of twin(h).
    h_all = np.arange(2 * E)
    twin = h_all ^ 1
    w = dst
    nxt = rot[start[w] + (pos[twin] - 1) % np.maximum(deg[w], 1)]

    seen = np.zeros(2 * E, dtype=bool)
    cycles, hcycles, areas = [], [], []
    for h0 in range(2 * E):
        if seen[h0]:
            continue
        hc = []
        h = h0
        while not seen[h]:
            seen[h] = True
            hc.append(h)
            h = int(nxt[h])
        if h != h0:  # cannot happen for a permutation; guard against corrupt input
            raise InvalidTopology("half-edge successor is not a permutation")
        vc = [int(src[x]) for x in hc]
        p = coords[vc]
        q = coords[[int(dst[x]) for x in hc]]
        areas.append(0.5 * float(np.sum(cross2(p, q))))
        cycles.append(vc)
        hcycles.append(hc)

    # connected components (union-find over edges) for the Euler check
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
    used = {find(u) for e in edges for u in e}
    return FaceTrace(cycles=cycles, halfedges=hcycles, areas=np.array(areas), n_components=len(used))


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

@dataclass
class Topology:
    """Combinatorial embedding of a fence configuration (see module docstring)."""
    n_vertices: int
    fences: np.ndarray                 # [n, 2] int endpoint vertex ids
    pins: dict[int, int]               # pinned vertex -> host fence
    chains: list[list[int]]            # per fence: ordered vertices along it
    faces: list[list[int]]             # bounded faces, CCW vertex cycles
    outer: list[int] = field(default_factory=list)   # outer boundary cycle (CW)

    @property
    def n(self) -> int:
        return len(self.fences)

    @property
    def pin_vertices(self) -> list[int]:
        return sorted(self.pins)

    def edges(self) -> list[tuple[int, int]]:
        return [(c[i], c[i + 1]) for c in self.chains for i in range(len(c) - 1)]

    def fence_vertex_sets(self) -> list[set[int]]:
        return [set(c) for c in self.chains]

    def incident_pairs(self) -> set[tuple[int, int]]:
        """Fence pairs that share a vertex (common endpoint, or an endpoint of one on the other)."""
        at = {}
        for f, c in enumerate(self.chains):
            for v in c:
                at.setdefault(v, []).append(f)
        pairs = set()
        for fs in at.values():
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    pairs.add((min(fs[i], fs[j]), max(fs[i], fs[j])))
        return pairs

    def features(self) -> dict:
        return {"faces": len(self.faces), "tjunctions": len(self.pins),
                "joints": self.n_vertices - len(self.pins)}

    def to_json(self) -> dict:
        return {
            "n_vertices": self.n_vertices,
            "fences": self.fences.tolist(),
            "pins": [[int(v), int(h)] for v, h in sorted(self.pins.items())],
            "chains": self.chains,
            "faces": self.faces,
            "outer": self.outer,
        }

    @staticmethod
    def from_json(d: dict) -> "Topology":
        return Topology(
            n_vertices=int(d["n_vertices"]),
            fences=np.asarray(d["fences"], dtype=np.int64).reshape(-1, 2),
            pins={int(v): int(h) for v, h in d["pins"]},
            chains=[list(map(int, c)) for c in d["chains"]],
            faces=[list(map(int, f)) for f in d["faces"]],
            outer=list(map(int, d.get("outer", []))),
        )


def build_chains(coords: np.ndarray, fences: np.ndarray, pins: dict[int, int]) -> list[list[int]]:
    """Order each fence's vertices along it: endpoint a, pinned vertices by t, endpoint b."""
    on = [[] for _ in range(len(fences))]
    for v, h in pins.items():
        a, b = fences[h]
        t = fence_param(coords, a, b, v)
        on[h].append((t, v))
    chains = []
    for i, (a, b) in enumerate(fences):
        mids = [v for _, v in sorted(on[i])]
        chains.append([int(a)] + mids + [int(b)])
    return chains


def check_embedding(coords: np.ndarray, fences: np.ndarray, pins: dict[int, int],
                    chains: list[list[int]], eps: float = GEOM_EPS) -> list[str]:
    """Float sanity checks that a sketch is a legal, planar fence embedding.

    Returns a list of human-readable problems (empty list == OK).
    """
    problems = []
    n = len(fences)
    V = len(coords)
    ends = {}
    for i, (a, b) in enumerate(fences):
        if a == b:
            problems.append(f"fence {i} has identical endpoints")
        ends.setdefault(int(a), []).append(i)
        ends.setdefault(int(b), []).append(i)
    for v in range(V):
        if v not in ends:
            problems.append(f"vertex {v} is not the endpoint of any fence")
    # Every endpoint must lie on another fence: either >= 2 fence ends meet at the
    # vertex (a joint), or the vertex is pinned in the interior of a host fence.
    for v, fs in ends.items():
        if len(fs) < 2 and v not in pins:
            problems.append(f"vertex {v} (end of fence {fs[0]}) is dangling")
    for v, h in pins.items():
        a, b = fences[h]
        if v in (a, b):
            problems.append(f"vertex {v} pinned on its own fence {h}")
            continue
        if h in ends.get(v, []):
            problems.append(f"fence {h} is pinned to itself at {v}")
        t = fence_param(coords, a, b, v)
        d = coords[b] - coords[a]
        off = abs(float(cross2(d, coords[v] - coords[a]))) / max(np.linalg.norm(d), 1e-300)
        if not (0.0 < t < 1.0):
            problems.append(f"pin {v} on fence {h} has t={t:.6g} outside (0,1)")
        if off > 1e-6 * max(1.0, np.linalg.norm(d)):
            problems.append(f"pin {v} is {off:.3g} off its host fence {h}")
    # Planarity: edges (chain pieces) may meet only at shared chain vertices.
    edges = [(c[k], c[k + 1]) for c in chains for k in range(len(c) - 1)]
    if edges:
        e = np.asarray(edges)
        P1, P2 = coords[e[:, 0]], coords[e[:, 1]]
        lens = np.linalg.norm(P2 - P1, axis=1)
        for k in np.where(lens < eps)[0]:
            problems.append(f"edge {edges[k]} has ~zero length")
        iu, ju = np.triu_indices(len(edges), 1)
        share = ((e[iu, 0] == e[ju, 0]) | (e[iu, 0] == e[ju, 1]) |
                 (e[iu, 1] == e[ju, 0]) | (e[iu, 1] == e[ju, 1]))
        # (a) edges without a common vertex must stay apart
        m = ~share
        dist = segment_distance(P1[iu[m]], P2[iu[m]], P1[ju[m]], P2[ju[m]])
        for k in np.where(dist < eps)[0][:10]:
            problems.append(f"edges {edges[iu[m][k]]} and {edges[ju[m][k]]} touch/cross")
        # (b) edges with a common vertex must not overlap (leave it in the same direction)
        for a_, b_ in zip(iu[share], ju[share]):
            ea, eb = edges[a_], edges[b_]
            s = set(ea) & set(eb)
            if len(s) == 2:
                problems.append(f"duplicate edge {ea}")
                continue
            c = s.pop()
            oa = ea[1] if ea[0] == c else ea[0]
            ob = eb[1] if eb[0] == c else eb[0]
            da, db = coords[oa] - coords[c], coords[ob] - coords[c]
            na, nb = np.linalg.norm(da), np.linalg.norm(db)
            if na < eps or nb < eps:
                continue
            if abs(float(cross2(da, db))) / (na * nb) < 1e-9 and float(np.dot(da, db)) > 0:
                problems.append(f"edges {ea} and {eb} overlap")
    return problems


def make_topology(coords: np.ndarray, fences, pins: dict[int, int], check: bool = True) -> Topology:
    """Build a validated Topology (chains + faces) from an exact sketch.

    `coords` must already satisfy the incidences in `pins` (pinned vertices on
    their hosts); lengths need NOT be 1 -- sketches are fixed up by the optimizer.
    """
    coords = np.asarray(coords, dtype=float)
    fences = np.asarray(fences, dtype=np.int64).reshape(-1, 2)
    pins = {int(v): int(h) for v, h in pins.items()}
    chains = build_chains(coords, fences, pins)
    if check:
        problems = check_embedding(coords, fences, pins, chains)
        if problems:
            raise InvalidTopology("; ".join(problems[:5]))
    edges = [(c[k], c[k + 1]) for c in chains for k in range(len(c) - 1)]
    tr = trace_faces(coords, edges)
    if tr.n_components != 1:
        raise InvalidTopology(f"configuration has {tr.n_components} connected components")
    V_used = len({u for e in edges for u in e})
    if len(tr.cycles) != len(edges) - V_used + 2:
        raise InvalidTopology("Euler check failed: embedding is not planar")
    neg = [i for i, a in enumerate(tr.areas) if a < 0]
    pos = [i for i, a in enumerate(tr.areas) if a > 0]
    if len(neg) != 1:
        raise InvalidTopology(f"expected one outer cycle, got {len(neg)} negative cycles")
    if len(pos) + len(neg) != len(tr.cycles):
        raise InvalidTopology("degenerate (zero-area) face cycle")
    faces = [tr.cycles[i] for i in pos]
    return Topology(n_vertices=len(coords), fences=fences, pins=pins, chains=chains,
                    faces=faces, outer=tr.cycles[neg[0]])


def face_areas(coords: np.ndarray, faces: list[list[int]]) -> np.ndarray:
    return np.array([polygon_signed_area(np.asarray(coords)[f]) for f in faces])


def segments_of(coords: np.ndarray, fences) -> np.ndarray:
    """[n, 4] array of (x1, y1, x2, y2) fence segments."""
    coords = np.asarray(coords, dtype=float)
    f = np.asarray(fences, dtype=np.int64).reshape(-1, 2)
    return np.concatenate([coords[f[:, 0]], coords[f[:, 1]]], axis=1)


def topology_from_segments(segs: np.ndarray, tol: float = 1e-7):
    """Recover (coords, fences, pins) from raw segments by snapping incidences.

    Endpoints closer than `tol` are merged into one joint vertex; an endpoint within
    `tol` of another fence's interior becomes a vertex pinned on that fence. Used to
    import configurations that only carry coordinates (float path, not a referee).
    """
    segs = np.asarray(segs, dtype=float).reshape(-1, 4)
    n = len(segs)
    pts = segs.reshape(-1, 2)  # endpoint 2i = a_i, 2i+1 = b_i
    parent = list(range(2 * n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(2 * n):
        for j in range(i + 1, 2 * n):
            if i // 2 != j // 2 and np.linalg.norm(pts[i] - pts[j]) < tol:
                parent[find(i)] = find(j)
    roots = sorted({find(i) for i in range(2 * n)})
    vid = {r: k for k, r in enumerate(roots)}
    coords = np.array([np.mean([pts[i] for i in range(2 * n) if find(i) == r], axis=0) for r in roots])
    fences = np.array([[vid[find(2 * i)], vid[find(2 * i + 1)]] for i in range(n)])
    pins = {}
    for v in range(len(coords)):
        for h in range(n):
            a, b = fences[h]
            if v in (a, b):
                continue
            d = point_segment_distance(coords[v], coords[a], coords[b])
            if d < tol and min(np.linalg.norm(coords[v] - coords[a]), np.linalg.norm(coords[v] - coords[b])) > tol:
                if v in pins:
                    raise InvalidTopology(f"vertex {v} lies on the interiors of two fences")
                pins[v] = h
    return coords, fences, pins


# ---------------------------------------------------------------------------
# Symmetry detection (MAP-Elites feature)
# ---------------------------------------------------------------------------

def rotational_symmetry_order(segs: np.ndarray, tol: float = 1e-4, max_order: int = 8) -> int:
    """Largest m such that rotating the segment set by 2*pi/m about its centroid maps it to itself."""
    segs = np.asarray(segs, dtype=float).reshape(-1, 4)
    pts = segs.reshape(-1, 2)
    c = pts.mean(axis=0)
    P = segs.reshape(-1, 2, 2) - c

    def canon(S):
        # order-independent representation: endpoints sorted within each segment
        a, b = S[:, 0], S[:, 1]
        swap = (a[:, 0] > b[:, 0] + 1e-9) | ((np.abs(a[:, 0] - b[:, 0]) <= 1e-9) & (a[:, 1] > b[:, 1]))
        a2 = np.where(swap[:, None], b, a)
        b2 = np.where(swap[:, None], a, b)
        return np.concatenate([a2, b2], axis=1)

    base = canon(P)
    best = 1
    for m in range(2, max_order + 1):
        th = 2 * math.pi / m
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        Q = canon(P @ R.T)
        # every rotated segment must match some original segment
        d = np.linalg.norm(Q[:, None, :] - base[None, :, :], axis=2)
        if np.all(d.min(axis=1) < tol):
            best = m
    return best


# ---------------------------------------------------------------------------
# Batched differentiable residuals (torch)
# ---------------------------------------------------------------------------

class TorchTopology:
    """Index tensors for one fixed topology + batched objective/residuals.

    Parameter layout per restart (a row of the [B, P] batch):
        z = [x_0, y_0, x_1, y_1, ..., x_{V-1}, y_{V-1}, t_{p_0}, ..., t_{p_{K-1}}]
    where p_k are the pinned vertices in sorted order.
    """

    def __init__(self, topo: Topology, device="cpu", dtype=None, coords0=None, cross_radius=None):
        """coords0 + cross_radius: only penalize crossings of non-incident fence pairs
        that are within cross_radius of each other in the starting sketch (a speed
        optimization; any crossing among the pruned far pairs is still caught by the
        float embedding check after polishing and by the referee)."""
        import torch

        self.torch = torch
        self.topo = topo
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32
        L = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=self.device)  # noqa: E731
        self.V = topo.n_vertices
        self.pin_list = topo.pin_vertices
        self.K = len(self.pin_list)
        self.n_params = 2 * self.V + self.K
        self.fa = L(topo.fences[:, 0])
        self.fb = L(topo.fences[:, 1])
        hosts = [topo.pins[v] for v in self.pin_list]
        self.pin_v = L(self.pin_list)
        self.pin_ha = L([topo.fences[h][0] for h in hosts])
        self.pin_hb = L([topo.fences[h][1] for h in hosts])
        # face edges for the shoelace sums
        fi, fj, ff = [], [], []
        for k, cyc in enumerate(topo.faces):
            for i in range(len(cyc)):
                fi.append(cyc[i]); fj.append(cyc[(i + 1) % len(cyc)]); ff.append(k)
        self.F = len(topo.faces)
        self.fe_i, self.fe_j, self.fe_f = L(fi), L(fj), L(ff)
        # corners of every face cycle (bounded faces AND the outer boundary)
        cu, cv, cw = [], [], []
        for cyc in list(topo.faces) + [topo.outer]:
            m = len(cyc)
            for i in range(m):
                cu.append(cyc[i - 1]); cv.append(cyc[i]); cw.append(cyc[(i + 1) % m])
        self.cu, self.cv, self.cw = L(cu), L(cv), L(cw)
        # non-incident fence pairs: must not intersect at all
        inc = topo.incident_pairs()
        qi, qj = [], []
        for i in range(topo.n):
            for j in range(i + 1, topo.n):
                if (i, j) not in inc:
                    qi.append(i); qj.append(j)
        self.n_pairs_total = len(qi)
        if coords0 is not None and cross_radius is not None and qi:
            X0 = np.asarray(coords0, dtype=float)
            P1, P2 = X0[topo.fences[:, 0]], X0[topo.fences[:, 1]]
            I, J = np.array(qi), np.array(qj)
            near = segment_distance(P1[I], P2[I], P1[J], P2[J]) < cross_radius
            qi, qj = I[near].tolist(), J[near].tolist()
        self.qi, self.qj = L(qi), L(qj)
        # ordering of pins along each host: indices into T_ext = [0, t_0..t_{K-1}, 1]
        kidx = {v: k + 1 for k, v in enumerate(self.pin_list)}
        lo, hi = [], []
        for c in topo.chains:
            seq = [0] + [kidx[v] for v in c[1:-1]] + [self.K + 1]
            for s in range(len(seq) - 1):
                lo.append(seq[s]); hi.append(seq[s + 1])
        self.ord_lo, self.ord_hi = L(lo), L(hi)

    # -- packing ----------------------------------------------------------
    # Public layout is batch-first Z[B, P]. Internally the optimizer keeps the
    # same numbers stored transposed, Zt[P, B] ("batch-last"), so every gather
    # is an index_select along dim 0 of contiguous B-vectors and its backward is
    # a coalesced index_add (much faster than batch-first fancy indexing).
    def split(self, Z):
        X = Z[:, : 2 * self.V].reshape(Z.shape[0], self.V, 2)
        T = Z[:, 2 * self.V:]
        return X, T

    def pack(self, X, T):
        return self.torch.cat([X.reshape(X.shape[0], -1), T], dim=1)

    def split_t(self, Zt):
        """Zt[P, B] -> X[V, 2, B], T[K, B] (views)."""
        return Zt[: 2 * self.V].view(self.V, 2, -1), Zt[2 * self.V:]

    def initial_t(self, X):
        """Projection parameters of pinned vertices onto their hosts (batch-first X[B, V, 2])."""
        A = X[:, self.pin_ha]
        Bv = X[:, self.pin_hb]
        P = X[:, self.pin_v]
        d = Bv - A
        return ((P - A) * d).sum(-1) / (d * d).sum(-1).clamp_min(1e-12)

    # -- objective --------------------------------------------------------
    @staticmethod
    def _cross(u, v):
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    def face_areas_t(self, X):
        """[F, B] signed shoelace areas of the bounded faces (X is [V, 2, B])."""
        xi = X.index_select(0, self.fe_i)
        xj = X.index_select(0, self.fe_j)
        c = self._cross(xi, xj)
        A = X.new_zeros(self.F, X.shape[-1])
        return 0.5 * A.index_add_(0, self.fe_f, c)

    def face_areas(self, X):
        """[B, F] areas for a batch-first X[B, V, 2]."""
        return self.face_areas_t(X.permute(1, 2, 0).contiguous()).T

    # -- constraint residuals ----------------------------------------------
    def residuals_t(self, Zt, turn_margin: float = 0.03, t_gap: float = 1e-3, min_face: float = 1e-3):
        """All constraint residuals as differentiable tensors, batch-last ([*, B]).

        length   : [n, B]    segment length - 1 (signed)              (a)
        cross    : [Q, B]    proper-intersection depth of non-incident fence pairs (b)
        area     : [F, B]    relu(face_area - 1)                       (c)
        pin_vec  : [K, 2, B] pinned vertex minus its host point        (d)
        pin      : [K, B]    |pin_vec|
        order    : [*, B]    relu(t_gap - (t_next - t_prev)) along each host (pins stay inside, in order)
        turn     : [C, B]    relu(|turn angle| - (pi - margin)) at face corners (no folding / overlap)
        face_pos : [F, B]    relu(min_face - face_area) (faces stay positively oriented)
        _areas   : [F, B]    the face areas themselves (objective = their sum)
        """
        torch = self.torch
        X, T = self.split_t(Zt)
        B = X.shape[-1]
        out = {}
        P1 = X.index_select(0, self.fa)
        P2 = X.index_select(0, self.fb)
        D = P2 - P1
        out["length"] = (D.pow(2).sum(1) + 1e-18).sqrt() - 1.0
        if self.K:
            A = X.index_select(0, self.pin_ha)
            Bv = X.index_select(0, self.pin_hb)
            pv = X.index_select(0, self.pin_v) - (A + T.unsqueeze(1) * (Bv - A))
            out["pin_vec"] = pv
            out["pin"] = (pv.pow(2).sum(1) + 1e-30).sqrt()
        else:
            out["pin_vec"] = X.new_zeros(0, 2, B)
            out["pin"] = X.new_zeros(0, B)
        areas = self.face_areas_t(X)
        out["area"] = torch.relu(areas - 1.0)
        out["face_pos"] = torch.relu(min_face - areas)
        if len(self.qi):
            p1, di = P1.index_select(0, self.qi), D.index_select(0, self.qi)
            q1, dj = P1.index_select(0, self.qj), D.index_select(0, self.qj)
            w = q1 - p1
            o1 = self._cross(di, w)                 # q1 relative to line p
            o2 = o1 + self._cross(di, dj)           # q2 = q1 + dj
            o3 = -self._cross(dj, w)                # p1 relative to line q
            o4 = o3 + self._cross(dj, di)           # p2 = p1 + di
            # positive iff both straddle tests fail (a proper crossing); its size
            # is the smaller of the two straddle products
            out["cross"] = torch.relu(torch.minimum(-o1 * o2, -o3 * o4))
        else:
            out["cross"] = X.new_zeros(0, B)
        Text = torch.cat([T.new_zeros(1, B), T, T.new_ones(1, B)], dim=0)
        out["order"] = torch.relu(t_gap - (Text.index_select(0, self.ord_hi) - Text.index_select(0, self.ord_lo)))
        Xu = X.index_select(0, self.cu)
        Xv = X.index_select(0, self.cv)
        Xw = X.index_select(0, self.cw)
        d1 = Xv - Xu
        d2 = Xw - Xv
        cz = self._cross(d1, d2)
        dt = (d1 * d2).sum(1)
        small = (cz.abs() + dt.abs()) < 1e-12          # zero-length edge: no defined angle
        turn = torch.atan2(torch.where(small, torch.zeros_like(cz), cz),
                           torch.where(small, torch.ones_like(dt), dt))
        out["turn"] = torch.relu(turn.abs() - (math.pi - turn_margin))
        out["_areas"] = areas
        return out

    def residuals(self, Z, **kw):
        """Batch-first convenience wrapper: Z[B, P] -> dict of [B, *] tensors (same content)."""
        out = self.residuals_t(Z.T.contiguous(), **kw)
        res = {}
        for k, v in out.items():
            res[k] = v.permute(*range(v.dim() - 1, -1, -1)) if k == "pin_vec" else v.T
        return res


# ---------------------------------------------------------------------------
# Configuration I/O
# ---------------------------------------------------------------------------

def fmt(x: float) -> str:
    """Round-trip exact decimal for a float64 (the referee parses these strings)."""
    return repr(float(x))


def config_to_json(topo: Topology, coords: np.ndarray, n: int | None = None, **extra) -> dict:
    coords = np.asarray(coords, dtype=float)
    segs = segments_of(coords, topo.fences)
    d = {
        "problem": "maximum-fence-area",
        "n": int(n if n is not None else topo.n),
        "segments": [[fmt(v) for v in s] for s in segs],
        "vertices": [[fmt(x), fmt(y)] for x, y in coords],
        "topology": topo.to_json(),
    }
    d.update(extra)
    return d


def config_from_json(d: dict) -> tuple[Topology, np.ndarray]:
    topo = Topology.from_json(d["topology"])
    coords = np.array([[float(x), float(y)] for x, y in d["vertices"]])
    return topo, coords


def save_config(path, topo: Topology, coords: np.ndarray, **extra) -> None:
    with open(path, "w") as fh:
        json.dump(config_to_json(topo, coords, **extra), fh, indent=1)


def verify(config, tol: float = 1e-9):
    """Referee entry point (re-exported). Implemented independently in fence/referee.py."""
    from fence.referee import verify as _verify
    return _verify(config, tol)
