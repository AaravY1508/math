"""Sketch-construction DSL used by evolved `propose_topology(n, rng)` programs.

A *sketch* is a planar fence embedding with explicit incidence (joints and
T-junction pins) whose fence lengths are only roughly 1 -- the optimizer
(fence.optimize) turns it into an exact unit-length configuration. The builder
guarantees that every sketch it returns is a legal embedding: every endpoint is
supported, nothing crosses, the graph is connected (validated with
fence.geometry.make_topology after every operation).

Building blocks
---------------
seeds     ring(k) | grid(r, c) | pinwheel_in_ring(k, K) | literal elite configuration
grow      add a path of m = 1..3 fences between two points P, Q of one face's
          boundary, bulging into that face, creating one new cell. In the outer
          face this grows the region; inside a bounded face it splits the face.
symmetry  apply each growth step in an orbit of rotations about the seed centre
edits     structural mutations: toggle_junction (T <-> joint), reattach an
          endpoint to another edge, remove_add (move a fence elsewhere)

`build_from_genome(n, rng, genome)` is the standard interpreter of a genome
dict; evolved programs usually just call it, but can use the primitives freely.
"""
from __future__ import annotations

import math

import numpy as np

from fence import geometry as G


def a_est(n: int) -> float:
    """Rough achievable area for n fences (fits the published records to ~5%)."""
    return max(0.4, 0.5 * n - 0.66 * math.sqrt(n))


class BuildError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _pip(pt, poly: np.ndarray) -> bool:
    """Even-odd point-in-polygon."""
    x, y = pt
    c = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                c = not c
    return c


def arc_points(Q: np.ndarray, P: np.ndarray, m: int, ell: float = 1.0):
    """Interior vertices of an m-segment polyline Q -> P with equal segments of length
    ~ell lying on a circular arc that bulges to the RIGHT of the direction Q->P
    (i.e. to the left of P->Q). Returns [A_1..A_{m-1}] ordered from Q to P, or None.
    """
    d = float(np.linalg.norm(P - Q))
    if m == 1:
        return []
    if d >= m * ell * 0.995 or d < 1e-9:
        return None
    r = d / ell
    # sin(m*phi/2) / sin(phi/2) = r, decreasing in phi on (0, 2*pi/m)
    lo, hi = 1e-9, 2 * math.pi / m - 1e-9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = math.sin(m * mid / 2) / math.sin(mid / 2)
        if f > r:
            lo = mid
        else:
            hi = mid
    phi = 0.5 * (lo + hi)
    u = (P - Q) / d
    th = math.atan2(u[1], u[0]) - (m - 1) * phi / 2
    pts = []
    cur = Q.astype(float).copy()
    for _ in range(m - 1):
        cur = cur + ell * np.array([math.cos(th), math.sin(th)])
        pts.append(cur.copy())
        th += phi
    # correct the tiny closing error so the last segment ends exactly at P
    end = cur + ell * np.array([math.cos(th), math.sin(th)])
    err = P - end
    for k in range(len(pts)):
        pts[k] = pts[k] + err * (k + 1) / m
    return pts


# ---------------------------------------------------------------------------
# Sketch
# ---------------------------------------------------------------------------

class Sketch:
    def __init__(self, P=None, fences=None, pins=None, center=(0.0, 0.0)):
        self.P = [list(map(float, p)) for p in (P if P is not None else [])]
        self.fences = [list(map(int, f)) for f in (fences if fences is not None else [])]
        self.pins = {int(k): int(v) for k, v in (pins or {}).items()}
        self.center = np.array(center, dtype=float)
        self._topo = None

    # -- basics ---------------------------------------------------------------
    def copy(self) -> "Sketch":
        s = Sketch(self.P, self.fences, self.pins, self.center)
        s._topo = self._topo
        return s

    @property
    def n(self) -> int:
        return len(self.fences)

    def coords(self) -> np.ndarray:
        return np.asarray(self.P, dtype=float).reshape(-1, 2)

    def _dirty(self):
        self._topo = None

    def topology(self) -> G.Topology:
        if self._topo is None:
            self._topo = G.make_topology(self.coords(), self.fences, self.pins)
        return self._topo

    def valid(self) -> bool:
        try:
            self.topology()
            return True
        except G.InvalidTopology:
            return False

    def add_vertex(self, x, y) -> int:
        self.P.append([float(x), float(y)])
        self._dirty()
        return len(self.P) - 1

    def add_fence(self, a, b) -> int:
        self.fences.append([int(a), int(b)])
        self._dirty()
        return len(self.fences) - 1

    def add_pin(self, host: int, t: float) -> int:
        a, b = self.fences[host]
        X = self.coords()
        p = (1 - t) * X[a] + t * X[b]
        v = self.add_vertex(*p)
        self.pins[v] = host
        return v

    def to_json(self) -> dict:
        return {"P": [[float(x), float(y)] for x, y in self.P], "fences": self.fences,
                "pins": [[int(v), int(h)] for v, h in sorted(self.pins.items())],
                "center": self.center.tolist()}

    @staticmethod
    def from_json(d) -> "Sketch":
        return Sketch(d["P"], d["fences"], {int(v): int(h) for v, h in d["pins"]}, d.get("center", (0, 0)))

    @staticmethod
    def from_config(topo: G.Topology, coords: np.ndarray) -> "Sketch":
        c = np.asarray(coords).mean(axis=0)
        return Sketch(np.asarray(coords) - c, topo.fences.tolist(), dict(topo.pins), (0.0, 0.0))

    def compact(self):
        """Drop vertices no fence uses; renumber."""
        used = sorted({v for f in self.fences for v in f})
        remap = {v: k for k, v in enumerate(used)}
        self.P = [self.P[v] for v in used]
        self.fences = [[remap[a], remap[b]] for a, b in self.fences]
        self.pins = {remap[v]: h for v, h in self.pins.items() if v in remap}
        self._dirty()

    # -- face bookkeeping ------------------------------------------------------
    def edge_fence_map(self) -> dict:
        topo = self.topology()
        emap = {}
        for f, c in enumerate(topo.chains):
            for k in range(len(c) - 1):
                emap[(c[k], c[k + 1])] = f
                emap[(c[k + 1], c[k])] = f
        return emap

    def face_cycles(self):
        """[(cycle, signed_area, is_outer)], cycles with their face on the left."""
        topo = self.topology()
        X = self.coords()
        out = [(c, G.polygon_signed_area(X[c]), False) for c in topo.faces]
        out.append((topo.outer, G.polygon_signed_area(X[topo.outer]), True))
        return out

    def locate(self, pt, tol=1e-7):
        """('vertex', v) | ('fence', f, t) | None for an existing point of the sketch."""
        X = self.coords()
        d = np.linalg.norm(X - pt, axis=1)
        v = int(np.argmin(d))
        if d[v] < tol:
            return ("vertex", v)
        for f, (a, b) in enumerate(self.fences):
            if G.point_segment_distance(np.asarray(pt), X[a], X[b]) < tol:
                d = X[b] - X[a]
                t = float(np.dot(pt - X[a], d) / np.dot(d, d))
                if 1e-9 < t < 1 - 1e-9:
                    return ("fence", f, t)
        return None

    def attach(self, pt, tol=1e-7) -> int:
        loc = self.locate(np.asarray(pt, dtype=float), tol)
        if loc is None:
            raise BuildError("attach point not on the sketch")
        if loc[0] == "vertex":
            return loc[1]
        return self.add_pin(loc[1], loc[2])

    # -- the growth primitive --------------------------------------------------
    def add_path_points(self, P, Q, inner) -> list[int]:
        """Add fences P -> inner... -> Q (P, Q must lie on the sketch)."""
        vp = self.attach(P)
        vq = self.attach(Q)
        chain = [vp] + [self.add_vertex(*a) for a in inner] + [vq]
        return [self.add_fence(chain[k], chain[k + 1]) for k in range(len(chain) - 1)]


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

def ring(k: int, side: float = 1.0, squash: float = 1.0, rot: float = 0.0) -> Sketch:
    R = side / (2 * math.sin(math.pi / k))
    P = [[R * math.cos(rot + 2 * math.pi * i / k) * squash, R * math.sin(rot + 2 * math.pi * i / k) / squash]
         for i in range(k)]
    return Sketch(P, [[i, (i + 1) % k] for i in range(k)], {})


def grid(rows: int, cols: int) -> Sketch:
    idx = {}
    P = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            idx[(r, c)] = len(P)
            P.append([c - cols / 2, r - rows / 2])
    F = []
    for r in range(rows + 1):
        for c in range(cols):
            F.append([idx[(r, c)], idx[(r, c + 1)]])
    for r in range(rows):
        for c in range(cols + 1):
            F.append([idx[(r, c)], idx[(r + 1, c)]])
    return Sketch(P, F, {})


def pinwheel_in_ring(k: int, K: int, s: float = 0.35, rot: float = 0.0) -> Sketch:
    """Central k-gon pinwheel (each fence's start pinned on the previous fence) whose
    outer ends are pinned on a surrounding ring of K fences."""
    sk = ring(K, rot=rot)
    X = sk.coords()
    Rc = s / (2 * math.sin(math.pi / k))
    c = [np.array([Rc * math.cos(2 * math.pi * j / k), Rc * math.sin(2 * math.pi * j / k)]) for j in range(k)]
    ends = []
    for j in range(k):
        a, b = c[j], c[(j + 1) % k]
        u = (b - a) / np.linalg.norm(b - a)
        # ray from a through b: first hit with the ring
        best = None
        for f, (p, q) in enumerate(sk.fences):
            P1, P2 = X[p], X[q]
            M = np.array([u, P1 - P2]).T
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            lam, mu = np.linalg.solve(M, P1 - a)
            if lam > np.linalg.norm(b - a) + 1e-6 and 0.02 < mu < 0.98 and (best is None or lam < best[0]):
                best = (lam, f, mu, a + lam * u)
        if best is None:
            raise BuildError("pinwheel ray missed the ring")
        ends.append(best)
    # fence j runs from c_j (pinned on fence j-1) through c_{j+1} to the ring
    ring_n = sk.n
    ring_pins = [sk.add_pin(f, mu) for (_, f, mu, _) in ends]
    starts = [sk.add_vertex(*c[j]) for j in range(k)]
    for j in range(k):
        sk.add_fence(starts[j], ring_pins[j])
    for j in range(k):
        sk.pins[starts[j]] = ring_n + (j - 1) % k       # c_j lies on fence j-1
    sk._dirty()
    return sk


# ---------------------------------------------------------------------------
# growth
# ---------------------------------------------------------------------------

# Growth paths of m >= 3 fences used to be laid down and scored in the wrong vertex order (see
# grow_once and sample_growth below), so no growth step ever added a 3-fence cell. The fix is
# EXPERIMENTS.md change 4: mean fill of a verified result rose from 0.868 to 0.911 at equal
# throughput.

DEFAULT_GROW = {
    "target_area": 0.95,     # desired area of each new cell
    "m_weights": [0.3, 0.45, 0.25],   # probability of paths with 1, 2, 3 fences
    "inside_prob": 0.3,      # probability to split a bounded face instead of growing outward
    "split_min_area": 1.2,   # only faces larger than this are split
    "snap_prob": 0.35,       # snap path ends to existing vertices (joints) instead of new T-pins
    "chord_tol": 0.3,        # |chord length - 1| tolerated for single-fence paths
    "candidates": 48,
    "greedy": 0.75,          # take the best-scoring candidate (else a random one of the top 5)
    "len_jitter": 0.06,      # relative jitter of path segment lengths
    "clearance": 0.04,       # min sketch distance of new geometry from existing edges
}


def _cycle_positions(X, cyc):
    pts = X[cyc]
    nxt = np.roll(pts, -1, axis=0)
    seg = np.linalg.norm(nxt - pts, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return pts, nxt, seg, cum


def _point_at(pts, nxt, seg, cum, a):
    L = cum[-1]
    a = a % L
    i = int(np.searchsorted(cum, a, side="right") - 1)
    i = min(i, len(seg) - 1)
    s = (a - cum[i]) / max(seg[i], 1e-300)
    return i, s, pts[i] + s * (nxt[i] - pts[i])


def _snap(i, s, seg, snap_dist):
    """Move a boundary position to the nearest cycle vertex if close enough."""
    if s * seg[i] < snap_dist:
        return i, 0.0
    if (1 - s) * seg[i] < snap_dist:
        return (i + 1) % len(seg), 0.0
    return i, s


def sample_growth(sk: Sketch, rng, params: dict, face=None, m_max=3):
    """Propose one cell-creating path; returns (P, Q, inner, meta) or None."""
    p = {**DEFAULT_GROW, **(params or {})}
    X = sk.coords()
    cycles = sk.face_cycles()
    emap = sk.edge_fence_map()
    if face is None:
        big = [c for c in cycles if not c[2] and c[1] > p["split_min_area"]]
        if big and rng.random() < p["inside_prob"]:
            w = np.array([c[1] for c in big])
            face = big[int(rng.choice(len(big), p=w / w.sum()))]
        else:
            face = cycles[-1]            # outer face
    cyc, farea, is_outer = face
    pts, nxt, seg, cum = _cycle_positions(X, cyc)
    L = cum[-1]
    mw = np.array(p["m_weights"][:m_max], dtype=float)
    if mw.sum() <= 0:
        return None
    mw = mw / mw.sum()
    E = np.array([[X[a], X[b]] for a, b in _all_edges(sk)])
    face_poly = X[cyc]
    vpos = cum[:-1]                      # arclength position of each cycle vertex
    ncyc = len(cyc)
    cands = []
    for _ in range(int(p["candidates"])):
        m = int(rng.choice(len(mw), p=mw)) + 1
        a0 = rng.random() * L
        # chain length along the boundary: ~ (4 - m) for square-ish unit cells
        c = rng.uniform(max(0.25, 2.3 - m), 4.5 - m)
        if c >= L - 0.2:
            continue
        i1, s1, _ = _point_at(pts, nxt, seg, cum, a0)
        i2, s2, _ = _point_at(pts, nxt, seg, cum, a0 + c)
        if rng.random() < p["snap_prob"]:
            i1, s1 = _snap(i1, s1, seg, 0.3)
        if rng.random() < p["snap_prob"]:
            i2, s2 = _snap(i2, s2, seg, 0.3)
        aP = vpos[i1] + s1 * seg[i1]
        aQ = vpos[i2] + s2 * seg[i2]
        span = (aQ - aP) % L
        if span < 0.15 or span > L - 0.15:
            continue
        Pp = pts[i1] + s1 * (nxt[i1] - pts[i1])
        Qp = pts[i2] + s2 * (nxt[i2] - pts[i2])
        # boundary chain P -> Q: the cycle vertices strictly between them (by arclength)
        rel = (vpos - aP) % L
        ks = [k for k in np.argsort(rel, kind="stable") if 1e-9 < rel[k] < span - 1e-9]
        chain = [Pp] + [pts[k] for k in ks] + [Qp]
        d = float(np.linalg.norm(Pp - Qp))
        if d < 0.15:
            continue
        if m == 1:
            # a single straight fence must not run along the fence(s) P or Q sit on
            fP = emap.get((cyc[i1], cyc[(i1 + 1) % ncyc])) if s1 > 0 else None
            fQ = emap.get((cyc[i2], cyc[(i2 + 1) % ncyc])) if s2 > 0 else None
            if abs(d - 1.0) > p["chord_tol"] or not ks or (fP is not None and fP == fQ):
                continue
            inner = []
        else:
            ell = 1.0 + p["len_jitter"] * rng.normal()
            inner = arc_points(Qp, Pp, m, ell)
            if inner is None:
                continue
        # arc_points returns the interior points ordered Q -> P, so the closed cell is
        # P -> (boundary chain) -> Q -> inner, not the reverse
        cell = np.array(chain + inner)
        area = G.polygon_signed_area(cell)
        if area <= 0.05:
            continue
        if not is_outer and farea - area < 0.08:
            continue
        cost = abs(area - p["target_area"]) + (0.5 * abs(d - 1.0) if m == 1 else 0.0) + 0.02 * rng.random()
        cands.append((cost, Pp, Qp, inner, m, area, i1, s1, i2, s2))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    order = list(range(len(cands)))
    if rng.random() > p["greedy"]:
        head = order[:5]
        rng.shuffle(head)
        order = head + order[5:]
    for idx in order[:12]:
        cost, Pp, Qp, inner, m, area, i1, s1, i2, s2 = cands[idx]
        path = [Qp] + list(inner) + [Pp]
        if _path_clear(X, E, cyc, face_poly, is_outer, path, p["clearance"]):
            return Pp, Qp, inner, {"m": m, "area": area, "outer": is_outer}
    return None


def _all_edges(sk: Sketch):
    return sk.topology().edges()


def _path_clear(X, E, cyc, face_poly, is_outer, path, clearance) -> bool:
    path = [np.asarray(q, dtype=float) for q in path]
    segs = [(path[k], path[k + 1]) for k in range(len(path) - 1)]
    A = E[:, 0]
    B = E[:, 1]
    for k, (s0, s1) in enumerate(segs):
        dist = G.segment_distance(np.broadcast_to(s0, A.shape), np.broadcast_to(s1, A.shape), A, B)
        # edges touching the path's end points are allowed to touch there
        touch = np.zeros(len(E), dtype=bool)
        for endpt in ([s0] if k == 0 else []) + ([s1] if k == len(segs) - 1 else []):
            touch |= G.point_segment_distance(np.broadcast_to(endpt, A.shape), A, B) < 1e-7
        bad = (~touch) & (dist < clearance)
        if bad.any():
            return False
        if touch.any():
            # the segment must leave its endpoint: its other end far from touching edges
            far = s1 if k == 0 else s0
            if len(segs) == 1:
                far = 0.5 * (s0 + s1)
            if (G.point_segment_distance(np.broadcast_to(far, A[touch].shape), A[touch], B[touch]) < clearance).any():
                return False
    # the path must run through the chosen face
    probes = [0.5 * (s0 + s1) for s0, s1 in segs] + path[1:-1]
    for q in probes:
        ins = _pip(q, face_poly)
        if is_outer and ins:
            return False
        if not is_outer and not ins:
            return False
    return True


def grow_once(sk: Sketch, rng, params=None, m_max=3, sym=1, face=None) -> bool:
    """Apply one growth step (and its rotated copies if sym > 1). True on success."""
    for _ in range(6):
        prop = sample_growth(sk, rng, params, face=face, m_max=m_max)
        if prop is None:
            continue
        Pp, Qp, inner, meta = prop
        trial = sk.copy()
        try:
            copies = [0.0] + [2 * math.pi * j / sym for j in range(1, sym)]
            # add_path_points lays the fences down as P -> inner... -> Q, so the interior points
            # must run P -> Q; arc_points ordered them Q -> P. Without the reversal a path of
            # three or more fences crosses itself and topology() rejects every such growth step.
            seq = inner[::-1]
            for th in copies:
                R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
                rot = lambda q: sk.center + R @ (np.asarray(q) - sk.center)  # noqa: E731
                trial.add_path_points(rot(Pp), rot(Qp), [rot(a) for a in seq])
            trial.topology()
        except (BuildError, G.InvalidTopology):
            continue
        sk.P, sk.fences, sk.pins, sk._topo = trial.P, trial.fences, trial.pins, trial._topo
        return True
    return False


# ---------------------------------------------------------------------------
# structural edits (all keep the sketch legal or leave it untouched)
# ---------------------------------------------------------------------------

def _ends_at(sk: Sketch):
    ends = {}
    for f, (a, b) in enumerate(sk.fences):
        ends.setdefault(a, []).append(f)
        ends.setdefault(b, []).append(f)
    return ends


def _commit(sk: Sketch, trial: Sketch) -> bool:
    try:
        trial.compact()
        trial.topology()
    except (BuildError, G.InvalidTopology):
        return False
    sk.P, sk.fences, sk.pins, sk._topo = trial.P, trial.fences, trial.pins, trial._topo
    return True


def remove_fence(sk: Sketch, rng, f=None) -> bool:
    hosts = set(sk.pins.values())
    ends = _ends_at(sk)
    order = [f] if f is not None else list(rng.permutation(sk.n))
    for f in order:
        f = int(f)
        if f in hosts:
            continue
        trial = sk.copy()
        a, b = trial.fences[f]
        del trial.fences[f]
        trial.pins = {v: (h if h < f else h - 1) for v, h in trial.pins.items()}
        for v in (a, b):
            if len(ends[v]) == 1:          # f was the only fence ending at v (a pin): drop it
                trial.pins.pop(v, None)
        if _commit(sk, trial):
            return True
    return False


def project_pins(sk: Sketch, rounds: int = 80, tol: float = 1e-15, clamp: float | None = None) -> None:
    """Put every pinned vertex exactly on its host's line, in place. Pins may sit on hosts whose ends
    are pins themselves (a pinwheel's ends form a cycle), so the projections alternate until nothing
    moves. The parameter along the host is kept in [clamp, 1 - clamp] when `clamp` is given, else
    left free (make_topology judges it)."""
    for _ in range(rounds):
        X = sk.coords()
        moved = 0.0
        for v, h in sk.pins.items():
            a, b = sk.fences[h]
            d = X[b] - X[a]
            dd = float(d @ d)
            if dd < 1e-18:
                continue
            t = float((X[v] - X[a]) @ d) / dd
            if clamp is not None:
                t = min(1 - clamp, max(clamp, t))
            q = X[a] + t * d
            moved = max(moved, float(np.abs(q - X[v]).max()))
            X[v] = q
            sk.P[v] = [float(q[0]), float(q[1])]
        sk._dirty()
        if moved < tol:
            break


def pinwheel_at(sk: Sketch, v: int, size: float = 0.05) -> Sketch | None:
    """The joint v, where k >= 3 fence ends meet, turned into a pinwheel: each of those fences stops
    just past v on the next one around it (a T-junction), so the k ends enclose a tiny k-gon. No fence
    is added; v stops being a joint, so the layout gains a face (F = n + 1 - J). Returns the new
    sketch, or None when v is not such a joint or neither handedness gives a legal layout.

    Geometry: fence i keeps its far end A_i and runs along the line through v + d*rot90(u_i) with
    direction u_i (u_i = unit vector from v to A_i, ends sorted by angle); its new end is where that
    line meets fence i+1's shifted line. One sign of d makes every end land inside the next fence."""
    if v in sk.pins:
        return None
    ends = [f for f, (a, b) in enumerate(sk.fences) if v in (a, b)]
    if len(ends) < 3:
        return None
    X = sk.coords()
    far = {f: (sk.fences[f][1] if sk.fences[f][0] == v else sk.fences[f][0]) for f in ends}
    U = {f: (X[far[f]] - X[v]) / max(1e-12, np.linalg.norm(X[far[f]] - X[v])) for f in ends}
    ends.sort(key=lambda f: math.atan2(U[f][1], U[f][0]))
    k = len(ends)
    for d in (size, -size):
        base = {f: X[v] + d * np.array([-U[f][1], U[f][0]]) for f in ends}
        P = {}
        ok = True
        for i, f in enumerate(ends):
            g = ends[(i + 1) % k]
            # base_f + s u_f = base_g + r u_g
            M = np.column_stack([U[f], -U[g]])
            if abs(np.linalg.det(M)) < 1e-9:
                ok = False
                break
            s_, r_ = np.linalg.solve(M, base[g] - base[f])
            P[f] = base[f] + s_ * U[f]
        if not ok:
            continue
        # fence i really runs from its fixed far end A_i to P_i, not along the shifted line: settle
        # every pin onto its host segment by alternating projections (converges in a few rounds)
        for _ in range(60):
            for i, f in enumerate(ends):
                g = ends[(i + 1) % k]
                a, b = X[far[g]], P[g]
                dg = b - a
                t = float((P[f] - a) @ dg) / float(dg @ dg)
                P[f] = a + t * dg
        trial = sk.copy()
        new = {}
        for f in ends:
            new[f] = trial.add_vertex(*P[f])
        for i, f in enumerate(ends):
            g = ends[(i + 1) % k]
            trial.fences[f] = [new[f] if u == v else u for u in trial.fences[f]]
            trial.pins[new[f]] = g
        project_pins(trial)          # pins already on the rotated fences follow them
        try:
            trial.compact()
            trial.topology()
        except (BuildError, G.InvalidTopology):
            continue
        return trial
    return None


def pinwheel(sk: Sketch, rng, size: float = 0.05, v: int | None = None) -> str | None:
    """Turn a random joint of >= 3 fence ends (or v) into a pinwheel (`pinwheel_at`), in place."""
    ends = _ends_at(sk)
    cands = [v] if v is not None else [u for u, fs in ends.items() if len(fs) >= 3 and u not in sk.pins]
    if v is None:
        cands = [cands[i] for i in rng.permutation(len(cands))]
    for u in cands[:8]:
        trial = pinwheel_at(sk, int(u), size) or pinwheel_at(sk, int(u), 0.4 * size)
        if trial is not None:
            sk.P, sk.fences, sk.pins, sk._topo = trial.P, trial.fences, trial.pins, trial._topo
            return f"joint {u} ({len(ends[u])} ends)"
    return None


def toggle_junction(sk: Sketch, rng) -> bool:
    ends = _ends_at(sk)
    X = sk.coords()
    opts = []
    for v, h in sk.pins.items():
        opts.append(("t2l", v, h))
    for v, fs in ends.items():
        if v not in sk.pins and len(fs) >= 3:
            opts.append(("l2t", v, None))
    if not opts:
        return False
    rng.shuffle(opts)
    for kind, v, h in opts[:6]:
        trial = sk.copy()
        if kind == "t2l":
            a, b = trial.fences[h]
            t = G.fence_param(X, a, b, v)
            tgt = a if t < 0.5 else b
            for f in ends[v]:
                trial.fences[f] = [tgt if u == v else u for u in trial.fences[f]]
            del trial.pins[v]
            if any(p == q for p, q in trial.fences):
                continue
        else:
            fs = list(ends[v])
            rng.shuffle(fs)
            f, g = fs[0], fs[1]
            ga, gb = trial.fences[g]
            other = gb if ga == v else ga
            t = rng.uniform(0.12, 0.3)
            q = (1 - t) * X[v] + t * X[other]
            nv = trial.add_vertex(*q)
            # new vertex lies in the interior of g, measured from g's endpoint a
            trial.pins[nv] = g
            trial.fences[f] = [nv if u == v else u for u in trial.fences[f]]
        if _commit(sk, trial):
            return True
    return False


def reattach(sk: Sketch, rng) -> bool:
    ends = _ends_at(sk)
    X = sk.coords()
    emap = sk.edge_fence_map()
    cycles = sk.face_cycles()
    for _ in range(8):
        f = int(rng.integers(sk.n))
        e = int(rng.integers(2))
        v = sk.fences[f][e]
        others = [g for g in ends[v] if g != f]
        if v not in sk.pins and len(others) < 2:
            continue                      # detaching would leave a lone endpoint
        cand = [c for c in cycles if v in c[0]]
        if not cand:
            continue
        cyc = cand[int(rng.integers(len(cand)))][0]
        k = int(rng.integers(len(cyc)))
        u, w = cyc[k], cyc[(k + 1) % len(cyc)]
        if v in (u, w):
            continue
        host = emap.get((u, w))
        if host is None or host == f:
            continue
        s = rng.uniform(0.15, 0.85)
        q = (1 - s) * X[u] + s * X[w]
        trial = sk.copy()
        nv = trial.add_vertex(*q)
        trial.pins[nv] = host
        trial.fences[f][e] = nv
        if not others:
            trial.pins.pop(v, None)
        if _commit(sk, trial):
            return True
    return False


def add_face(sk: Sketch, rng, grow=None) -> bool:
    """Grow one new cell anywhere (adds 1-3 fences)."""
    return grow_once(sk, rng, grow, m_max=3)


def remove_face(sk: Sketch, rng, grow=None) -> bool:
    """Delete one removable fence, merging the two faces it separated."""
    return remove_fence(sk, rng)


def move_fence(sk: Sketch, rng, grow=None) -> bool:
    """remove_face followed by a single-fence add_face elsewhere (n unchanged)."""
    snap = sk.copy()
    if remove_fence(sk, rng) and grow_once(sk, rng, grow, m_max=1):
        return True
    sk.P, sk.fences, sk.pins, sk._topo = snap.P, snap.fences, snap.pins, snap._topo
    return False


# ---------------------------------------------------------------------------
# face-aware edits (on the sketch of a verified configuration)
# ---------------------------------------------------------------------------
#
# A connected layout with n fences and J joints (vertices that are not T-pins) has exactly
# F = n + 1 - J bounded faces (Euler: E = n + #pins, V = J + #pins). An m-fence cell adds
# m - 1 joints and removing a pin-free fence removes none, so at fixed n only an edit that
# removes a joint can raise F. `face_fill` reads the face areas off the sketch itself (a
# sketch made from a verified configuration carries its exact coordinates) and applies one of
#   relocate  a face below `dead` loses the boundary fence whose removal costs the least area;
#             the fence comes back as a one-fence cell (J unchanged, so F unchanged)
#   dissolve  a two-fence corner joint goes away: one of its fences is removed, the other's
#             loose end is pinned on the nearest edge it can reach, and the removed fence comes
#             back as a one-fence cell (J - 1, so F + 1)

DEAD_FACE = 0.9          # a bounded face below this area is under-filled
MIN_FACE = 1e-3          # the optimizer's min_face: a sketch face below this starts infeasible
MIN_EDGE = 0.1           # new pins only on edges at least this long ...
PIN_T = (0.15, 0.85)     # ... and this far inside them
FENCE_LEN = (0.6, 1.4)   # sketch lengths an edit may leave (the optimizer makes them exactly 1)
CELL_GROW = {"target_area": 1.0, "candidates": 48, "greedy": 1.0}


def face_sides(sk: Sketch):
    """(bounded face cycles, their areas, {directed edge (u, w): face on its left}); the
    unbounded face is -1."""
    topo = sk.topology()
    left = {}
    for i, cyc in enumerate(list(topo.faces) + [topo.outer]):
        fid = i if i < len(topo.faces) else -1
        for k in range(len(cyc)):
            left[(cyc[k], cyc[(k + 1) % len(cyc)])] = fid
    return topo.faces, G.face_areas(sk.coords(), topo.faces), left


def removal_cost(sides, areas) -> float | None:
    """Area lost when the fence between two faces goes: merged bounded faces keep at most 1,
    a face opened to the outside is lost. None for a fence with one face on both sides
    (removing it would disconnect the layout)."""
    a, b = sides
    if a == b:
        return None
    if a == -1 or b == -1:
        return float(areas[b if a == -1 else a])
    return max(0.0, float(areas[a] + areas[b]) - 1.0)


def _corner(sk: Sketch, ends, v) -> bool:
    """v is a joint where exactly two fences end."""
    return v not in sk.pins and len(ends[v]) == 2


def _drop_fence(sk: Sketch, f: int) -> None:
    """Delete fence f (hosting no pin) without validating; pins whose only fence was f go too."""
    ends = _ends_at(sk)
    a, b = sk.fences[f]
    del sk.fences[f]
    sk.pins = {v: (h if h < f else h - 1) for v, h in sk.pins.items()}
    for v in (a, b):
        if len(ends[v]) == 1:
            sk.pins.pop(v, None)
    sk._dirty()


def _one_fence_cell(sk: Sketch, rng, greedy: float = 1.0) -> str | None:
    """Add one fence that closes a new cell: in a notch of the outer boundary or across a
    bounded face of area >= 0.5 (largest first), in random order of the two. Returns where.
    greedy < 1 sometimes takes one of the five best placements instead of the best."""
    grow = {**CELL_GROW, "greedy": greedy}
    places = ["outer", "split"] if rng.random() < 0.5 else ["split", "outer"]
    for where in places:
        if where == "outer":
            if grow_once(sk, rng, grow, m_max=1, face=sk.face_cycles()[-1]):
                return where
            continue
        for face in sorted((c for c in sk.face_cycles()[:-1] if c[1] >= 0.5), key=lambda c: -c[1])[:3]:
            if grow_once(sk, rng, {**grow, "target_area": 0.5 * face[1]}, m_max=1, face=face):
                return where
    return None


def _acceptable(sk: Sketch, n: int, faces_expected: int, reject=None) -> bool:
    topo = sk.topology()
    if sk.n != n or len(topo.faces) != faces_expected:
        return False
    X = sk.coords()
    if G.face_areas(X, topo.faces).min(initial=1.0) < MIN_FACE:
        return False
    L = np.linalg.norm(X[topo.fences[:, 1]] - X[topo.fences[:, 0]], axis=1)
    return bool(np.all((L >= FENCE_LEN[0]) & (L <= FENCE_LEN[1]))) and not (reject and reject(sk))


def _with_cell(base: Sketch, rng, n: int, faces_expected: int, reject, tries: int = 3):
    """(where, base plus a one-fence cell) passing `_acceptable`, or None after `tries` cells
    (the best placement first, then less greedy ones)."""
    for k in range(tries):
        trial = base.copy()
        where = _one_fence_cell(trial, rng, greedy=1.0 if k == 0 else 0.3)
        if where is not None and _acceptable(trial, n, faces_expected, reject):
            return where, trial
    return None


def _relocate(sk: Sketch, rng, faces, areas, left, dead_ids, reject=None):
    hosts = set(sk.pins.values())
    ends = _ends_at(sk)
    emap = sk.edge_fence_map()
    for d in dead_ids:
        cand = {}
        cyc = faces[d]
        for k in range(len(cyc)):
            f = emap[(cyc[k], cyc[(k + 1) % len(cyc)])]
            if f in cand or f in hosts or any(_corner(sk, ends, v) for v in sk.fences[f]):
                continue
            a, b = sk.fences[f]
            sides = (left[(a, b)], left[(b, a)])
            c = removal_cost(sides, areas)
            if c is not None:
                cand[f] = (c, sides)
        for f, (c, sides) in sorted(cand.items(), key=lambda kv: (kv[1][0], rng.random())):
            base = sk.copy()
            if not remove_fence(base, rng, f=f):
                continue
            trial = _with_cell(base, rng, sk.n, len(faces), reject)
            if trial is None:
                continue
            where, trial = trial
            sk.P, sk.fences, sk.pins, sk._topo = trial.P, trial.fences, trial.pins, trial._topo
            shared = all(s != -1 and areas[s] < DEAD_FACE for s in sides)
            return {"move": "relocate", "where": where, "face_area": round(float(areas[d]), 6),
                    "cost": round(c, 6), "shared": shared}
    return None


def _repin_targets(sk: Sketch, v: int, w: int, skip: set[int]):
    """Points on edges of the sketch where a fence end now at v could be pinned instead, nearest
    first: (distance, host fence, point). Edges of fences in `skip` and edges touching w (the
    fence's other end) are left out."""
    X = sk.coords()
    topo = sk.topology()
    out = []
    for h, chain in enumerate(topo.chains):
        if h in skip:
            continue
        for p, q in zip(chain[:-1], chain[1:]):
            if w in (p, q) or v in (p, q):
                continue
            d = X[q] - X[p]
            ln = float(np.linalg.norm(d))
            if ln < MIN_EDGE:
                continue
            s = float(np.clip(np.dot(X[v] - X[p], d) / (ln * ln), *PIN_T))
            Q = X[p] + s * d
            if FENCE_LEN[0] <= np.linalg.norm(Q - X[w]) <= FENCE_LEN[1]:
                out.append((float(np.linalg.norm(Q - X[v])), h, Q))
    out.sort(key=lambda t: t[0])
    return out


def _dissolve(sk: Sketch, rng, faces, areas, left, only_faces=None, reject=None):
    hosts = set(sk.pins.values())
    ends = _ends_at(sk)
    near = set().union(*(faces[i] for i in only_faces)) if only_faces is not None else None
    cand = []
    for v, fs in ends.items():
        if not _corner(sk, ends, v) or (near is not None and v not in near):
            continue
        for f, g in ((fs[0], fs[1]), (fs[1], fs[0])):
            u = sk.fences[f][1] if sk.fences[f][0] == v else sk.fences[f][0]
            if f in hosts or _corner(sk, ends, u):
                continue                   # f must go cleanly: no pins on it, no second loose end
            a, b = sk.fences[f]
            c = removal_cost((left[(a, b)], left[(b, a)]), areas)
            if c is not None:
                cand.append((c, rng.random(), v, f, g))
    cand.sort()
    for c, _, v, f, g in cand[:6]:
        w = sk.fences[g][1] if sk.fences[g][0] == v else sk.fences[g][0]
        for dist, h, Q in _repin_targets(sk, v, w, {f, g})[:4]:
            trial = sk.copy()
            nv = trial.add_vertex(*Q)
            trial.pins[nv] = h
            trial.fences[g] = [nv if x == v else x for x in trial.fences[g]]
            _drop_fence(trial, f)
            try:
                trial.compact()
                trial.topology()
            except (BuildError, G.InvalidTopology):
                continue
            got = _with_cell(trial, rng, sk.n, len(faces) + 1, reject)
            if got is None:
                continue
            where, trial = got
            sk.P, sk.fences, sk.pins, sk._topo = trial.P, trial.fences, trial.pins, trial._topo
            return {"move": "dissolve", "where": where, "cost": round(c, 6), "repin_dist": round(dist, 4)}
    return None


def face_fill(sk: Sketch, rng, target_faces: int | None = None, dead: float = DEAD_FACE,
              reject=None) -> dict | None:
    """Face-aware edit of the sketch of a verified configuration (see above).

    A face below `dead` is relocated first; if no fence of an under-filled face can be moved,
    a corner joint on one of them is dissolved. Otherwise, with fewer faces than
    `target_faces`, any corner joint is dissolved. Edits sk in place and returns what was done
    ({"move", "where", ...}), or returns None with sk untouched. The result is a legal embedding
    with the same number of fences, the predicted number of faces (relocate: F, dissolve: F + 1),
    no face below MIN_FACE and every fence's sketch length within FENCE_LEN. `reject(sketch)`
    may veto candidate results (e.g. ones with the parent's topology); the next one is tried.
    """
    faces, areas, left = face_sides(sk)
    dead_ids = [int(i) for i in np.argsort(areas) if areas[i] < dead]
    if dead_ids:
        info = _relocate(sk, rng, faces, areas, left, dead_ids, reject)
        if info is None:
            info = _dissolve(sk, rng, faces, areas, left, only_faces=dead_ids, reject=reject)
            if info is not None:
                info["move"] = "dissolve@dead"
        if info is not None:
            return info
    if target_faces is not None and len(faces) < target_faces:
        return _dissolve(sk, rng, faces, areas, left, reject=reject)
    return None


# Structural mutation vocabulary (names used in genomes and by the random mutator):
#   add_face / remove_face  -- change the face structure (fence count is fixed up afterwards)
#   split_edge              -- detach a fence end and pin it on another edge (new T-junction)
#   toggle_junction         -- T-junction <-> endpoint joint
#   move_fence              -- remove one fence and re-add it somewhere else
EDITS = {
    "add_face": add_face,
    "remove_face": remove_face,
    "split_edge": lambda sk, rng, grow=None: reattach(sk, rng),
    "toggle_junction": lambda sk, rng, grow=None: toggle_junction(sk, rng),
    "move_fence": move_fence,
    # backwards-compatible aliases
    "reattach": lambda sk, rng, grow=None: reattach(sk, rng),
    "remove_add": move_fence,
}


# ---------------------------------------------------------------------------
# genome interpreter
# ---------------------------------------------------------------------------

def make_seed(n: int, rng, seed: dict) -> Sketch:
    kind = seed.get("kind", "ring")
    if kind == "ring":
        if "k" in seed:
            k = int(seed["k"])
        else:
            k = int(round(2 * math.sqrt(math.pi * a_est(n)) * seed.get("ring_scale", 1.0)))
        k = max(3, min(k, n))
        return ring(k, squash=seed.get("squash", 1.0), rot=rng.uniform(0, 2 * math.pi))
    if kind == "literal_grid":
        # an explicit (possibly too large) grid; fit_count prunes interior fences down to n
        return grid(int(seed["rows"]), int(seed["cols"]))
    if kind == "grid":
        r, c = int(seed.get("rows", 0)), int(seed.get("cols", 0))
        if r <= 0 or c <= 0 or 2 * r * c + r + c > n:
            # largest near-square grid that fits in n fences (r x c grid uses 2rc + r + c)
            r, c = 1, 1
            for rr in range(1, 12):
                for cc in range(rr, rr + 3):
                    if 2 * rr * cc + rr + cc <= n and rr * cc > r * c:
                        r, c = rr, cc
        return grid(r, c)
    if kind == "pinwheel":
        k = int(seed.get("k", 3))
        K = int(seed.get("K", max(k + 3, int(round(2 * math.sqrt(math.pi * min(a_est(n), 6.0)))))))
        for KK in (K, K + 1, K + 2):
            for s in (seed.get("s", 0.35), 0.25, 0.5):
                try:
                    return pinwheel_in_ring(k, KK, s=s)
                except BuildError:
                    continue
        raise BuildError("pinwheel seed could not be placed")
    if kind == "literal":
        return Sketch.from_json(seed["sketch"])
    raise BuildError(f"unknown seed kind {kind!r}")


def build_from_genome(n: int, rng, genome: dict) -> Sketch:
    """Standard interpreter: seed -> symmetric/asymmetric growth to n fences -> edits."""
    seed = dict(genome.get("seed", {"kind": "ring"}))
    sym = int(genome.get("sym", 1))
    if sym > 1 and seed.get("kind") == "ring":
        # a C_sym-symmetric start needs a ring size divisible by sym
        k = seed.get("k") or int(round(2 * math.sqrt(math.pi * a_est(n)) * seed.get("ring_scale", 1.0)))
        seed["k"] = max(sym, int(round(k / sym)) * sym)
        seed["squash"] = 1.0
    sk = make_seed(n, rng, seed)
    grow = {**DEFAULT_GROW, **genome.get("grow", {})}
    fit_count(sk, n, rng, grow, sym)
    for ed in genome.get("edits", []):
        name = ed.get("op") if isinstance(ed, dict) else ed
        if name not in EDITS:
            continue
        # each edit is tried on a copy and kept only if the fence count can be restored
        trial = sk.copy()
        try:
            if EDITS[name](trial, rng, grow):
                fit_count(trial, n, rng, grow, 1)
                sk.P, sk.fences, sk.pins, sk._topo = trial.P, trial.fences, trial.pins, trial._topo
        except (BuildError, G.InvalidTopology):
            pass
    if sk.n != n:
        raise BuildError(f"built {sk.n} fences, wanted {n}")
    sk.topology()
    return sk


def fit_count(sk: Sketch, n: int, rng, grow: dict, sym: int = 1) -> None:
    """Prune or grow the sketch until it has exactly n fences."""
    guard = 0
    while sk.n > n and guard < 4 * n:
        guard += 1
        if not remove_fence(sk, rng):
            raise BuildError(f"cannot prune sketch from {sk.n} to {n} fences")
    guard = 0
    while sk.n < n and guard < 6 * n:
        guard += 1
        rem = n - sk.n
        s = sym if sym > 1 and rem >= sym else 1
        ok = grow_once(sk, rng, grow, m_max=min(3, rem // s), sym=s)
        if not ok and s > 1:
            ok = grow_once(sk, rng, grow, m_max=min(3, rem), sym=1)
        if not ok:
            # widen the search: try every face in turn
            for face in sk.face_cycles():
                if grow_once(sk, rng, {**grow, "chord_tol": 0.45, "candidates": 96}, m_max=min(3, rem), face=face):
                    ok = True
                    break
        if not ok:
            raise BuildError(f"growth stalled at {sk.n}/{n} fences")
