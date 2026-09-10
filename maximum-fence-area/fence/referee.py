"""Independent referee for Maximum Fence Area configurations (50-digit mpmath).

This module is the *only* thing allowed to certify a result. It deliberately
shares NO code with the optimization path (fence.geometry / fence.optimize /
fence.builder): it imports only the standard library and mpmath, and it re-derives
everything -- incidences, the planar arrangement, the faces and their areas --
from nothing but the raw segment coordinates. A test asserts this import policy.

Input: a configuration dict with key "segments": a list of n entries
[x1, y1, x2, y2] (decimal strings are parsed exactly at 50 digits; Python floats
are converted exactly from binary), or a path to such a JSON file, or the bare
list of segments.

Checks (all arithmetic at mp.dps = 50; `tol` is an absolute tolerance at unit
scale):
  1. every fence has length 1:                 | |P1P2| - 1 | <= tol
  2. every endpoint lies on ANOTHER fence:      dist(endpoint, other segment) <= tol
     (anywhere on it: its endpoint or its interior -- T-junctions are legal)
  3. no two fences cross:                       no proper crossing (all four
     orientation distances > tol with opposite signs), no vertex lying on the
     interiors of two fences, and no collinear overlap longer than tol
  4. every bounded face has area <= 1 + tol, where faces come from a half-edge
     traversal of the arrangement (components nested inside a face are
     subtracted from it as holes).
The reported total area is the sum of the bounded face areas at 50 digits.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import cmp_to_key

import mpmath
from mpmath import mp, mpf

DPS = 50


@dataclass
class VerifyResult:
    valid: bool
    n: int
    total_area: str                     # 50-significant-digit decimal
    face_areas: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    max_length_error: str = "0"
    max_endpoint_gap: str = "0"
    max_face_excess: str = "0"
    n_faces: int = 0
    n_tjunctions: int = 0
    tol: float = 0.0
    dps: int = DPS

    @property
    def area(self) -> float:
        return float(self.total_area)

    def to_json(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        tag = "VALID" if self.valid else "INVALID"
        s = (f"{tag} n={self.n} area={self.total_area[:22]} faces={self.n_faces} "
             f"max|len-1|={float(self.max_length_error):.2e} max_gap={float(self.max_endpoint_gap):.2e} "
             f"max_face_excess={float(self.max_face_excess):.2e} (tol={self.tol:g}, {self.dps} digits)")
        if self.reasons:
            s += " :: " + "; ".join(self.reasons[:4])
        return s


# ---------------------------------------------------------------------------
# exact-ish primitives at 50 digits
# ---------------------------------------------------------------------------

def _num(v) -> mpf:
    if isinstance(v, str):
        return mpf(v.strip())
    if isinstance(v, (int, float)):
        return mpf(v)          # exact binary conversion
    return mpf(str(v))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _norm(a):
    return mpmath.sqrt(_dot(a, a))


def _pt_seg(p, a, b):
    """(distance from p to segment [a,b], projection parameter t along a->b)."""
    d = _sub(b, a)
    dd = _dot(d, d)
    t = _dot(_sub(p, a), d) / dd
    tc = min(max(t, mpf(0)), mpf(1))
    q = (a[0] + tc * d[0], a[1] + tc * d[1])
    return _norm(_sub(p, q)), t


def _line_dist(p, a, b):
    """signed distance of p from the directed line a->b (positive on the left)."""
    d = _sub(b, a)
    return _cross(d, _sub(p, a)) / _norm(d)


def _load(config):
    if isinstance(config, str):
        with open(config) as fh:
            config = json.load(fh)
    if isinstance(config, dict):
        segs = config["segments"]
        n_claim = config.get("n")
    else:
        segs, n_claim = config, None
    return [tuple(_num(v) for v in s) for s in segs], n_claim


# ---------------------------------------------------------------------------
# the referee
# ---------------------------------------------------------------------------

def verify(config, tol: float = 1e-9) -> VerifyResult:
    with mp.workdps(DPS):
        return _verify(config, tol)


def _verify(config, tol) -> VerifyResult:
    segs, n_claim = _load(config)
    n = len(segs)
    T = mpf(tol)
    reasons: list[str] = []
    if n_claim is not None and int(n_claim) != n:
        reasons.append(f"config claims n={n_claim} but has {n} segments")
    if n < 3:
        reasons.append("fewer than 3 fences")
    P = [((s[0], s[1]), (s[2], s[3])) for s in segs]

    # 1. unit lengths ------------------------------------------------------
    max_len_err = mpf(0)
    for i, (a, b) in enumerate(P):
        e = abs(_norm(_sub(b, a)) - 1)
        max_len_err = max(max_len_err, e)
        if e > T:
            reasons.append(f"fence {i} has length error {mpmath.nstr(e, 5)}")
    if any(_norm(_sub(b, a)) <= T for a, b in P):
        return VerifyResult(False, n, "0", reasons=reasons + ["degenerate fence"], tol=tol)

    # 2./3. pairwise relations ----------------------------------------------
    # dist_ep[(i, e)][j] = distance from endpoint e (0/1) of fence i to fence j
    ep_support = {(i, e): False for i in range(n) for e in (0, 1)}
    max_gap = mpf(0)
    for i in range(n):
        p1, p2 = P[i]
        for j in range(i + 1, n):
            q1, q2 = P[j]
            dq1, dq2 = _line_dist(q1, p1, p2), _line_dist(q2, p1, p2)
            dp1, dp2 = _line_dist(p1, q1, q2), _line_dist(p2, q1, q2)
            if (abs(dq1) > T and abs(dq2) > T and (dq1 > 0) != (dq2 > 0) and
                    abs(dp1) > T and abs(dp2) > T and (dp1 > 0) != (dp2 > 0)):
                reasons.append(f"fences {i} and {j} cross")
                continue
            collinear = None
            if abs(dq1) <= T and abs(dq2) <= T:
                collinear = (p1, p2, q1, q2)      # fence j lies (nearly) on fence i's line
            elif abs(dp1) <= T and abs(dp2) <= T:
                collinear = (q1, q2, p1, p2)      # fence i lies (nearly) on fence j's line
            if collinear is not None:
                # measure the overlap along the host direction; touching end-to-end is fine
                a1, a2, b1, b2 = collinear
                d = _sub(a2, a1)
                L = _norm(d)
                s = sorted([_dot(_sub(b1, a1), d) / L, _dot(_sub(b2, a1), d) / L])
                overlap = min(L, s[1]) - max(mpf(0), s[0])
                if overlap > T:
                    reasons.append(f"fences {i} and {j} overlap collinearly ({mpmath.nstr(overlap, 5)})")
                    continue
            # endpoint-on-fence support (either direction)
            for e, pt in enumerate((p1, p2)):
                dist, _ = _pt_seg(pt, q1, q2)
                if dist <= T:
                    ep_support[(i, e)] = True
            for e, pt in enumerate((q1, q2)):
                dist, _ = _pt_seg(pt, p1, p2)
                if dist <= T:
                    ep_support[(j, e)] = True
    for (i, e), ok in ep_support.items():
        if not ok:
            pt = P[i][e]
            gap = min(_pt_seg(pt, *P[j])[0] for j in range(n) if j != i) if n > 1 else mpf(1)
            max_gap = max(max_gap, gap)
            reasons.append(f"endpoint {e} of fence {i} is not on any other fence (gap {mpmath.nstr(gap, 5)})")
    if reasons:
        return VerifyResult(False, n, "0", reasons=reasons, max_length_error=mpmath.nstr(max_len_err, 10),
                            max_endpoint_gap=mpmath.nstr(max_gap, 10), tol=tol)

    # record the largest (sub-tolerance) endpoint gap for reporting
    for i in range(n):
        for e in (0, 1):
            g = min(_pt_seg(P[i][e], *P[j])[0] for j in range(n) if j != i)
            max_gap = max(max_gap, g)

    # 4. arrangement ----------------------------------------------------------
    # 4a. vertices = clusters of endpoints closer than tol (union-find)
    pts = [P[i][e] for i in range(n) for e in (0, 1)]   # endpoint id = 2*i + e
    parent = list(range(2 * n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(2 * n):
        for b in range(a + 1, 2 * n):
            if _norm(_sub(pts[a], pts[b])) <= T:
                parent[find(a)] = find(b)
    roots = sorted({find(a) for a in range(2 * n)})
    vid = {r: k for k, r in enumerate(roots)}
    members = {}
    for a in range(2 * n):
        members.setdefault(vid[find(a)], []).append(a)
    V = len(roots)
    vpos = []
    for v in range(V):
        ms = members[v]
        vpos.append((sum(pts[a][0] for a in ms) / len(ms), sum(pts[a][1] for a in ms) / len(ms)))
    fence_ends = [(vid[find(2 * i)], vid[find(2 * i + 1)]) for i in range(n)]
    for i, (va, vb) in enumerate(fence_ends):
        if va == vb:
            return VerifyResult(False, n, "0", reasons=[f"fence {i} collapses to a point"], tol=tol)

    # 4b. vertices lying in the interior of a fence (T-junctions) -> split points
    on_fence = [[] for _ in range(n)]
    interior_hosts = {}
    for i in range(n):
        a, b = P[i]
        va, vb = fence_ends[i]
        on_fence[i].append((mpf(0), va))
        on_fence[i].append((mpf(1), vb))
        for v in range(V):
            if v in (va, vb):
                continue
            dist, t = _pt_seg(vpos[v], a, b)
            if dist <= T:
                on_fence[i].append((t, v))
                interior_hosts.setdefault(v, []).append(i)
    for v, hs in interior_hosts.items():
        if len(hs) > 1:
            reasons.append(f"vertex {v} lies in the interiors of fences {hs} (crossing)")
    if reasons:
        return VerifyResult(False, n, "0", reasons=reasons, tol=tol)

    edges = set()
    for i in range(n):
        seq = [v for _, v in sorted(on_fence[i], key=lambda tv: tv[0])]
        for k in range(len(seq) - 1):
            u, w = seq[k], seq[k + 1]
            if u == w:
                continue
            key = (min(u, w), max(u, w))
            if key in edges:
                return VerifyResult(False, n, "0", reasons=[f"edge {key} is shared by two fences (overlap)"], tol=tol)
            edges.add(key)
    edges = sorted(edges)

    # 4c. half-edges and the rotation system --------------------------------
    # Half-edge 2k is edges[k] traversed u->w, 2k+1 is w->u, so twin(h) = h ^ 1.
    src = []
    dst = []
    for u, w in edges:
        src += [u, w]
        dst += [w, u]
    H = len(src)
    out = [[] for _ in range(V)]
    for h in range(H):
        out[src[h]].append(h)

    def direction(h):
        return _sub(vpos[dst[h]], vpos[src[h]])

    def half(d):
        # 0 for angles in [0, pi), 1 for [pi, 2pi): a total order on directions
        return 0 if (d[1] > 0 or (d[1] == 0 and d[0] > 0)) else 1

    def ccw_cmp(h1, h2):
        d1, d2 = direction(h1), direction(h2)
        a, b = half(d1), half(d2)
        if a != b:
            return a - b
        c = _cross(d1, d2)
        if c > 0:
            return -1
        if c < 0:
            return 1
        return 0

    pos = [0] * H
    for v in range(V):
        out[v].sort(key=cmp_to_key(ccw_cmp))
        for k in range(len(out[v]) - 1):
            if ccw_cmp(out[v][k], out[v][k + 1]) == 0:
                return VerifyResult(False, n, "0", reasons=[f"two edges leave vertex {v} in the same direction (overlap)"], tol=tol)
        for k, h in enumerate(out[v]):
            pos[h] = k

    # 4d. face walk: keep the face on the LEFT. Arriving at w along h=(u->w),
    # leave along the outgoing half-edge immediately clockwise of twin(h)=(w->u),
    # i.e. the previous entry in w's counter-clockwise rotation.
    nxt = [0] * H
    for h in range(H):
        w = dst[h]
        ring = out[w]
        nxt[h] = ring[(pos[h ^ 1] - 1) % len(ring)]

    seen = [False] * H
    cycles = []
    for h0 in range(H):
        if seen[h0]:
            continue
        cyc = []
        h = h0
        while not seen[h]:
            seen[h] = True
            cyc.append(h)
            h = nxt[h]
        area = sum(_cross(vpos[src[x]], vpos[dst[x]]) for x in cyc) / 2
        cycles.append((cyc, area))

    # 4e. components; each component's outer boundary is its clockwise cycle(s)
    cpar = list(range(V))

    def cfind(x):
        while cpar[x] != x:
            cpar[x] = cpar[cpar[x]]
            x = cpar[x]
        return x

    for u, w in edges:
        cpar[cfind(u)] = cfind(w)
    positive = [(cyc, a) for cyc, a in cycles if a > 0]
    negative = [(cyc, a) for cyc, a in cycles if a <= 0]

    def inside(pt, cyc):
        # even-odd ray casting against the polygon of a half-edge cycle
        x, y = pt
        c = False
        for h in cyc:
            (x1, y1), (x2, y2) = vpos[src[h]], vpos[dst[h]]
            if (y1 > y) != (y2 > y):
                xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if xi > x:
                    c = not c
        return c

    face_area = [a for _, a in positive]
    for cyc, a in negative:
        comp = cfind(src[cyc[0]])
        probe = vpos[src[cyc[0]]]
        best = None
        for k, (pc, pa) in enumerate(positive):
            if cfind(src[pc[0]]) == comp:
                continue
            if inside(probe, pc) and (best is None or pa < positive[best][1]):
                best = k
        if best is not None:
            face_area[best] += a          # subtract the enclosed component (a <= 0)

    # 5. face constraints and total -------------------------------------------
    max_excess = mpf(0)
    for k, a in enumerate(face_area):
        if a - 1 > max_excess:
            max_excess = a - 1
        if a - 1 > T:
            reasons.append(f"face {k} has area {mpmath.nstr(a, 12)} > 1")
        if a <= 0:
            reasons.append(f"face {k} has non-positive area after removing holes")
    total = sum(face_area, mpf(0))
    n_t = sum(1 for v in interior_hosts)
    return VerifyResult(
        valid=not reasons,
        n=n,
        total_area=mpmath.nstr(total, DPS),
        face_areas=[mpmath.nstr(a, 20) for a in face_area],
        reasons=reasons,
        max_length_error=mpmath.nstr(max_len_err, 10),
        max_endpoint_gap=mpmath.nstr(max_gap, 10),
        max_face_excess=mpmath.nstr(max_excess, 10),
        n_faces=len(face_area),
        n_tjunctions=n_t,
        tol=tol,
    )
