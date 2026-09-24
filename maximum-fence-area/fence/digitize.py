"""Digitize a record picture from the fence page into a topology the optimizer can polish.

The page publishes a line drawing of every record (`<n>.gif`: black fences on white). Only
the *topology* has to come out right -- which vertex touches which fence, and where; the
optimizer fixes lengths and angles from any topologically-correct sketch.

Pipeline (numpy / scipy only):
  1. threshold -> boolean mask of the drawn lines (`load_mask`);
  2. Zhang-Suen thinning -> one-pixel skeleton (`thin`);
  3. skeleton -> planar graph: junction / end pixels become nodes, the pixel paths between them
     edges, each path split at its corners by Douglas-Peucker; nearby nodes are merged and short
     spurs dropped (`skeleton_graph`);
  4. graph -> fences (`fences_from_graph`): edges that continue straight through a node are
     chained into *lines*; the picture is rescaled so the total line length is n (fences never
     overlap, so the drawing's ink is exactly n unit fences); a DP (`_cut_line`) cuts every line
     into pieces of near-integer length, each piece into round(length) equal fences, with a forced
     cut wherever two lines would pass through one node; a node inside a fence becomes a pin
     (T-junction), a node at a cut a joint.
  5. `make_topology` validates the embedding; if the count is not exactly n the unit length is
     nudged, then (with `lams`) a per-fence price is added to the DP.

Optional graph repairs, off unless asked for (the v2 grid of `digitize_records.py` turns them on):
`ring_r` (ring-branch junction test for junctions thinning hides), `merge_close`, `tee_shift`.

`digitize(path, n, **kw)` runs all of it and returns (Topology, coords, info). EXPERIMENTS.md
change 11 has what each step fixed.
"""
from __future__ import annotations

import subprocess

import numpy as np
from scipy import ndimage

from fence import geometry as G


# ---------------------------------------------------------------------------
# image -> mask -> skeleton
# ---------------------------------------------------------------------------

def load_mask(path: str, scale: int = 3, threshold: int = 128) -> np.ndarray:
    """Boolean ink mask of an image file, upscaled `scale`x (ImageMagick does the decoding)."""
    pgm = subprocess.run(["convert", path, "-flatten", "-colorspace", "gray", "-scale", f"{100 * scale}%",
                          "-depth", "8", "pgm:-"], check=True, capture_output=True).stdout
    # binary PGM: "P5\n<w> <h>\n<max>\n" then w*h bytes
    toks, pos = [], 0
    while len(toks) < 4:
        while pgm[pos:pos + 1].isspace():
            pos += 1
        if pgm[pos:pos + 1] == b"#":
            pos = pgm.index(b"\n", pos) + 1
            continue
        end = pos
        while not pgm[end:end + 1].isspace():
            end += 1
        toks.append(pgm[pos:end])
        pos = end
    w, h = int(toks[1]), int(toks[2])
    img = np.frombuffer(pgm[pos + 1: pos + 1 + w * h], dtype=np.uint8).reshape(h, w)
    return img < threshold


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning to a one-pixel-wide, 8-connected skeleton."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p = img
            P2, P3, P4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            P8, P9 = p[1:-1, :-2], p[:-2, :-2]
            nb = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(x.astype(np.int32) for x in nb)
            seq = nb + [P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                c = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                c = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            kill = (p[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if kill.any():
                img[1:-1, 1:-1][kill] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def _rdp(pts: np.ndarray, eps: float) -> list[int]:
    """Douglas-Peucker: indices of the polyline's kept points."""
    if len(pts) < 3:
        return list(range(len(pts)))
    a, b = pts[0], pts[-1]
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-12:
        dist = np.linalg.norm(pts - a, axis=1)
    else:
        dist = np.abs(d[0] * (pts[:, 1] - a[1]) - d[1] * (pts[:, 0] - a[0])) / L
    k = int(np.argmax(dist))
    if dist[k] <= eps:
        return [0, len(pts) - 1]
    left = _rdp(pts[: k + 1], eps)
    right = _rdp(pts[k:], eps)
    return left + [k + i for i in right[1:]]


def _ring_branches(sk: np.ndarray, r: int) -> np.ndarray:
    """Number of separate skeleton runs crossing the square ring at Chebyshev radius r around each
    pixel: 2 along a line, >= 3 near a junction -- also where thinning left a two-pixel blob that
    hides the junction from the crossing number."""
    ring = [(-r, dx) for dx in range(-r, r)] + [(dy, r) for dy in range(-r, r)] + \
           [(r, -dx) for dx in range(-r, r)] + [(-dy, -r) for dy in range(-r, r)]
    p = np.pad(sk, r)
    H, W = sk.shape
    vals = [p[r + dy: r + dy + H, r + dx: r + dx + W] for dy, dx in ring]
    return sum(((vals[i] == 0) & (vals[(i + 1) % len(vals)] == 1)).astype(np.int32) for i in range(len(vals)))


def skeleton_graph(skel: np.ndarray, rdp_eps: float = 2.0, merge_px: float = 6.0, spur_px: float = 8.0,
                   ring_r: int = 0):
    """Planar straight-line graph of a skeleton: (nodes [V, 2] as (x, y) pixels, edges [(u, v)])."""
    sk = skel.astype(np.uint8)
    # crossing number (0 -> 1 transitions around the 8-cycle): 2 on a line -- including the
    # staircase pixels of a diagonal, whose plain neighbour count is 3 -- 1 at an end, >= 3 at a junction
    p = np.pad(sk, 1)
    cyc = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
    A = sum(((cyc[i] == 0) & (cyc[(i + 1) % 8] == 1)).astype(np.int32) for i in range(8))
    node_mask = (sk == 1) & (A != 2)
    if ring_r:
        node_mask |= (sk == 1) & (_ring_branches(sk, ring_r) >= 3)
    # branches touch diagonally around a junction pixel: cut its whole 3x3 neighbourhood out
    node_mask = ndimage.binary_dilation(node_mask, structure=np.ones((3, 3))) & (sk == 1)
    lab, nl = ndimage.label(node_mask, structure=np.ones((3, 3)))
    nodes = [np.array(ndimage.center_of_mass(node_mask, lab, i + 1))[::-1] for i in range(nl)]
    # node cluster id per pixel
    path_mask = (sk == 1) & ~node_mask
    plab, npl = ndimage.label(path_mask, structure=np.ones((3, 3)))
    H, W = sk.shape
    polylines = []   # (node_a, node_b, [pixel points (x, y)])
    for c in range(1, npl + 1):
        ys, xs = np.nonzero(plab == c)
        pix = set(zip(ys.tolist(), xs.tolist()))
        # order pixels: start at a pixel with one neighbour inside the component
        def nbrs(p):
            y, x = p
            return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and (y + dy, x + dx) in pix]
        # order pixels by BFS distance from one end (robust to small blobs a greedy walk stalls on)
        def bfs(src):
            dist, frontier = {src: 0}, [src]
            while frontier:
                nxt = []
                for q in frontier:
                    for r in nbrs(q):
                        if r not in dist:
                            dist[r] = dist[q] + 1
                            nxt.append(r)
                frontier = nxt
            return dist
        d0 = bfs(next(iter(pix)))
        start = max(d0, key=d0.get)
        dist = bfs(start)
        order = sorted(pix, key=lambda q: dist[q])
        # one point per BFS level (mean of its pixels): a monotone path Douglas-Peucker can simplify
        levels = {}
        for q in order:
            levels.setdefault(dist[q], []).append(q)
        mean_pts = [tuple(np.mean(levels[k], axis=0)) for k in sorted(levels)]
        # attached node clusters at each end
        def touching(p):
            y, x = p
            out = set()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and lab[yy, xx]:
                        out.add(int(lab[yy, xx]) - 1)
            return out
        ta, tb = touching(order[0]), touching(order[-1])
        pts = np.array([(x, y) for y, x in mean_pts], dtype=float)
        if not ta and not tb:
            continue          # closed loop without junctions: not expected in these pictures
        a = min(ta, key=lambda k: np.linalg.norm(nodes[k] - pts[0])) if ta else None
        b = min(tb, key=lambda k: np.linalg.norm(nodes[k] - pts[-1])) if tb else None
        if a is None:
            nodes.append(pts[0].copy()); a = len(nodes) - 1
        if b is None:
            nodes.append(pts[-1].copy()); b = len(nodes) - 1
        polylines.append((a, b, pts))
    # merge nodes closer than merge_px (union-find), then re-point polylines
    V = len(nodes)
    parent = list(range(V))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    P = np.array(nodes)
    for i in range(V):
        for j in range(i + 1, V):
            if np.linalg.norm(P[i] - P[j]) < merge_px:
                parent[find(i)] = find(j)
    # short polylines between two nodes also collapse
    for a, b, pts in polylines:
        if len(pts) + 2 < merge_px and find(a) != find(b):
            parent[find(a)] = find(b)
    roots = sorted({find(i) for i in range(V)})
    rid = {r: k for k, r in enumerate(roots)}
    NP = np.array([P[[i for i in range(V) if find(i) == r]].mean(0) for r in roots])
    # build straight edges: polyline node_a -> pixels -> node_b, split at DP corners
    coords = [p for p in NP]
    edges = []
    for a, b, pts in polylines:
        A, B = rid[find(a)], rid[find(b)]
        if A == B and len(pts) < 3 * merge_px:
            continue
        full = np.vstack([coords[A], pts, coords[B]])
        keep = _rdp(full, rdp_eps)
        ids = [A] + [None] * (len(keep) - 2) + [B]
        for k in range(1, len(keep) - 1):
            coords.append(full[keep[k]].copy())
            ids[k] = len(coords) - 1
        for k in range(len(ids) - 1):
            if ids[k] != ids[k + 1]:
                edges.append((ids[k], ids[k + 1]))
    coords = np.array(coords)
    # drop spurs: degree-1 nodes whose edge is short (skeleton artefacts at thick corners)
    while True:
        dg = np.zeros(len(coords), int)
        for u, v in edges:
            dg[u] += 1
            dg[v] += 1
        bad = [i for i, (u, v) in enumerate(edges)
               if (dg[u] == 1 or dg[v] == 1) and np.linalg.norm(coords[u] - coords[v]) < spur_px]
        if not bad:
            break
        edges = [e for i, e in enumerate(edges) if i not in set(bad)]
    edges = sorted({(min(u, v), max(u, v)) for u, v in edges if u != v})
    used = sorted({u for e in edges for u in e})
    remap = {u: k for k, u in enumerate(used)}
    return coords[used], [(remap[u], remap[v]) for u, v in edges]


def remove_straight_nodes(coords: np.ndarray, edges, max_dev_px: float = 3.5):
    """Dissolve degree-2 nodes lying within `max_dev_px` of the chord of their two neighbours
    (pixel staircase and Douglas-Peucker noise), smallest deviation first."""
    coords = np.asarray(coords, float)
    edges = [tuple(e) for e in edges]
    while True:
        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        best = None
        for x, nb in adj.items():
            if len(nb) != 2 or nb[0] == nb[1]:
                continue
            a, b = nb
            d = coords[b] - coords[a]
            L = np.linalg.norm(d)
            dev = abs(d[0] * (coords[x, 1] - coords[a, 1]) - d[1] * (coords[x, 0] - coords[a, 0])) / max(L, 1e-12)
            if dev < max_dev_px and (best is None or dev < best[0]):
                best = (dev, x, a, b)
        if best is None:
            break
        _, x, a, b = best
        edges = [e for e in edges if x not in e] + [(a, b)]
    used = sorted({u for e in edges for u in e})
    remap = {u: k for k, u in enumerate(used)}
    return coords[used], [(remap[u], remap[v]) for u, v in edges]


def contract_short_edges(coords: np.ndarray, edges, min_px: float = 9.0):
    """Contract edges shorter than `min_px` (skeleton stubs at thick junctions) into their
    higher-degree endpoint, shortest first."""
    coords = np.asarray(coords, float).copy()
    edges = [tuple(e) for e in edges]
    while True:
        deg = {}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        lens = [np.linalg.norm(coords[u] - coords[v]) for u, v in edges]
        k = int(np.argmin(lens)) if edges else -1
        if k < 0 or lens[k] >= min_px:
            break
        u, v = edges[k]
        keep, drop = (u, v) if deg[u] >= deg[v] else (v, u)
        if deg[u] == deg[v]:
            coords[keep] = 0.5 * (coords[u] + coords[v])
        edges = [(keep if a == drop else a, keep if b == drop else b) for a, b in edges]
        edges = sorted({(min(a, b), max(a, b)) for a, b in edges if a != b})
    used = sorted({u for e in edges for u in e})
    remap = {u: k for k, u in enumerate(used)}
    return coords[used], [(remap[u], remap[v]) for u, v in edges]


def merge_close_nodes(coords: np.ndarray, edges, dist: float):
    """Merge nodes closer than `dist` whether or not an edge joins them (two skeleton junctions of
    one thick crossing need not be adjacent after the stubs between them were dropped)."""
    coords = np.asarray(coords, float)
    V = len(coords)
    parent = list(range(V))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(V):
        for j in range(i + 1, V):
            if np.linalg.norm(coords[i] - coords[j]) < dist:
                parent[find(i)] = find(j)
    roots = sorted({find(i) for i in range(V)})
    rid = {r: k for k, r in enumerate(roots)}
    C = np.array([coords[[i for i in range(V) if find(i) == r]].mean(0) for r in roots])
    E = sorted({(min(rid[find(u)], rid[find(v)]), max(rid[find(u)], rid[find(v)]))
                for u, v in edges if find(u) != find(v)})
    return C, E


def drop_spurs(coords: np.ndarray, edges, max_len: float):
    """Remove dangling edges shorter than `max_len` (a fence end is never free, so a degree-1 node
    is a skeleton artefact), repeatedly."""
    edges = [tuple(e) for e in edges]
    while True:
        deg = {}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        bad = {i for i, (u, v) in enumerate(edges)
               if (deg[u] == 1 or deg[v] == 1) and np.linalg.norm(coords[u] - coords[v]) < max_len}
        if not bad:
            break
        edges = [e for i, e in enumerate(edges) if i not in bad]
    used = sorted({u for e in edges for u in e})
    remap = {u: k for k, u in enumerate(used)}
    return np.asarray(coords)[used], [(remap[u], remap[v]) for u, v in edges]


def straighten_tees(coords: np.ndarray, edges, max_shift: float, max_turn_deg: float = 40.0):
    """A skeleton pulls a T-junction towards its stem, which on a short edge reads as a false bend
    of 20 degrees or more. At every node of degree >= 3, project it onto the chord of its two most
    nearly collinear edges when that moves it less than `max_shift` (real bends sit much further
    from the chord)."""
    coords = np.asarray(coords, float).copy()
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    for x, nb in adj.items():
        if len(nb) < 3:
            continue
        best = None
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                t = _turn(coords, nb[i], x, nb[j])
                if best is None or t < best[0]:
                    best = (t, nb[i], nb[j])
        t, a, b = best
        if t > max_turn_deg:
            continue
        d = coords[b] - coords[a]
        q = coords[a] + np.dot(coords[x] - coords[a], d) / np.dot(d, d) * d
        if np.linalg.norm(q - coords[x]) < max_shift:
            coords[x] = q
    return coords


# ---------------------------------------------------------------------------
# graph -> unit fences
# ---------------------------------------------------------------------------

def _turn(coords, a, x, b) -> float:
    """Turn angle in degrees when a path goes a -> x -> b (0 = straight on)."""
    d1, d2 = coords[x] - coords[a], coords[b] - coords[x]
    c = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def lines_of(coords: np.ndarray, edges, straight_deg: float = 16.0):
    """Chain edges that continue (almost) straight through a node into lines (node lists)."""
    adj = {}
    for k, (u, v) in enumerate(edges):
        adj.setdefault(u, []).append(k)
        adj.setdefault(v, []).append(k)
    other = lambda k, x: edges[k][1] if edges[k][0] == x else edges[k][0]
    # at every node pair incident edges greedily by straightness
    through = {}   # (node, edge) -> edge continuing straight through node
    for x, ks in adj.items():
        cands = []
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                t = _turn(coords, other(ks[i], x), x, other(ks[j], x))
                if t < straight_deg:
                    cands.append((t, ks[i], ks[j]))
        used = set()
        for t, ki, kj in sorted(cands):
            if ki in used or kj in used:
                continue
            used |= {ki, kj}
            through[(x, ki)] = kj
            through[(x, kj)] = ki
    seen, lines = set(), []
    for k0 in range(len(edges)):
        if k0 in seen:
            continue
        # walk back to the start of this line
        k, x = k0, edges[k0][0]
        guard = 0
        while (x, k) in through and guard < len(edges):
            k2 = through[(x, k)]
            x = other(k2, x)
            k = k2
            guard += 1
            if k == k0:
                break
        # now walk forward from x along k
        start = x
        nodes = [start]
        cur_k, cur_x = k, start
        while True:
            seen.add(cur_k)
            nxt = other(cur_k, cur_x)
            nodes.append(nxt)
            if (nxt, cur_k) not in through:
                break
            cur_k = through[(nxt, cur_k)]
            cur_x = nxt
            if cur_k in seen:
                break
        lines.append(nodes)
    return lines


def _cut_line(coords, nodes, s, bend_deg: float, forced=(), lam: float = 0.0) -> tuple[list[int], float]:
    """Choose the joints along a line (indices into `nodes`, ends included) by dynamic programming:
    every piece between joints should have near-integer length (it is a whole number of unit
    fences), and a bend left inside a piece costs its angle -- pixel noise bends a few degrees,
    a real joint between two fences more. Indices in `forced` must be joints; `lam` is a price per
    fence, which moves the total fence count towards n when the picture alone does not. Returns
    (cuts, cost)."""
    m = len(nodes) - 1
    forced = set(forced)
    turn = [0.0] + [_turn(coords, nodes[i - 1], nodes[i], nodes[i + 1]) for i in range(1, m)] + [0.0]

    def cost(i, j):
        L = s[j] - s[i]
        k = max(1, round(L))
        c = 25.0 * (L - k) ** 2 / k + lam * k
        c += sum(max(0.0, turn[q] - bend_deg) ** 2 * 1e-3 for q in range(i + 1, j))
        return c
    best = [0.0] + [np.inf] * m
    prev = [0] * (m + 1)
    for j in range(1, m + 1):
        for i in range(j):
            if any(i < q < j for q in forced):
                continue
            c = best[i] + cost(i, j) + 0.01
            if c < best[j]:
                best[j], prev[j] = c, i
    cuts, j = [m], m
    while j:
        j = prev[j]
        cuts.append(j)
    return cuts[::-1], float(best[m])


def fences_from_graph(coords: np.ndarray, edges, n: int, straight_deg: float = 16.0,
                      bend_deg: float = 3.0, snap: float = 0.2, length_scale: float = 1.0, lam: float = 0.0):
    """Unit-fence sketch (coords, fences, pins, report) from a planar straight-line graph.

    Coordinates are rescaled so the total edge length is n."""
    coords = np.asarray(coords, float).copy()
    total = sum(np.linalg.norm(coords[u] - coords[v]) for u, v in edges)
    coords *= length_scale * n / total
    deg = np.zeros(len(coords), int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    lines = lines_of(coords, edges, straight_deg)
    P = [c for c in coords]
    fences, pin_at, report = [], {}, []
    arcs = []
    for nodes in lines:
        seg_len = [np.linalg.norm(coords[nodes[i + 1]] - coords[nodes[i]]) for i in range(len(nodes) - 1)]
        arcs.append(np.concatenate([[0.0], np.cumsum(seg_len)]))
    forced = [set() for _ in lines]
    line_cuts = [_cut_line(coords, nodes, arcs[li], bend_deg, lam=lam)[0] for li, nodes in enumerate(lines)]
    # two lines cannot both pass through one node (fences never cross): cut all but the one
    # whose forced cut would cost the most
    for _ in range(20):
        through = {}
        for li, nodes in enumerate(lines):
            for i in range(1, len(nodes) - 1):
                if i not in line_cuts[li]:
                    through.setdefault(nodes[i], []).append((li, i))
        clash = {v: ls for v, ls in through.items() if len(ls) > 1}
        if not clash:
            break
        for v, ls in clash.items():
            extra = []
            for li, i in ls:
                base = _cut_line(coords, lines[li], arcs[li], bend_deg, forced[li], lam)[1]
                extra.append((_cut_line(coords, lines[li], arcs[li], bend_deg, forced[li] | {i}, lam)[1] - base, li, i))
            extra.sort()
            for _, li, i in extra[:-1]:
                forced[li].add(i)
        line_cuts = [_cut_line(coords, nodes, arcs[li], bend_deg, forced[li], lam)[0] for li, nodes in enumerate(lines)]
    for li, nodes in enumerate(lines):
        seg_len = np.diff(arcs[li])
        s = arcs[li]
        cut_idx = line_cuts[li]
        for a_i, b_i in zip(cut_idx[:-1], cut_idx[1:]):
            L = s[b_i] - s[a_i]
            k = max(1, int(round(L)))
            report.append({"nodes": [int(nodes[a_i]), int(nodes[b_i])], "length": round(float(L), 3), "fences": k})
            # breakpoints at equal spacing, snapped to a node when one is close
            inner = list(range(a_i + 1, b_i))
            cuts = [a_i]
            brk_vid = [nodes[a_i]]
            for j in range(1, k):
                target = s[a_i] + L * j / k
                near = [i for i in inner if abs(s[i] - target) < snap]
                if near:
                    i = min(near, key=lambda i: abs(s[i] - target))
                    brk_vid.append(nodes[i])
                    cuts.append(i)
                else:
                    # new joint vertex on the run
                    # locate segment containing target
                    i = int(np.searchsorted(s, target) - 1)
                    i = min(max(i, a_i), b_i - 1)
                    t = (target - s[i]) / max(seg_len[i], 1e-12)
                    p = (1 - t) * coords[nodes[i]] + t * coords[nodes[i + 1]]
                    P.append(p)
                    brk_vid.append(len(P) - 1)
                    cuts.append(i + t)
            brk_vid.append(nodes[b_i])
            cuts.append(b_i)
            for j in range(k):
                f = len(fences)
                fences.append([int(brk_vid[j]), int(brk_vid[j + 1])])
                lo, hi = cuts[j], cuts[j + 1]
                for i in inner:
                    # a node only this line passes through is a noise bend inside the fence: dropped
                    if lo < i < hi and nodes[i] not in brk_vid and deg[nodes[i]] >= 3:
                        pin_at[int(nodes[i])] = f
    P = np.array(P)
    used = sorted({v for f in fences for v in f} | set(pin_at))
    remap = {v: k for k, v in enumerate(used)}
    P = P[used]
    fences = [[remap[a], remap[b]] for a, b in fences]
    pin_at = {remap[v]: h for v, h in pin_at.items()}
    # put pins exactly on their hosts (repeat: a host's endpoint may itself be a pin)
    for _ in range(200):
        moved = 0.0
        for v, h in pin_at.items():
            a, b = fences[h]
            d = P[b] - P[a]
            t = np.clip(np.dot(P[v] - P[a], d) / np.dot(d, d), 0.02, 0.98)
            q = P[a] + t * d
            moved = max(moved, float(np.linalg.norm(q - P[v])))
            P[v] = q
        if moved < 1e-13:
            break
    return P, fences, pin_at, report


def digitize(path: str, n: int, scale: int = 3, **kw):
    """Picture -> (Topology, coords, report). Raises InvalidTopology if the read is not a legal sketch."""
    mask = load_mask(path, scale=scale)
    skel = thin(mask)
    g_kw = {k: kw[k] for k in ("rdp_eps", "merge_px", "spur_px", "ring_r") if k in kw}
    nodes, edges = skeleton_graph(skel, **g_kw)
    nodes = nodes * np.array([1.0, -1.0])          # image y points down
    nodes, edges = remove_straight_nodes(nodes, edges, kw.get("max_dev_px", 3.5))
    # stubs are judged against the fence length: total ink / n pixels per unit
    unit = sum(np.linalg.norm(nodes[u] - nodes[v]) for u, v in edges) / n
    nodes, edges = contract_short_edges(nodes, edges, kw.get("min_edge", 0.15) * unit)
    if kw.get("merge_close"):
        nodes, edges = merge_close_nodes(nodes, edges, kw["merge_close"] * unit)
    nodes, edges = drop_spurs(nodes, edges, kw.get("spur", 0.3) * unit)
    nodes, edges = remove_straight_nodes(nodes, edges, kw.get("max_dev_px", 3.5))
    if kw.get("tee_shift"):
        nodes = straighten_tees(nodes, edges, kw["tee_shift"] * unit)
    f_kw = {k: kw[k] for k in ("straight_deg", "bend_deg", "snap") if k in kw}
    # the ink is n fences only up to line-thickness effects at corners and junctions: if the read
    # does not give exactly n, retry with the unit length nudged
    err = None
    tries = [(ls, 0.0) for ls in (1.0, 0.98, 1.02, 0.96, 1.04, 0.94, 1.06, 0.92, 1.08)]
    tries += [(1.0, lam) for lam in kw.get("lams", ())]
    for ls, lam in tries:
        P, fences, pins, report = fences_from_graph(nodes, edges, n, length_scale=ls, lam=lam, **f_kw)
        if len(fences) != n:
            err = G.InvalidTopology(f"digitized {len(fences)} fences, expected {n}")
            continue
        try:
            topo = G.make_topology(P, fences, pins)
        except G.InvalidTopology as e:
            err = e
            continue
        info = {"graph_nodes": len(nodes), "graph_edges": len(edges), "fences": len(fences), "pins": len(pins),
                "length_scale": ls, "lam": lam, "runs": report}
        return topo, P, info
    raise err
