"""Overlay an optimized layout on the page's record picture and say where the two disagree.

The digitized topologies (fence/digitize.py) optimize to within ~0.01-0.04 of the page at the n that
still trail; the hypothesis of EXPERIMENTS.md change 15 is that each remaining gap is a local misread.
This module locates them:

  1. `fit`: a similarity transform (scale, rotation, translation; reflection allowed) mapping the
     layout onto the picture's skeleton, by ICP -- points sampled along the fences against the
     skeleton pixels, with correspondences taken both ways (layout -> nearest skeleton pixel and
     skeleton pixel -> nearest layout sample, so neither side can shrink onto part of the other),
     from 24 rotations x 2 handednesses; the lowest two-way mean distance wins.
  2. `score`: each fence gets the mean distance from its samples to the nearest skeleton pixel (a
     distance transform of the skeleton, scipy.ndimage), each skeleton branch the mean distance of
     its pixels to the nearest fence, and each face of the layout the IoU with the picture face it
     overlaps most (faces of the picture = connected components of the background, rasterized in
     pixel space). Distances are in fence units (pixels / fitted scale).
  3. High scores flag disagreement: a fence with nothing drawn under it (extra fence, or a T on the
     wrong side), a drawn branch with no fence on it (missing fence), a face the picture does not
     have. `report["focus"]` lists the flagged places as [x, y, r] in the layout's own coordinates,
     the format neighbourhood.py --focus takes.

`render` writes a PNG of the overlay (picture grey, fences green where they agree and red where
flagged, flagged picture branches blue) for looking at.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from fence import digitize as D
from fence import geometry as G


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def layout_of(config: dict):
    topo, X = G.config_from_json(config)
    return topo, np.asarray(X, dtype=float)


def fence_samples(X: np.ndarray, fences, step: float = 0.02):
    """Points every ~step along every fence (units), and the fence each belongs to."""
    pts, own = [], []
    for f, (a, b) in enumerate(fences):
        m = max(2, int(math.ceil(np.linalg.norm(X[b] - X[a]) / step)) + 1)
        t = np.linspace(0.0, 1.0, m)[:, None]
        pts.append(X[a] * (1 - t) + X[b] * t)
        own.append(np.full(m, f))
    return np.vstack(pts), np.concatenate(own)


def picture(path: str, scale: int = 3):
    """(ink mask, skeleton) of a record picture, image coordinates (row, col)."""
    mask = D.load_mask(path, scale=scale)
    return mask, D.thin(mask).astype(bool)


def to_xy(rc: np.ndarray) -> np.ndarray:
    """(row, col) pixels -> (x, y) with y up, the frame the fit works in."""
    return np.column_stack([rc[:, 1].astype(float), -rc[:, 0].astype(float)])


def to_rc(xy: np.ndarray) -> np.ndarray:
    return np.column_stack([-xy[:, 1], xy[:, 0]])


# ---------------------------------------------------------------------------
# the similarity fit
# ---------------------------------------------------------------------------

def umeyama(A: np.ndarray, B: np.ndarray):
    """Least-squares proper similarity B ~ s R A + t (det R = +1)."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    C = B0.T @ A0 / len(A)
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    Dm = np.diag([1.0, d])
    R = U @ Dm @ Vt
    var = (A0 ** 2).sum() / len(A)
    s = float(np.trace(np.diag(S) @ Dm) / var) if var > 0 else 1.0
    return s, R, mb - s * R @ ma


class Similarity:
    """x -> s R (F x) + t, F = diag(-1, 1) when reflected."""

    def __init__(self, s: float, R: np.ndarray, t: np.ndarray, reflect: bool):
        self.s, self.R, self.t, self.reflect = float(s), np.asarray(R, float), np.asarray(t, float), bool(reflect)

    def F(self, P):
        return P * np.array([-1.0, 1.0]) if self.reflect else P

    def __call__(self, P):
        return self.s * self.F(P) @ self.R.T + self.t

    def inverse(self, Q):
        return self.F((Q - self.t) @ self.R / self.s)

    def to_json(self):
        return {"scale_px_per_unit": self.s, "rotation_deg": math.degrees(math.atan2(self.R[1, 0], self.R[0, 0])),
                "t": self.t.tolist(), "reflect": self.reflect}


def fit(P: np.ndarray, S: np.ndarray, angles: int = 24, iters: int = 40, sub: int = 4000, seed: int = 0):
    """ICP of layout samples P (units) onto skeleton points S (pixels, y up). Returns (Similarity, cost)."""
    rng = np.random.default_rng(seed)
    Ss = S if len(S) <= sub else S[rng.choice(len(S), sub, replace=False)]
    treeS = cKDTree(Ss)
    cP, cS = P.mean(0), Ss.mean(0)
    s0 = math.sqrt(((Ss - cS) ** 2).sum(1).mean() / ((P - cP) ** 2).sum(1).mean())
    best = (math.inf, None)
    for reflect in (False, True):
        F = np.array([-1.0, 1.0]) if reflect else np.array([1.0, 1.0])
        PF = P * F
        cPF = PF.mean(0)
        for k in range(angles):
            th = 2 * math.pi * k / angles
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            s, t = s0, cS - s0 * R @ cPF
            cost = math.inf
            for it in range(iters):
                Q = s * PF @ R.T + t
                d1, i1 = treeS.query(Q)
                d2, i2 = cKDTree(Q).query(Ss)
                A = np.vstack([PF, PF[i2]])
                B = np.vstack([Ss[i1], Ss])
                s_new, R_new, t_new = umeyama(A, B)
                new_cost = 0.5 * (d1.mean() + d2.mean())
                s, R, t = s_new, R_new, t_new
                if abs(cost - new_cost) < 1e-6 * max(1.0, new_cost):
                    cost = new_cost
                    break
                cost = new_cost
            if cost < best[0]:
                best = (cost, Similarity(s, R, t, reflect))
    return best[1], best[0]


# ---------------------------------------------------------------------------
# rasterizing
# ---------------------------------------------------------------------------

def points_in_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd test of the points (px, py) against a polygon [k, 2]."""
    inside = np.zeros(px.shape, dtype=bool)
    k = len(poly)
    for i in range(k):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % k]
        cond = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (px < xint)
    return inside


def raster_segments(shape, segs_xy: np.ndarray, step_px: float = 0.5) -> np.ndarray:
    """Boolean image with the pixel of every point along the segments (x, y-up pixel coords) set."""
    img = np.zeros(shape, dtype=bool)
    for x1, y1, x2, y2 in segs_xy:
        m = max(2, int(math.hypot(x2 - x1, y2 - y1) / step_px) + 1)
        t = np.linspace(0, 1, m)
        r = np.round(-(y1 + t * (y2 - y1))).astype(int)
        c = np.round(x1 + t * (x2 - x1)).astype(int)
        ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        img[r[ok], c[ok]] = True
    return img


def branches(skel: np.ndarray):
    """Label the skeleton's branches: skeleton pixels minus the 3x3 neighbourhoods of junction pixels
    (crossing number >= 3, as fence/digitize.py finds them)."""
    sk = skel.astype(np.uint8)
    p = np.pad(sk, 1)
    cyc = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
    A = sum(((cyc[i] == 0) & (cyc[(i + 1) % 8] == 1)).astype(np.int32) for i in range(8))
    junc = (sk == 1) & (A >= 3)
    junc = ndimage.binary_dilation(junc, structure=np.ones((3, 3)), iterations=2) & (sk == 1)
    lab, k = ndimage.label((sk == 1) & ~junc, structure=np.ones((3, 3)))
    return lab, k


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(config: dict, gif: str, scale: int = 3, fence_tol: float = 0.04, branch_tol: float = 0.06,
          iou_tol: float = 0.85, focus_r: float = 0.5) -> dict:
    """Fit the layout onto the picture and score fences, branches and faces (see module doc)."""
    topo, X = layout_of(config)
    fences = np.asarray(topo.fences)
    mask, skel = picture(gif, scale)
    S = to_xy(np.argwhere(skel))
    P, own = fence_samples(X, fences)
    T, cost_px = fit(P, S)
    unit = T.s                                           # pixels per fence unit
    # fences: distance of their samples to the skeleton
    dt_skel = ndimage.distance_transform_edt(~skel)
    Q = T(P)
    rc = np.round(to_rc(Q)).astype(int)
    rc[:, 0] = rc[:, 0].clip(0, skel.shape[0] - 1)
    rc[:, 1] = rc[:, 1].clip(0, skel.shape[1] - 1)
    d_s = dt_skel[rc[:, 0], rc[:, 1]] / unit
    fence_score = np.array([d_s[own == f].mean() for f in range(len(fences))])
    fence_max = np.array([d_s[own == f].max() for f in range(len(fences))])
    # branches: distance of their pixels to the rasterized fences
    segs = np.array([[*T(X[a:a + 1])[0], *T(X[b:b + 1])[0]] for a, b in fences])
    ras = raster_segments(skel.shape, segs)
    dt_fence = ndimage.distance_transform_edt(~ras)
    lab, nb = branches(skel)
    br = []
    for k in range(1, nb + 1):
        pix = np.argwhere(lab == k)
        if len(pix) < 3:
            continue
        d = dt_fence[pix[:, 0], pix[:, 1]] / unit
        c = T.inverse(to_xy(pix.mean(0, keepdims=True)))[0]
        br.append({"id": k, "pixels": int(len(pix)), "length": len(pix) / unit, "mean": float(d.mean()),
                   "max": float(d.max()), "centre": c.tolist()})
    # faces: layout faces (rasterized) vs picture faces (background components not touching the border)
    bg, nf = ndimage.label(~mask)
    border = set(np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]))) - {0}
    rr, cc = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    px, py = cc.ravel().astype(float), -rr.ravel().astype(float)
    pic_area = {int(l): float(c) / unit ** 2 for l, c in zip(*np.unique(bg[bg > 0], return_counts=True))
                if l not in border}
    faces = []
    areas = G.face_areas(X, topo.faces)
    for fi, cyc in enumerate(topo.faces):
        poly = T(X[cyc])
        lo, hi = poly.min(0), poly.max(0)
        sel = (px >= lo[0] - 1) & (px <= hi[0] + 1) & (py >= lo[1] - 1) & (py <= hi[1] + 1)
        inside = np.zeros(px.shape, bool)
        inside[sel] = points_in_polygon(px[sel], py[sel], poly)
        ins = inside.reshape(mask.shape)
        labs, cnt = np.unique(bg[ins & (bg > 0)], return_counts=True)
        labs_ok = [(int(l), int(c)) for l, c in zip(labs, cnt) if l not in border]
        if labs_ok:
            l, c = max(labs_ok, key=lambda x: x[1])
            comp = bg == l
            # the layout face is drawn to the fence centre lines, the picture's face stops at the ink:
            # compare the picture face grown by half the line width against the layout face
            grown = ndimage.binary_dilation(comp, iterations=max(1, int(round(scale * 1.5))))
            inter = float((ins & grown).sum())
            union = float((ins | grown).sum())
            iou, pa = inter / union if union else 0.0, pic_area[l]
        else:
            l, iou, pa = None, 0.0, None
        faces.append({"face": fi, "area": float(areas[fi]), "pic_label": l, "pic_area": pa, "iou": iou,
                      "centre": X[cyc].mean(0).tolist(), "vertices": [int(v) for v in cyc]})
    # picture faces no layout face claims
    claimed = {f["pic_label"] for f in faces}
    unclaimed = []
    for l, a in pic_area.items():
        if l not in claimed and a > 0.05:
            pix = np.argwhere(bg == l)
            unclaimed.append({"pic_label": l, "pic_area": a,
                              "centre": T.inverse(to_xy(pix.mean(0, keepdims=True)))[0].tolist()})
    flagged_f = [int(f) for f in np.where(fence_score > fence_tol)[0]]
    flagged_b = [b for b in br if b["mean"] > branch_tol and b["length"] > 0.1]
    flagged_faces = [f for f in faces if f["iou"] < iou_tol]
    focus = []
    for f in flagged_f:
        a, b = fences[f]
        focus.append([*((X[a] + X[b]) / 2).tolist(), focus_r])
    for b in flagged_b:
        focus.append([*b["centre"], focus_r])
    for f in flagged_faces:
        focus.append([*f["centre"], focus_r])
    for u in unclaimed:
        focus.append([*u["centre"], focus_r])
    return {"n": int(len(fences)), "transform": T.to_json(), "fit_cost_units": cost_px / unit,
            "fence_score": fence_score.tolist(), "fence_max": fence_max.tolist(), "branches": br,
            "faces": faces, "unclaimed_picture_faces": unclaimed, "picture_faces": len(pic_area),
            "layout_faces": len(topo.faces),
            "flagged": {"fences": flagged_f, "branches": [b["id"] for b in flagged_b],
                        "faces": [f["face"] for f in flagged_faces]},
            "focus": focus, "_T": T, "_skel": skel, "_mask": mask, "_lab": lab, "_segs": segs}


def picture_fit(config: dict, gif: str, scale: int = 3, steps: int = 600, lr: float = 0.01,
                length_w: float = 0.2, rep: dict | None = None) -> np.ndarray:
    """Bend the layout's own geometry onto the picture: Adam on the vertex coordinates (pins kept on
    their hosts by their host parameter) minimizing the squared distance of the fence samples to the
    skeleton (bilinear on the distance transform, in fence units) plus a small pull of every fence
    towards unit length. The topology is unchanged; returns the fitted coordinates (layout frame),
    a start for the optimizer that sits in the picture's basin where the topology allows it."""
    import torch
    import torch.nn.functional as Fn
    topo, X = layout_of(config)
    rep = rep or score(config, gif, scale)
    T, skel = rep["_T"], rep["_skel"]
    dt = ndimage.distance_transform_edt(~skel) / T.s
    H, W = dt.shape
    img = torch.tensor(dt, dtype=torch.float64)[None, None]
    fences = np.asarray(topo.fences)
    pins = dict(topo.pins)
    # every vertex is free; a pin is held on its host line by a penalty (pins may form cycles --
    # a pinwheel's three ends each sit on the next fence -- so they cannot be solved in order)
    Xf = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    pv = torch.tensor(sorted(pins), dtype=torch.long)
    ph_a = torch.tensor([fences[pins[v]][0] for v in sorted(pins)], dtype=torch.long)
    ph_b = torch.tensor([fences[pins[v]][1] for v in sorted(pins)], dtype=torch.long)
    ts = torch.linspace(0, 1, 25, dtype=torch.float64)[None, :, None]
    Rm = torch.tensor(T.R, dtype=torch.float64)
    Fm = torch.tensor([-1.0 if T.reflect else 1.0, 1.0], dtype=torch.float64)
    tvec = torch.tensor(T.t, dtype=torch.float64)
    opt = torch.optim.Adam([Xf], lr=lr)

    def coords():
        return Xf

    def pin_off(P):
        if len(pv) == 0:
            return torch.zeros((), dtype=torch.float64)
        A, B, V = P[ph_a], P[ph_b], P[pv]
        d = B - A
        cr = d[:, 0] * (V - A)[:, 1] - d[:, 1] * (V - A)[:, 0]
        return ((cr / d.norm(dim=1).clamp_min(1e-9)) ** 2).mean()

    fa = torch.tensor(fences[:, 0])
    fb = torch.tensor(fences[:, 1])
    for _ in range(steps):
        opt.zero_grad()
        P = coords()
        A, B = P[fa][:, None, :], P[fb][:, None, :]
        S = (A + ts * (B - A)).reshape(-1, 2)
        Q = T.s * (S * Fm) @ Rm.T + tvec                    # pixels, y up
        gx = Q[:, 0] / (W - 1) * 2 - 1
        gy = (-Q[:, 1]) / (H - 1) * 2 - 1
        g = torch.stack([gx, gy], -1)[None, None]
        d = Fn.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True).reshape(-1)
        L = (P[fb] - P[fa]).norm(dim=1)
        loss = (d ** 2).mean() + length_w * ((L - 1) ** 2).mean() + 10.0 * pin_off(P)
        loss.backward()
        opt.step()
    return coords().detach().numpy()


def summary(rep: dict) -> str:
    fs = np.array(rep["fence_score"])
    lines = [f"n={rep['n']}  fit {rep['fit_cost_units']:.4f} u  scale {rep['transform']['scale_px_per_unit']:.1f} px/u "
             f"rot {rep['transform']['rotation_deg']:.1f} refl {rep['transform']['reflect']}  "
             f"faces layout {rep['layout_faces']} picture {rep['picture_faces']}",
             f"  fences: median {np.median(fs):.4f} max {fs.max():.4f}; flagged {rep['flagged']['fences']} "
             f"({', '.join(f'{fs[f]:.3f}' for f in rep['flagged']['fences'])})"]
    for f in rep["faces"]:
        if f["face"] in rep["flagged"]["faces"]:
            pa = f"{f['pic_area']:.3f}" if f["pic_area"] is not None else "-"
            lines.append(f"  face {f['face']}: area {f['area']:.3f} iou {f['iou']:.3f} picture face area {pa} "
                         f"at ({f['centre'][0]:.2f}, {f['centre'][1]:.2f})")
    for b in rep["branches"]:
        if b["id"] in rep["flagged"]["branches"]:
            lines.append(f"  branch {b['id']}: len {b['length']:.2f} mean {b['mean']:.3f} max {b['max']:.3f} "
                         f"at ({b['centre'][0]:.2f}, {b['centre'][1]:.2f})")
    for u in rep["unclaimed_picture_faces"]:
        lines.append(f"  picture face {u['pic_label']} (area {u['pic_area']:.3f}) has no layout face, "
                     f"at ({u['centre'][0]:.2f}, {u['centre'][1]:.2f})")
    return "\n".join(lines)


def render(rep: dict, path: str) -> None:
    """PNG: picture ink light grey, skeleton flagged branches blue, fences green (agree) / red (flagged)."""
    mask, lab = rep["_mask"], rep["_lab"]
    img = np.full(mask.shape + (3,), 255, dtype=np.uint8)
    img[mask] = (205, 205, 205)
    for b in rep["flagged"]["branches"]:
        img[ndimage.binary_dilation(lab == b, iterations=2)] = (40, 90, 255)
    fl = set(rep["flagged"]["fences"])
    for f, seg in enumerate(rep["_segs"]):
        ras = raster_segments(mask.shape, seg[None, :], 0.5)
        ras = ndimage.binary_dilation(ras, iterations=1)
        img[ras] = (230, 30, 30) if f in fl else (20, 150, 60)
    ppm = f"P6 {mask.shape[1]} {mask.shape[0]} 255\n".encode() + img.tobytes()
    subprocess.run(["convert", "ppm:-", path], input=ppm, check=True)


def public(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if not k.startswith("_")}


def main(argv=None):
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="overlay library layouts on the page's record pictures")
    ap.add_argument("--ns", required=True)
    ap.add_argument("--gif-dir", required=True)
    ap.add_argument("--library", default=str(Path(__file__).resolve().parent.parent / "minima"))
    ap.add_argument("--config", default=None, help="a config file instead of the library record ('{n}' -> n)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--focus-r", type=float, default=0.5)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    focus = {}
    for n in [int(x) for x in a.ns.split(",")]:
        if a.config:
            cfg = json.load(open(a.config.format(n=n)))
            cfg = cfg.get("config", cfg)
        else:
            cfg = json.load(open(Path(a.library) / "records" / f"N{n}.json"))["config"]
        rep = score(cfg, str(Path(a.gif_dir) / f"{n}.gif"), focus_r=a.focus_r)
        print(summary(rep), flush=True)
        render(rep, str(out / f"overlay-n{n:03d}.png"))
        json.dump(public(rep), open(out / f"overlay-n{n:03d}.json", "w"), indent=1)
        focus[str(n)] = rep["focus"]
    json.dump(focus, open(out / "focus.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
