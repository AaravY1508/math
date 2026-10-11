#!/usr/bin/env python3
"""Render certified n=37..50 placements as GIFs in the style of the light-problem page.

Matches the published images (e.g. source/36.gif): 232x230 canvas, 4 px black square,
8 px red lights, blue points attaining the minimum intensity I, and black contours
at 1.5*I and 3*I (levels fitted to the published 30, 33, 35 and 36 GIFs).
"""
import numpy as np
from PIL import Image
from scipy.optimize import minimize

W, H = 232, 230          # canvas of the published 36.gif
X0, Y0, SIDE = 7.5, 5.5, 217.0   # square corners in pixel coords (pixel centres at integer+0.5)
STAMP = [4, 6, 8, 8, 8, 8, 6, 4]  # 8 px disc, 52 px, as in the published images


def F(q, P):
    d2 = ((q[..., None, :] - P) ** 2).sum(-1)
    return (1 / d2).sum(-1)


def gradF(q, P):
    dx = q[..., None, :] - P
    d2 = (dx ** 2).sum(-1)
    return (-2 * dx / d2[..., None] ** 2).sum(-2)


def to_px(u):
    return X0 + u[..., 0] * SIDE, Y0 + (1 - u[..., 1]) * SIDE


def local_minima(P, grid=400):
    """All local minima of F on the closed unit square: interior, edges, corners."""
    found = []
    def refine(x0, bounds):
        r = minimize(lambda z: F(z, P), x0, jac=lambda z: gradF(z, P), method='L-BFGS-B',
                     bounds=bounds, options=dict(ftol=1e-15, gtol=1e-12, maxiter=500))
        return r.x
    t = (np.arange(grid) + 0.5) / grid
    g = np.stack(np.meshgrid(t, t, indexing='xy'), -1)
    v = F(g, P)
    pad = np.pad(v, 1, constant_values=np.inf)
    ismin = np.ones_like(v, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                ismin &= v <= pad[1 + dy:1 + dy + grid, 1 + dx:1 + dx + grid]
    for iy, ix in zip(*np.nonzero(ismin)):
        found.append(refine(g[iy, ix], [(0, 1), (0, 1)]))
    s = np.linspace(0, 1, 4 * grid + 1)
    for fixed_axis in (0, 1):
        for val in (0.0, 1.0):
            q = np.zeros((s.size, 2)); q[:, fixed_axis] = val; q[:, 1 - fixed_axis] = s
            v = F(q, P)
            for i in range(s.size):
                if (i == 0 or v[i] <= v[i - 1]) and (i == s.size - 1 or v[i] <= v[i + 1]):
                    b = [(0, 1), (0, 1)]; b[fixed_axis] = (val, val)
                    found.append(refine(q[i], b))
    # keep true local minima on the square (projected gradient check), dedupe
    out = []
    for m in found:
        m = np.clip(m, 0, 1)
        g_ = gradF(m, P)
        ok = True
        for k in range(2):
            if 0 < m[k] < 1 and abs(g_[k]) > 1e-3 * F(m, P):
                ok = False
            if m[k] == 0 and g_[k] < -1e-3 * F(m, P): ok = False
            if m[k] == 1 and g_[k] > 1e-3 * F(m, P): ok = False
        if ok and all(np.hypot(*(m - o)) > 2e-3 for o in out):
            out.append(m)
    return np.array(out)


def stamp(img, cx, cy, color):
    # stamp centre is at the corner between pixels; choose the nearest such corner
    ox, oy = int(round(cx)) - 4, int(round(cy)) - 4
    for r, w in enumerate(STAMP):
        y = oy + r
        x_start = ox + (8 - w) // 2
        for x in range(x_start, x_start + w):
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                img[y, x] = color


def render(P, levels, minima=None, peak=0.85, half=1.0):
    P = np.asarray(P, float)
    px = np.arange(W) + 0.5; py = np.arange(H) + 0.5
    X, Y = np.meshgrid(px, py)
    U = np.stack([(X - X0) / SIDE, 1 - (Y - Y0) / SIDE], -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = F(U, P)
        gn = np.hypot(*np.moveaxis(gradF(U, P), -1, 0)) / SIDE   # per pixel
        ink = np.zeros((H, W))
        for L in levels:
            d = np.abs(f - L) / gn                                   # distance to level set in px
            ink = np.maximum(ink, peak * np.clip(1.0 - d / half, 0, 1))
    ink = np.nan_to_num(ink)
    img = np.repeat((255 * (1 - ink))[..., None], 3, -1).round().astype(np.uint8)
    # square border, 4 px, centred on the boundary
    a, b = int(round(X0 - 2)), int(round(X0 + SIDE - 2))
    c, d = int(round(Y0 - 2)), int(round(Y0 + SIDE - 2))
    img[c:d + 4, a:a + 4] = 0; img[c:d + 4, b:b + 4] = 0
    img[c:c + 4, a:b + 4] = 0; img[d:d + 4, a:b + 4] = 0
    if minima is None:
        minima = local_minima(P)
    for m in minima:
        stamp(img, *to_px(np.asarray(m)), (0, 0, 255))
    for p in P:
        stamp(img, *to_px(p), (255, 0, 0))
    return Image.fromarray(img), minima


def main():
    import argparse, json
    from pathlib import Path
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=here / 'results')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in range(37, 51):
        d = json.loads((args.results / f'n{n}' / 'solution.json').read_text())
        P = np.array(d['points'])
        assert len(P) == n and d['valid'] and d['certified']
        lo, hi = d['objective_lower'], d['objective_upper']
        allm = local_minima(P)
        mins = allm[F(allm, P) <= hi * (1 + 1e-6)]   # blue = points attaining I
        img, _ = render(P, [1.5 * lo, 3 * lo], minima=mins)
        img.convert('P', palette=Image.Palette.ADAPTIVE, colors=128).save(args.out / f'{n}.gif')
        rows.append(dict(n=n, lower=lo, upper=hi, darkest_points=len(mins),
                         other_local_minima=len(allm) - len(mins)))
        print(rows[-1], flush=True)
    (args.out / 'values.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
