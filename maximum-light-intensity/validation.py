"""Independent validation of inverse-square illumination on the closed unit square.

Coordinates in the supplied binary64 array are treated as the exact light positions.
The certificate covers the entire square, including its boundary and light poles.
Every arithmetic operation contributing to interval bounds is rounded outwards;
interval summation includes a conservative floating point accumulation bound.
No sampled-grid claim is used as a proof of the global minimum.

A useful global inequality is 1/t >= 2/s - t/s**2 for positive t,s.
Applying it at the rectangle centre gives the cancellation-aware lower bound
    f(c) - |grad_x f(c)| hx - |grad_y f(c)| hy
         - sum_i 1/|c-p_i|**4 * (hx**2 + hy**2).
The maximum of this and independent farthest-distance bounds is valid even when
a light lies inside the rectangle.  Away from poles a cubic Taylor remainder
also exploits positive curvature near minima.  The Hessian of 1/|r|**2 has
Lipschitz constant at most 24/|r|**5 along any segment, so |R_3| <=
4 * |delta|**3 * sum_i d_min_i**-5.  To verify the constant, rotate r to
(R,0): D_u H = (8/R**5) * [[-3*u_x,u_y],[u_y,u_x]], whose spectral norm
is 8*(sqrt(4*u_x**2+u_y**2)+abs(u_x))/R**5 <= 24/R**5 for |u|=1.
"""
from __future__ import annotations

import heapq
import hashlib
import math
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

_INF = float("inf")
_EPS = np.finfo(float).eps
_TINY = np.nextafter(0.0, 1.0)


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _sum_interval(lo, hi):
    """Enclose real sums; gamma_n bound is deliberately larger than required."""
    n = len(lo)
    # numpy's summation may be sequential or pairwise. This bound covers either.
    magnitude = float(np.sum(np.maximum(np.abs(lo), np.abs(hi))))
    gamma = float(_up((n + 2) * _EPS / (1.0 - (n + 2) * _EPS)))
    magnitude = float(_up(magnitude / float(_down(1.0 - gamma))))
    error = float(_up(float(_up(gamma * magnitude)) + (n + 2) * _TINY))
    return float(_down(float(np.sum(lo)) - error)), float(_up(float(np.sum(hi)) + error))


def _mul_interval(alo, ahi, blo, bhi):
    v = np.array([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return _down(v.min(axis=0)), _up(v.max(axis=0))


def _square_interval(lo, hi):
    a, b = lo * lo, hi * hi
    low = np.where((lo <= 0) & (hi >= 0), 0.0, np.maximum(0.0, _down(np.minimum(a, b))))
    return low, _up(np.maximum(a, b))


def _distance_bounds(points, xlo, xhi, ylo, yhi):
    dxlo = np.maximum(0.0, np.maximum(_down(xlo - points[:, 0]), _down(points[:, 0] - xhi)))
    dylo = np.maximum(0.0, np.maximum(_down(ylo - points[:, 1]), _down(points[:, 1] - yhi)))
    dxhi = np.maximum(_up(np.abs(xlo - points[:, 0])), _up(np.abs(xhi - points[:, 0])))
    dyhi = np.maximum(_up(np.abs(ylo - points[:, 1])), _up(np.abs(yhi - points[:, 1])))
    rlo = np.maximum(0.0, _down(np.maximum(0.0, _down(dxlo * dxlo)) + np.maximum(0.0, _down(dylo * dylo))))
    rhi = _up(_up(dxhi * dxhi) + _up(dyhi * dyhi))
    return rlo, rhi


def _point_intervals(points, q, hessian=False):
    dx = q[0] - points[:, 0]
    dy = q[1] - points[:, 1]
    xlo, xhi = _down(dx), _up(dx)
    ylo, yhi = _down(dy), _up(dy)
    xxlo, xxhi = _square_interval(xlo, xhi)
    yylo, yyhi = _square_interval(ylo, yhi)
    rlo = np.maximum(0.0, _down(xxlo + yylo))
    rhi = _up(xxhi + yyhi)
    if np.any(rlo == 0):
        return None
    vlo, vhi = _down(1.0 / rhi), _up(1.0 / rlo)
    wlo, whi = _down(vlo * vlo), _up(vhi * vhi)
    flo, fhi = _sum_interval(vlo, vhi)
    gxlo, gxhi = _mul_interval(xlo, xhi, wlo, whi)
    gylo, gyhi = _mul_interval(ylo, yhi, wlo, whi)
    gxlo, gxhi = _sum_interval(_down(-2 * gxhi), _up(-2 * gxlo))
    gylo, gyhi = _sum_interval(_down(-2 * gyhi), _up(-2 * gylo))
    _, s4 = _sum_interval(wlo, whi)
    result = [flo, fhi, max(abs(gxlo), abs(gxhi)), max(abs(gylo), abs(gyhi)), s4]
    if hessian:
        ulo, uhi = _down(wlo * vlo), _up(whi * vhi)
        axlo = _down(_down(8 * _down(xxlo * ulo)) - _up(2 * whi))
        axhi = _up(_up(8 * _up(xxhi * uhi)) - _down(2 * wlo))
        aylo = _down(_down(8 * _down(yylo * ulo)) - _up(2 * whi))
        ayhi = _up(_up(8 * _up(yyhi * uhi)) - _down(2 * wlo))
        bxylo, bxyhi = _mul_interval(xlo, xhi, ylo, yhi)
        blo, bhi = _mul_interval(bxylo, bxyhi, ulo, uhi)
        alo, ahi = _sum_interval(axlo, axhi)
        dlo, dhi = _sum_interval(aylo, ayhi)
        blo, bhi = _sum_interval(_down(8 * blo), _up(8 * bhi))
        bmag = max(abs(blo), abs(bhi))
        diff = max(abs(float(_down(alo - dlo))), abs(float(_up(alo - dlo))))
        rad = float(_up(math.sqrt(float(_up(float(_up(diff * diff)) + float(_up(4 * float(_up(bmag * bmag)))))))))
        eigen_lower = float(_down(0.5 * float(_down(float(_down(alo + dlo)) - rad))))
        result.append(eigen_lower)
    return result


def intensity(points, q):
    """Ordinary (uncertified) binary64 objective evaluation."""
    points = np.asarray(points, dtype=float)
    q = np.asarray(q, dtype=float)
    r2 = np.sum((points - q) ** 2, axis=1)
    if np.any(r2 == 0):
        return _INF
    return float(np.sum(1 / r2))


def intensity_interval(points, q):
    """Outward-rounded enclosure of intensity at one exact binary64 point."""
    points = np.asarray(points, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any(np.all(points == q, axis=1)):
        return [_INF, _INF]
    d = _point_intervals(points, q)
    if d is None:
        return [0.0, _INF]
    return [d[0], d[1]]


def numerical_minimum(points, seeds=None, grid=7):
    """Find candidate minima; deliberately separate from the global certificate."""
    points = np.asarray(points, dtype=float)
    candidates = []

    def fg(q):
        delta = q - points
        r2 = np.maximum(np.sum(delta * delta, axis=1), 1e-150)
        v = np.sum(1 / r2)
        g = -2 * np.sum(delta / r2[:, None] ** 2, axis=0)
        return float(v), g

    def add(q):
        q = np.clip(np.asarray(q), 0, 1)
        v = intensity(points, q)
        if np.isfinite(v):
            candidates.append((v, q.copy()))

    for q in ((0, 0), (0, 1), (1, 0), (1, 1)):
        add(q)
    # Every boundary component gets independent one-dimensional minimizations.
    knots = np.unique(np.concatenate(([0., 1.], np.linspace(0, 1, grid * 2), points.ravel())))
    for axis in range(2):
        for fixed in (0., 1.):
            def edge(t):
                q = [t, t]
                q[axis] = fixed
                return intensity(points, q)
            for a, b in zip(knots[:-1], knots[1:]):
                r = minimize_scalar(edge, bounds=(a, b), method="bounded", options={"xatol": 2e-14})
                q = [r.x, r.x]
                q[axis] = fixed
                add(q)
    starts = list(np.asarray(seeds, dtype=float)) if seeds is not None else []
    coords = (np.arange(grid) + .5) / grid
    starts.extend((x, y) for x in coords for y in coords)
    for q in starts:
        if min(np.sum((points - q) ** 2, axis=1)) < 1e-20:
            continue
        r = minimize(fg, np.asarray(q), jac=True, bounds=((0, 1), (0, 1)), method="L-BFGS-B", options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 150})
        add(r.x)
    candidates.sort(key=lambda z: z[0])
    unique = []
    for v, q in candidates:
        if all(np.linalg.norm(q - z[1]) > 1e-6 for z in unique):
            unique.append((v, q))
    return {"objective": unique[0][0], "witness": unique[0][1].tolist(), "minima": [{"value": v, "point": q.tolist()} for v, q in unique]}


def _box_bound(points, box, use_cubic=True):
    xlo, xhi, ylo, yhi = box
    cx, cy = (xlo + xhi) * .5, (ylo + yhi) * .5
    hx = float(_up(max(float(_up(cx-xlo)), float(_up(xhi-cx)))))
    hy = float(_up(max(float(_up(cy-ylo)), float(_up(yhi-cy)))))
    rlo, rhi = _distance_bounds(points, *box)
    reciprocal = _down(1.0 / rhi)
    lower, _ = _sum_interval(reciprocal, _up(1.0 / rhi))
    cubic = bool(use_cubic and np.all(rlo > 0))
    center = _point_intervals(points, (cx, cy), hessian=cubic)
    if center is None:
        return max(0.0, lower), _INF, (cx, cy)
    flo, fhi, gx, gy, s4 = center[:5]
    rho2 = float(_up(float(_up(hx * hx)) + float(_up(hy * hy))))
    drop = float(_up(float(_up(gx * hx)) + float(_up(gy * hy))))
    tangent = float(_down(float(_down(flo - drop)) - float(_up(s4 * rho2))))
    lower = max(lower, tangent)
    if cubic:
        lam = center[5]
        # Lower bound for each one-dimensional quadratic; both alternatives
        # below are valid when lam > 0, hence their maximum is also valid.
        def quadratic(g, h):
            gh = float(_up(g * h))
            if lam > 0:
                unconstrained = -float(_up(float(_up(g * g)) / float(_down(2 * lam))))
                return max(-gh, unconstrained)
            return float(_down(-gh + float(_down(.5 * float(_down(lam * float(_up(h * h))))))))
        ix = _up(1.0 / rlo)
        s5terms = _up(_up(ix * ix) * _up(1.0 / np.maximum(_TINY, _down(np.sqrt(rlo)))))
        _, s5 = _sum_interval(s5terms, s5terms)
        rho3 = float(_up(rho2 * float(_up(math.sqrt(rho2)))))
        remainder = float(_up(4 * float(_up(s5 * rho3))))
        third = float(_down(float(_down(float(_down(flo + quadratic(gx, hx))) + quadratic(gy, hy))) - remainder))
        lower = max(lower, third)
    return max(0.0, lower), fhi, (cx, cy)


def validate(points, tolerance=1e-5, max_boxes=1_000_000, time_limit=120.0,
             numerical=True, witness=None, use_cubic=True, progress=None):
    """Certify a placement's continuous minimum to an absolute target tolerance.

    Returns JSON-serializable bounds even if limits prevent reaching tolerance.
    `valid` validates the input geometry; `certified` means the requested gap was
    achieved. `objective_lower` is the guaranteed performance to report.  A
    `witness` from the search may be supplied to reduce preprocessing time.
    The proof applies to exact binary64 coordinates, not rounded diagram labels.
    """
    start = time.monotonic()
    points = np.asarray(points, dtype=float)
    reasons = []
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) == 0:
        reasons.append("Expected a nonempty n by 2 coordinate array.")
    elif not np.all(np.isfinite(points)):
        reasons.append("Coordinates must be finite.")
    elif np.any(points < 0) or np.any(points > 1):
        reasons.append("A light lies outside the closed unit square.")
    if reasons:
        return {"valid": False, "certified": False, "errors": reasons, "elapsed_seconds": time.monotonic() - start}
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    upper = _INF
    best_q = None
    numerics = None
    candidates = [(0., 0.), (0., 1.), (1., 0.), (1., 1.)]
    if witness is not None:
        q = np.asarray(witness, dtype=float)
        if q.shape == (2,) and np.all(np.isfinite(q)) and np.all(q >= 0) and np.all(q <= 1):
            candidates.append(q)
    if numerical:
        numerics = numerical_minimum(points)
        candidates.append(numerics["witness"])
    for q in candidates:
        qhi = intensity_interval(points, q)[1]
        if qhi < upper:
            upper, best_q = qhi, tuple(float(x) for x in q)
    root = (0., 1., 0., 1.)
    lb, ub, q = _box_bound(points, root, use_cubic)
    if ub < upper:
        upper, best_q = ub, q
    heap = [(lb, 0, root)]
    serial = 0
    processed = 0
    terminal_lower = _INF
    status = "certified"
    last_progress = start
    while heap:
        global_lower = min(terminal_lower, heap[0][0])
        if float(_up(upper - global_lower)) <= tolerance:
            break
        if processed >= max_boxes:
            status = "box_limit"
            break
        if time.monotonic() - start >= time_limit:
            status = "time_limit"
            break
        lb, _, box = heapq.heappop(heap)
        if lb >= float(_up(upper - tolerance)):
            terminal_lower = min(terminal_lower, lb)
            continue
        xlo, xhi, ylo, yhi = box
        split_x = xhi - xlo >= yhi - ylo
        if split_x:
            mid = (xlo + xhi) * .5
            children = [(xlo, mid, ylo, yhi), (mid, xhi, ylo, yhi)]
        else:
            mid = (ylo + yhi) * .5
            children = [(xlo, xhi, ylo, mid), (xlo, xhi, mid, yhi)]
        if (mid in (xlo, xhi)) if split_x else (mid in (ylo, yhi)):
            # Cannot shrink further in binary64: retain the valid bound.
            terminal_lower = min(terminal_lower, lb)
            status = "floating_point_resolution"
            break
        for child in children:
            clb, cub, cq = _box_bound(points, child, use_cubic)
            # Parent bounds apply to every descendant and sometimes are better.
            clb = max(lb, clb)
            if cub < upper:
                upper, best_q = cub, cq
            serial += 1
            if clb < float(_up(upper - tolerance)):
                heapq.heappush(heap, (clb, serial, child))
            else:
                terminal_lower = min(terminal_lower, clb)
        processed += 1
        now = time.monotonic()
        if progress and now - last_progress >= 5:
            progress({"lower": min(terminal_lower, heap[0][0] if heap else _INF), "upper": upper, "boxes": processed, "seconds": now - start})
            last_progress = now
    lower = min(terminal_lower, heap[0][0] if heap else _INF)
    gap = float(_up(upper - lower))
    certified = gap <= tolerance
    if certified:
        status = "certified"
    elif status == "certified":
        status = "tolerance_rounding"
    return {
        "valid": True, "certified": bool(certified), "status": status,
        "n": len(points), "objective_lower": float(lower), "objective_upper": float(upper),
        "objective_estimate": numerics["objective"] if numerics else intensity(points, best_q),
        "gap": gap, "tolerance": float(tolerance), "witness": list(best_q),
        "witness_intensity_interval": intensity_interval(points, best_q),
        "processed_boxes": processed, "remaining_boxes": len(heap),
        "elapsed_seconds": time.monotonic() - start,
        "domain": [0.0, 1.0, 0.0, 1.0],
        "coordinate_interpretation": "exact IEEE-754 binary64 values",
        "coordinates_sha256_float64_little_endian": hashlib.sha256(np.asarray(points, dtype="<f8").tobytes(order="C")).hexdigest(),
        "bound_method": "outward-rounded spatial branch and bound; reciprocal tangents, farthest distances, cubic Taylor bounds",
        "numerical_minima": numerics["minima"] if numerics else [],
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="JSON coordinate array or object containing points")
    p.add_argument("--tolerance", type=float, default=1e-5)
    p.add_argument("--time-limit", type=float, default=120)
    p.add_argument("--output")
    args = p.parse_args()
    data = json.loads(Path(args.input).read_text())
    points = data.get("points", data) if isinstance(data, dict) else data
    result = validate(points, tolerance=args.tolerance, time_limit=args.time_limit)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
