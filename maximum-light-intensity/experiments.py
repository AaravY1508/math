"""Independent optimization experiments: moving-witness minimax polishing.

The optimizer tracks smooth local-minimum branches of the illumination field.
Its constraint Jacobian is exact by the envelope theorem: derivatives of the
minimizing witness positions do not enter the light-position derivatives.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize


def _field(points, sites, hessian=False):
    d = sites[:, None, :] - points[None, :, :]
    r2 = np.maximum(np.sum(d*d, axis=2), 1e-24)
    ir2 = 1.0/r2
    vals = ir2.sum(axis=1)
    grad = (-2*d*ir2[:, :, None]**2).sum(axis=1)
    if not hessian:
        return vals, grad
    hes = 8*np.einsum('mni,mnj,mn->mij', d, d, ir2**3)
    hes[:, 0, 0] -= 2*(ir2**2).sum(axis=1)
    hes[:, 1, 1] -= 2*(ir2**2).sum(axis=1)
    return vals, grad, hes


def track_sites(points, reference, maxiter=35, tol=1e-9):
    """Damped vectorized Newton minimization, preserving square boundary faces.

    Reference sites on an edge are constrained to that edge. Corners are fixed.
    Interior sites are allowed to reach the square boundary if their branch dies.
    The routine is deterministic for a fixed reference and point configuration.
    """
    points = np.asarray(points, float)
    q = np.array(reference, dtype=float, copy=True)
    fixed = (q < 1e-10) | (q > 1-1e-10)
    free = ~fixed
    for iteration in range(maxiter):
        vals, g, h = _field(points, q, hessian=True)
        g[fixed] = 0
        # Projected gradient at a boundary reached by an interior branch.
        g[(q <= 1e-12) & (g > 0)] = 0
        g[(q >= 1-1e-12) & (g < 0)] = 0
        done = np.max(np.abs(g), axis=1) <= tol*np.maximum(1, vals)
        if np.all(done):
            break
        h[:, 0, 0] = np.where(fixed[:, 0], 1, h[:, 0, 0])
        h[:, 1, 1] = np.where(fixed[:, 1], 1, h[:, 1, 1])
        h[:, 0, 1] = h[:, 1, 0] = np.where(fixed.any(axis=1), 0, h[:, 0, 1])
        # Positive-definite modified Newton direction, even far from a branch.
        eigmin = 0.5*(h[:, 0, 0]+h[:, 1, 1]-np.hypot(h[:, 0, 0]-h[:, 1, 1], 2*h[:, 0, 1]))
        shift = np.maximum(0, 1e-3*np.maximum(1, vals)-eigmin)
        h[:, 0, 0] += shift
        h[:, 1, 1] += shift
        det = h[:, 0, 0]*h[:, 1, 1]-h[:, 0, 1]**2
        step = np.column_stack((h[:, 1, 1]*g[:, 0]-h[:, 0, 1]*g[:, 1],
                                h[:, 0, 0]*g[:, 1]-h[:, 0, 1]*g[:, 0]))/np.maximum(det[:, None], 1e-30)
        step[fixed] = 0
        step[done] = 0
        # Limit branch travel and backtrack each witness independently.
        step *= np.minimum(1, 0.08/np.maximum(np.linalg.norm(step, axis=1), 1e-30))[:, None]
        alpha = np.ones(len(q))
        gd = (g*step).sum(axis=1)
        for _ in range(18):
            candidate = np.clip(q-alpha[:, None]*step, 0, 1)
            cv, _ = _field(points, candidate)
            accept = (cv <= vals-1e-4*alpha*gd) | done | (alpha < 1e-5)
            if np.all(accept):
                break
            alpha[~accept] *= 0.5
        delta = np.max(np.abs(candidate-q), axis=1)
        q = candidate
        if np.max(delta) < 1e-13:
            break
    values, gradients = _field(points, q)
    return values, q


def optimize_tracked(points, maxiter=300, grid=41, rounds=5,
                     ftol=2e-10, radius=0.25, verbose=False, oracle_fn=None, **kwargs):
    """Optimize moving minimum branches and rediscover all minima each round.

    Returns the same essential data as a semi-infinite exchange optimizer:
    points, objective, witnesses, witness_values, evaluations, elapsed, history.
    """
    from solver import oracle as default_oracle, intensity_jac
    oracle = default_oracle if oracle_fn is None else oracle_fn
    start = time.monotonic()
    p = np.array(points, float, copy=True)
    best = p.copy()
    initial_values, initial_sites = oracle(p, grid=grid)
    best_value = float(np.min(initial_values))
    scale = max(best_value, 1.0)
    history = []
    evaluations = 0
    reference = initial_sites
    remaining = int(maxiter)
    for outer in range(rounds):
        if remaining <= 0:
            break
        current = p.copy()
        count = min(remaining, max(30, int(np.ceil(remaining/(rounds-outer)))))
        cache_x = None
        cache_c = cache_j = cache_sites = None
        def evaluate(x):
            nonlocal cache_x, cache_c, cache_j, cache_sites, evaluations
            if cache_x is not None and np.array_equal(x, cache_x):
                return cache_c, cache_j
            lights = x[:-1].reshape((-1, 2))
            vals, sites = track_sites(lights, reference)
            vals, jac = intensity_jac(lights, sites)
            cache_x = x.copy()
            cache_c = vals/scale-x[-1]
            cache_j = np.column_stack((jac/scale, -np.ones(len(sites))))
            cache_sites = sites
            evaluations += 1
            return cache_c, cache_j
        x0 = np.r_[p.ravel(), float(np.min(_field(p, reference)[0]))/scale]
        lower = np.maximum(0, current.ravel()-radius)
        upper = np.minimum(1, current.ravel()+radius)
        bounds = list(zip(lower, upper)) + [(0, None)]
        result = minimize(lambda x: -x[-1], x0,
                          jac=lambda x: np.r_[np.zeros(len(x)-1), -1.0],
                          method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': lambda x: evaluate(x)[0],
                                       'jac': lambda x: evaluate(x)[1]},
                          options={'maxiter': count, 'ftol': ftol, 'disp': False})
        remaining -= int(result.nit)
        p = np.clip(result.x[:-1].reshape((-1, 2)), 0, 1)
        values, sites = oracle(p, grid=grid)
        value = float(np.min(values))
        history.append({'round': outer, 'objective': value,
                        'surrogate': float(result.x[-1]*scale),
                        'iterations': int(result.nit), 'success': bool(result.success),
                        'message': str(result.message), 'witnesses': len(sites)})
        if verbose:
            print(history[-1], flush=True)
        if value > best_value:
            best_value, best = value, p.copy()
        reference = sites
        if abs(value-result.x[-1]*scale) < 1e-7 and result.success:
            break
    values, sites = oracle(best, grid=grid)
    return {'points': best, 'objective': float(np.min(values)),
            'witnesses': sites, 'witness_values': values,
            'evaluations': evaluations, 'elapsed': time.monotonic()-start,
            'history': history}


def boundary_grid_seed(n, rng=None, boundary_fraction=0.6):
    """A square boundary ring with staggered interior grid.

    Near-boundary lights cover low-background corners/edges. Remaining lights
    form an offset square or triangular lattice inside the central square.
    """
    rng = np.random.default_rng() if rng is None else rng
    k = round((n*boundary_fraction-4)/4)
    k = max(2, k)
    ring_count = 4*(k+1)
    while ring_count > n-4:
        k -= 1
        ring_count = 4*(k+1)
    offset = 0.032
    s = np.linspace(offset, 1-offset, k+2)
    ring = np.array([(x, offset) for x in s] + [(x, 1-offset) for x in s]
                    + [(offset, y) for y in s[1:-1]] + [(1-offset, y) for y in s[1:-1]])
    inner_n = n-len(ring)
    rows = max(2, round(np.sqrt(inner_n)))
    counts = np.full(rows, inner_n//rows)
    counts[:inner_n%rows] += 1
    rng.shuffle(counts)
    interior = np.array([(x, y) for row, (y, count) in enumerate(zip(np.linspace(.20, .80, rows), counts))
                         for x in np.linspace(.20, .80, count)])
    points = np.concatenate((ring, interior))
    points += rng.normal(0, .004, points.shape)
    return np.clip(points, .005, .995)

# Numba is optional, but substantially accelerates branch tracking during search.
try:
    from numba import njit
except ImportError:
    njit = None

if njit is not None:
    @njit(cache=True)
    def _point_field(p, x, y):
        value = gx = gy = hxx = hxy = hyy = 0.0
        for j in range(len(p)):
            dx = x-p[j, 0]
            dy = y-p[j, 1]
            r2 = max(dx*dx+dy*dy, 1e-24)
            inv = 1.0/r2
            inv2 = inv*inv
            inv3 = inv2*inv
            value += inv
            gx -= 2*dx*inv2
            gy -= 2*dy*inv2
            hxx += 8*dx*dx*inv3-2*inv2
            hxy += 8*dx*dy*inv3
            hyy += 8*dy*dy*inv3-2*inv2
        return value, gx, gy, hxx, hxy, hyy

    @njit(cache=True)
    def _point_value(p, x, y):
        value = 0.0
        for j in range(len(p)):
            dx = x-p[j, 0]
            dy = y-p[j, 1]
            value += 1.0/max(dx*dx+dy*dy, 1e-24)
        return value

    @njit(cache=True)
    def _track_core(p, reference, maxiter, tol):
        q = reference.copy()
        values = np.empty(len(q))
        for i in range(len(q)):
            x, y = q[i, 0], q[i, 1]
            fixedx = x < 1e-10 or x > 1-1e-10
            fixedy = y < 1e-10 or y > 1-1e-10
            for it in range(maxiter):
                value, gx, gy, hxx, hxy, hyy = _point_field(p, x, y)
                if fixedx or (x <= 1e-12 and gx > 0) or (x >= 1-1e-12 and gx < 0):
                    gx = 0.0
                if fixedy or (y <= 1e-12 and gy > 0) or (y >= 1-1e-12 and gy < 0):
                    gy = 0.0
                if max(abs(gx), abs(gy)) <= tol*max(1.0, value):
                    break
                if fixedx:
                    hxx = 1.0
                if fixedy:
                    hyy = 1.0
                if fixedx or fixedy:
                    hxy = 0.0
                eigmin = 0.5*(hxx+hyy-np.hypot(hxx-hyy, 2*hxy))
                shift = max(0.0, 1e-3*max(1.0, value)-eigmin)
                hxx += shift
                hyy += shift
                det = max(hxx*hyy-hxy*hxy, 1e-30)
                sx = (hyy*gx-hxy*gy)/det
                sy = (hxx*gy-hxy*gx)/det
                if fixedx:
                    sx = 0.0
                if fixedy:
                    sy = 0.0
                fraction = min(1.0, .08/max(np.hypot(sx, sy), 1e-30))
                sx *= fraction
                sy *= fraction
                gd = gx*sx+gy*sy
                alpha = 1.0
                for bt in range(18):
                    nx = min(1.0, max(0.0, x-alpha*sx))
                    ny = min(1.0, max(0.0, y-alpha*sy))
                    nv = _point_value(p, nx, ny)
                    if nv <= value-1e-4*alpha*gd or alpha < 1e-5:
                        break
                    alpha *= .5
                delta = max(abs(nx-x), abs(ny-y))
                x, y = nx, ny
                if delta < 1e-13:
                    break
            q[i, 0], q[i, 1] = x, y
            values[i] = _point_value(p, x, y)
        return values, q

    track_sites_numpy = track_sites

    def track_sites(points, reference, maxiter=35, tol=1e-9):
        return _track_core(np.ascontiguousarray(points, dtype=float),
                           np.ascontiguousarray(reference, dtype=float),
                           maxiter, tol)

if njit is not None:
    @njit(cache=True)
    def _values_core(points, sites):
        vals = np.empty(len(sites))
        for i in range(len(sites)):
            vals[i] = _point_value(points, sites[i, 0], sites[i, 1])
        return vals


def fast_oracle(points, grid=39, extra=None):
    """Fast candidate-generation oracle; final verification must be independent.

    Uses grid and Voronoi starts plus independently searched edge intervals,
    refining all starts with the compiled damped Newton tracker. It preserves
    the (values sorted ascending, witness sites) interface of solver.oracle.
    """
    from scipy.ndimage import minimum_filter
    from scipy.spatial import Voronoi, QhullError
    from solver import unique_sites, intensity
    p = np.ascontiguousarray(points, dtype=float)
    def values(sites):
        s = np.ascontiguousarray(sites, dtype=float)
        return _values_core(p, s) if njit is not None else intensity(p, s)
    t = np.linspace(0, 1, grid)
    xx, yy = np.meshgrid(t, t)
    grid_sites = np.column_stack((xx.ravel(), yy.ravel()))
    gv = values(grid_sites).reshape((grid, grid))
    mask = gv == minimum_filter(gv, size=3, mode='constant', cval=np.inf)
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    starts = [np.array([0., 0.]), np.array([0., 1.]),
              np.array([1., 0.]), np.array([1., 1.])]
    starts.extend(grid_sites[mask.ravel()])
    try:
        vv = Voronoi(p).vertices
        starts.extend(vv[np.all((vv > 0) & (vv < 1), axis=1)])
    except QhullError:
        pass
    if extra is not None:
        starts.extend(extra)
    et = np.linspace(0, 1, max(grid*3, 121))
    for axis in (0, 1):
        for fixed in (0., 1.):
            sites = np.zeros((len(et), 2))
            sites[:, axis] = et
            sites[:, 1-axis] = fixed
            ev = values(sites)
            ix = np.where((ev[1:-1] <= ev[:-2]) & (ev[1:-1] <= ev[2:]))[0]+1
            starts.extend(sites[ix])
    _, sites = track_sites(p, np.array(starts), maxiter=80, tol=1e-10)
    sites = unique_sites(sites, eps=1e-5)
    vs = values(sites)
    ix = np.argsort(vs)
    return vs[ix], sites[ix]
