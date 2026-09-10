"""Stage 3 sanity: analytic Jacobians, and end-to-end optimize -> polish -> referee on tiny cases."""
import math

import numpy as np
import pytest

from fence import builder as Bd
from fence import geometry as G
from fence import optimize as O

torch = pytest.importorskip("torch")


def _fd_check(fun, jac, z, h=1e-6):
    J = jac(z)
    Jn = np.zeros_like(J)
    for k in range(len(z)):
        e = np.zeros_like(z)
        e[k] = h
        Jn[:, k] = (np.atleast_1d(fun(z + e)) - np.atleast_1d(fun(z - e))) / (2 * h)
    return np.abs(J - Jn).max()


def test_numpy_jacobians_match_finite_differences():
    sk = Bd.build_from_genome(14, np.random.default_rng(3), {"seed": {"kind": "ring"}, "grow": {"inside_prob": 0.6}})
    topo = sk.topology()
    prob = O.NumpyProblem(topo)
    X = sk.coords()
    T = np.array([G.fence_param(X, *topo.fences[topo.pins[v]], v) for v in prob.pins])
    z = np.concatenate([X.ravel(), T]) + 1e-3 * np.random.default_rng(0).normal(size=prob.N)
    assert _fd_check(prob.eq, prob.eq_jac, z) < 1e-6
    assert _fd_check(prob.ineq, prob.ineq_jac, z) < 1e-6
    f = lambda q: prob.objective(q)[0]  # noqa: E731
    g = prob.objective(z)[1]
    gn = np.array([(f(z + h) - f(z - h)) / 2e-6 for h in 1e-6 * np.eye(prob.N)])
    assert np.abs(g - gn).max() < 1e-6


def _run(n, genome, seed=0, steps=400, batch=128):
    sk = Bd.build_from_genome(n, np.random.default_rng(seed), genome)
    cfg = O.OptConfig(batch=batch, steps=steps, topk=4, verify_top=2, polish_iters=600)
    return O.optimize_topology(sk.topology(), sk.coords(), n, cfg, device="cpu", seed=seed)


def test_square_reaches_area_one_and_is_verified():
    res = _run(4, {"seed": {"kind": "ring", "k": 4}})
    assert res["best"] is not None, res
    assert abs(res["best"]["area"] - 1.0) < 1e-9
    assert res["best"]["config"]["verification"]["valid"]


def test_triangle_is_rigid():
    res = _run(3, {"seed": {"kind": "ring", "k": 3}})
    assert abs(res["best"]["area"] - math.sqrt(3) / 4) < 1e-9


def test_pentagon_face_capped_at_one():
    # a ring of 5 unit fences could enclose 1.72, but a single face may not exceed 1
    res = _run(5, {"seed": {"kind": "ring", "k": 5}})
    assert res["best"] is not None
    assert res["best"]["area"] <= 1.0 + 1e-9
    assert res["best"]["area"] > 1.0 - 1e-6


def test_torch_lagrangian_gradients_finite():
    sk = Bd.build_from_genome(12, np.random.default_rng(1), {"seed": {"kind": "ring"}, "grow": {"inside_prob": 0.5}})
    topo = sk.topology()
    tt = G.TorchTopology(topo, dtype=torch.float64)
    gen = torch.Generator()
    gen.manual_seed(0)
    Zt = O.init_batch(tt, sk.coords(), 16, O.OptConfig(), gen).T.contiguous().requires_grad_(True)
    alm = O.ALMState(tt, 16, Zt.dtype, Zt.device)
    alm.rho.fill_(100.0)
    alm.w.fill_(100.0)
    alm.lagrangian(tt.residuals_t(Zt)).sum().backward()
    assert torch.isfinite(Zt.grad).all()
