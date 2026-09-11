"""Stage 3: batched local optimization of one FIXED topology.

Pipeline for a topology (vertex coords + pin parameters t are the free variables):

1. GPU: B independent random restarts, conceptually one [B, P] tensor (stored
   transposed as [P, B] for coalesced gathers). Adam minimizes an augmented
   Lagrangian of
        maximize total_area  s.t.  |fence| = 1, pins on hosts, face_area <= 1
   plus pure penalties for the "embedding safety" constraints (no crossings,
   pins ordered along hosts, no folded corners, faces positively oriented).
   The penalty parameter rho and the safety weight w are annealed from soft to
   stiff so restarts explore first and are forced feasible later instead of
   "cheating" early; Lagrange multipliers are updated periodically (PHR method),
   which lets the batch converge to near-KKT points without the ill-conditioning
   of a pure penalty method. The whole Adam step runs as a replayed CUDA graph.
2. The top-k distinct near-feasible restarts are polished on the CPU in float64
   with scipy SLSQP (exact equality constraints for lengths and pins, inequality
   constraints for face areas / pin order), in chunks with an embedding check after
   each chunk (revert on any crossing), followed by a Gauss-Newton projection onto
   the active constraints (residuals ~1e-15).
3. Candidates are then certified by fence.referee.verify (50 digits); only a
   verified area ever counts as a result.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from fence import geometry as G

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class OptConfig:
    batch: int = 16384           # restarts per topology (capped by memory, see mem_cap_gb)
    steps: int = 1000            # Adam steps
    lr: float = 0.03
    lr_end: float = 2e-4
    rho_start: float = 10.0      # augmented-Lagrangian penalty parameter schedule
    rho_end: float = 3000.0
    w_start: float = 10.0        # weight of the pure "safety" penalties
    w_end: float = 1e4
    anneal_frac: float = 0.7     # fraction of steps over which rho / w ramp up
    mult_start: float = 0.2      # multiplier updates begin at this fraction of steps ...
    mult_every: int = 40         # ... and happen every this many steps
    noise_max: float = 0.18      # per-restart init noise sigma ~ U(0, noise_max)
    scale_jitter: float = 0.10   # per-restart global scale ~ U(1-sj, 1+sj)
    topk: int = 8                # candidates sent to the CPU polish
    polish_iters: int = 1200
    verify_top: int = 3          # best polished candidates sent to the referee
    tol: float = 1e-9            # referee tolerance
    mem_cap_gb: float = 14.0     # hard cap for the GPU working set of one batch
    feas_gate: float = 5e-3      # max violation for a restart to be polished
    cross_radius: float = 1.6    # crossing penalty only for fence pairs this close in the sketch
    avoid_height: float = 2.0    # repulsion from known minima: bump height, in units of area ...
    avoid_sigma: float = 0.15    # ... and width, as RMS per-vertex distance after rigid alignment


SAFETY = {"cross": 3.0, "order": 1.0, "turn": 0.3, "face_pos": 1.0}


def default_batch(n: int) -> int:
    """Default restarts per topology as a function of n (overridable by flag)."""
    if n <= 20:
        return 16384
    if n <= 35:
        return 12288
    return 10240


# ---------------------------------------------------------------------------
# GPU stage
# ---------------------------------------------------------------------------

def _bytes_per_restart(tt) -> int:
    Q = len(tt.qi)
    FE = len(tt.fe_i)
    C = len(tt.cu)
    # rough count of float32 temporaries kept for autograd (x3 safety margin)
    floats = 14 * Q + 8 * FE + 16 * C + 12 * tt.K + 12 * tt.topo.n + 8 * tt.n_params
    return 4 * floats * 3


def init_batch(tt, coords0: np.ndarray, B: int, cfg: OptConfig, gen):
    """Random restarts around the sketch, returned batch-first [B, P]: global scale
    jitter + per-restart gaussian noise, then pins are projected back onto hosts.
    Restart 0 is the unperturbed sketch."""
    torch = tt.torch
    X0 = torch.as_tensor(np.asarray(coords0) - np.asarray(coords0).mean(0), dtype=tt.dtype, device=tt.device)
    scale = 1.0 + cfg.scale_jitter * (2 * torch.rand(B, 1, 1, generator=gen, device=tt.device, dtype=tt.dtype) - 1)
    sigma = cfg.noise_max * torch.rand(B, 1, 1, generator=gen, device=tt.device, dtype=tt.dtype)
    sigma[0] = 0.0
    scale[0] = 1.0
    X = X0[None] * scale + sigma * torch.randn(B, tt.V, 2, generator=gen, device=tt.device, dtype=tt.dtype)
    T = tt.initial_t(X).clamp(0.02, 0.98) if tt.K else X.new_zeros(B, 0)
    for _ in range(4):  # Gauss-Seidel sweeps also handle cyclic pin dependencies (pinwheels)
        if not tt.K:
            break
        A = X[:, tt.pin_ha]
        Bv = X[:, tt.pin_hb]
        X[:, tt.pin_v] = A + T.unsqueeze(-1) * (Bv - A)
        T = tt.initial_t(X).clamp(0.02, 0.98)
    return tt.pack(X, T)


class ALMState:
    """Per-restart Lagrange multipliers (batch-last) + schedule scalars on device."""

    def __init__(self, tt, B, dtype, device):
        torch = tt.torch
        self.lam_len = torch.zeros(tt.topo.n, B, dtype=dtype, device=device)
        self.lam_pin = torch.zeros(tt.K, 2, B, dtype=dtype, device=device)
        # an active face has multiplier 1 at a KKT point (d total_area / d face_area = 1)
        self.lam_area = torch.ones(tt.F, B, dtype=dtype, device=device)
        self.rho = torch.tensor(1.0, dtype=dtype, device=device)
        self.w = torch.tensor(1.0, dtype=dtype, device=device)

    def lagrangian(self, r):
        """Per-restart augmented Lagrangian [B] (to be minimized)."""
        rho, w = self.rho, self.w
        c = r["length"]
        L = -r["_areas"].sum(0)
        L = L + (self.lam_len * c).sum(0) + 0.5 * rho * c.pow(2).sum(0)
        if self.lam_pin.numel():
            p = r["pin_vec"]
            L = L + (self.lam_pin * p).sum((0, 1)) + 0.5 * rho * p.pow(2).sum((0, 1))
        g = r["_areas"] - 1.0
        # PHR term for the inequality face_area - 1 <= 0
        L = L + ((self.lam_area + rho * g).clamp_min(0.0).pow(2) - self.lam_area.pow(2)).sum(0) / (2 * rho)
        safe = 0.0
        for k, cf in SAFETY.items():
            v = r[k]
            if v.numel() == 0:
                continue
            safe = safe + (cf * v.sum(0) if k == "cross" else cf * v.pow(2).sum(0))
        return L + w * safe

    def update(self, r):
        rho = self.rho
        self.lam_len.add_(rho * r["length"])
        if self.lam_pin.numel():
            self.lam_pin.add_(rho * r["pin_vec"])
        self.lam_area.copy_((self.lam_area + rho * (r["_areas"] - 1.0)).clamp_min(0.0))
        for lam in (self.lam_len, self.lam_pin, self.lam_area):
            lam.nan_to_num_(0.0, 0.0, 0.0).clamp_(-1e3, 1e3)


class Repulsion:
    """Steers restarts away from known minima of this topology (deflation).

    Each known minimum, given in this topology's vertex labeling, adds a Gaussian bump of
    height `avoid_height` to every restart's Lagrangian. Its argument is the RMS per-vertex
    distance between the restart and the minimum after the best rotation, translation and
    reflection, which in 2D has a closed form:
        min |Xc - R Yc|^2 = |Xc|^2 + |Yc|^2 - 2 sqrt(<Xc, Yc>^2 + <Xc, Yc_perp>^2)
    so the bump means the same thing however the optimizer happens to place the layout.
    A restart that starts inside a known basin is pushed out of it while the constraints
    are still soft; restarts elsewhere never feel it.
    """

    def __init__(self, tt, avoid, cfg: OptConfig):
        torch = tt.torch
        self.torch = torch
        Y = torch.as_tensor(np.asarray(avoid, dtype=float), dtype=tt.dtype, device=tt.device)
        Y = Y - Y.mean(1, keepdim=True)
        mirror = torch.tensor([1.0, -1.0], dtype=tt.dtype, device=tt.device)
        self.M = Y.shape[0]
        self.Y = torch.cat([Y, Y * mirror], 0)                           # [2M, V, 2]
        self.Yp = torch.stack([self.Y[..., 1], -self.Y[..., 0]], -1)     # each point turned by -90 degrees
        self.syy = self.Y.pow(2).sum((1, 2))
        self.V = tt.V
        self.height = cfg.avoid_height
        self.inv = 1.0 / (2.0 * cfg.avoid_sigma ** 2)

    def rms2(self, X):
        """[2M, B] squared aligned RMS distance of every restart (X is [V, 2, B]) to every image."""
        Xc = X - X.mean(0, keepdim=True)
        sxx = Xc.pow(2).sum((0, 1))
        a = self.torch.einsum("vcb,mvc->mb", Xc, self.Y)
        b = self.torch.einsum("vcb,mvc->mb", Xc, self.Yp)
        return (sxx + self.syy[:, None] - 2.0 * (a * a + b * b + 1e-18).sqrt()).clamp_min(0.0) / self.V

    def penalty(self, X):
        """[B] repulsion added to each restart's Lagrangian."""
        return self.height * self.torch.exp(-self.rms2(X) * self.inv).sum(0)


def _schedule(cfg: OptConfig, step: int):
    frac = step / max(1, cfg.steps - 1)
    ramp = min(1.0, frac / cfg.anneal_frac)
    rho = cfg.rho_start * (cfg.rho_end / cfg.rho_start) ** ramp
    w = cfg.w_start * (cfg.w_end / cfg.w_start) ** ramp
    lr = cfg.lr_end + 0.5 * (cfg.lr - cfg.lr_end) * (1 + math.cos(math.pi * frac))
    return rho, w, lr


def _run_adam(tt, Zt, alm: ALMState, cfg: OptConfig, use_graph: bool, rep: Repulsion | None = None):
    """Augmented-Lagrangian Adam on the batch-last parameter tensor Zt[P, B] (in place).

    On CUDA the step (forward, backward, NaN guard, Adam update) is captured once as
    a CUDA graph and replayed; schedule scalars and multipliers live in device
    tensors that are updated in place between replays.
    """
    import torch

    dev = Zt.device
    lr_t = torch.tensor(cfg.lr, device=dev, dtype=torch.float32)
    opt = torch.optim.Adam([Zt], lr=lr_t if use_graph else cfg.lr, capturable=use_graph)

    def body():
        r = tt.residuals_t(Zt)
        lag = alm.lagrangian(r)
        if rep is not None:
            lag = lag + rep.penalty(tt.split_t(Zt)[0])
        loss = lag.sum()
        loss.backward()
        Zt.grad.nan_to_num_(0.0, 0.0, 0.0)   # a diverged restart must not poison Adam state
        opt.step()

    def set_sched(step):
        rho, w, lr = _schedule(cfg, step)
        alm.rho.fill_(rho)
        alm.w.fill_(w)
        if use_graph:
            lr_t.fill_(lr)
        else:
            for g in opt.param_groups:
                g["lr"] = lr

    def maybe_update(step):
        if step >= cfg.mult_start * cfg.steps and step % cfg.mult_every == 0:
            with torch.no_grad():
                alm.update(tt.residuals_t(Zt))

    if not use_graph:
        for step in range(cfg.steps):
            set_sched(step)
            maybe_update(step)
            opt.zero_grad(set_to_none=True)
            body()
        return
    warm = 3
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s):
        for step in range(warm):
            set_sched(step)
            opt.zero_grad(set_to_none=True)
            body()
    torch.cuda.current_stream(dev).wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        body()
    for step in range(warm, cfg.steps):
        set_sched(step)
        maybe_update(step)
        graph.replay()
    torch.cuda.synchronize(dev)
    del graph


def max_violation_t(r):
    """[B] max constraint violation (inf if any non-incident pair crosses)."""
    import torch

    parts = []
    for k in ("length", "pin", "area", "order", "turn", "face_pos"):
        v = r[k]
        if v.numel():
            parts.append(v.abs().amax(0))
    mv = torch.stack(parts, 0).amax(0)
    if r["cross"].numel():
        mv = torch.where(r["cross"].amax(0) > 0, torch.full_like(mv, float("inf")), mv)
    return mv


def adam_stage(topo: G.Topology, coords0: np.ndarray, cfg: OptConfig, device="cuda", seed=0, log=None,
               avoid: np.ndarray | None = None):
    """Run the batched augmented-Lagrangian Adam.

    `avoid` [M, V, 2]: known minima of this topology in its vertex labeling
    (fence.minima.avoid_images); restarts are repelled from them. Restarts can still
    polish back into a known basin, so callers also drop polished candidates whose area
    matches a known minimum (`distinct_candidates`).

    Returns (Z float64 [k, P] batch-first rows of the best distinct restarts, their
    areas, stats).
    """
    import torch

    tt = G.TorchTopology(topo, device=device, dtype=torch.float32, coords0=coords0, cross_radius=cfg.cross_radius)
    rep = Repulsion(tt, avoid, cfg) if avoid is not None and len(avoid) and cfg.avoid_height > 0 else None
    per = _bytes_per_restart(tt)
    B = int(max(64, min(cfg.batch, cfg.mem_cap_gb * 1e9 // per)))
    gen = torch.Generator(device=tt.device)
    gen.manual_seed(int(seed) % (2 ** 63))
    use_graph = tt.device.type == "cuda" and os.environ.get("FENCE_NO_CUDA_GRAPH") != "1"
    t0 = time.time()
    while True:
        try:
            Zt = init_batch(tt, coords0, B, cfg, gen).T.contiguous().requires_grad_(True)
            alm = ALMState(tt, B, Zt.dtype, Zt.device)
            try:
                _run_adam(tt, Zt, alm, cfg, use_graph, rep)
            except torch.cuda.OutOfMemoryError:
                raise
            except RuntimeError as exc:
                if not use_graph or "out of memory" in str(exc):
                    raise
                # graph capture unsupported for this op mix: fall back to eager
                if log:
                    log({"event": "cuda_graph_fallback", "error": repr(exc)[:300]})
                use_graph = False
                Zt = init_batch(tt, coords0, B, cfg, gen).T.contiguous().requires_grad_(True)
                alm = ALMState(tt, B, Zt.dtype, Zt.device)
                _run_adam(tt, Zt, alm, cfg, False, rep)
            break
        except torch.cuda.OutOfMemoryError:
            Zt = alm = None
            torch.cuda.empty_cache()
            if B <= 256:
                raise
            B //= 2
            if log:
                log({"event": "oom_backoff", "batch": B})
    with torch.no_grad():
        r = tt.residuals_t(Zt)
        area = r["_areas"].sum(0)
        mv = max_violation_t(r)
        finite = torch.isfinite(Zt).all(0) & torch.isfinite(area)
        ok = finite & (mv < cfg.feas_gate)
        stats = {"batch": B, "adam_s": round(time.time() - t0, 2), "feasible_frac": round(float(ok.float().mean()), 4),
                 "best_raw_area": float(torch.where(ok, area, torch.full_like(area, -1e9)).max())}
        if rep is not None:
            stats["avoid_images"] = rep.M
        idx = torch.nonzero(ok).flatten()
        if len(idx) == 0:
            return np.zeros((0, tt.n_params)), np.zeros(0), stats
        order = idx[torch.argsort(area[idx], descending=True)][: 50 * cfg.topk]
        Zs = Zt[:, order].T.double().cpu().numpy()
        As = area[order].double().cpu().numpy()
    del Zt, alm, r
    # keep distinct local optima (by area) -- restarts in the same basin are redundant
    keep, seen = [], []
    for i, a in enumerate(As):
        if all(abs(a - s) > 2e-5 for s in seen):
            keep.append(i)
            seen.append(a)
        if len(keep) >= cfg.topk:
            break
    return Zs[keep], As[keep], stats


# ---------------------------------------------------------------------------
# CPU polish (float64): SLSQP + Gauss-Newton projection
# ---------------------------------------------------------------------------

class NumpyProblem:
    """float64 objective/constraints with analytic Jacobians for one topology."""

    def __init__(self, topo: G.Topology, t_gap=1e-3, min_face=1e-4):
        self.topo = topo
        self.V = topo.n_vertices
        self.pins = topo.pin_vertices
        self.K = len(self.pins)
        self.N = 2 * self.V + self.K
        self.fa = topo.fences[:, 0]
        self.fb = topo.fences[:, 1]
        self.host = np.array([topo.pins[v] for v in self.pins], dtype=np.int64)
        self.pv = np.array(self.pins, dtype=np.int64)
        self.ha = topo.fences[self.host, 0] if self.K else np.zeros(0, dtype=np.int64)
        self.hb = topo.fences[self.host, 1] if self.K else np.zeros(0, dtype=np.int64)
        fi, fj, ff = [], [], []
        for k, cyc in enumerate(topo.faces):
            for i in range(len(cyc)):
                fi.append(cyc[i]); fj.append(cyc[(i + 1) % len(cyc)]); ff.append(k)
        self.fi, self.fj, self.ff = np.array(fi), np.array(fj), np.array(ff)
        self.F = len(topo.faces)
        kidx = {v: k for k, v in enumerate(self.pins)}
        lo, hi = [], []
        for c in topo.chains:
            # -1 stands for t = 0 (chain start) and -2 for t = 1 (chain end)
            seq = [-1] + [kidx[v] for v in c[1:-1]] + [-2]
            for s in range(len(seq) - 1):
                lo.append(seq[s]); hi.append(seq[s + 1])
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.t_gap, self.min_face = t_gap, min_face
        m = len(self.lo)
        self.Jo = np.zeros((m, self.N))
        for r in range(m):
            if self.hi[r] >= 0:
                self.Jo[r, 2 * self.V + self.hi[r]] += 1.0
            if self.lo[r] >= 0:
                self.Jo[r, 2 * self.V + self.lo[r]] -= 1.0

    def split(self, z):
        return z[: 2 * self.V].reshape(self.V, 2), z[2 * self.V:]

    def areas(self, X):
        c = X[self.fi, 0] * X[self.fj, 1] - X[self.fj, 0] * X[self.fi, 1]
        return 0.5 * np.bincount(self.ff, weights=c, minlength=self.F)

    def area_jac(self, X):
        J = np.zeros((self.F, self.N))
        xi, yi = 2 * self.fi, 2 * self.fi + 1
        xj, yj = 2 * self.fj, 2 * self.fj + 1
        np.add.at(J, (self.ff, xi), 0.5 * X[self.fj, 1])
        np.add.at(J, (self.ff, yj), 0.5 * X[self.fi, 0])
        np.add.at(J, (self.ff, xj), -0.5 * X[self.fi, 1])
        np.add.at(J, (self.ff, yi), -0.5 * X[self.fj, 0])
        return J

    def objective(self, z):
        X, _ = self.split(z)
        J = self.area_jac(X)
        return -float(self.areas(X).sum()), -J.sum(0)

    def eq(self, z):
        X, T = self.split(z)
        d = X[self.fb] - X[self.fa]
        r_len = (d * d).sum(1) - 1.0
        if not self.K:
            return r_len
        A = X[self.ha]
        Bv = X[self.hb]
        r_pin = (X[self.pv] - (A + T[:, None] * (Bv - A))).ravel()
        return np.concatenate([r_len, r_pin])

    def eq_jac(self, z):
        X, T = self.split(z)
        n = len(self.fa)
        J = np.zeros((n + 2 * self.K, self.N))
        d = X[self.fb] - X[self.fa]
        r = np.arange(n)
        np.add.at(J, (r, 2 * self.fb), 2 * d[:, 0])
        np.add.at(J, (r, 2 * self.fb + 1), 2 * d[:, 1])
        np.add.at(J, (r, 2 * self.fa), -2 * d[:, 0])
        np.add.at(J, (r, 2 * self.fa + 1), -2 * d[:, 1])
        for k in range(self.K):
            for c in (0, 1):
                row = n + 2 * k + c
                J[row, 2 * self.pv[k] + c] += 1.0
                J[row, 2 * self.ha[k] + c] -= 1.0 - T[k]
                J[row, 2 * self.hb[k] + c] -= T[k]
                J[row, 2 * self.V + k] -= X[self.hb[k], c] - X[self.ha[k], c]
        return J

    def ineq(self, z):
        X, T = self.split(z)
        A = self.areas(X)
        Te = np.concatenate([T, [1.0, 0.0]])       # Te[-1] = 0 (chain start), Te[-2] = 1 (chain end)
        g_order = Te[self.hi] - Te[self.lo] - self.t_gap
        return np.concatenate([1.0 - A, A - self.min_face, g_order])

    def ineq_jac(self, z):
        X, _ = self.split(z)
        JA = self.area_jac(X)
        return np.vstack([-JA, JA, self.Jo])

    def violations(self, z) -> dict:
        """Violations of the TRUE constraints (not the optimizer's safety margins):
        unit lengths, pins on hosts, face area <= 1, faces positively oriented, and
        pins strictly inside their host and in their recorded order."""
        X, T = self.split(z)
        d = X[self.fb] - X[self.fa]
        e = self.eq(z)
        A = self.areas(X)
        Te = np.concatenate([T, [1.0, 0.0]])
        gap = Te[self.hi] - Te[self.lo]
        return {"length": float(np.abs(np.sqrt((d * d).sum(1)) - 1.0).max(initial=0)),
                "pin": float(np.abs(e[len(self.fa):]).max(initial=0)),
                "area": float((A - 1.0).max(initial=0)),
                "face_pos": float(np.maximum(0.0, 1e-12 - A).max(initial=0)),
                "order": float(np.maximum(0.0, 1e-12 - gap).max(initial=0))}

    def max_violation(self, z):
        return max(self.violations(z).values())


def project(prob: NumpyProblem, z, iters=10, active_tol=1e-7):
    """Gauss-Newton (minimum-norm) projection onto equalities + active face-area constraints."""
    for _ in range(iters):
        X, _ = prob.split(z)
        A = prob.areas(X)
        act = np.where(A > 1.0 - active_tol)[0]
        r = np.concatenate([prob.eq(z), A[act] - 1.0])
        if np.abs(r).max(initial=0) < 1e-15:
            break
        J = np.vstack([prob.eq_jac(z), prob.area_jac(X)[act]])
        dz, *_ = np.linalg.lstsq(J, -r, rcond=None)
        z = z + dz
    return z


def polish(topo_json: dict, z0: np.ndarray, iters: int = 1500, chunk: int = 100, delta: float = 0.12) -> dict:
    """Chunked, trust-region SLSQP polish of one candidate (runs in a CPU pool worker).

    SLSQP has no notion of fence crossings, so each chunk is confined to a box of
    half-width `delta` around its starting point and the embedding is re-checked
    afterwards. A chunk that breaks the embedding is discarded and retried with a
    smaller box; the last good iterate is always kept.
    """
    from scipy.optimize import minimize

    topo = G.Topology.from_json(topo_json)
    prob = NumpyProblem(topo)
    z = np.asarray(z0, dtype=float)
    t0 = time.time()
    cons = [{"type": "eq", "fun": prob.eq, "jac": prob.eq_jac},
            {"type": "ineq", "fun": prob.ineq, "jac": prob.ineq_jac}]
    status, used, reverts = -1, 0, 0
    while used < iters and reverts < 4:
        bounds = list(zip(z - delta, z + delta))
        try:
            res = minimize(prob.objective, z, jac=True, method="SLSQP", constraints=cons, bounds=bounds,
                           options={"maxiter": chunk, "ftol": 1e-15})
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": repr(exc)}
        used += max(1, int(res.nit))
        zn = res.x
        X, _ = prob.split(zn)
        if not np.all(np.isfinite(zn)) or G.check_embedding(X, topo.fences, topo.pins, topo.chains, eps=1e-9):
            reverts += 1
            delta *= 0.4
            continue
        moved = float(np.abs(zn - z).max())
        z = zn
        status = int(res.status)
        if status == 9 or moved > 0.98 * delta:
            continue               # iteration limit or pressing on the box: re-centre and go on
        break                      # converged inside the box (or SLSQP gave up for good)
    z = project(prob, z)
    X, _ = prob.split(z)
    viol = prob.violations(z)
    mv = max(viol.values())
    area = float(prob.areas(X).sum())
    problems = G.check_embedding(X, topo.fences, topo.pins, topo.chains, eps=1e-9)
    return {"ok": mv < 1e-10 and not problems, "z": z, "area": area, "maxviol": mv, "violations": viol,
            "slsqp_status": status, "slsqp_iters": used, "reverts": reverts,
            "embedding_problems": problems[:3], "polish_s": round(time.time() - t0, 3)}


def distinct_candidates(polished: list[dict], exclude_areas=(), eps: float = 1e-6) -> tuple[list[dict], int]:
    """Successful polish results, best first, one per distinct area. A candidate whose area
    matches `exclude_areas` (known minima of this topology) within `eps` is left out: on a
    fixed topology that is the same minimum. Returns (candidates, number excluded)."""
    good = sorted([p for p in polished if p.get("ok")], key=lambda p: -p["area"])
    uniq, excluded = [], 0
    for p in good:
        if any(abs(p["area"] - a) <= eps for a in exclude_areas):
            excluded += 1
        elif all(abs(p["area"] - q["area"]) > 1e-9 for q in uniq):
            uniq.append(p)
    return uniq, excluded


def verify_candidate(topo_json: dict, z: np.ndarray, n: int, tol: float) -> dict:
    """Referee check of a polished candidate (runs in a CPU pool worker)."""
    from fence.referee import verify

    topo = G.Topology.from_json(topo_json)
    X = np.asarray(z[: 2 * topo.n_vertices]).reshape(-1, 2)
    cfg = G.config_to_json(topo, X, n=n)
    t0 = time.time()
    res = verify(cfg, tol)
    cfg["verification"] = res.to_json()
    return {"valid": res.valid, "area": res.area if res.valid else None, "total_area": res.total_area,
            "config": cfg, "summary": res.summary(), "verify_s": round(time.time() - t0, 3)}


def pool_init():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"


# ---------------------------------------------------------------------------
# synchronous convenience wrapper (tests / smoke runs)
# ---------------------------------------------------------------------------

def optimize_topology(topo: G.Topology, coords0, n: int, cfg: OptConfig | None = None, device="cpu",
                      seed=0, pool=None, avoid=None, exclude_areas=()) -> dict:
    """Adam -> polish -> verify for one topology. Returns the best *verified* result.

    avoid / exclude_areas: geometries and areas of known minima of this topology, which
    the result must not be (see adam_stage and distinct_candidates)."""
    cfg = cfg or OptConfig()
    Zs, As, stats = adam_stage(topo, coords0, cfg, device=device, seed=seed, avoid=avoid)
    tj = topo.to_json()
    if pool is not None:
        pol = pool.starmap(polish, [(tj, z, cfg.polish_iters) for z in Zs])
    else:
        pol = [polish(tj, z, cfg.polish_iters) for z in Zs]
    good = [p for p in pol if p.get("ok")]
    uniq, n_excluded = distinct_candidates(pol, exclude_areas)
    tops = uniq[: cfg.verify_top]
    if pool is not None:
        ver = pool.starmap(verify_candidate, [(tj, p["z"], n, cfg.tol) for p in tops])
    else:
        ver = [verify_candidate(tj, p["z"], n, cfg.tol) for p in tops]
    valid = [v for v in ver if v["valid"]]
    best = max(valid, key=lambda v: v["area"]) if valid else None
    return {"stats": stats, "n_polished_ok": len(good), "best": best, "adam_areas": [float(a) for a in As[:5]],
            "polished_areas": [p["area"] for p in uniq[:5]], "excluded_known": n_excluded,
            "polish_fail": [{k: v for k, v in p.items() if k not in ("z",)} for p in pol if not p.get("ok")][:3],
            "rejected": [v["summary"] for v in ver if not v["valid"]]}


def config_as_dict(cfg: OptConfig) -> dict:
    return asdict(cfg)
