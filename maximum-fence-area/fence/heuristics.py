"""Ranking heuristics for candidate nets -- soft scores, never rejections.

Each heuristic is registered separately with a default weight and can be switched
off on its own (`HeuristicSet.from_flags`), so a run can measure what each one is
worth. A heuristic returns a score in [0, 1] and a short detail string; the rank
of a net is the weighted mean of its enabled scores.

Two of the heuristics come with targets taken from a square-grid model, and the
problem statement says to trust the records where they disagree. They do
disagree, so the targets here are the measured values. Measured on the 39 best configurations in `minima/`
(each referee-verified; 1 ties its record, the rest sit 0-3% below), see
`fence.heuristics.calibrate`:

  boundary size b     model 2*sqrt(2n) = 2.83*sqrt(n); measured 3.31*sqrt(n)
                      (relative spread +-17%, and strongly split: 2.59*sqrt(n)
                      for the T-junction-free grid layouts, 3.65*sqrt(n) for the
                      T-rich ones). The model under-counts the boundary of every
                      good T-rich layout.
  face count F        model n/2; measured 0.402*n, and exactly ceil(A_record) on
                      all 39 of 39. The face-count target is therefore the
                      record-based bound of hard rule 14 itself, not n/2, which
                      overshoots by ~25% and would rank the right nets last.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fence import geometry as G
from fence.builder import a_est
from fence.rules import NetCandidate, isoperimetric_cap

# Boundary-size models: b ~ c * sqrt(n).
#
# The measured population is bimodal, not scattered around one coefficient: a layout with
# no T-junctions is a grid of cells and keeps a short boundary (2.59 sqrt(n)), while a
# T-rich layout spends fences on boundary detail (3.65 sqrt(n)). Scoring both against the
# pooled mean 3.31 is worse than useless -- it ranked the n=12 record, the 2x2 grid with
# b=8, at 0.21 and seventh, below four layouts that reach less area. So the default target
# is conditioned on the candidate's own T-junction count, which is known before any
# optimization: T = 0 scores against the grid regime, T >= T_RICH against the T-rich one,
# and in between the coefficient interpolates. Over the 39 verified configurations this
# raises the mean boundary score from 0.663 to 0.946 and the worst from 0.21 to 0.56.
BOUNDARY_C_GRID = 2.59          # measured over the T-junction-free configurations
BOUNDARY_C_RICH = 3.65          # measured over the T-rich ones (T >= 5)
T_RICH = 8                      # T at which the coefficient reaches the T-rich value
BOUNDARY_SPREAD = 0.17          # relative sigma of b / target across the 39 configs


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

@dataclass
class Heuristic:
    id: str
    number: int
    summary: str
    fn: object
    weight: float = 1.0
    needs_coords: bool = False


HEURISTICS: dict[str, Heuristic] = {}


def heuristic(id: str, number: int, summary: str, weight: float = 1.0, needs_coords: bool = False):
    def deco(fn):
        HEURISTICS[id] = Heuristic(id, number, summary, fn, weight, needs_coords)
        return fn
    return deco


def _bell(x: float, target: float, spread: float) -> float:
    """1 at the target, falling off with relative distance."""
    if target <= 0:
        return 0.5
    return math.exp(-0.5 * ((x / target - 1.0) / max(spread, 1e-6)) ** 2)


def unit_scaled(cand: NetCandidate) -> np.ndarray | None:
    """Coordinates rescaled so the mean fence length is 1.

    Sketches are built at roughly unit length but not exactly, and every area
    comparison below is against the unit-fence caps, so the scale has to be fixed
    before the areas mean anything.
    """
    if cand.coords is None:
        return None
    X = np.asarray(cand.coords, dtype=float)
    f = cand.topo.fences
    L = np.linalg.norm(X[f[:, 1]] - X[f[:, 0]], axis=1)
    m = float(np.mean(L))
    return X / m if m > 1e-12 else X


# -- 1 -----------------------------------------------------------------------
@heuristic("boundary", 1, "outer boundary size b near its target for this layout's T-count",
           weight=1.0)
def _boundary(c: NetCandidate, ctx: dict) -> tuple[float, str]:
    target = boundary_target(c.n, c.stats.T, ctx)
    s = _bell(c.stats.b, target, ctx.get("boundary_spread", BOUNDARY_SPREAD))
    return s, f"b={c.stats.b} target={target:.1f} (T={c.stats.T})"


def boundary_target(n: int, T: int, ctx: dict) -> float:
    """Expected outer-boundary size, interpolated on the layout's own T-junction count."""
    w = min(1.0, T / float(T_RICH))
    return (BOUNDARY_C_GRID + (BOUNDARY_C_RICH - BOUNDARY_C_GRID) * w) * math.sqrt(n)


# -- 2 -----------------------------------------------------------------------
@heuristic("face_count", 2, "face count F near its target (record-based, above rule 14's bound)",
           weight=1.2)
def _face_count(c: NetCandidate, ctx: dict) -> tuple[float, str]:
    target = face_target(c.n, ctx)
    F = c.stats.F
    if F < target:
        # below the target is also below the record bound: rule 14 usually removed
        # these already, so this only matters when rule 14 is switched off
        return max(0.0, 1.0 - 0.5 * (target - F)), f"F={F} target={target} (short)"
    return _bell(F, target, ctx.get("face_spread", 0.12)), f"F={F} target={target}"


def face_target(n: int, ctx: dict) -> int:
    """Soft target for the face count: ceil(A_record), rule 14's bound (n/2 overshoots by ~25%)."""
    rec = ctx.get("record_area")
    return max(1, math.ceil((rec if rec is not None else a_est(n)) - 1e-12))


# -- 3 -----------------------------------------------------------------------
FAMILY_PRIOR = {
    "polyomino": 1.0,          # near-square minimum-perimeter cluster of unit squares
    "polyomino+appendage": 1.0,  # ... plus a small cell to use up leftover fences
    "grid": 0.9,
    "library": 0.85,           # topologies already found by the search
    "polyomino+t": 0.8,        # a cluster with boundary T-junction edits
    "ring": 0.5,
    "pinwheel": 0.45,
    "?": 0.6,
}


@heuristic("seed_family", 3, "prefer near-square unit-square clusters plus a small appendage",
           weight=0.8)
def _seed_family(c: NetCandidate, ctx: dict) -> tuple[float, str]:
    p = FAMILY_PRIOR.get(c.family, FAMILY_PRIOR["?"])
    return p, f"family={c.family}"


# -- 4 -----------------------------------------------------------------------
@heuristic("t_at_boundary", 4, "T-junctions belong on the boundary and in appendages, not the bulk",
           weight=0.7)
def _t_at_boundary(c: NetCandidate, ctx: dict) -> tuple[float, str]:
    T = c.stats.T
    if T == 0:
        return 0.5, "T=0"
    outer = set(c.topo.outer)
    on = sum(1 for v in c.topo.pins if v in outer)
    return on / T, f"{on}/{T} pins on the boundary"


# 5 (one_cluster) was removed: rules 6 and 13 already require one component, so it scored
# every surviving net 1.0.


# -- 6 -----------------------------------------------------------------------
@heuristic("convex_outer", 6, "nearly convex outer boundary: penalize reflex outer corners",
           weight=0.8, needs_coords=True)
def _convex_outer(c: NetCandidate, ctx: dict) -> tuple[float, str]:
    X = unit_scaled(c)
    cyc = c.topo.outer
    if X is None or len(cyc) < 3:
        return 0.5, "no outer cycle"
    P = X[cyc]
    d1 = P - np.roll(P, 1, axis=0)
    d2 = np.roll(P, -1, axis=0) - P
    turn = G.cross2(d1, d2)
    # the outer cycle runs clockwise, so a convex corner turns clockwise (cross < 0)
    reflex = int(np.sum(turn > 1e-9))
    return 1.0 - reflex / len(cyc), f"{reflex}/{len(cyc)} reflex outer corners"


# -- 7 -----------------------------------------------------------------------
@heuristic("face_quality", 7, "faces near the best area their perimeter allows", weight=1.0,
           needs_coords=True)
def _face_quality(c: NetCandidate, ctx: dict) -> tuple[float, str]:
    X = unit_scaled(c)
    if X is None or not c.topo.faces:
        return 0.0, "no faces"
    fill = []
    for cyc in c.topo.faces:
        a = max(0.0, G.polygon_signed_area(X[cyc]))
        cap = isoperimetric_cap(len(cyc))
        if cap <= 0:
            fill.append(0.0)
        else:
            # under the cap this is how full the face is; above it the face cannot be
            # realized at unit edge lengths at all, and the overshoot is what has to be
            # squeezed out, so the score falls away again rather than saturating at 1
            fill.append(a / cap if a <= cap else cap / a)
    fill = np.array(fill)
    return float(fill.mean()), f"mean fill {fill.mean():.2f}, worst {fill.min():.2f}"


# -- 8 -----------------------------------------------------------------------
@heuristic("symmetry", 8, "prefer symmetric parameterizations: fewer free parameters",
           weight=0.5, needs_coords=True)
def _symmetry(c: NetCandidate, ctx: dict) -> tuple[float, str]:
    X = unit_scaled(c)
    if X is None:
        return 0.5, "no geometry"
    segs = G.segments_of(X, c.topo.fences)
    m = G.rotational_symmetry_order(segs, tol=ctx.get("sym_tol", 0.05))
    refl = _has_reflection(segs, tol=ctx.get("sym_tol", 0.05))
    s = {1: 0.35, 2: 0.8}.get(m, 1.0)
    if refl:
        s = min(1.0, s + 0.25)
    return s, f"C{m}" + (" + mirror" if refl else "")


def _has_reflection(segs: np.ndarray, tol: float = 0.05, n_axes: int = 12) -> bool:
    """True if some axis through the centroid maps the segment set to itself."""
    segs = np.asarray(segs, dtype=float).reshape(-1, 4)
    P = segs.reshape(-1, 2, 2)
    ctr = P.reshape(-1, 2).mean(axis=0)
    Q = P - ctr

    def canon(S):
        a, b = S[:, 0], S[:, 1]
        swap = (a[:, 0] > b[:, 0] + 1e-9) | ((np.abs(a[:, 0] - b[:, 0]) <= 1e-9) & (a[:, 1] > b[:, 1]))
        return np.concatenate([np.where(swap[:, None], b, a), np.where(swap[:, None], a, b)], axis=1)

    base = canon(Q)
    for k in range(n_axes):
        th = math.pi * k / n_axes
        c2, s2 = math.cos(2 * th), math.sin(2 * th)
        M = np.array([[c2, s2], [s2, -c2]])
        R = canon(Q @ M.T)
        d = np.linalg.norm(R[:, None, :] - base[None, :, :], axis=2)
        if np.all(d.min(axis=1) < tol):
            return True
    return False


# -- 9 -----------------------------------------------------------------------
@heuristic("lattice_snap", 9, "grid part on integer/half-integer coordinates", weight=0.5,
           needs_coords=True)
def _lattice_snap(c: NetCandidate, ctx: dict) -> tuple[float, str]:
    X = unit_scaled(c)
    if X is None or len(X) == 0:
        return 0.0, "no geometry"
    X = X - X.mean(axis=0)
    f = c.topo.fences
    D = X[f[:, 1]] - X[f[:, 0]]
    angles = np.arctan2(D[:, 1], D[:, 0]) % (math.pi / 2)
    best = 0.0
    tol = ctx.get("snap_tol", 0.06)
    for th in list(angles) + [0.0]:
        R = np.array([[math.cos(-th), -math.sin(-th)], [math.sin(-th), math.cos(-th)]])
        Y = X @ R.T
        Y = Y - Y.min(axis=0)
        r = np.abs(Y * 2 - np.round(Y * 2)) / 2
        frac = float(np.mean(np.all(r < tol, axis=1)))
        best = max(best, frac)
    return best, f"{best:.0%} of vertices on the half-integer lattice"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@dataclass
class Score:
    total: float
    parts: dict[str, float]
    details: dict[str, str]

    def explain(self, top: int = 99) -> str:
        items = sorted(self.parts.items(), key=lambda kv: -kv[1])[:top]
        return f"{self.total:.3f} [" + ", ".join(f"{k}={v:.2f}" for k, v in items) + "]"


class HeuristicSet:
    """The enabled heuristics and their weights."""

    def __init__(self, weights: dict[str, float] | None = None, ctx: dict | None = None):
        self.weights = {k: h.weight for k, h in HEURISTICS.items()}
        for k, v in (weights or {}).items():
            if k not in HEURISTICS:
                raise ValueError(f"unknown heuristic {k!r}")
            self.weights[k] = float(v)
        self.ctx = dict(ctx or {})

    @staticmethod
    def from_flags(off: list[str] | None = None, weights: list[str] | None = None,
                   ctx: dict | None = None) -> "HeuristicSet":
        """`off`: ids to zero out. `weights`: "id=value" strings."""
        w = {}
        for spec in weights or []:
            k, _, v = spec.partition("=")
            w[k.strip()] = float(v)
        for k in off or []:
            if k not in HEURISTICS:
                raise ValueError(f"unknown heuristic {k!r}")
            w[k] = 0.0
        return HeuristicSet(w, ctx)

    def enabled(self) -> list[str]:
        return [k for k in sorted(HEURISTICS, key=lambda k: HEURISTICS[k].number)
                if self.weights.get(k, 0.0) > 0]

    def disabled(self) -> list[str]:
        return [k for k in sorted(HEURISTICS, key=lambda k: HEURISTICS[k].number)
                if self.weights.get(k, 0.0) <= 0]

    def score(self, cand: NetCandidate, ctx: dict | None = None) -> Score:
        c = {**self.ctx, **(ctx or {})}
        parts, details, wsum, acc = {}, {}, 0.0, 0.0
        for k in sorted(HEURISTICS, key=lambda k: HEURISTICS[k].number):
            h = HEURISTICS[k]
            w = self.weights.get(k, 0.0)
            if w <= 0:
                continue
            if h.needs_coords and cand.coords is None:
                continue
            s, detail = h.fn(cand, c)
            s = float(min(1.0, max(0.0, s)))
            parts[k] = s
            details[k] = detail
            acc += w * s
            wsum += w
        return Score(acc / wsum if wsum else 0.0, parts, details)


def describe_heuristics() -> str:
    lines = []
    for h in sorted(HEURISTICS.values(), key=lambda h: h.number):
        lines.append(f"{h.number:>2}. {h.id:<14} w={h.weight:<4} {h.summary}"
                     + ("  [needs geometry]" if h.needs_coords else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# calibration against the records
# ---------------------------------------------------------------------------

def calibrate(configs: list[tuple[int, G.Topology, np.ndarray, float]]) -> dict:
    """Fit the two model-based targets to a set of (n, topo, coords, area) configurations.

    Returns the measured coefficient of b ~ c*sqrt(n), the measured F/n slope, and
    how often F equals ceil(area) -- i.e. whether the record-based face bound of
    hard rule 14 is tight.
    """
    if not configs:
        return {}
    n = np.array([c[0] for c in configs], dtype=float)
    b = np.array([len(c[1].outer) for c in configs], dtype=float)
    F = np.array([len(c[1].faces) for c in configs], dtype=float)
    T = np.array([len(c[1].pins) for c in configs], dtype=float)
    A = np.array([c[3] for c in configs], dtype=float)
    cb = float((b * np.sqrt(n)).sum() / n.sum())
    rel = b / (cb * np.sqrt(n))
    rich = T >= 5
    out = {
        "n_configs": len(configs),
        "boundary_c_measured": round(cb, 3),
        "boundary_c_model": round(2 * math.sqrt(2), 3),
        "boundary_spread": round(float(np.std(rel)), 3),
        "boundary_c_t_rich": round(float((b[rich] * np.sqrt(n[rich])).sum() / n[rich].sum()), 3)
        if rich.any() else None,
        "boundary_c_t_free": round(float((b[~rich] * np.sqrt(n[~rich])).sum() / n[~rich].sum()), 3)
        if (~rich).any() else None,
        "face_slope_measured": round(float((F * n).sum() / (n * n).sum()), 4),
        "face_slope_model": 0.5,
        "face_equals_ceil_area": int(np.sum(np.abs(F - np.ceil(A - 1e-12)) < 0.5)),
    }
    return out
