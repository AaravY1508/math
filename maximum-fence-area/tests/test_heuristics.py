"""Heuristics: each scores separately, each can be switched off, and the two model-based
targets are the measured ones rather than the square-grid ones they came from."""
import glob
import json
import math
from pathlib import Path

import numpy as np
import pytest

from fence import geometry as G
from fence import heuristics as H
from fence import rules as R
from tests.test_rules import cand_from_segments, records
from tests import configs as C

HERE = Path(__file__).resolve().parent.parent
HAS_LIB = bool(list((HERE / "minima" / "records").glob("N*.json")))


def test_every_numbered_heuristic_is_registered():
    assert sorted(h.number for h in H.HEURISTICS.values()) == list(range(1, 10))


def test_heuristics_can_be_switched_off_one_at_a_time():
    for hid in H.HEURISTICS:
        hs = H.HeuristicSet.from_flags([hid])
        assert hid in hs.disabled() and hid not in hs.enabled()
    with pytest.raises(ValueError):
        H.HeuristicSet.from_flags(["not_a_heuristic"])
    with pytest.raises(ValueError):
        H.HeuristicSet.from_flags(weights=["nope=1"])


def test_weights_change_the_ranking_not_the_parts():
    cand = cand_from_segments(C.GRID_2x2, 12)
    a = H.HeuristicSet().score(cand, {"record_area": 4.0})
    b = H.HeuristicSet.from_flags(weights=["boundary=0"]).score(cand, {"record_area": 4.0})
    assert a.parts["face_count"] == b.parts["face_count"]
    assert "boundary" in a.parts and "boundary" not in b.parts
    assert a.total != b.total


def test_score_is_bounded_and_explains_itself():
    cand = cand_from_segments(C.GRID_2x2, 12)
    s = H.HeuristicSet().score(cand, {"record_area": 4.0})
    assert 0.0 <= s.total <= 1.0
    assert all(0.0 <= v <= 1.0 for v in s.parts.values())
    assert "face_count" in s.explain()


def test_a_disabled_heuristic_costs_no_work():
    """With every heuristic off the score is 0 and no part is computed."""
    cand = cand_from_segments(C.SQUARE, 4)
    hs = H.HeuristicSet.from_flags(list(H.HEURISTICS))
    s = hs.score(cand, {})
    assert s.parts == {} and s.total == 0.0


# -- individual heuristics ---------------------------------------------------

def test_face_count_target_follows_the_record_not_half_n():
    assert H.face_target(22, {"record_area": 8.07384}) == 9      # ceil(record)
    assert H.face_target(22, {"face_model": "half-n"}) == 11     # the square-grid model
    assert H.face_target(22, {}) > 0                             # falls back to a_est


def test_face_count_prefers_the_record_bound():
    cand = cand_from_segments(C.GRID_2x2, 12)                    # F = 4
    hs = H.HeuristicSet()
    at = hs.score(cand, {"record_area": 4.0}).parts["face_count"]
    over = hs.score(cand, {"record_area": 4.0, "face_model": "half-n"}).parts["face_count"]
    assert at == pytest.approx(1.0)
    assert over < at            # n/2 = 6 wants two more faces than the record needs


def test_boundary_models_are_all_available_and_differ():
    cand = cand_from_segments(C.GRID_2x2, 12)
    got = {k: H.HeuristicSet().score(cand, {"boundary_c": v}).parts["boundary"]
           for k, v in H.BOUNDARY_C.items()}
    assert len(set(got.values())) == len(H.BOUNDARY_C)
    assert H.BOUNDARY_C["square-grid"] == pytest.approx(2 * math.sqrt(2))


def test_boundary_target_follows_the_layouts_own_t_count():
    """The measured population is bimodal; one pooled coefficient fits neither mode."""
    grid = H.boundary_target(12, 0, {})
    rich = H.boundary_target(12, 12, {})
    assert grid < rich
    assert grid == pytest.approx(H.BOUNDARY_C_GRID * math.sqrt(12))
    assert rich == pytest.approx(H.BOUNDARY_C_RICH * math.sqrt(12))
    assert H.boundary_target(12, 0, {"boundary_c": 3.31}) == pytest.approx(3.31 * math.sqrt(12))


def test_the_grid_record_is_not_demoted_by_the_boundary_heuristic():
    """Regression: pooled at 3.31 sqrt(n) this scored 0.21 and ranked the n=12 record 7th."""
    cand = cand_from_segments(C.GRID_2x2, 12)         # the record at n=12, b=8, T=0
    assert H.HeuristicSet().score(cand, {"record_area": 4.0}).parts["boundary"] > 0.75


@pytest.mark.skipif(not HAS_LIB, reason="no minima library in this checkout")
def test_the_boundary_heuristic_scores_the_records_well_at_both_extremes():
    hs = H.HeuristicSet()
    scores = []
    for n, topo, X, area in records():
        cand = R.NetCandidate(n=n, topo=topo, coords=X, family="library", exact=True)
        scores.append(hs.score(cand, {"record_area": area}).parts["boundary"])
    assert float(np.mean(scores)) > 0.85        # 0.66 under a single pooled coefficient
    assert min(scores) > 0.4                    # 0.21 under a single pooled coefficient


def test_one_cluster_and_convexity_on_a_square():
    cand = cand_from_segments(C.SQUARE, 4)
    s = H.HeuristicSet().score(cand, {})
    assert s.parts["one_cluster"] == 1.0
    assert s.parts["convex_outer"] == pytest.approx(1.0)     # no reflex corner on a square
    assert s.parts["face_quality"] == pytest.approx(1.0)     # a unit square fills its cap
    assert s.parts["symmetry"] == pytest.approx(1.0)         # C4 and mirrors
    assert s.parts["lattice_snap"] == pytest.approx(1.0)


def test_convexity_penalizes_a_reflex_corner():
    cand = cand_from_segments(C.DOMINO, 7)
    straight = H.HeuristicSet().score(cand, {}).parts["convex_outer"]
    # an L-shaped tromino has exactly one reflex outer corner
    L = C.DOMINO + [C.seg(0, 1, 0, 2), C.seg(0, 2, 1, 2), C.seg(1, 1, 1, 2)]
    lc = cand_from_segments(L, 10)
    assert H.HeuristicSet().score(lc, {}).parts["convex_outer"] < straight


def test_t_at_boundary_rewards_boundary_pins():
    cand = cand_from_segments(C.SPLIT_SQUARE, 5)     # both pins sit on the outer boundary
    assert H.HeuristicSet().score(cand, {}).parts["t_at_boundary"] == pytest.approx(1.0)
    assert H.HeuristicSet().score(cand_from_segments(C.SQUARE, 4), {}).parts["t_at_boundary"] == 0.5


def test_unit_scaled_normalizes_the_sketch_scale():
    cand = cand_from_segments(C.SQUARE, 4)
    cand.coords = cand.coords * 3.0
    X = H.unit_scaled(cand)
    L = np.linalg.norm(X[cand.topo.fences[:, 1]] - X[cand.topo.fences[:, 0]], axis=1)
    assert np.allclose(L, 1.0)


def test_seed_family_prior_prefers_clusters_over_rings():
    assert H.FAMILY_PRIOR["polyomino"] > H.FAMILY_PRIOR["ring"]
    assert H.FAMILY_PRIOR["polyomino+appendage"] > H.FAMILY_PRIOR["pinwheel"]


# -- the calibration claim ---------------------------------------------------

@pytest.mark.skipif(not HAS_LIB, reason="no minima library in this checkout")
def test_calibration_says_the_square_grid_targets_are_off():
    """Heuristics 1 and 2 are stated as unverified; the records disagree with both.

    This test pins the mismatch that the defaults were changed for, so that if the
    library ever moves far enough to agree with the square-grid model again, it fails
    and the defaults get revisited.
    """
    cal = H.calibrate(records())
    assert cal["n_configs"] >= 20
    assert cal["boundary_c_measured"] > cal["boundary_c_model"] * 1.10
    assert cal["face_slope_measured"] < 0.45                 # the model says 0.5
    assert cal["face_equals_ceil_area"] >= 0.9 * cal["n_configs"]
    assert H.BOUNDARY_C["measured"] == pytest.approx(cal["boundary_c_measured"], abs=0.15)


@pytest.mark.skipif(not HAS_LIB, reason="no minima library in this checkout")
def test_records_score_well_under_the_default_weights():
    hs = H.HeuristicSet()
    scores = []
    for n, topo, X, area in records():
        cand = R.NetCandidate(n=n, topo=topo, coords=X, family="library", exact=True)
        scores.append(hs.score(cand, {"record_area": area}).total)
    assert min(scores) > 0.5 and float(np.mean(scores)) > 0.7


def test_face_quality_penalizes_an_oversized_face_as_well_as_an_empty_one():
    """A face above the area its perimeter allows is not realizable; it must not score 1."""
    good = cand_from_segments(C.SQUARE, 4)
    assert H.HeuristicSet().score(good, {}).parts["face_quality"] == pytest.approx(1.0)
    big = cand_from_segments(C.HEXAGON, 6)      # one face of area 3 sqrt(3) / 2, cap 1
    s = H.HeuristicSet().score(big, {}).parts["face_quality"]
    assert s < 0.45
