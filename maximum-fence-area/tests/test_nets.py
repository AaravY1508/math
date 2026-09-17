"""Stage 1: the counting envelope, the polyomino families, and the filter/rank loop."""
import math

import numpy as np
import pytest

from fence import geometry as G
from fence import heuristics as H
from fence import nets as N
from fence import rules as R

RNG = lambda s=0: np.random.default_rng(s)      # noqa: E731


# -- the counting envelope ---------------------------------------------------

def test_envelope_entries_all_satisfy_the_counting_rules():
    for n in (6, 12, 17, 23):
        specs = N.feasible_counts(n, None, None)
        assert specs
        for s in specs:
            assert s.V - (s.n + s.T) + s.F == 1           # rule 6
            assert s.T <= 2 * s.n                          # rule 8
            assert s.T <= 2 * s.F - 2                      # rule 9, via the degree sum
            assert 2 * (s.n + s.T) - s.b >= 3 * s.F        # rules 7 + 11
            assert s.b >= 3                                # rule 10
            assert s.E == s.n + s.T


def test_envelope_contains_the_real_configurations():
    """Every hand-built configuration's (F, T, V, b) must be inside the envelope for its n."""
    from tests.test_rules import cand_from_segments
    from tests import configs as C

    for segs, n in ((C.TRIANGLE, 3), (C.SQUARE, 4), (C.SPLIT_SQUARE, 5), (C.DOMINO, 7),
                    (C.GRID_2x2, 12), (C.GRID_2x3, 17)):
        cand = cand_from_segments(segs, n)
        s = cand.stats
        env = {sp.as_tuple() for sp in N.feasible_counts(n, None, None)}
        assert (s.F, s.T, s.V, s.b) in env, f"n={n}"


def test_record_bound_shrinks_the_envelope():
    big = N.feasible_counts(22, None, None)
    small = N.feasible_counts(22, 9, 8.07384)
    assert len(small) < len(big)
    assert all(s.F >= 9 for s in small)
    # the isoperimetric bound on the outer boundary also bites: b^2/(4 pi) >= area
    assert all(s.b * s.b / (4 * math.pi) >= 8.07384 - 1e-9 for s in small)


def test_disabling_a_rule_widens_the_envelope():
    on = N.feasible_counts(20, 8, 7.14732)
    off = N.feasible_counts(20, 8, 7.14732, enabled=set(R.RULES) - {"record_faces"})
    assert len(off) > len(on)


def test_outer_isoperimetric_bound_holds_on_the_records():
    """Total area <= b^2 / (4 pi): the bounded area is what the outer cycle encloses,
    and each of its b graph-edges is at most 1 long."""
    from tests.test_rules import records

    for n, topo, X, area in records():
        b = len(topo.outer)
        assert b * b / (4 * math.pi) >= area - 1e-9, f"n={n}"


# -- polyomino families ------------------------------------------------------

def test_min_perimeter_matches_the_lattice_bound():
    from fence.evolve import lattice_max

    for c in range(1, 26):
        assert N.min_perimeter(c) == 2 * math.ceil(2 * math.sqrt(c))
    for n in (12, 17, 24, 31, 40, 49):
        c = lattice_max(n)
        assert 2 * c + math.ceil(2 * math.sqrt(c)) <= n


@pytest.mark.parametrize("c", list(range(1, 17)))
def test_polyomino_shapes_are_connected_and_the_right_size(c):
    for cells in N.polyomino_shapes(c):
        assert len(cells) == c
        s = set(cells)
        seen = {next(iter(s))}
        stack = [next(iter(s))]
        while stack:
            i, j = stack.pop()
            for q in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if q in s and q not in seen:
                    seen.add(q)
                    stack.append(q)
        assert seen == s, f"c={c} shape is disconnected"


@pytest.mark.parametrize("c", list(range(1, 15)))
def test_polyomino_sketch_is_a_legal_net(c):
    for cells in N.polyomino_shapes(c, limit=3):
        sk = N.polyomino_sketch(cells)
        assert sk.n == N.polyomino_fences(cells)
        topo = sk.topology()
        cand = R.NetCandidate(n=sk.n, topo=topo, coords=sk.coords(), family="polyomino",
                              exact=True)
        assert R.RuleSet.from_flags(["record_faces"]).check_all(cand, {}) == []
        assert len(topo.faces) == c                        # one face per cell


def test_square_cluster_uses_the_expected_fence_count():
    # a p x q rectangle of cells uses 2pq + p + q fences
    for p, q in ((1, 1), (2, 2), (2, 3), (3, 3), (3, 4)):
        cells = [(i, j) for i in range(p) for j in range(q)]
        assert N.polyomino_fences(cells) == 2 * p * q + p + q


def test_canon_cells_is_invariant_under_the_symmetries():
    cells = [(0, 0), (0, 1), (1, 0)]
    rot = [(j, -i) for i, j in cells]
    refl = [(-i, j) for i, j in cells]
    assert N._canon_cells(cells) == N._canon_cells(rot) == N._canon_cells(refl)
    assert N._canon_cells(cells) != N._canon_cells([(0, 0), (0, 1), (0, 2)])


# -- the stage ---------------------------------------------------------------

def test_stage_returns_only_nets_that_pass_every_enabled_rule():
    res = N.enumerate_nets(12, RNG(1), R.RuleSet(), H.HeuristicSet(),
                           {"record_area": 4.0}, max_candidates=120, keep=20)
    assert res.ranked
    rs = R.RuleSet()
    for r in res.ranked:
        assert rs.check(r.cand, {"record_area": 4.0, "min_faces": 4}) is None
        assert r.cand.topo.n == 12


def test_stage_ranks_in_descending_score_and_dedups_topologies():
    res = N.enumerate_nets(12, RNG(2), R.RuleSet(), H.HeuristicSet(),
                           {"record_area": 4.0}, max_candidates=200, keep=30)
    scores = [r.score.total for r in res.ranked]
    assert scores == sorted(scores, reverse=True)
    sigs = [r.signature for r in res.ranked if r.signature]
    assert len(sigs) == len(set(sigs))


def test_ledger_names_the_rule_that_rejected_each_candidate():
    res = N.enumerate_nets(22, RNG(3), R.RuleSet(), H.HeuristicSet(),
                           {"record_area": 8.07384}, max_candidates=150, keep=10)
    assert res.ledger.seen > 0
    assert "record_faces" in res.ledger.counts          # 22 is a face-short n
    assert res.ledger.samples["record_faces"]
    assert all(k in R.RULES or k == "build_error" for k in res.ledger.counts)


def test_switching_off_rule_14_keeps_more_nets():
    ctx = {"record_area": 8.07384}
    on = N.enumerate_nets(22, RNG(4), R.RuleSet(), H.HeuristicSet(), ctx,
                          max_candidates=150, keep=99)
    off = N.enumerate_nets(22, RNG(4), R.RuleSet.from_flags(["record_faces"]),
                           H.HeuristicSet(), ctx, max_candidates=150, keep=99)
    assert len(off.ranked) > len(on.ranked)
    assert off.ledger.counts.get("record_faces", 0) == 0


def test_face_target_generator_reaches_the_record_bound():
    """The generator that exists because everything else is face-short at these n."""
    ctx = {"record_area": 8.07384, "min_faces": 9}
    got = [sk for sk, fam, org in N._gen_face_target(22, RNG(5), ctx)]
    assert got, "no face-target net was built"
    assert any(len(sk.topology().faces) >= 9 for sk in got)


def test_unknown_generator_is_an_error():
    with pytest.raises(ValueError):
        N.enumerate_nets(12, RNG(), generators=["nope"])


def test_coverage_reports_nets_inside_the_envelope():
    res = N.enumerate_nets(12, RNG(6), R.RuleSet(), H.HeuristicSet(),
                           {"record_area": 4.0}, max_candidates=120, keep=20)
    cov = N.coverage(res)
    assert cov["nets"] > 0
    assert cov["outside_envelope"] == []          # every net built must be in the envelope
    assert cov["inside_envelope"] == cov["nets"]


def test_duplicates_are_counted_separately_from_rule_rejections():
    """A net an earlier generator already built is dropped, but it did not break a rule."""
    res = N.enumerate_nets(12, RNG(8), R.RuleSet(), H.HeuristicSet(),
                           {"record_area": 4.0}, max_candidates=300, keep=99)
    assert res.distinct == len(res.ranked)
    assert res.ledger.accepted == res.distinct + res.duplicates
    assert res.duplicates >= 0


def test_coverage_uses_every_distinct_net_not_just_the_kept_ones():
    full = N.enumerate_nets(16, RNG(9), R.RuleSet(), H.HeuristicSet(),
                            {"record_area": 5.53706}, max_candidates=300, keep=99)
    cut = N.enumerate_nets(16, RNG(9), R.RuleSet(), H.HeuristicSet(),
                           {"record_area": 5.53706}, max_candidates=300, keep=2)
    assert len(cut.ranked) == 2
    assert N.coverage(cut)["nets"] == N.coverage(full)["nets"]
    assert N.coverage(cut)["outside_envelope"] == []
