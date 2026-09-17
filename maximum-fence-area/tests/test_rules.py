"""The hard rules: each fires on the thing it is meant to catch, and none fires on a record."""
import glob
import json
import math
from pathlib import Path

import numpy as np
import pytest

from fence import geometry as G
from fence import rules as R
from tests import configs as C

HERE = Path(__file__).resolve().parent.parent


def cand_from_segments(segs, n=None, exact=True, family="test"):
    arr = np.array([[float(x) for x in s] for s in segs])
    coords, fences, pins = G.topology_from_segments(arr, tol=1e-7)
    topo = G.make_topology(coords, fences, pins, check=False)
    return R.NetCandidate(n=n or len(segs), topo=topo, coords=coords, family=family, exact=exact)


def records():
    out = []
    for p in sorted(glob.glob(str(HERE / "minima" / "records" / "N*.json")),
                    key=lambda s: int(Path(s).stem[1:])):
        d = json.loads(Path(p).read_text())
        topo, X = G.config_from_json(d["config"])
        out.append((d["n"], topo, X, d["area"]))
    return out


# -- the registry ------------------------------------------------------------

def test_every_numbered_rule_is_registered():
    assert sorted(r.number for r in R.RULES.values()) == list(range(1, 15))
    assert len({r.id for r in R.RULES.values()}) == 14


def test_rules_can_be_switched_off_one_at_a_time():
    for rid in R.RULES:
        rs = R.RuleSet.from_flags([rid])
        assert rs.disabled() == [rid]
        assert rid not in rs.enabled
    assert R.RuleSet.from_flags(only=["euler"]).enabled == {"euler"}
    with pytest.raises(ValueError):
        R.RuleSet.from_flags(["not_a_rule"])


# -- rules pass on things that are known good --------------------------------

@pytest.mark.parametrize("segs,n", [(C.SQUARE, 4), (C.SPLIT_SQUARE, 5), (C.DOMINO, 7),
                                    (C.GRID_2x2, 12), (C.GRID_2x3, 17)])
def test_valid_configurations_pass_every_rule(segs, n):
    cand = cand_from_segments(segs, n)
    assert R.RuleSet().check_all(cand, {}) == []


def test_the_unit_triangle_passes_every_rule():
    # 3 fences, one face of area sqrt(3)/4: the smallest legal configuration there is
    cand = cand_from_segments(C.TRIANGLE, 3)
    assert R.RuleSet().check_all(cand, {}) == []


@pytest.mark.skipif(not list((HERE / "minima" / "records").glob("N*.json")),
                    reason="no minima library in this checkout")
def test_no_structural_rule_rejects_a_verified_record():
    """Rule 14 may reject (a config below its record is exactly what it is for); nothing else may."""
    rs = R.RuleSet.from_flags(["record_faces"])
    for n, topo, X, area in records():
        cand = R.NetCandidate(n=n, topo=topo, coords=X, family="library", exact=True)
        assert rs.check_all(cand, {}) == [], f"n={n}"


# -- each rule fires on its own violation ------------------------------------

def test_rule_1_wrong_fence_count_and_wrong_length():
    cand = cand_from_segments(C.SQUARE, 4)
    cand.n = 5
    assert R.RuleSet().check(cand, {}).rule == "unit_fences"
    bad = cand_from_segments(C.BAD_LENGTH, 4)
    assert R.RuleSet(enabled={"unit_fences"}).check(bad, {}).rule == "unit_fences"


def test_rule_1_length_is_not_checked_on_an_inexact_sketch():
    bad = cand_from_segments(C.BAD_LENGTH, 4, exact=False)
    assert R.RuleSet(enabled={"unit_fences"}).check(bad, {}) is None


def test_rule_2_free_end():
    arr = np.array([[float(x) for x in s] for s in C.DANGLING])
    coords, fences, pins = G.topology_from_segments(arr, tol=1e-7)
    chains = G.build_chains(coords, fences, pins)
    topo = G.Topology(len(coords), fences, pins, chains, faces=[], outer=[])
    cand = R.NetCandidate(n=5, topo=topo, coords=coords, exact=True)
    rej = R.RuleSet(enabled={"no_free_ends"}).check(cand, {})
    assert rej is not None and rej.rule == "no_free_ends"


def test_rule_3_crossing_and_collinear_overlap():
    for segs, n in ((C.CROSSING, 6), (C.OVERLAP, 6)):
        arr = np.array([[float(x) for x in s] for s in segs])
        try:
            coords, fences, pins = G.topology_from_segments(arr, tol=1e-7)
        except G.InvalidTopology:
            continue                      # the importer already refuses it, which is stricter
        chains = G.build_chains(coords, fences, pins)
        topo = G.Topology(len(coords), fences, pins, chains, faces=[], outer=[])
        cand = R.NetCandidate(n=n, topo=topo, coords=coords, exact=True)
        assert R.RuleSet(enabled={"no_crossings"}).check(cand, {}) is not None


def test_rule_4_oversized_face():
    cand = cand_from_segments(C.HEXAGON, 6)
    rej = R.RuleSet(enabled={"face_area_max"}).check(cand, {})
    assert rej is not None and "area" in rej.reason
    # and it is skipped on a sketch, whose scale the optimizer has not fixed yet
    cand.exact = False
    assert R.RuleSet(enabled={"face_area_max"}).check(cand, {}) is None


def test_rule_6_euler_holds_on_every_record():
    for n, topo, X, _ in records():
        V, T, F = topo.n_vertices, len(topo.pins), len(topo.faces)
        assert V - (n + T) + F == 1, f"n={n}"


def test_rule_7_incidence_identity_holds_on_every_record():
    for n, topo, X, _ in records():
        T, b = len(topo.pins), len(topo.outer)
        assert sum(len(f) for f in topo.faces) == 2 * (n + T) - b, f"n={n}"


def test_rule_6_and_7_fire_when_the_counts_are_corrupted():
    cand = cand_from_segments(C.GRID_2x2, 12)
    cand.topo.faces = cand.topo.faces[:-1]          # drop a face: Euler and incidence both break
    cand._stats = None
    assert R.RuleSet(enabled={"euler"}).check(cand, {}).rule == "euler"
    assert R.RuleSet(enabled={"incidence"}).check(cand, {}).rule == "incidence"


def test_rule_8_t_bound_holds_and_is_computed_from_touches():
    cand = cand_from_segments(C.SPLIT_SQUARE, 5)
    assert cand.stats.T == 2 and cand.stats.t_touches == 2
    assert R.RuleSet(enabled={"t_bound"}).check(cand, {}) is None


def test_rule_9_and_10_reject_a_pendant_fence():
    """A square with a fence hanging off it: degree 1 at the tip, and a bridge."""
    segs = C.SQUARE + [C.seg(1, 0, 2, 0)]
    arr = np.array([[float(x) for x in s] for s in segs])
    coords, fences, pins = G.topology_from_segments(arr, tol=1e-7)
    chains = G.build_chains(coords, fences, pins)
    topo = G.Topology(len(coords), fences, pins, chains, faces=[[0, 1, 2, 3]], outer=[])
    cand = R.NetCandidate(n=5, topo=topo, coords=coords, exact=True)
    assert R.RuleSet(enabled={"min_degree"}).check(cand, {}) is not None
    assert R.RuleSet(enabled={"bridgeless"}).check(cand, {}) is not None


def test_rule_10_finds_no_bridge_in_a_grid():
    cand = cand_from_segments(C.GRID_2x3, 17)
    assert cand.stats.bridges == []
    assert R.RuleSet(enabled={"bridgeless"}).check(cand, {}) is None


def test_rule_11_isoperimetric_cap():
    assert R.isoperimetric_cap(2) == 0.0
    assert R.isoperimetric_cap(3) == pytest.approx(math.sqrt(3) / 4)     # unit equilateral triangle
    assert R.isoperimetric_cap(4) == pytest.approx(1.0)                  # the unit square, capped at 1
    assert R.isoperimetric_cap(12) == 1.0                                # rule 4 caps everything at 1
    for m in range(3, 40):
        assert R.isoperimetric_cap(m) <= m * m / (4 * math.pi) + 1e-12


def test_rule_11_triangle_area_matches_its_cap():
    cand = cand_from_segments(C.TRIANGLE, 3)
    assert cand.stats.face_sizes == [3]
    assert float(cand.stats.areas[0]) == pytest.approx(math.sqrt(3) / 4, abs=1e-9)
    assert R.RuleSet(enabled={"face_size"}).check(cand, {}) is None


def test_rule_12_t_junction_host_stays_straight():
    cand = cand_from_segments(C.SPLIT_SQUARE, 5)
    assert R.RuleSet(enabled={"t_straight"}).check(cand, {}) is None
    v = next(iter(cand.topo.pins))
    cand.coords = cand.coords.copy()
    cand.coords[v] = cand.coords[v] + np.array([0.0, 0.02])       # lift the pin off its host
    cand.exact = True
    assert R.RuleSet(enabled={"t_straight"}).check(cand, {}) is not None


def test_rule_13_embedding_covers_every_half_edge():
    cand = cand_from_segments(C.GRID_2x2, 12)
    assert R.RuleSet(enabled={"embedding"}).check(cand, {}) is None
    cand.topo.outer = []
    cand._stats = None
    assert R.RuleSet(enabled={"embedding"}).check(cand, {}) is not None


def test_rule_14_face_bound():
    assert R.min_faces_for(None) is None
    assert R.min_faces_for(4.0) == 4
    assert R.min_faces_for(8.07384) == 9
    cand = cand_from_segments(C.GRID_2x2, 12)         # 4 faces
    assert R.RuleSet(enabled={"record_faces"}).check(cand, {"min_faces": 4}) is None
    rej = R.RuleSet(enabled={"record_faces"}).check(cand, {"min_faces": 5})
    assert rej is not None and rej.number == 14


# -- the ledger --------------------------------------------------------------

def test_ledger_records_which_rule_rejected_what():
    led = R.Ledger()
    led.note(None, "good")
    led.note(R.Rejection("euler", 6, "V - (n+T) + F = 2, not 1"), "bad/one")
    led.note(R.Rejection("euler", 6, "another"), "bad/two")
    assert led.accepted == 1 and led.seen == 3
    assert led.counts["euler"] == 2
    assert "bad/one" in led.samples["euler"][0]
    other = R.Ledger()
    other.note(R.Rejection("bridgeless", 10, "cut edge"), "x")
    led.merge(other)
    assert led.counts["bridgeless"] == 1 and led.seen == 4
    assert "euler" in led.table() and "rule  6" in led.table()


def test_find_bridges_handles_parallel_edges():
    # two vertices joined by two distinct edges: a 2-gon, no bridge
    assert R.find_bridges(2, [(0, 1), (0, 1)]) == []
    assert R.find_bridges(3, [(0, 1), (1, 2)]) != []
    assert R.find_bridges(4, [(0, 1), (1, 2), (2, 0), (2, 3)]) == [(2, 3)]
