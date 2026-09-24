"""Growth paths of three or more fences (fence.builder, EXPERIMENTS.md change 4).

`arc_points(Q, P, m, ell)` returns the m - 1 interior points of the path ordered from Q to P.
Two places used that order the wrong way round, and both are reachable only when there are at
least two interior points, i.e. m >= 3:

  * `sample_growth` closed the proposed cell as `chain + inner[::-1]`, which walks
    P -> (boundary) -> Q -> A_{m-1} -> ... -> A_1: a bowtie whose signed area is about 0, so
    the candidate lost the cost comparison or was dropped outright by the `area <= 0.05` test.
  * `grow_once` laid the path down with `add_path_points(P, Q, inner)`, which builds
    P -> A_1 -> ... -> Q while A_1 sits next to Q: the path crosses itself and `topology()`
    rejects it.

Together they meant no growth step ever added a 3-fence cell -- and a cell of three unit
fences on a unit-long piece of boundary is exactly a unit square, the densest cell there is.
Both are fixed; the switch that restored them was removed.
"""
import numpy as np
import pytest

from fence import builder as Bd
from fence import geometry as G


def total_area(sk: Bd.Sketch) -> float:
    return float(sum(G.face_areas(sk.coords(), sk.topology().faces)))


def grow_m3(sk: Bd.Sketch, rng, **grow):
    """One outer growth step restricted to 3-fence paths; True if it was built."""
    params = {"m_weights": [0.0, 0.0, 1.0], "target_area": 1.0, "greedy": 1.0, **grow}
    return Bd.grow_once(sk, rng, params, m_max=3, face=sk.face_cycles()[-1])


def test_arc_points_of_three_unit_fences_on_a_unit_chord_is_a_unit_square():
    Q, P = np.array([1.0, 0.0]), np.array([0.0, 0.0])
    A1, A2 = Bd.arc_points(Q, P, 3, 1.0)
    # the path Q -> A1 -> A2 -> P closes a unit square with the chord P -> Q
    for a, b in ((Q, A1), (A1, A2), (A2, P)):
        assert np.linalg.norm(np.asarray(b) - np.asarray(a)) == pytest.approx(1.0, abs=1e-9)
    assert G.polygon_signed_area(np.array([P, Q, A1, A2])) == pytest.approx(1.0, abs=1e-9)
    # the order the old code used instead is a bowtie of zero area
    assert G.polygon_signed_area(np.array([P, Q, A2, A1])) == pytest.approx(0.0, abs=1e-9)


def test_three_fence_growth_adds_a_near_unit_cell():
    rng = np.random.default_rng(0)
    built, added = 0, []
    for k in range(40):
        sk = Bd.ring(4 + k % 3, 1.0)
        before = total_area(sk)
        if grow_m3(sk, rng):
            built += 1
            added.append(total_area(sk) - before)
            sk.topology()                      # a legal embedding, not just a vertex list
    assert built >= 36
    assert float(np.mean(added)) > 0.9
    assert sum(a >= 0.9 for a in added) >= 0.8 * built


def test_grown_cell_has_the_area_sample_growth_predicted():
    """The candidate's score and what the step actually adds must agree; they did not before."""
    rng = np.random.default_rng(3)
    checked = 0
    for k in range(30):
        sk = Bd.ring(5, 1.0)
        prop = Bd.sample_growth(sk, rng, {"m_weights": [0.0, 0.0, 1.0], "target_area": 1.0, "greedy": 1.0},
                                face=sk.face_cycles()[-1], m_max=3)
        if prop is None:
            continue
        Pp, Qp, inner, meta = prop
        before = total_area(sk)
        trial = sk.copy()
        trial.add_path_points(Pp, Qp, inner[::-1])
        trial.topology()
        assert total_area(trial) - before == pytest.approx(meta["area"], abs=1e-9)
        checked += 1
    assert checked >= 20


def test_three_fence_split_of_a_bounded_face_stays_legal():
    rng = np.random.default_rng(11)
    ok = 0
    for k in range(30):
        sk = Bd.ring(6, 1.0)                   # one big bounded face to split
        face = sk.face_cycles()[0]
        if Bd.grow_once(sk, rng, {"m_weights": [0.0, 0.0, 1.0], "target_area": 1.0, "greedy": 1.0},
                        m_max=3, face=face):
            topo = sk.topology()
            assert G.face_areas(sk.coords(), topo.faces).min() > 0
            ok += 1
    assert ok >= 10


def test_fit_count_reaches_n_with_three_fence_cells():
    """The interpreter still lands on exactly n fences once m = 3 paths can be built."""
    rng = np.random.default_rng(5)
    for n in (13, 20, 26):
        sk = Bd.ring(5, 1.0)
        Bd.fit_count(sk, n, rng, {**Bd.DEFAULT_GROW, "target_area": 1.0}, 1)
        assert sk.n == n
        sk.topology()
