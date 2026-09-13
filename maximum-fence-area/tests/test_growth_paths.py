"""Growth paths of three or more fences (fence.builder.M3_GROWTH).

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
With M3_GROWTH off both mistakes are kept, so a run reproduces the previous behaviour.
"""
import numpy as np
import pytest

from fence import builder as Bd
from fence import geometry as G


@pytest.fixture
def m3(monkeypatch):
    """Pin the fix on, whatever the module default is."""
    monkeypatch.setattr(Bd, "M3_GROWTH", True)


def test_the_validated_default_is_on():
    assert Bd.M3_GROWTH is True            # EXPERIMENTS.md change 4


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


def test_three_fence_growth_never_builds_without_the_fix(monkeypatch):
    monkeypatch.setattr(Bd, "M3_GROWTH", False)
    rng = np.random.default_rng(0)
    built = 0
    for k in range(40):
        sk = Bd.ring(4 + k % 3, 1.0)
        built += grow_m3(sk, rng)
    assert built == 0


def test_three_fence_growth_adds_a_near_unit_cell(m3):
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


def test_grown_cell_has_the_area_sample_growth_predicted(m3):
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


def test_m3_growth_leaves_paths_of_at_most_two_fences_untouched(monkeypatch):
    """The flag must change only m >= 3: reversing at most one interior point is a no-op."""
    def build(on):
        monkeypatch.setattr(Bd, "M3_GROWTH", on)
        out = []
        for k in range(25):
            sk = Bd.ring(5, 1.0)
            rng = np.random.default_rng(100 + k)
            for _ in range(3):
                Bd.grow_once(sk, rng, {"m_weights": [0.5, 0.5, 0.0]}, m_max=2)
            out.append((sk.to_json(), round(total_area(sk), 12)))
        return out
    assert build(False) == build(True)


def test_three_fence_split_of_a_bounded_face_stays_legal(m3):
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


def test_fit_count_reaches_n_with_three_fence_cells(m3):
    """The interpreter still lands on exactly n fences once m = 3 paths can be built."""
    rng = np.random.default_rng(5)
    for n in (13, 20, 26):
        sk = Bd.ring(5, 1.0)
        Bd.fit_count(sk, n, rng, {**Bd.DEFAULT_GROW, "target_area": 1.0}, 1)
        assert sk.n == n
        sk.topology()


def test_worker_flag_sets_the_builder_toggle(monkeypatch):
    from fence import evolve as E
    monkeypatch.setattr(Bd, "M3_GROWTH", False)
    assert E.worker_parser().parse_args(["--gpu", "0", "--ns", "13", "--run-dir", "/tmp/x",
                                         "--deadline", "0"]).m3_growth == "off"
    monkeypatch.setattr(Bd, "M3_GROWTH", True)
    assert E.worker_parser().parse_args(["--gpu", "0", "--ns", "13", "--run-dir", "/tmp/x",
                                         "--deadline", "0"]).m3_growth == "on"
    assert E.worker_parser().parse_args(["--gpu", "0", "--ns", "13", "--run-dir", "/tmp/x",
                                         "--deadline", "0", "--m3-growth", "off"]).m3_growth == "off"
