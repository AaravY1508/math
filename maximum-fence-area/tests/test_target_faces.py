"""--target-faces: face-short n seed and breed only from layouts with the target face count.

target_faces(record) = floor(record) + 1 is the fewest faces that could beat the published record
(every face holds at most area 1). An n whose all-time record has fewer faces than that is
face-short. Its best layouts are almost all the integer grid answer at F - 1, so seeding the
archive by area fills it with the grid: on the 6 h run of 2026-09-13 all 11 face-short n were
seeded with six grid layouts and no target-F layout, although the library held 20-159 of them
per n. --target-faces restricts seeding, parent choice and elite sources to target-F layouts for
those n only; every other n, and every n with the flag off, is unchanged.
"""
import copy
import random

import pytest

from fence import evolve as E
from fence import geometry as G
from fence import minima as M
from tests.test_evolve import make_worker
from tests.test_library_seeding import library_for_20, verified_record


def add_faces(lib, n, rows):
    """Minima at n with a given face count (the library does not re-verify): (proposer, sig, area, F)."""
    cfg = verified_record(n)
    topo, X = G.config_from_json(cfg)
    form = M.canonical_form(topo)
    for prop, sig, area, F in rows:
        c = copy.deepcopy(cfg)
        c["verification"]["total_area"] = area
        c["verification"]["n_faces"] = F
        lib.add_minimum(prop, n, c, sig, X[form.orders[0]])


def seeded(w, n):
    return sorted(((round(e.area, 9), e.features["faces"]) for isl in w.state[n].islands
                   for e in isl.cells.values()), reverse=True)


@pytest.fixture
def face_short_20(tmp_path):
    """n=20: the grid record 7.0 at 7 faces (target 8), grid-like 7-face minima, and 8-face minima."""
    lib, _ = library_for_20(tmp_path / "minima")
    add_faces(lib, 20, [("pD", "sigD", "6.93", 8), ("pE", "sigE", "6.85", 8), ("pF", "sigF", "6.10", 8)])
    return tmp_path


def test_face_short_is_an_n_whose_record_is_below_its_target(face_short_20):
    assert E.target_faces(E.load_baselines()[20]) == 8
    w = make_worker(face_short_20, "20", extra=["--target-faces", "on"])
    assert w.face_short == {20}
    assert make_worker(face_short_20 / "off", "20", minima=False).face_short == set()


def test_off_seeds_by_area_as_before(face_short_20):
    w = make_worker(face_short_20, "20", extra=["--seed-elites", "3"])
    assert w.face_short == set()
    assert seeded(w, 20) == [(7.0, 7), (6.93, 8), (6.9, 7)]           # the grid record leads


def test_on_seeds_only_target_face_layouts(face_short_20):
    w = make_worker(face_short_20, "20", extra=["--seed-elites", "3", "--target-faces", "on"])
    got = seeded(w, 20)
    assert got == [(6.93, 8), (6.85, 8), (6.1, 8)]                    # best 8-face first; no grid
    assert all(F >= 8 for _, F in got)


def test_on_picks_parents_and_elite_sources_from_target_faces_only(face_short_20):
    w = make_worker(face_short_20, "20", extra=["--seed-elites", "3", "--target-faces", "on"])
    st = w.state[20]
    # a better 7-face elite turns up during the run: it may live in the archive, never parent
    grid = copy.copy(next(e for isl in st.islands for e in isl.cells.values()))
    grid.features = {**grid.features, "faces": 7}
    grid.fitness, grid.area = 2.0, 7.0
    st.islands[0].cells[("grid",)] = grid
    ok = w._faces_ok(20)
    rng = random.Random(0)
    for _ in range(200):
        for isl in st.islands:
            e = isl.sample(rng, keep=ok)
            assert e is None or e.features["faces"] >= 8
    src = w._elite_sketches(20, with_blocked=True)
    assert src and all(s["n"] == 20 and s["area"] < 7.0 for s in src)  # own 8-face elites; no neighbours


def test_an_n_that_meets_its_target_is_unaffected(tmp_path):
    lib, _ = library_for_20(tmp_path / "minima")
    rec = verified_record(20)
    rec["verification"]["n_faces"] = 8                                # as if the record met its target
    lib.update_record(20, {**rec, "verification": {**rec["verification"], "total_area": "7.1"}}, {})
    add_faces(lib, 20, [("pD", "sigD", "6.93", 8)])
    on = make_worker(tmp_path, "20", extra=["--seed-elites", "3", "--target-faces", "on"])
    off = make_worker(tmp_path / "b", "20", extra=["--seed-elites", "3", "--minima-dir", str(tmp_path / "minima")])
    assert on.face_short == set()
    assert seeded(on, 20) == seeded(off, 20)
    assert on._faces_ok(20) is None


def test_archive_sample_keep_predicate():
    a = E.Archive()
    for F, fit in ((7, 1.0), (8, 0.9), (8, 0.8)):
        p = E.Program.from_genome({"seed": {"kind": "ring"}})
        a.try_insert(E.Elite(p, fit, fit, {"faces": F, "tjunctions": 0, "sym": 1}, {}))
    rng = random.Random(1)
    assert {a.sample(rng).features["faces"] for _ in range(100)} == {7, 8}
    assert {a.sample(rng, keep=lambda e: e.features["faces"] >= 8).features["faces"] for _ in range(100)} == {8}
    assert a.sample(rng, keep=lambda e: False) is None


def test_flags():
    assert E.TARGET_FACES is False
    base = ["--gpu", "0", "--ns", "20", "--run-dir", "/tmp/x", "--deadline", "0"]
    assert E.worker_parser().parse_args(base).target_faces == "off"
    assert E.worker_parser().parse_args(base + ["--target-faces", "on"]).target_faces == "on"
