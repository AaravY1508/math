"""Minima library: topology signatures, per-proposer indexing, trivial markers, records, import."""
import json

import numpy as np
import pytest

from fence import builder as Bd
from fence import evolve as E
from fence import geometry as G
from fence import minima as M
from tests import configs as C


def verified(segs, n):
    s = np.array([[float(v) for v in q] for q in segs])
    coords, fences, pins = G.topology_from_segments(s, tol=1e-9)
    topo = G.make_topology(coords, fences, pins)
    cfg = G.config_to_json(topo, coords, n=n)
    cfg["verification"] = G.verify(cfg, 1e-9).to_json()
    assert cfg["verification"]["valid"]
    return cfg, topo, coords


def relabeled(topo, X, rng, mirror=False):
    """The same layout with vertices and fences renumbered, fence ends swapped, and optionally mirrored."""
    p, q = rng.permutation(topo.n_vertices), rng.permutation(topo.n)
    Xn = np.empty_like(X)
    Xn[p] = X * ([-1.0, 1.0] if mirror else [1.0, 1.0]) + [3.0, -2.0]
    fences = np.empty((topo.n, 2), dtype=np.int64)
    for f, (a, b) in enumerate(topo.fences):
        fences[q[f]] = [p[b], p[a]] if rng.random() < 0.5 else [p[a], p[b]]
    return G.make_topology(Xn, fences, {int(p[v]): int(q[h]) for v, h in topo.pins.items()}), Xn


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_signature_ignores_labels_motion_and_reflection(seed):
    rng = np.random.default_rng(seed)
    sk = Bd.build_from_genome(20, np.random.default_rng(seed), {"seed": {"kind": "ring"}, "grow": {"inside_prob": 0.5}})
    topo, X = sk.topology(), sk.coords()
    sig = M.canonical_form(topo).signature
    for mirror in (False, True, False, True):
        assert M.canonical_form(relabeled(topo, X, rng, mirror)[0]).signature == sig


def test_signature_separates_layouts_and_sees_the_fence_partition():
    sigs = {M.canonical_form(verified(segs, n)[1]).signature
            for segs, n in [(C.GRID_2x2, 12), (C.GRID_2x3, 17), (C.DOMINO, 7), (C.SQUARE, 4)]}
    assert len(sigs) == 4
    # same plane graph as SPLIT_SQUARE, but the stem ends in joints instead of T-junctions
    joints = [C.seg(0, 0, "0.5", 0), C.seg("0.5", 0, 1, 0), C.seg(1, 0, 1, 1), C.seg(1, 1, "0.5", 1),
              C.seg("0.5", 1, 0, 1), C.seg(0, 1, 0, 0), C.seg("0.5", 0, "0.5", 1)]
    s = np.array([[float(v) for v in q] for q in joints])
    t_joint = G.make_topology(*G.topology_from_segments(s, tol=1e-9))
    t_pin = verified(C.SPLIT_SQUARE, 5)[1]
    assert sorted(map(sorted, t_joint.edges())) == sorted(map(sorted, t_pin.edges()))
    assert M.canonical_form(t_joint).signature != M.canonical_form(t_pin).signature


def test_automorphism_images():
    _, topo, X = verified(C.GRID_2x2, 12)
    form = M.canonical_form(topo)
    assert len(form.orders) == 8                      # the square's symmetry group D4
    # a D4-symmetric geometry: every relabeled copy is the same point set up to motion
    assert len(M.avoid_images(form, [{"canon_coords": X[form.orders[0]]}])) == 1
    Xa = X.copy()
    Xa[0] += [0.05, 0.02]                             # break the symmetry of the geometry only
    imgs = M.avoid_images(form, [{"canon_coords": Xa[form.orders[0]]}])
    assert len(imgs) == 8
    assert min(M.rigid_rms(Xa, Y) for Y in imgs) < 1e-12


def test_rigid_rms_is_invariant():
    X = np.random.default_rng(0).normal(size=(9, 2))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert M.rigid_rms(X, X @ R.T + [4.0, 1.0]) < 1e-7
    assert M.rigid_rms(X, X * [-1.0, 1.0]) < 1e-7
    assert M.rigid_rms(X, X + np.random.default_rng(1).normal(scale=0.1, size=X.shape)) > 1e-2


def _with_area(cfg, area):
    return {**cfg, "verification": {**cfg["verification"], "total_area": repr(area)}}


def test_library_indexes_distinct_minima_per_proposer(tmp_path):
    lib = M.MinimaLibrary(tmp_path)
    cfg, topo, X = verified(C.TRIANGLE, 3)
    form = M.canonical_form(topo)
    a = float(cfg["verification"]["total_area"])
    add = lambda p, c, sig=form.signature: lib.add_minimum(p, 3, c, sig, X[form.orders[0]])  # noqa: E731
    assert add("p1", cfg)[1]["name"] == "N3.0"
    new, doc = add("p1", _with_area(cfg, a + 5e-7))            # same topology, within 1e-6: same minimum
    assert not new and doc["name"] == "N3.0"
    assert add("p1", _with_area(cfg, a + 1e-5))[1]["name"] == "N3.1"
    assert add("p1", cfg, sig="0" * 16)[1]["name"] == "N3.2"   # another topology with the same area
    assert add("p2", cfg)[1]["name"] == "N3.0"                  # k counts per proposer
    assert (tmp_path / "proposers" / "p1" / "N3.2.json").exists()
    assert {p: [m["k"] for m in ms] for p, ms in lib.minima(3).items()} == {"p1": [0, 1, 2], "p2": [0]}
    assert lib.minima(30) == {}                                 # N3.* must not match N30.*


def test_trivial_markers_and_records(tmp_path):
    lib = M.MinimaLibrary(tmp_path)
    cfg = verified(C.GRID_2x2, 12)[0]
    lib.mark_trivial("p1", 12, cfg)
    lib.mark_trivial("p1", 12, cfg)
    assert lib.trivial_proposers(12) == {"p1"}
    assert json.loads((tmp_path / "proposers" / "p1" / "N12.trivial.json").read_text())["count"] == 2
    assert lib.minima(12) == {}                                 # a marker is not an indexed minimum
    assert lib.update_record(12, _with_area(cfg, 3.9)) and lib.record(12)["area"] == 3.9
    assert not lib.update_record(12, _with_area(cfg, 3.8))
    assert lib.update_record(12, cfg) and lib.record(12)["area"] == 4.0


def test_import_run_records_and_seed_lineages(tmp_path):
    run = tmp_path / "runs" / "old"
    (run / "best").mkdir(parents=True)
    (run / "worker0" / "programs").mkdir(parents=True)
    cfg = verified(C.GRID_2x2, 12)[0]
    seed0 = E.render_program(E.seed_genomes(12)[0])
    (run / "best" / "n012.json").write_text(json.dumps({**cfg, "program_origin": "seed:0"}))
    (run / "worker0" / "programs" / "n012_best.py").write_text(seed0)
    tri = verified(C.TRIANGLE, 3)[0]
    (run / "best" / "n003.json").write_text(json.dumps({**tri, "program_origin": "random:param:x"}))
    lib = M.MinimaLibrary(tmp_path / "lib")
    s = M.import_run(lib, run)
    assert sorted(s["records"]) == [3, 12]
    assert lib.trivial_proposers(12) == {E.proposer_id(seed0)} == {E.Program.from_genome(E.seed_genomes(12)[0]).proposer}
    assert lib.trivial_proposers(3) == set()                     # not grid-trivial


def test_a_lineage_less_record_is_seeded_once_indexed_as_digitized(tmp_path):
    # a record written without a proposer (as the 2026-09-24 digitized ones were) is invisible to
    # best(), which seeds later runs; indexing it under the digitized lineage makes it the first seed
    cfg, topo, X = verified(C.GRID_2x3, 17)
    lib = M.MinimaLibrary(tmp_path)
    lib.update_record(17, cfg, {"via": "digitized"})
    assert lib.best(17, 3) == []
    form = M.canonical_form(topo)
    lib.add_minimum(M.DIGITIZED_PROPOSER, 17, cfg, form.signature, X[form.orders[0]], {"via": "digitized"})
    top = lib.best(17, 3)
    assert top[0]["name"] == "record:N17" and top[0]["proposer"] == M.DIGITIZED_PROPOSER
