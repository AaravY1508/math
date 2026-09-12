"""Archive seeding from the minima library, the forced edit on its children, and --focus."""
import copy
import random

import numpy as np
import pytest

from fence import evolve as E
from fence import geometry as G
from fence import minima as M
from tests.test_evolve import make_worker
from tests.test_face_edits import RECORDS


def verified_record(n):
    cfg = {k: RECORDS[str(n)][k] for k in ("n", "segments", "vertices", "topology")}
    cfg["verification"] = G.verify(cfg, 1e-9).to_json()
    assert cfg["verification"]["valid"]
    return cfg


def library_for_20(root):
    """n=20: the real record layout (7.0) as minimum N20.0 of lineage pA and as the record, plus
    two minima with made-up topology signatures and areas (the library does not re-verify)."""
    lib = M.MinimaLibrary(root)
    cfg = verified_record(20)
    topo, X = G.config_from_json(cfg)
    form = M.canonical_form(topo)
    lib.add_minimum("pA", 20, cfg, form.signature, X[form.orders[0]])
    for prop, sig, area in (("pB", "sigB", "6.9"), ("pC", "sigC", "6.8"), ("pC", "sigB", "6.5")):
        c = copy.deepcopy(cfg)
        c["verification"]["total_area"] = area
        lib.add_minimum(prop, 20, c, sig, X[form.orders[0]])
    lib.update_record(20, cfg, {})                    # a record written without its lineage
    return lib, form.signature


def test_library_best_is_distinct_and_best_first(tmp_path):
    lib, sig = library_for_20(tmp_path / "minima")
    best = lib.best(20, 10)
    # one per topology, the best one (sigB's 6.5 minimum is dropped), largest area first
    assert [d["signature"] for d in best] == [sig, "sigB", "sigC"]
    assert [d["area"] for d in best] == pytest.approx([7.0, 6.9, 6.8])
    assert best[0]["name"] == "record:N20" and best[0]["proposer"] == "pA"   # lineage from its minimum
    assert lib.best(20, 1) == best[:1] and lib.best(21, 3) == []


def test_worker_seeds_each_archive_on_a_fresh_start_only(tmp_path):
    library_for_20(tmp_path / "minima")
    w = make_worker(tmp_path, "20", extra=["--seed-elites", "2"])
    st = w.state[20]
    elites = [e for isl in st.islands for e in isl.cells.values()]
    assert sorted(e.area for e in elites) == pytest.approx([6.9, 7.0])
    assert all(e.program.origin.startswith("library:") and e.program.genome["avoid_signature"] for e in elites)
    assert st.seeded == len(E.seed_genomes(20))        # no initial sweep of the seed programs
    w.checkpoint()
    for isl in st.islands:
        isl.cells.clear()
    w.checkpoint()
    w2 = make_worker(tmp_path, "20", extra=["--seed-elites", "2", "--resume"])
    assert all(len(isl) == 0 for isl in w2.state[20].islands)          # a resumed archive is kept as it was
    w3 = make_worker(tmp_path / "other", "20", extra=["--seed-elites", "0"])
    assert all(len(isl) == 0 for isl in w3.state[20].islands)


def test_elite_children_of_a_library_minimum_get_an_edit():
    mut = E.RandomMutator(random.Random(0))
    parent = E.Program.from_genome(E.seed_genomes(20)[2])
    src = {"sketch": E._config_to_sketch(RECORDS["20"]), "note": "n20:7.0000", "proposer": "pA", "n": 20,
           "area": 7.0, "avoid_signature": "sigX"}
    elite = [c for c in (mut.mutate(parent, {"n": 20, "elite_sketches": [src]}) for _ in range(60))
             if c.origin.startswith("random:elite")]
    assert elite and all(c.genome["edits"] and c.genome["avoid_signature"] == "sigX" for c in elite)


def test_no_child_re_creates_a_seeded_minimum(tmp_path):
    library_for_20(tmp_path / "minima")
    w = make_worker(tmp_path, "20", extra=["--seed-elites", "3"])
    jobs = [j for j in (w.make_job(20) for _ in range(40)) if j is not None]
    library_children = [j for j in jobs if j["program"].genome and j["program"].genome.get("avoid_signature")]
    assert len(library_children) >= 10
    assert all(j["sig"] != j["program"].genome["avoid_signature"] for j in library_children)


def test_focus_picks_the_n_closest_to_their_records():
    b = E.load_baselines()
    records = {12: 4.0, 13: 4.048, 14: 4.7326, 20: 7.0, 30: None, 50: b[50]["upper"] + 1e-3}
    assert E.focus_ns(records, b, records, 2) == pytest.approx({13: b[13]["A"] - 4.048, 14: b[14]["A"] - 4.7326})
    assert list(E.focus_ns(records, b, records, 10)) == [13, 14, 20]   # 12: grid record, 30: no area, 50: beaten


def test_record_lineage_from_a_grid_trivial_marker(tmp_path):
    # a record imported from a run that did not track lineages, whose layout is the grid-trivial
    # answer some lineage stabilized on (such answers are marked, not indexed as minima)
    lib = M.MinimaLibrary(tmp_path / "minima")
    cfg = verified_record(20)
    lib.update_record(20, cfg, {})
    assert lib.best(20, 3) == []                      # no lineage known: left out
    lib.mark_trivial("pGrid", 20, cfg)
    [d] = lib.best(20, 3)
    assert d["name"] == "record:N20" and d["proposer"] == "pGrid"


def test_seeded_minima_are_known_to_every_lineage(tmp_path):
    # (library_for_20's made-up minima reuse the record's layout, so seed only the record)
    lib, sig = library_for_20(tmp_path / "minima")
    w = make_worker(tmp_path, "20", extra=["--seed-elites", "1"])
    st = w.state[20]
    assert [m["area"] for m in st.seeded_minima] == pytest.approx([7.0])
    # another lineage landing on the seeded record's topology is steered away from it ...
    [m] = w._known_minima(20, "someone-else", sig)
    assert m["name"] == "seeded:record:N20" and np.shape(m["canon_coords"]) == (G.config_from_json(RECORDS["20"])[1].shape)
    # ... and a result there with its area is a duplicate, not a new minimum
    assert w._dup_of(20, "someone-else", sig, 7.0 + 1e-9) == "seeded:record:N20"
    assert w._dup_of(20, "someone-else", sig, 6.95) is None
    # the record's own lineage knows it once, not twice
    assert len(w._known_minima(20, "pA", sig)) == 1
    w.checkpoint()
    w2 = make_worker(tmp_path, "20", extra=["--seed-elites", "1", "--resume"])
    assert [m["name"] for m in w2.state[20].seeded_minima] == [m["name"] for m in st.seeded_minima]
