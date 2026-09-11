"""Stage 4 sanity: programs, mutations, builder validity, archive and status classification,
lineages, grid-trivial handling and the worker's minima bookkeeping."""
import json
import pickle
import random
import time

import numpy as np
import pytest

from fence import builder as Bd
from fence import evolve as E
from fence import geometry as G
from fence import minima as M
from tests import configs as C
from tests.test_minima import verified


@pytest.mark.parametrize("n", [8, 13, 20, 35, 50])
def test_seed_programs_produce_valid_sketches(n):
    ok = 0
    for k, g in enumerate(E.seed_genomes(n)):
        prog = E.Program.from_genome(g)
        try:
            sk = prog.propose(n, seed=k)
        except Bd.BuildError:
            continue
        topo = sk.topology()           # raises if not a legal embedding
        assert topo.n == n
        ok += 1
    assert ok >= len(E.seed_genomes(n)) - 2


def test_random_mutations_compile_and_mostly_build():
    rng = random.Random(0)
    mut = E.RandomMutator(rng)
    n = 18
    sk = E.Program.from_genome(E.seed_genomes(n)[0]).propose(n, 1)
    elite = {"sketch": sk.to_json(), "note": "test"}
    parent = E.Program.from_genome(E.seed_genomes(n)[2])
    built = 0
    for i in range(30):
        child = mut.mutate(parent, {"n": n, "elite_sketches": [elite]})
        E.compile_program(child.source)
        try:
            s = child.propose(n, i)
            assert s.topology().n == n
            built += 1
        except Bd.BuildError:
            pass
        parent = child
    assert built >= 20


def test_structural_edits_keep_sketch_legal():
    rng = np.random.default_rng(5)
    sk = Bd.build_from_genome(24, rng, {"seed": {"kind": "ring"}, "grow": {"inside_prob": 0.5}})
    for name in E.EDIT_OPS:
        for _ in range(3):
            trial = sk.copy()
            Bd.EDITS[name](trial, rng, None)
            trial.topology()


def test_classify_semantics():
    b = {"A": 1.47585, "upper": 1.47586, "status": "best_known", "display": "1.47585+"}
    assert E.classify(1.4758529, b) == "TIES"
    assert E.classify(1.4758601, b) == "TIES"      # inside the 1e-7 safety margin: not claimed
    assert E.classify(1.4758603, b) == "BEATS"
    assert E.classify(1.4755, b) == "NEAR"
    assert E.classify(1.2, b) == "BELOW"
    exact = {"A": 4.0, "upper": 4.0, "status": "trivial", "display": "4"}
    assert E.classify(4.0 + 1e-12, exact) == "TIES"
    assert E.classify(4.0 + 1e-6, exact) == "BEATS"


def test_archive_keeps_best_per_cell():
    arc = E.Archive()
    p = E.Program.from_genome({"seed": {"kind": "ring"}})
    f = {"faces": 5, "tjunctions": 3, "sym": 1}
    assert arc.try_insert(E.Elite(p, 0.9, 4.0, f, {}))
    assert not arc.try_insert(E.Elite(p, 0.8, 3.5, f, {}))
    assert arc.try_insert(E.Elite(p, 0.95, 4.2, f, {}))
    assert arc.try_insert(E.Elite(p, 0.5, 2.0, {"faces": 6, "tjunctions": 3, "sym": 1}, {}))
    assert len(arc) == 2 and arc.best().fitness == 0.95


def test_baselines_loader_computes_upper_bounds():
    b = E.load_baselines()
    assert abs(b[6]["upper"] - 1.47586) < 1e-12
    assert b[12]["upper"] == 4.0
    assert abs(b[3]["upper"] - 3 ** 0.5 / 4) < 1e-15


# -- lineages, grid-trivial answers, scheduling -----------------------------------

def test_lattice_max_and_grid_trivial():
    for n, b in E.load_baselines().items():
        if b["status"] == "trivial" and n >= 4:
            assert E.lattice_max(n) == b["A"]
    assert {n: E.lattice_max(n) for n in (20, 22, 23, 25, 29, 32, 38, 39, 41, 48, 50)} == \
        {20: 7, 22: 8, 23: 8, 25: 9, 29: 11, 32: 12, 38: 15, 39: 15, 41: 16, 48: 19, 50: 20}
    assert E.is_grid_trivial(20.0 - 1e-11, 50) and E.is_grid_trivial(4.0 + 1e-12, 12)
    assert not E.is_grid_trivial(20.35492, 50)
    assert not E.is_grid_trivial(21.0, 50)          # more unit squares than 50 fences can hold
    assert not E.is_grid_trivial(None, 50)


def test_schedule_weights():
    b = E.load_baselines()
    assert E.schedule_weight(24, b[24], None) == 0.4
    assert E.schedule_weight(24, b[24], 9.0) == pytest.approx(0.4 * E.TRIVIAL_TIE_WEIGHT)
    assert E.schedule_weight(20, b[20], 7.0) == 1.0     # grid-trivial but below the record: full share
    assert E.schedule_weight(50, b[50], 20.0) == E.LARGE_N_WEIGHT
    assert E.schedule_weight(34, b[34], None) == E.LARGE_N_WEIGHT
    assert E.schedule_weight(8, b[8], None) == 0.25


def test_children_inherit_or_adopt_a_lineage():
    mut = E.RandomMutator(random.Random(3))
    n = 18
    elite = {"sketch": E.Program.from_genome(E.seed_genomes(n)[0]).propose(n, 1).to_json(), "note": "e",
             "proposer": "elite-lineage"}
    parent = E.Program.from_genome(E.seed_genomes(n)[2])
    assert parent.proposer == E.Program.from_genome(E.seed_genomes(n)[2]).proposer == E.proposer_id(parent.source)
    seen = set()
    for _ in range(150):
        child = mut.mutate(parent, {"n": n, "elite_sketches": [elite]})
        op = child.origin.split(":")[1]
        seen.add(op)
        if op == "elite":
            assert child.proposer == "elite-lineage"
        elif op == "fresh":
            assert child.proposer == E.proposer_id(child.source)
        else:
            assert child.proposer == parent.proposer
    assert {"elite", "fresh", "param"} <= seen


# -- worker (CPU only: no GPU is touched before Worker.run) ---------------------

def make_worker(tmp_path, ns, minima=True, extra=()):
    argv = ["--gpu", "0", "--ns", ns, "--run-dir", str(tmp_path / "run"), "--deadline", str(time.time() + 3600),
            "--minima-dir", str(tmp_path / "minima") if minima else "", *extra]
    return E.Worker(E.worker_parser().parse_args(argv))


def test_heartbeat_is_replaced_atomically(tmp_path):
    w = make_worker(tmp_path, "13", minima=False)
    w.heartbeat()
    ino = w.hb_path.stat().st_ino
    float(w.hb_path.read_text())
    w.heartbeat()
    assert w.hb_path.stat().st_ino != ino           # renamed into place, never truncated and rewritten
    assert not list(w.hb_path.parent.glob("*.tmp"))


def test_restarted_worker_keeps_a_better_saved_best(tmp_path):
    w = make_worker(tmp_path, "24")
    prog = E.Program.from_genome(E.seed_genomes(24)[0])
    cfg = lambda a: {"n": 24, "verification": {"total_area": repr(a), "dps": 50}}  # noqa: E731
    saved = w.run_dir / "best" / "n024.json"
    w._new_best(24, prog, 8.9, cfg(8.9), {}, None)
    w.state[24].best["area"] = 8.5                  # as if restored from an older checkpoint
    w._new_best(24, prog, 8.7, cfg(8.7), {}, None)
    assert json.loads(saved.read_text())["verification"]["total_area"] == repr(8.9)
    assert w.state[24].best["area"] == 8.9
    w._new_best(24, prog, 8.95, cfg(8.95), {}, None)
    assert json.loads(saved.read_text())["verification"]["total_area"] == repr(8.95)
    assert w.lib.record(24)["area"] == 8.95


def test_trivial_lineage_is_blocked_only_while_best_is_trivial(tmp_path):
    grid = E.Program.from_genome(E.seed_genomes(24)[0])          # the 3x3 grid proposer
    M.MinimaLibrary(tmp_path / "minima").mark_trivial(grid.proposer, 24, {"verification": {"total_area": "9.0"}})
    w = make_worker(tmp_path, "24")
    assert w._blocked(24) == set()                  # n's best not known yet
    w.records[24] = 9.0                             # the grid answer, a tie with the record
    assert w._blocked(24) == {grid.proposer}
    for _ in range(10):
        job = w.make_job(24)
        assert job is None or job["program"].proposer != grid.proposer
    w.state[24].best = {"area": 9.2}                # beaten by a non-grid layout
    assert w._blocked(24) == set()


def test_stabilized_minimum_is_indexed_once(tmp_path):
    w = make_worker(tmp_path, "3", extra=["--stable-after", "3"])
    cfg = verified(C.TRIANGLE, 3)[0]
    prog = E.Program.from_genome(E.seed_genomes(3)[0])
    sig = M.canonical_form(G.config_from_json(cfg)[0]).signature
    area = float(cfg["verification"]["total_area"])
    w._track_epoch(3, prog, area, cfg, sig, None)
    for _ in range(3):
        w._track_epoch(3, prog, None, None, None, None)
    [m] = w.known[3][prog.proposer]
    pdir = tmp_path / "minima" / "proposers" / prog.proposer
    assert m["name"] == "N3.0" and (pdir / "N3.0.json").exists() and (pdir / "proposer.json").exists()
    dup = w._dup_of(3, prog.proposer, sig, area + 1e-7)
    assert dup == "N3.0"
    for _ in range(8):                              # re-convergence is never a new result
        w._track_epoch(3, prog, area, cfg, sig, dup)
    assert len(w.lib.minima_of(prog.proposer, 3)) == 1


def test_trivial_stabilization_marks_the_lineage(tmp_path):
    w = make_worker(tmp_path, "12", extra=["--stable-after", "2"])
    cfg = verified(C.GRID_2x2, 12)[0]
    prog = E.Program.from_genome(E.seed_genomes(12)[0])
    w._track_epoch(12, prog, 4.0, cfg, "sig", None)
    w._track_epoch(12, prog, 3.5, cfg, "sig", None)
    w._track_epoch(12, prog, None, None, None, None)
    assert w.trivial[12] == {prog.proposer} and w.lib.trivial_proposers(12) == {prog.proposer}
    assert w.known[12] == {}


def test_exhausted_topology_is_reproposed(tmp_path):
    w = make_worker(tmp_path, "14")
    prog = E.Program.from_genome({"seed": {"kind": "ring"}, "grow": {"inside_prob": 0.5}})
    sig = M.canonical_form(w._build(14, prog, 7)[1]).signature
    w.known[14] = {prog.proposer: [{"name": f"N14.{k}", "signature": sig, "area": 4.0 + k} for k in range(3)]}
    child, _, _, form, _, rerolls = w._propose(14, prog, 7)
    assert rerolls >= 1 and child.proposer == prog.proposer
    assert form.signature != sig or rerolls == 2
    w.known[14] = {}
    w.state[14].barren[(prog.proposer, sig)] = 2    # two steered attempts found nothing new
    assert w._propose(14, prog, 7)[5] >= 1


def test_resume_from_a_checkpoint_without_lineages(tmp_path):
    w = make_worker(tmp_path, "13")
    p = E.Program.from_genome(E.seed_genomes(13)[0])
    w.state[13].islands[0].try_insert(E.Elite(p, 0.9, 4.0, {"faces": 4, "tjunctions": 0, "sym": 1}, {}))
    w.checkpoint()
    blob = pickle.loads(w.ckpt_path.read_bytes())
    for s in blob["state"].values():                # what checkpoints written before this change hold
        del s.__dict__["epochs"], s.__dict__["barren"]
        for isl in s.islands:
            for e in isl.cells.values():
                del e.program.__dict__["proposer"]
    w.ckpt_path.write_bytes(pickle.dumps(blob))
    w2 = make_worker(tmp_path, "13", extra=["--resume"])
    s2 = w2.state[13]
    assert s2.epochs == {} and s2.barren == {}
    [e] = s2.islands[0].cells.values()
    assert e.program.proposer == E.proposer_id(p.source)
