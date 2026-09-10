"""Stage 4 sanity: programs, mutations, builder validity, archive and status classification."""
import random

import numpy as np
import pytest

from fence import builder as Bd
from fence import evolve as E


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
