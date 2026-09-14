"""Mutation operator WEIGHTS (fence.evolve.RandomMutator, --op-mix).

`legacy` is the mix every run so far used. `tuned` re-weights it against the run-best win rate
measured per operator over the 58469 evaluations of the 6 h run of 2026-09-12; it did not pass
its A/B (EXPERIMENTS.md change 5) and is not the default.

The `remove_edit` fall-through is a separate concern on its own flag: tests/test_remove_edit_fix.py.
"""
import random

import pytest

from fence import evolve as E

LEGACY = [("jitter", 0.25), ("add_edit", 0.12), ("remove_edit", 0.05), ("seed", 0.08), ("symmetry", 0.08),
          ("elite", 0.36), ("fresh", 0.06)]


def test_legacy_is_the_previous_mix_and_is_the_module_default():
    assert E.OP_MIX == "legacy"
    m = E.RandomMutator(random.Random(0))
    assert m.op_mix == "legacy"
    assert m.op_weights(True, False) == LEGACY
    assert E.make_mutator("random", random.Random(0)).ops == LEGACY


def test_tuned_mix_is_a_distribution_that_favours_the_measured_winners():
    w = dict(E.RandomMutator(random.Random(0), op_mix="tuned").op_weights(True, False))
    assert sum(w.values()) == pytest.approx(1.0)
    legacy = dict(LEGACY)
    assert set(w) == set(legacy)
    assert w["elite"] > legacy["elite"]                       # 11.7 wins / 1000 evals
    for k in ("fresh", "seed", "jitter", "add_edit", "symmetry", "remove_edit"):
        assert w[k] < legacy[k], k
    assert all(v > 0 for v in w.values())                     # every operator keeps a floor


def test_the_face_op_still_takes_its_share_from_the_regrowing_operators_under_tuned():
    w = dict(E.RandomMutator(random.Random(0), face_weight=0.25, op_mix="tuned").op_weights(True, True))
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["face"] == 0.25
    assert w["elite"] == pytest.approx(0.56)                  # elite keeps its full share
    assert w["jitter"] == pytest.approx(0.20 * (0.44 - 0.25) / 0.44)


def test_seed_program_injection_rate_by_mix():
    assert E.SEED_INJECT["legacy"] == 0.04
    assert E.SEED_INJECT["tuned"] < E.SEED_INJECT["legacy"]


def test_worker_passes_the_op_mix(tmp_path):
    from tests.test_evolve import make_worker
    assert make_worker(tmp_path, "20", minima=False).mutator.ops == LEGACY
    w = make_worker(tmp_path / "t", "20", minima=False, extra=["--op-mix", "tuned"])
    assert w.mutator.op_mix == "tuned" and w.mutator.ops == E.RandomMutator.TUNED_OPS
    assert E.SEED_INJECT[w.args.op_mix] == E.SEED_INJECT["tuned"]


def test_tuned_children_still_compile_and_build():
    rng = random.Random(1)
    mut = E.RandomMutator(rng, op_mix="tuned")
    n = 18
    sk = E.Program.from_genome(E.seed_genomes(n)[0]).propose(n, 1)
    parent = E.Program.from_genome(E.seed_genomes(n)[2])
    built = 0
    for i in range(30):
        child = mut.mutate(parent, {"n": n, "elite_sketches": [{"sketch": sk.to_json(), "note": "test"}]})
        E.compile_program(child.source)
        try:
            child.propose(n, i)
            built += 1
        except Exception:  # noqa: BLE001 -- a broken program is just an unfit program
            pass
        parent = child
    assert built >= 20
