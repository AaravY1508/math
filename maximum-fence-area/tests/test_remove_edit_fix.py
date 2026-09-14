"""The `remove_edit` fall-through (fence.evolve.RandomMutator, --remove-edit-fix).

`mutate` chooses an operator by weight and then walks an if/elif chain. The `remove_edit` arm
reads `elif op == "remove_edit" and g.get("edits")`, so a draw on a genome with no edits misses
it, falls through to the chain's `else` branch, and silently becomes `fresh` -- the worst-scoring
operator, and not what any weight asked for. Face-operator children always carry `"edits": []`,
so they were its main source; over the 6 h run of 2026-09-12 an estimated 600-750 of 58469
evaluations went that way.

This is a bug, not a weight, so it sits on its own flag and is independent of --op-mix: the
fix applies (or not) identically under both mixes. `off` reproduces the previous behaviour.
"""
import random

import pytest

from fence import evolve as E

MIXES = ["legacy", "tuned"]


def forced(op: str, mix: str = "legacy", fix: bool = False, seed: int = 0) -> E.RandomMutator:
    """A mutator that always picks `op` (weights are the only thing that chooses one)."""
    mut = E.RandomMutator(random.Random(seed), face_weight=0.0, op_mix=mix, remove_edit_fix=fix)
    mut.ops = [(op, 1.0)]
    return mut


def edit_free_parent() -> E.Program:
    p = E.Program.from_genome({"seed": {"kind": "ring"}, "grow": {"inside_prob": 0.5}})
    assert not p.genome.get("edits")
    return p


@pytest.mark.parametrize("mix", MIXES)
def test_off_reproduces_the_silent_fall_through_to_fresh(mix):
    mut = forced("remove_edit", mix, fix=False)
    origins = [mut.mutate(edit_free_parent(), {"n": 18}).origin for _ in range(8)]
    assert all(o == "random:fresh" for o in origins), origins


@pytest.mark.parametrize("mix", MIXES)
def test_on_jitters_instead(mix):
    mut = forced("remove_edit", mix, fix=True)
    origins = [mut.mutate(edit_free_parent(), {"n": 18}).origin for _ in range(8)]
    assert all(o.startswith("random:param:") for o in origins), origins


@pytest.mark.parametrize("mix", MIXES)
@pytest.mark.parametrize("fix", [False, True])
def test_a_genome_with_edits_is_unaffected_either_way(mix, fix):
    parent = E.Program.from_genome({"seed": {"kind": "ring"}, "edits": ["split_edge", "move_fence"]})
    child = forced("remove_edit", mix, fix).mutate(parent, {"n": 18})
    assert child.origin == "random:edit-"
    assert len(child.genome["edits"]) == 1


def test_the_fix_is_independent_of_the_operator_mix():
    """Change 5's reweighting must not carry the fix in, nor the fix carry the weights in."""
    assert E.RandomMutator(random.Random(0), op_mix="tuned").remove_edit_fix is E.REMOVE_EDIT_FIX
    assert E.RandomMutator(random.Random(0), remove_edit_fix=True).ops == E.RandomMutator.BASE_OPS
    assert E.RandomMutator(random.Random(0), op_mix="tuned", remove_edit_fix=False).ops == \
        E.RandomMutator.TUNED_OPS


def test_the_fix_does_not_change_any_other_operator():
    """Only the remove_edit-with-nothing-to-remove draw may differ."""
    for op in ("jitter", "add_edit", "seed", "symmetry", "fresh"):
        a = forced(op, fix=False, seed=3).mutate(edit_free_parent(), {"n": 18})
        b = forced(op, fix=True, seed=3).mutate(edit_free_parent(), {"n": 18})
        assert a.origin == b.origin and a.genome == b.genome, op


def test_worker_flag(tmp_path):
    from tests.test_evolve import make_worker
    assert make_worker(tmp_path, "20", minima=False).mutator.remove_edit_fix is E.REMOVE_EDIT_FIX
    w = make_worker(tmp_path / "on", "20", minima=False, extra=["--remove-edit-fix", "on"])
    assert w.mutator.remove_edit_fix is True
    w = make_worker(tmp_path / "off", "20", minima=False, extra=["--remove-edit-fix", "off"])
    assert w.mutator.remove_edit_fix is False
