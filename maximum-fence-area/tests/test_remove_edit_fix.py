"""The `remove_edit` fall-through (fence.evolve.RandomMutator, --remove-edit-fix).

`mutate` chooses an operator by weight and then walks an if/elif chain. The `remove_edit` arm
reads `elif op == "remove_edit" and g.get("edits")`, so a draw on a genome with no edits misses
it, falls through to the chain's `else` branch, and silently becomes `fresh` -- the worst-scoring
operator, and not what any weight asked for. Face-operator children always carry `"edits": []`,
so they were its main source; over the 6 h run of 2026-09-12 an estimated 600-750 of 58469
evaluations went that way.

This is a bug, not a weight, so it sits on its own flag. `off` reproduces the previous behaviour.
"""
import random

import pytest

from fence import evolve as E

def forced(op: str, fix: bool = False, seed: int = 0) -> E.RandomMutator:
    """A mutator that always picks `op` (weights are the only thing that chooses one)."""
    mut = E.RandomMutator(random.Random(seed), face_weight=0.0, remove_edit_fix=fix)
    mut.ops = [(op, 1.0)]
    return mut


def edit_free_parent() -> E.Program:
    p = E.Program.from_genome({"seed": {"kind": "ring"}, "grow": {"inside_prob": 0.5}})
    assert not p.genome.get("edits")
    return p


def test_off_reproduces_the_silent_fall_through_to_fresh():
    mut = forced("remove_edit", fix=False)
    origins = [mut.mutate(edit_free_parent(), {"n": 18}).origin for _ in range(8)]
    assert all(o == "random:fresh" for o in origins), origins


def test_on_jitters_instead():
    mut = forced("remove_edit", fix=True)
    origins = [mut.mutate(edit_free_parent(), {"n": 18}).origin for _ in range(8)]
    assert all(o.startswith("random:param:") for o in origins), origins


@pytest.mark.parametrize("fix", [False, True])
def test_a_genome_with_edits_is_unaffected_either_way(fix):
    parent = E.Program.from_genome({"seed": {"kind": "ring"}, "edits": ["split_edge", "move_fence"]})
    child = forced("remove_edit", fix).mutate(parent, {"n": 18})
    assert child.origin == "random:edit-"
    assert len(child.genome["edits"]) == 1
