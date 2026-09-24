"""Mutation operator WEIGHTS (fence.evolve.RandomMutator).

A mix re-weighted against measured win rates failed its A/B (EXPERIMENTS.md change 5) and was
removed; these pin the one remaining mix. The `remove_edit` fall-through is a separate concern on
its own flag: tests/test_remove_edit_fix.py.
"""
import random

import pytest

from fence import evolve as E

WEIGHTS = [("jitter", 0.25), ("add_edit", 0.12), ("remove_edit", 0.05), ("seed", 0.08), ("symmetry", 0.08),
           ("elite", 0.36), ("fresh", 0.06)]


def test_operator_weights():
    assert E.RandomMutator(random.Random(0)).op_weights(True, False) == WEIGHTS
    assert E.make_mutator("random", random.Random(0)).ops == WEIGHTS


def test_the_face_op_takes_its_share_from_the_regrowing_operators():
    w = dict(E.RandomMutator(random.Random(0), face_weight=0.25).op_weights(True, True))
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["face"] == 0.25
    assert w["elite"] == pytest.approx(0.36)                  # elite keeps its full share
    assert w["jitter"] == pytest.approx(0.25 * (0.64 - 0.25) / 0.64)


def test_seed_program_injection_rate():
    assert E.SEED_INJECT == 0.04


def test_worker_uses_the_weights(tmp_path):
    from tests.test_evolve import make_worker
    assert make_worker(tmp_path, "20", minima=False).mutator.ops == WEIGHTS
