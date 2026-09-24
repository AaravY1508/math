"""The onion net generator (EXPERIMENTS.md change 13): legal nets of the right size, off by default."""
import numpy as np
import pytest

from fence import nets as N
from fence import rules as R


def test_onion_is_registered_off_by_default():
    assert "onion" in N.GENERATORS
    assert N.GENERATORS["onion"].default is False


@pytest.mark.parametrize("outer_pin", [False, True])
def test_onion_sketch_counts(outer_pin):
    # diamond core, one ring of a 10-gon with 8 spokes: n = 4 + 10 + 8, F = 1 + 8
    sk = N.onion_sketch(4, [(10, 8)], outer_pin=outer_pin)
    topo = sk.topology()
    assert topo.n == 22
    assert len(topo.faces) == 9
    assert len(topo.pins) >= (8 if outer_pin else 0)


def test_onion_specs_hit_n_exactly():
    for n in (17, 24, 41, 50):
        specs = N.onion_specs(n)
        assert specs
        for core, rings in specs:
            assert core + sum(s + m for s, m in rings) == n


def test_onion_generator_yields_rule_passing_nets():
    n = 30
    ctx = {"min_faces": R.min_faces_for(11.53492)}
    rs = R.RuleSet()
    legal = [sk for sk, fam, _ in N._gen_onion(n, np.random.default_rng(0), {})
             if rs.check(R.NetCandidate(n=n, topo=sk.topology(), coords=sk.coords(), family=fam), ctx) is None]
    assert legal
