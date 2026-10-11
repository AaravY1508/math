"""transplant.py: seeds one fence away from the n-1 / n+1 layouts, legal and distinct."""
import numpy as np

import transplant as TP
from fence import builder as Bd
from fence import minima as Mn
from fence import netevolve as NE
from fence import optimize as O


def sk_of(rows, cols):
    g = Bd.grid(rows, cols)
    return Bd.Sketch.from_config(g.topology(), g.coords())


def test_up_and_down_give_distinct_legal_n_fence_layouts():
    # n = 17: up from the 2x3 grid is impossible (that is 17 already), so use 2x2+1 row pieces:
    # up from n=12 (2x2 grid) to 13, down from n=17 (2x3 grid) to 16
    for src, n, d in ((sk_of(2, 2), 13, "up"), (sk_of(2, 3), 16, "down")):
        ev = NE.NetEvolver(n, src, ctx={}, cfg=O.OptConfig(batch=8, steps=5), device="cpu")
        seen = set()
        got = TP.transplants(ev, src, n, d, 12, seen)
        assert got, d
        sigs = [s for _, s, _ in got]
        assert len(sigs) == len(set(sigs)) and set(sigs) <= seen
        for how, sig, sk in got:
            assert sk.n == n and how.startswith(d)
            assert Mn.canonical_form(sk.topology()).signature == sig
        again = TP.transplants(ev, src, n, d, 12, seen)
        assert again == []                              # same rng seeds: nothing new
