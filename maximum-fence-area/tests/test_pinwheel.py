"""The pinwheel topology move (fence.builder.pinwheel_at, netevolve move `pinwheel`, off by default).

n=17's record (6.01086+) is a ring of five cells around a central one, with a pinwheel -- three fences
each stopping on the next around a triangle of area 0.0109 -- where the ring layout would otherwise
have a three-way joint. Collapsing that triangle back to the joint gives a 6-face layout (6.0, the
2x3 grid's value); the move must rebuild the 7-face record topology from it."""
import json
from pathlib import Path

import numpy as np
import pytest

from fence import builder as Bd
from fence import geometry as G
from fence import minima as Mn
from fence import netevolve as NE
from fence import optimize as O

DATA = Path(__file__).parent / "data"


def record17():
    return G.config_from_json(json.load(open(DATA / "record-17-config.json")))


def collapse_smallest_face(topo, X) -> Bd.Sketch:
    A = G.face_areas(X, topo.faces)
    f = topo.faces[int(np.argmin(A))]
    sk = Bd.Sketch.from_config(topo, X)
    w = sk.add_vertex(*sk.coords()[f].mean(0))
    for v in f:
        sk.pins.pop(v, None)
    sk.fences = [[w if u in f else u for u in fe] for fe in sk.fences]
    sk.compact()
    return sk


def test_move_is_registered_off_by_default():
    assert NE.MOVES["pinwheel"].kind == "topology"
    assert "pinwheel" not in NE.MoveSet().enabled()
    assert "pinwheel" in NE.MoveSet({"pinwheel": 1.0}).enabled()


def test_pinwheel_rebuilds_the_n17_record_topology():
    topo, X = record17()
    assert topo.features() == {"faces": 7, "tjunctions": 12, "joints": 11}
    y = collapse_smallest_face(topo, X)
    assert y.topology().features() == {"faces": 6, "tjunctions": 9, "joints": 12}
    target = Mn.canonical_form(topo).signature
    ends = Bd._ends_at(y)
    made = []
    for v, fs in ends.items():
        if len(fs) >= 3 and v not in y.pins:
            pw = Bd.pinwheel_at(y, v, 0.05)
            if pw is not None:
                t = pw.topology()
                assert t.n == 17 and t.features()["faces"] == 7 and t.features()["joints"] == 11
                assert min(G.face_areas(pw.coords(), t.faces)) < 0.02
                made.append(Mn.canonical_form(t).signature)
    assert target in made


def test_pinwheel_needs_every_gap_below_half_a_turn():
    sk = Bd.grid(2, 3)
    ends = Bd._ends_at(sk)
    X = sk.coords()
    for v, fs in ends.items():
        if len(fs) == 3:           # boundary joints of the grid: two ends are collinear
            assert Bd.pinwheel_at(sk, v, 0.05) is None
        if len(fs) == 4:           # interior joints: a tiny square
            t = Bd.pinwheel_at(sk, v, 0.05).topology()
            assert t.features() == {"faces": 7, "tjunctions": 4, "joints": 11}
    assert Bd.pinwheel(sk.copy(), np.random.default_rng(0)) is not None


@pytest.mark.slow
def test_pinwheel_optimizes_to_the_record_on_cpu():
    topo, X = record17()
    y = collapse_smallest_face(topo, X)
    for v, fs in Bd._ends_at(y).items():
        if len(fs) >= 3 and v not in y.pins and (pw := Bd.pinwheel_at(y, v, 0.05)) is not None:
            if Mn.canonical_form(pw.topology()).signature == Mn.canonical_form(topo).signature:
                r = O.optimize_topology(pw.topology(), pw.coords(), 17, O.OptConfig(batch=64, steps=600),
                                        device="cpu", seed=0)
                assert r["best"] is not None and r["best"]["area"] >= 6.0108
                return
    pytest.fail("no pinwheel rebuilt the record topology")


def test_project_pins_settles_pins_exactly_and_can_clamp():
    topo, X = record17()
    sk = Bd.Sketch.from_config(topo, X)
    rng = np.random.default_rng(1)
    sk.P = (sk.coords() + 1e-3 * rng.standard_normal(sk.coords().shape)).tolist()
    sk._dirty()
    Bd.project_pins(sk)
    sk.topology()                                       # legal again: every pin back on its host
    Y = sk.coords()
    for v, h in sk.pins.items():
        a, b = sk.fences[h]
        d = Y[b] - Y[a]
        assert abs(d[0] * (Y[v] - Y[a])[1] - d[1] * (Y[v] - Y[a])[0]) < 1e-12
    v, h = next(iter(sk.pins.items()))
    a, b = sk.fences[h]
    sk.P[v] = (Y[a] + 1.3 * (Y[b] - Y[a])).tolist()     # past the host's end
    Bd.project_pins(sk, clamp=0.02)
    t = G.fence_param(sk.coords(), a, b, v)
    assert abs(t - 0.98) < 1e-9
