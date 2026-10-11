"""fence/overlay.py on n=17: the record's own layout fits its picture tightly and flags nothing; a
layout with a displaced vertex scores high exactly there; picture_fit keeps the topology and brings
a distorted layout back onto the picture."""
import json
from pathlib import Path

import numpy as np

from fence import geometry as G
from fence import overlay as OV

DATA = Path(__file__).parent / "data"
GIF = str(DATA / "record-17.gif")


def cfg17():
    return json.load(open(DATA / "record-17-config.json"))


def test_umeyama_recovers_a_similarity():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 2))
    th, s, t = 0.7, 2.5, np.array([3.0, -1.0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    s2, R2, t2 = OV.umeyama(A, s * A @ R.T + t)
    assert abs(s2 - s) < 1e-9 and np.allclose(R2, R) and np.allclose(t2, t)


def test_record_fits_its_own_picture():
    rep = OV.score(cfg17(), GIF)
    assert rep["fit_cost_units"] < 0.01
    assert rep["layout_faces"] == rep["picture_faces"] == 7
    assert not rep["flagged"]["fences"] and not rep["flagged"]["branches"]
    assert max(rep["fence_score"]) < 0.01


def test_a_displaced_vertex_is_flagged_where_it_moved():
    cfg = cfg17()
    topo, X = G.config_from_json(cfg)
    joints = [v for v in range(len(X)) if v not in topo.pins and v in set(topo.outer)]
    v = joints[0]
    Y = X.copy()
    Y[v] += (Y[v] - Y.mean(0)) / np.linalg.norm(Y[v] - Y.mean(0)) * 0.3      # push it outwards
    rep = OV.score(G.config_to_json(topo, Y, n=17), GIF)
    moved = [f for f, (a, b) in enumerate(topo.fences) if v in (a, b)]
    worst = int(np.argmax(rep["fence_score"]))
    assert worst in moved and rep["fence_score"][worst] > 0.04
    assert any(np.hypot(x - Y[v][0], y - Y[v][1]) < 0.6 for x, y, _ in rep["focus"])


def test_picture_fit_pulls_a_distorted_layout_back():
    cfg = cfg17()
    topo, X = G.config_from_json(cfg)
    Y = X * np.array([1.08, 0.94])                     # anisotropic distortion, same topology
    c2 = G.config_to_json(topo, Y, n=17)
    before = OV.score(c2, GIF)["fit_cost_units"]
    Xf = OV.picture_fit(c2, GIF, steps=600)
    L = np.linalg.norm(Xf[topo.fences[:, 1]] - Xf[topo.fences[:, 0]], axis=1)
    after = OV.score(G.config_to_json(topo, Xf, n=17), GIF)["fit_cost_units"]
    # the drawing is to scale, so fitted fences come back near unit length (the pinwheel's two host
    # fences converge slowest from an 8% distortion)
    assert after < 0.5 * before and np.median(np.abs(L - 1)) < 0.01 and np.abs(L - 1).max() < 0.05
