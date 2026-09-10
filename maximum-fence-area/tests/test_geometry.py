"""Float optimization-path geometry: DCEL faces, topology validation, torch residuals."""
import math

import numpy as np
import pytest

from fence import geometry as G
from tests import configs as C


def load(segs):
    s = np.array([[float(v) for v in q] for q in segs])
    coords, fences, pins = G.topology_from_segments(s, tol=1e-9)
    return coords, fences, pins


@pytest.mark.parametrize("name,segs,expected,faces", [
    ("triangle", C.TRIANGLE, math.sqrt(3) / 4, 1),
    ("square", C.SQUARE, 1.0, 1),
    ("split_square_T", C.SPLIT_SQUARE, 1.0, 2),
    ("domino", C.DOMINO, 2.0, 2),
    ("grid_2x2", C.GRID_2x2, 4.0, 4),
    ("grid_2x3", C.GRID_2x3, 6.0, 6),
])
def test_faces_and_areas_exact(name, segs, expected, faces):
    coords, fences, pins = load(segs)
    topo = G.make_topology(coords, fences, pins)
    assert len(topo.faces) == faces
    fa = G.face_areas(coords, topo.faces)
    assert np.all(fa > 0)
    assert abs(fa.sum() - expected) < 1e-12
    # outer boundary is clockwise with the same magnitude
    assert abs(G.polygon_signed_area(coords[topo.outer]) + expected) < 1e-12


def test_split_square_has_two_pins_and_half_faces():
    coords, fences, pins = load(C.SPLIT_SQUARE)
    assert len(pins) == 2
    topo = G.make_topology(coords, fences, pins)
    assert sorted(np.round(G.face_areas(coords, topo.faces), 12)) == [0.5, 0.5]
    # the host fences are split into two edges each
    assert sorted(len(c) for c in topo.chains) == [2, 2, 2, 3, 3]


def test_trace_faces_euler_on_t_junction_graph():
    # a "comb": square with two stems from the bottom to the top (non-unit is fine here)
    coords = np.array([[0, 0], [3, 0], [3, 1], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], float)
    edges = [(0, 4), (4, 6), (6, 1), (1, 2), (2, 7), (7, 5), (5, 3), (3, 0), (4, 5), (6, 7)]
    tr = G.trace_faces(coords, edges)
    assert len(tr.cycles) == len(edges) - 8 + 2
    pos = sorted(a for a in tr.areas if a > 0)
    assert np.allclose(pos, [1, 1, 1]) and np.isclose(min(tr.areas), -3)


def test_crossing_rejected_by_topology_builder():
    coords, fences, pins = load(C.SQUARE)
    # add two crossing chords with pins on the square's sides
    coords = np.vstack([coords, [[0, .5], [1, .5], [.5, 0], [.5, 1]]])
    V = len(coords)
    fences = np.vstack([fences, [[V - 4, V - 3], [V - 2, V - 1]]])
    sq = {tuple(np.round(coords[a], 6)) : i for i, (a, b) in enumerate(fences[:4])}
    pins = {}
    for v in range(V - 4, V):
        for h in range(4):
            a, b = fences[h]
            if G.point_segment_distance(coords[v], coords[a], coords[b]) < 1e-12:
                pins[v] = h
    with pytest.raises(G.InvalidTopology):
        G.make_topology(coords, fences, pins)


def test_dangling_rejected():
    coords, fences, pins = load(C.DANGLING)
    with pytest.raises(G.InvalidTopology):
        G.make_topology(coords, fences, pins)


def test_torch_residuals_zero_on_valid_config():
    torch = pytest.importorskip("torch")
    for segs, expected in [(C.SPLIT_SQUARE, 1.0), (C.GRID_2x2, 4.0), (C.TRIANGLE, math.sqrt(3) / 4)]:
        coords, fences, pins = load(segs)
        topo = G.make_topology(coords, fences, pins)
        tt = G.TorchTopology(topo, dtype=torch.float64)
        X = torch.as_tensor(coords)[None]
        T = tt.initial_t(X)
        Z = tt.pack(X, T)
        r = tt.residuals(Z)
        for k in ("length", "pin", "cross", "order", "turn", "face_pos"):
            assert float(r[k].abs().max()) < 1e-12 if r[k].numel() else True, k
        assert float(r["area"].max()) < 1e-12
        assert abs(float(r["_areas"].sum()) - expected) < 1e-12


def test_torch_crossing_penalty_detects_crossing():
    torch = pytest.importorskip("torch")
    # two disjoint unit squares, then drag one so a fence crosses the other square
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [3, 0], [4, 0], [4, 1], [3, 1]], float)
    fences = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]])
    # join them with a fence so the graph is connected: pins at the middles of sides 1 and 7
    coords = np.vstack([coords, [[1, .5], [3, .5]]])
    fences = np.vstack([fences, [[8, 9]]])
    topo = G.make_topology(coords, fences, {8: 1, 9: 7})
    tt = G.TorchTopology(topo, dtype=torch.float64)
    X = torch.as_tensor(coords)[None].clone()
    Z = tt.pack(X, tt.initial_t(X))
    assert float(tt.residuals(Z)["cross"].max()) == 0.0
    X2 = X.clone()
    X2[0, 4:8, 0] -= 2.5          # second square now overlaps the first
    Z2 = tt.pack(X2, tt.initial_t(X2))
    assert float(tt.residuals(Z2)["cross"].max()) > 0.0


def test_rotational_symmetry_detection():
    s = np.array([[float(v) for v in q] for q in C.GRID_2x2])
    assert G.rotational_symmetry_order(s) == 4
    s = np.array([[float(v) for v in q] for q in C.DOMINO])
    assert G.rotational_symmetry_order(s) == 2
