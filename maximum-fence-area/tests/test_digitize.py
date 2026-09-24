"""The picture digitizer must turn a record's line drawing into a legal topology with the right
fence and face counts (EXPERIMENTS.md change 11)."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fence import digitize as D
from fence import geometry as G

DATA = Path(__file__).parent / "data"
needs_convert = pytest.mark.skipif(shutil.which("convert") is None, reason="ImageMagick `convert` not installed")


def _draw(tmp_path, segs, px=60, pad=12, width=3):
    """Render unit segments as a black-on-white GIF, as the fence page draws its records."""
    segs = np.asarray(segs, float)
    lo = segs.reshape(-1, 2).min(0)
    hi = segs.reshape(-1, 2).max(0)
    W, H = ((hi - lo) * px + 2 * pad).astype(int)
    draw = []
    for x1, y1, x2, y2 in segs:
        a = ((x1 - lo[0]) * px + pad, (hi[1] - y1) * px + pad)
        b = ((x2 - lo[0]) * px + pad, (hi[1] - y2) * px + pad)
        draw += ["-draw", f"line {a[0]:.1f},{a[1]:.1f} {b[0]:.1f},{b[1]:.1f}"]
    out = tmp_path / "pic.gif"
    subprocess.run(["convert", "-size", f"{W}x{H}", "xc:white", "-stroke", "black", "-strokewidth", str(width),
                    *draw, str(out)], check=True)
    return out


def test_cut_line_joints():
    # straight, length 2, nodes at 0.8 and 1.0: one piece of two fences (the breakpoint snaps later)
    coords = np.array([[0.0, 0], [0.8, 0], [1.0, 0], [2.0, 0]])
    s = np.array([0.0, 0.8, 1.0, 2.0])
    cuts, _ = D._cut_line(coords, [0, 1, 2, 3], s, bend_deg=3.0)
    assert cuts == [0, 3]
    # a forced joint is honoured
    cuts, _ = D._cut_line(coords, [0, 1, 2, 3], s, bend_deg=3.0, forced={1})
    assert 1 in cuts
    # a 30 degree bend after one unit is a joint between two fences
    c30 = np.array([[0.0, 0], [1.0, 0], [1 + np.cos(np.pi / 6), np.sin(np.pi / 6)]])
    cuts, _ = D._cut_line(c30, [0, 1, 2], np.array([0.0, 1.0, 2.0]), bend_deg=3.0)
    assert cuts == [0, 1, 2]


def test_fences_from_graph_domino():
    # domino: 2x1 rectangle split by a middle fence; graph drawn at an arbitrary pixel scale
    k = 37.0
    nodes = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1]], float) * k
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)]
    P, fences, pins, _ = D.fences_from_graph(nodes, edges, 7)
    topo = G.make_topology(P, fences, pins)
    assert topo.n == 7 and len(topo.faces) == 2 and not topo.pins


@needs_convert
def test_digitize_drawn_grid(tmp_path):
    # 2x2 grid of unit squares: 12 fences, 4 faces, no T-junctions
    segs = [[x, y, x + 1, y] for y in range(3) for x in range(2)] + [[x, y, x, y + 1] for x in range(3) for y in range(2)]
    topo, P, info = D.digitize(str(_draw(tmp_path, segs)), 12, min_edge=0.1)
    assert topo.n == 12
    assert topo.features()["faces"] == 4


@needs_convert
def test_digitize_record_17_picture():
    # the page's own n=17 picture (6.01086+): six unit-ish cells and a small pinwheel triangle
    topo, P, info = D.digitize(str(DATA / "record-17.gif"), 17, min_edge=0.1)
    assert topo.n == 17
    assert topo.features()["faces"] == 7


def test_graph_cleanup_helpers():
    # a T whose junction node sits 0.04 off the through-line is put back on it
    coords = np.array([[0.0, 0.0], [0.3, 0.04], [1.0, 0.0], [0.3, -1.0]])
    edges = [(0, 1), (1, 2), (1, 3)]
    C = D.straighten_tees(coords, edges, max_shift=0.1)
    assert abs(C[1, 1]) < 1e-12
    assert D._turn(C, 0, 1, 2) < 1e-6
    # ... but a real 30 degree bend is left alone
    bent = np.array([[0.0, 0.0], [1.0, 0.0], [1 + np.cos(np.pi / 6), np.sin(np.pi / 6)], [1.0, -1.0]])
    assert np.allclose(D.straighten_tees(bent, [(0, 1), (1, 2), (1, 3)], max_shift=0.1), bent)
    # two unconnected nodes 0.02 apart merge; a short dangling stub goes
    C, E = D.merge_close_nodes(np.array([[0.0, 0], [1, 0], [1.02, 0], [2, 0]]), [(0, 1), (2, 3)], 0.05)
    assert len(C) == 3 and E == [(0, 1), (1, 2)]
    C, E = D.drop_spurs(np.array([[0.0, 0], [1, 0], [1, 1], [0, 1], [1.1, 0.05]]),
                        [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4)], 0.3)
    assert len(E) == 4 and len(C) == 4
