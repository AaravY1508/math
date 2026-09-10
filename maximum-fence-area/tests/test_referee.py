"""The referee must be exact on known configurations and must reject illegal ones."""
import ast
from pathlib import Path

import pytest
from mpmath import mp, mpf

from fence.referee import verify
from tests import configs as C

TOL = 1e-9


def area(res):
    with mp.workdps(50):
        return mpf(res.total_area)


@pytest.mark.parametrize("name,segs,expected,faces", [
    ("triangle", C.TRIANGLE, C.TRIANGLE_AREA, 1),
    ("square", C.SQUARE, 1, 1),
    ("split_square_T", C.SPLIT_SQUARE, 1, 2),
    ("domino", C.DOMINO, 2, 2),
    ("grid_2x2", C.GRID_2x2, 4, 4),
    ("grid_2x3", C.GRID_2x3, 6, 6),
])
def test_known_configs_exact(name, segs, expected, faces):
    res = verify({"segments": segs}, TOL)
    assert res.valid, res.summary()
    assert res.n_faces == faces
    with mp.workdps(50):
        assert abs(area(res) - expected) < mpf("1e-40"), (name, res.total_area)
    # far tighter than the 1e-12 the spec asks for
    assert abs(float(res.total_area) - float(expected)) < 1e-12


def test_split_square_faces_are_halves():
    res = verify({"segments": C.SPLIT_SQUARE}, TOL)
    assert sorted(float(a) for a in res.face_areas) == [0.5, 0.5]
    assert res.n_tjunctions == 2


def test_float_coordinates_accepted_within_tol():
    import math
    segs = [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.5, math.sqrt(3) / 2], [0.5, math.sqrt(3) / 2, 0.0, 0.0]]
    res = verify(segs, TOL)
    assert res.valid, res.summary()
    assert abs(res.area - math.sqrt(3) / 4) < 1e-12


@pytest.mark.parametrize("name,segs,needle", [
    ("crossing", C.CROSSING, "cross"),
    ("hexagon_face_too_big", C.HEXAGON, "> 1"),
    ("lopsided_T_face_too_big", C.LOPSIDED, "> 1"),
    ("dangling", C.DANGLING, "not on any other fence"),
    ("overlap", C.OVERLAP, "overlap"),
    ("bad_length", C.BAD_LENGTH, "length"),
])
def test_rejections(name, segs, needle):
    res = verify({"segments": segs}, TOL)
    assert not res.valid, f"{name} was accepted: {res.summary()}"
    assert any(needle in r for r in res.reasons), res.reasons


def test_hexagon_area_reported_exactly_even_when_rejected():
    res = verify({"segments": C.HEXAGON}, TOL)
    with mp.workdps(50):
        assert abs(mpf(res.face_areas[0]) - C.HEXAGON_AREA) < mpf("1e-15")


def test_face_with_hole_is_subtracted():
    """A component nested inside a face must be subtracted from that face."""
    res = verify({"segments": C.HEX_WITH_HOLE}, TOL)
    assert not res.valid                                   # the holed face is ~1.598 > 1
    fa = sorted(float(a) for a in res.face_areas)
    assert abs(fa[0] - 1.0) < 1e-12
    assert abs(fa[1] - (float(C.HEXAGON_AREA) - 1.0)) < 1e-12
    assert abs(res.area - float(C.HEXAGON_AREA)) < 1e-12   # not 3.598: no double counting


def test_translation_rotation_invariance():
    import math
    th = 0.7
    c, s = math.cos(th), math.sin(th)
    rot = [[3 + c * x1 - s * y1, -2 + s * x1 + c * y1, 3 + c * x2 - s * y2, -2 + s * x2 + c * y2]
           for x1, y1, x2, y2 in [[float(v) for v in q] for q in C.SPLIT_SQUARE]]
    res = verify(rot, TOL)
    assert res.valid, res.summary()
    assert abs(res.area - 1.0) < 1e-12


def test_n_mismatch_rejected():
    res = verify({"n": 5, "segments": C.SQUARE}, TOL)
    assert not res.valid


def test_referee_shares_no_code_with_optimizer():
    """Import policy: the referee may only use the standard library and mpmath."""
    src = Path(__file__).resolve().parents[1] / "fence" / "referee.py"
    tree = ast.parse(src.read_text())
    allowed = {"__future__", "json", "dataclasses", "functools", "mpmath"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in referee"
            names = [node.module.split(".")[0]]
        else:
            continue
        for nm in names:
            assert nm in allowed, f"referee imports {nm!r}"
