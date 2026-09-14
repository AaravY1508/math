"""Record images (render_records.py) and the face labels of report.render_svg.

The referee returns `face_areas` in its own face order, which differs from the topology's face
order at 20 of the 33 library records (2026-09-14). A renderer that pairs the topology's faces
with the referee's list writes the wrong area into the wrong face; report.render_svg did exactly
that. Both renderers now recompute each face's area from the face they draw.
"""
import json
import re
import xml.etree.ElementTree as ET

import pytest

import render_records as R
import report
from fence import geometry as G
from tests.test_face_edits import RECORDS
from tests.test_library_seeding import verified_record

SVG = {"s": "http://www.w3.org/2000/svg"}


def shoelace(pts):
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])))


def misordered_record():
    """A fixture record whose referee face order differs from its topology's."""
    for n in sorted(RECORDS, key=int):
        cfg = verified_record(int(n))
        topo, X = G.config_from_json(cfg)
        aligned = [float(a) for a in G.face_areas(X, topo.faces)]
        ref = [float(a) for a in cfg["verification"]["face_areas"]]
        if any(abs(a - b) > 1e-6 for a, b in zip(aligned, ref)):
            return cfg, aligned, ref
    pytest.skip("no fixture record with a differing face order")


def test_each_face_carries_its_own_area():
    cfg, aligned, ref = misordered_record()
    lay = R.layout(cfg)
    assert lay["areas"] == pytest.approx(aligned)
    for pts, a in zip(lay["faces"], lay["areas"]):
        assert shoelace(pts) == pytest.approx(a, abs=1e-9)
    assert sorted(lay["areas"]) == pytest.approx(sorted(ref))      # same faces, different order
    assert lay["areas"] != pytest.approx(ref)


def test_panel_draws_every_face_fence_and_pin():
    cfg = verified_record(20)
    lay = R.layout(cfg)
    body, W, H = R.panel(lay, 400, ["t"])
    root = ET.fromstring(R.svg(body, W, H))
    assert len(root.findall(".//s:polygon", SVG)) == lay["n_faces"]
    assert len(root.findall(".//s:line", SVG)) == 20
    assert len(root.findall(".//s:circle", SVG)) == len(lay["pins"])
    labels = [float(t.text) for t in root.findall(".//s:text", SVG) if re.fullmatch(r"\d\.\d{3}", t.text or "")]
    assert labels == pytest.approx([round(a, 3) for a in lay["areas"] if a >= 0.05], abs=6e-4)


def test_report_svg_labels_follow_the_drawn_faces():
    cfg, aligned, _ = misordered_record()
    root = ET.fromstring(report.render_svg(cfg, "t"))
    labels = [float(t.text) for t in root.findall(".//s:text", SVG) if re.fullmatch(r"\d\.\d{4}", t.text or "")]
    assert labels == pytest.approx([round(a, 4) for a in aligned], abs=6e-5)


def test_render_writes_one_image_per_n_and_a_contact_sheet(tmp_path):
    from tests.test_library_seeding import library_for_20
    lib, _ = library_for_20(tmp_path / "minima")
    res = R.render([20, 21], tmp_path / "minima", tmp_path / "out", size=300, thumb=150)
    assert res["rendered"] == 1 and res["missing"] == [21]
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"n020.svg", "records.svg"}
    for f in ("n020.svg", "records.svg"):
        ET.parse(tmp_path / "out" / f)                               # well-formed


def test_a_record_whose_areas_do_not_add_up_is_refused():
    cfg = json.loads(json.dumps(verified_record(20)))
    cfg["verification"]["total_area"] = "7.5"
    with pytest.raises(ValueError):
        R.layout(cfg)
