"""Render the current record-holder layout of each n as an SVG image.

    ./venv/bin/python render_records.py                         # the 33 non-grid n -> runs/record-images/
    ./venv/bin/python render_records.py --ns 20,34 --out /tmp/imgs
    ./venv/bin/python render_records.py --minima-dir minima-snapshots/20260914

Writes one SVG per n (n034.svg) and a contact sheet of all of them (records.svg). Reads the
minima library's records (minima/records/N<n>.json): the best verified layout for n over all
runs. Nothing here certifies anything; the areas shown are the ones the referee verified.

Each face is shaded by how full it is (area / 1, the most a face may hold): full faces light
blue, near-full paler, under-filled (< 0.9) amber, near-empty (< 0.5) red, and labelled with its
area. Fences are black, T-junction pins red dots. Face areas are recomputed in the topology's own
face order: the referee lists `face_areas` in ITS face order, which differs from the topology's
at most records, so pairing the two lists would put areas in the wrong faces.
"""
from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fence import evolve as E        # noqa: E402
from fence import geometry as G      # noqa: E402
from fence import minima as Mn       # noqa: E402

NON_GRID = [13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37,
            38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 50]
DEAD = 0.9


def fill_colour(a: float) -> str:
    if a >= 0.999:
        return "#cfe3f7"      # full
    if a >= DEAD:
        return "#e4edf8"      # near-full
    if a >= 0.5:
        return "#f6d98f"      # under-filled
    return "#f2aaa3"          # near-empty


def centroid(pts) -> tuple[float, float]:
    """Area centroid of a simple polygon (the vertex mean can fall outside a concave face)."""
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        c = x0 * y1 - x1 * y0
        a += c
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    if abs(a) < 1e-12:
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return cx / (3 * a), cy / (3 * a)


def layout(cfg: dict) -> dict:
    """What a drawing needs, with face areas aligned to the faces drawn."""
    topo, X = G.config_from_json(cfg)
    faces = [[(float(X[v][0]), float(X[v][1])) for v in cyc] for cyc in topo.faces]
    areas = [float(a) for a in G.face_areas(X, topo.faces)]
    total = float(cfg["verification"]["total_area"])
    if abs(sum(areas) - total) > 1e-6:
        raise ValueError(f"recomputed face areas sum to {sum(areas):.9f}, referee says {total:.9f}")
    segs = [[float(v) for v in s] for s in cfg["segments"]]
    pins = [(float(X[v][0]), float(X[v][1])) for v in topo.pins]
    return {"faces": faces, "areas": areas, "segs": segs, "pins": pins, "total": total,
            "n": int(cfg["n"]), "n_faces": len(faces)}


def panel(lay: dict, size: float, title: list[str], labels: bool = True) -> tuple[str, float, float]:
    """(SVG body, width, height) of one layout drawn into a size x size box under a title."""
    xs = [c for s in lay["segs"] for c in (s[0], s[2])]
    ys = [c for s in lay["segs"] for c in (s[1], s[3])]
    pad = 0.25
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    k = size / max(x1 - x0, y1 - y0)
    head = 16 * len(title) + 8
    W, H = (x1 - x0) * k, (y1 - y0) * k + head

    def P(x, y):
        return f"{(x - x0) * k:.2f},{head + (y1 - y) * k:.2f}"

    out = [f'<rect width="{W:.1f}" height="{H:.1f}" fill="white"/>']
    for i, line in enumerate(title):
        weight = ' font-weight="bold"' if i == 0 else ""
        out.append(f'<text x="4" y="{16 * (i + 1)}" font-family="monospace" font-size="{13 if i == 0 else 11}" '
                   f'fill="#223"{weight}>{html.escape(line)}</text>')
    for pts, a in zip(lay["faces"], lay["areas"]):
        out.append(f'<polygon points="{" ".join(P(*p) for p in pts)}" fill="{fill_colour(a)}" stroke="none"/>')
    for s in lay["segs"]:
        (ax, ay), (bx, by) = (P(s[0], s[1]).split(","), P(s[2], s[3]).split(","))
        out.append(f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" stroke="#111" '
                   f'stroke-width="{max(1.0, k * 0.022):.2f}" stroke-linecap="round"/>')
    r = max(1.4, k * 0.028)
    for x, y in lay["pins"]:
        cx, cy = P(x, y).split(",")
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="#d0342c"/>')
    if labels:
        fs = max(7.0, min(12.0, k * 0.16))
        for pts, a in zip(lay["faces"], lay["areas"]):
            if a < 0.05:
                continue
            cx, cy = P(*centroid(pts)).split(",")
            out.append(f'<text x="{cx}" y="{cy}" font-family="monospace" font-size="{fs:.1f}" '
                       f'text-anchor="middle" dominant-baseline="middle" fill="#334">{a:.3f}</text>')
    return "\n".join(out), W, H


def titles(n: int, lay: dict, base: dict) -> tuple[list[str], list[str]]:
    """(full title lines, compact title lines) for n's record layout."""
    b = base.get(n)
    tgt = E.target_faces(b)
    dead = sum(a < DEAD for a in lay["areas"])
    status = E.classify(lay["total"], b) if b else "NO_BASELINE"
    gap = lay["total"] - b["A"] if b else float("nan")
    full = [f"n = {n}   area {lay['total']:.8f}   {status}",
            f"record {b['display'] if b else '?'}   gap {gap:+.5f}",
            f"faces {lay['n_faces']} / {tgt} needed   under-filled {dead}   T-pins {len(lay['pins'])}"]
    short = [f"n={n}  {lay['total']:.5f}  {status}",
             f"gap {gap:+.4f}  F {lay['n_faces']}/{tgt}"]
    return full, short


def svg(body: str, W: float, H: float) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
            f'viewBox="0 0 {W:.1f} {H:.1f}">\n{body}\n</svg>\n')


def render(ns: list[int], minima_dir: Path, out: Path, size: float = 520, thumb: float = 250, cols: int = 6) -> dict:
    base = E.load_baselines()
    lib = Mn.MinimaLibrary(minima_dir)
    out.mkdir(parents=True, exist_ok=True)
    tiles, missing = [], []
    for n in ns:
        rec = lib.record(n)
        if rec is None:
            missing.append(n)
            continue
        lay = layout(rec["config"])
        full, short = titles(n, lay, base)
        body, W, H = panel(lay, size, full)
        (out / f"n{n:03d}.svg").write_text(svg(body, W, H))
        tiles.append(panel(lay, thumb, short, labels=False))
    # contact sheet: tiles on a grid, each cell as big as the largest tile
    cw = max(t[1] for t in tiles) + 12 if tiles else 0
    ch = max(t[2] for t in tiles) + 12 if tiles else 0
    rows = math.ceil(len(tiles) / cols)
    sheet = [f'<rect width="{cols * cw:.0f}" height="{rows * ch:.0f}" fill="#f4f5f7"/>']
    for i, (body, W, H) in enumerate(tiles):
        sheet.append(f'<g transform="translate({(i % cols) * cw + 6:.1f},{(i // cols) * ch + 6:.1f})">\n{body}\n</g>')
    (out / "records.svg").write_text(svg("\n".join(sheet), cols * cw, rows * ch))
    return {"out": str(out), "rendered": len(tiles), "missing": missing}


def main(argv=None):
    ap = argparse.ArgumentParser(description="SVG images of the record-holder layout of each n")
    ap.add_argument("--ns", default=",".join(map(str, NON_GRID)), help="n values (default: the 33 non-grid n)")
    ap.add_argument("--minima-dir", default=str(HERE / "minima"))
    ap.add_argument("--out", default=str(HERE / "runs" / "record-images"))
    args = ap.parse_args(argv)
    ns = sorted({int(x) for x in args.ns.split(",") if x.strip()})
    res = render(ns, Path(args.minima_dir), Path(args.out))
    print(f"{res['rendered']} layouts -> {res['out']}/n<NNN>.svg and records.svg"
          + (f"; no record for n = {res['missing']}" if res["missing"] else ""))


if __name__ == "__main__":
    main()
