"""Summarize a run: table of n | best_known | ours | delta | verified, plus SVGs.

    ./venv/bin/python report.py runs/<timestamp>            # print table, write report + SVGs
    ./venv/bin/python report.py runs/<timestamp> --reverify # re-run the 50-digit referee on every best
    ./venv/bin/python report.py runs/<timestamp> --svg-all  # also render configurations below baseline

Status column:
  BEATS  verified area exceeds the largest value consistent with the page's printed figure
         (printed value + one unit in its last decimal when printed with '+'), plus 1e-7
  TIES   verified area >= the printed figure (could equal or be just under the true record)
  NEAR   verified, within 1e-3 below the printed figure
  BELOW  verified, further below
A result only ever counts if the independent referee (fence/referee.py, mpmath 50
digits) accepted it. Unverified numbers are never shown as results.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fence.evolve import classify, load_baselines  # noqa: E402


def _reverify(path: str, tol: float):
    from fence.referee import verify
    cfg = json.loads(Path(path).read_text())
    return verify(cfg, tol).to_json()


def render_svg(cfg: dict, title: str, size: int = 520) -> str:
    segs = [[float(v) for v in s] for s in cfg["segments"]]
    xs = [p for s in segs for p in (s[0], s[2])]
    ys = [p for s in segs for p in (s[1], s[3])]
    pad = 0.35
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    scale = (size - 20) / max(x1 - x0, y1 - y0)
    W, H = int((x1 - x0) * scale) + 20, int((y1 - y0) * scale) + 60

    def tx(x):
        return 10 + (x - x0) * scale

    def ty(y):
        return 50 + (y1 - y) * scale

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="10" y="20" font-family="monospace" font-size="13">{html.escape(title)}</text>']
    topo = cfg.get("topology")
    verts = cfg.get("vertices")
    if topo and verts:
        V = [[float(a), float(b)] for a, b in verts]
        for cyc in topo.get("faces", []):
            pts = " ".join(f"{tx(V[v][0]):.2f},{ty(V[v][1]):.2f}" for v in cyc)
            out.append(f'<polygon points="{pts}" fill="#dbe8fb" stroke="none"/>')
        areas = cfg.get("verification", {}).get("face_areas", [])
        if len(areas) == len(topo.get("faces", [])):
            for cyc, a in zip(topo["faces"], areas):
                cx = sum(V[v][0] for v in cyc) / len(cyc)
                cy = sum(V[v][1] for v in cyc) / len(cyc)
                out.append(f'<text x="{tx(cx):.1f}" y="{ty(cy):.1f}" font-family="monospace" font-size="9" '
                           f'text-anchor="middle" fill="#445">{float(a):.4f}</text>')
    for s in segs:
        out.append(f'<line x1="{tx(s[0]):.2f}" y1="{ty(s[1]):.2f}" x2="{tx(s[2]):.2f}" y2="{ty(s[3]):.2f}" '
                   f'stroke="black" stroke-width="2.2" stroke-linecap="round"/>')
    if topo and verts:
        for v, _h in topo.get("pins", []):
            out.append(f'<circle cx="{tx(V[v][0]):.2f}" cy="{ty(V[v][1]):.2f}" r="2.6" fill="#d22"/>')
    out.append("</svg>")
    return "\n".join(out)


def build_report(run_dir: Path, reverify: bool = False, svg_all: bool = False, tol: float = 1e-9,
                 time_limit: float = 150.0) -> dict:
    t0 = time.time()
    base = load_baselines()
    best_dir = run_dir / "best"
    files = sorted(best_dir.glob("n*.json")) if best_dir.exists() else []
    rows = {}
    for f in files:
        cfg = json.loads(f.read_text())
        n = int(cfg["n"])
        ver = cfg.get("verification", {})
        rows[n] = {"file": str(f), "cfg": cfg, "verified": bool(ver.get("valid")),
                   "area": float(ver["total_area"]) if ver.get("valid") else None,
                   "area_50": ver.get("total_area"), "reverified": None}
    if reverify and rows:
        with ProcessPoolExecutor(max_workers=min(16, len(rows))) as ex:
            futs = {n: ex.submit(_reverify, r["file"], tol) for n, r in rows.items()}
            for n, fu in futs.items():
                left = time_limit - (time.time() - t0)
                try:
                    res = fu.result(timeout=max(1.0, left))
                    rows[n]["reverified"] = bool(res["valid"])
                    rows[n]["verified"] = bool(res["valid"])
                    rows[n]["area"] = float(res["total_area"]) if res["valid"] else None
                    rows[n]["area_50"] = res["total_area"]
                except FutTimeout:
                    rows[n]["reverified"] = None      # keep the worker's original referee verdict
    ns = sorted(set(base) | set(rows))
    table = []
    for n in ns:
        b = base.get(n)
        r = rows.get(n)
        area = r["area"] if r and r["verified"] else None
        st = classify(area, b)
        table.append({"n": n, "best_known": b["A"] if b else None, "best_known_display": b["display"] if b else None,
                      "site_status": b["status"] if b else None, "ours": area,
                      "ours_50_digits": r["area_50"] if r else None,
                      "delta": (area - b["A"]) if (area is not None and b) else None,
                      "verified": bool(r and r["verified"]), "reverified": r["reverified"] if r else None,
                      "status": st if r else "NOT_RUN"})
    # SVGs
    svg_dir = run_dir / "svg"
    svg_dir.mkdir(exist_ok=True)
    written = []
    for row in table:
        r = rows.get(row["n"])
        if not r or not row["verified"]:
            continue
        good = row["status"] in ("BEATS", "TIES")
        if good or svg_all:
            sub = svg_dir if good else svg_dir / "all"
            sub.mkdir(exist_ok=True)
            title = (f"n={row['n']}  A={row['ours']:.8f}  best known {row['best_known_display']}  "
                     f"[{row['status']}, referee-verified 50 digits]")
            p = sub / f"n{row['n']:03d}_{row['status']}.svg"
            p.write_text(render_svg(r["cfg"], title))
            written.append(str(p))
    rep = {"run_dir": str(run_dir), "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": table, "svgs": written,
           "improvements": [row for row in table if row["status"] == "BEATS"],
           "ties": [row for row in table if row["status"] == "TIES"]}
    return rep


def format_table(rep: dict, only_run: bool = True) -> str:
    lines = [f"{'n':>3} | {'best_known':>10} | {'ours':>12} | {'delta':>11} | {'verified':>8} | status",
             "-" * 66]
    for row in rep["rows"]:
        if only_run and row["status"] == "NOT_RUN":
            continue
        bk = row["best_known_display"] or "-"
        ours = f"{row['ours']:.8f}" if row["ours"] is not None else "-"
        delta = f"{row['delta']:+.6f}" if row["delta"] is not None else "-"
        lines.append(f"{row['n']:>3} | {bk:>10} | {ours:>12} | {delta:>11} | {str(row['verified']):>8} | {row['status']}")
    return "\n".join(lines)


def loud_summary(rep: dict) -> str:
    imp = rep["improvements"]
    bar = "#" * 78
    if imp:
        body = [bar, f"#  {len(imp)} VERIFIED IMPROVEMENT(S) OVER THE BEST KNOWN AREA  (referee: mpmath, 50 digits)", bar]
        for row in imp:
            body.append(f"#  n={row['n']}: ours {row['ours_50_digits'][:24]}  vs best known {row['best_known_display']}"
                        f"  (delta {row['delta']:+.7f})")
        body.append(bar)
    else:
        body = [bar, "#  NO IMPROVEMENTS over the best known areas in this run.", bar]
    if rep["ties"]:
        body.append("#  ties with the printed record value: " + ", ".join(f"n={r['n']}" for r in rep["ties"]))
        body.append(bar)
    return "\n".join(body)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--reverify", action="store_true", help="re-run the referee on every best configuration")
    ap.add_argument("--svg-all", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--time-limit", type=float, default=150.0)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    rep = build_report(run_dir, reverify=args.reverify, svg_all=args.svg_all, tol=args.tol, time_limit=args.time_limit)
    txt = format_table(rep) + "\n\n" + loud_summary(rep) + "\n"
    (run_dir / "report.txt").write_text(txt)
    (run_dir / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(txt)
    return rep


if __name__ == "__main__":
    main()
