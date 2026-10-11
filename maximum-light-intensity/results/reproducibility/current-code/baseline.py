#!/usr/bin/env python3
"""Archive Friedman's problem page and extract source-aware published baselines.

The digitized point set is only a seed: a small GIF cannot encode the published
configuration at enough precision to certify its displayed objective.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.request import urlopen

SOURCE_URL = "https://erich-friedman.github.io/packing/light/"
HERE = Path(__file__).resolve().parent


def archive(destination: Path, refresh: bool = False) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    index = destination / "index.html"
    if refresh or not index.exists():
        index.write_bytes(urlopen(SOURCE_URL, timeout=30).read())
    text = index.read_text()
    names = re.findall(r'<img src="([^"]+)"', text, flags=re.I)
    def fetch(name):
        path = destination / name
        if refresh or not path.exists():
            path.write_bytes(urlopen(SOURCE_URL + name, timeout=30).read())
        return {"file": name, "url": SOURCE_URL + name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(fetch, names))
    files.insert(0, {"file": "index.html", "url": SOURCE_URL,
                    "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                    "bytes": index.stat().st_size})
    manifest_path = destination / "manifest.json"
    retrieved = datetime.now(timezone.utc).isoformat()
    if manifest_path.exists() and not refresh:
        retrieved = json.loads(manifest_path.read_text())["retrieved_at_utc"]
    manifest = {"retrieved_at_utc": retrieved, "source_url": SOURCE_URL, "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    entries = {}
    pattern = r'<font size=\+3>(\d+)\.<br><img src="([^"]+)"[^>]*><br><font size=\+1>I = ([^<]+)<br>(.*?)(?=<td|</TR>)'
    for m in re.finditer(pattern, text, flags=re.I | re.S):
        n, image, displayed, attribution = m.groups()
        displayed = displayed.strip()
        value = float(displayed.split("=")[-1].strip().rstrip("+"))
        attribution = re.sub(r"<[^>]+>", " ", attribution)
        attribution = " ".join(html.unescape(attribution).split())
        entries[n] = {"n": int(n), "displayed": displayed, "value": value,
                      "has_trailing_plus": displayed.endswith("+"),
                      "attribution": attribution, "image": image,
                      "exact_coordinates_published": False}
    if len(entries) != len(names):
        raise ValueError(f"Parsed {len(entries)} entries from {len(names)} source images")
    baseline = {"source_url": SOURCE_URL, "retrieved_at_utc": retrieved,
                "largest_published_n": max(map(int, entries)),
                "comparison_caveat": "Trailing plus values are displayed lower thresholds, not exact records. No baseline is inferred for unlisted n.",
                "entries": entries}
    (destination / "published.json").write_text(json.dumps(baseline, indent=2) + "\n")
    return baseline


def digitize36(destination: Path) -> dict:
    import numpy as np
    from PIL import Image
    from scipy.ndimage import center_of_mass, label
    pixels = np.asarray(Image.open(destination / "36.gif").convert("RGB")).astype(float)
    red = ((pixels[:, :, 0] > 100) & (pixels[:, :, 0] > pixels[:, :, 1] + 60)
           & (pixels[:, :, 0] > pixels[:, :, 2] + 60))
    blue = ((pixels[:, :, 2] > 100) & (pixels[:, :, 2] > pixels[:, :, 1] + 60)
            & (pixels[:, :, 2] > pixels[:, :, 0] + 60))
    def centroids(mask):
        labels, count = label(mask)
        return np.asarray(center_of_mass(mask, labels, np.arange(1, count + 1)))
    red_rc, blue_rc = centroids(red), centroids(blue)
    if len(red_rc) != 36:
        raise ValueError(f"Expected 36 red light components; got {len(red_rc)}")
    # Centers of blue corner markers, rounded to the image's half-pixel raster.
    # The axes map pixels x=7.5..224.5 and y=5.5..222.5 to the closed unit square.
    x0, y0, x1, y1 = 7.5, 5.5, 224.5, 222.5
    def normalize(rc):
        return np.clip(np.column_stack(((rc[:, 1] - x0) / (x1 - x0),
                                       (y1 - rc[:, 0]) / (y1 - y0))), 0, 1).tolist()
    points = normalize(red_rc)
    result = {"n": 36, "points": points, "witnesses": normalize(blue_rc),
              "metadata": {"kind": "approximate_raster_seed", "source_url": SOURCE_URL + "36.gif",
                           "source_image_sha256": hashlib.sha256((destination / "36.gif").read_bytes()).hexdigest(),
                           "pixel_centroids_row_column": red_rc.tolist(),
                           "unit_square_pixel_bounds_xyxy": [x0, y0, x1, y1],
                           "coordinate_precision_warning": "Raster digitization only; nominal uncertainty roughly one image pixel / 217. This is not exact published data and has no certified objective."}}
    (destination / "seed36.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=HERE / "source")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = archive(args.source_dir, args.refresh)
    seed = digitize36(args.source_dir)
    print(f"Archived {len(result['entries'])} published entries; digitized {len(seed['points'])} lights.")

if __name__ == "__main__":
    main()
