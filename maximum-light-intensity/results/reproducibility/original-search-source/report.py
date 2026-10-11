#!/usr/bin/env python3
"""Create precise, source-aware submission artifacts from validated checkpoints.

Example:
  python3 report.py --checkpoints checkpoints --results results
Validation is intentionally external: each n requires a validation.json next to
its checkpoint or in the corresponding output folder. Missing/uncertified
validation is rejected unless --allow-unvalidated is explicitly supplied.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE_URL = "https://erich-friedman.github.io/packing/light/"


def format_bound(value: float, decimals: int = 9, upper: bool = False) -> str:
    """Round displayed certified endpoints outward, preserving their inequality."""
    rounding = ROUND_CEILING if upper else ROUND_FLOOR
    return format(Decimal.from_float(float(value)).quantize(Decimal(1).scaleb(-decimals), rounding=rounding), f".{decimals}f")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def points_hash(points: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(points, dtype="<f8").tobytes(order="C")).hexdigest()


def intensity(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    result = np.empty(len(queries))
    with np.errstate(divide="ignore", invalid="ignore"):
        for start in range(0, len(queries), 8192):
            displacement = queries[start:start + 8192, None, :] - points[None, :, :]
            result[start:start + 8192] = (1.0 / np.sum(displacement**2, axis=-1)).sum(axis=1)
    return result


def find_checkpoint(root: Path, n: int) -> Path:
    paths = [root / f"n{n}" / "best.json", root / str(n) / "best.json",
             root / f"best_{n}.json", root / f"best_n{n}.json", root / f"n{n}.json",
             root / f"{n}.json", root / f"n{n:02d}" / "best.json"]
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No checkpoint for n={n} under {root}")


def read_validation(checkpoint_path: Path, output_dir: Path, checkpoint: dict) -> tuple[dict, Path | None]:
    candidates = [output_dir / "validation.json", checkpoint_path.parent / "validation.json",
                  checkpoint_path.with_name(checkpoint_path.stem + "_validation.json")]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text()), path
    if isinstance(checkpoint.get("validation"), dict):
        return checkpoint["validation"], None
    return {"valid": False, "certified": False, "reason": "No validation supplied"}, None


def normalize_witnesses(checkpoint: dict, validation: dict) -> np.ndarray:
    candidates = []
    for obj in (checkpoint, validation):
        raw = list(obj.get("witnesses", [])) + list(obj.get("numerical_minima", []))
        if isinstance(raw, dict):
            raw = list(raw.values())
        for item in raw:
            if isinstance(item, dict):
                item = item.get("point", item.get("coordinates", item.get("position", [])))
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                candidates.append(item[:2])
        for name in ("witness", "worst_point", "minimum_point"):
            item = obj.get(name)
            if isinstance(item, dict):
                item = item.get("point", item.get("coordinates", []))
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                candidates.append(item[:2])
    if not candidates:
        return np.empty((0, 2))
    raw = np.asarray(candidates, dtype=float)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    unique = []
    for point in raw:
        if not unique or min(np.linalg.norm(point - p) for p in unique) > 1e-9:
            unique.append(point)
    return np.asarray(unique)


def plot_solution(n: int, points: np.ndarray, witnesses: np.ndarray, values: np.ndarray,
                  lower: float | None, upper: float | None, objective: float,
                  certified: bool, output: Path, dpi: int = 350, grid: int = 801) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.linewidth": 1.1,
                         "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10, 10.65))
    fig.subplots_adjust(left=0.095, right=0.965, bottom=0.14, top=0.91)
    axis = np.linspace(0.0, 1.0, grid)
    xx, yy = np.meshgrid(axis, axis)
    field = intensity(points, np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    reference = upper if upper is not None else objective
    levels = reference * np.array([1.025, 1.05, 1.10, 1.25, 1.50, 2.0])
    contours = ax.contour(xx, yy, field, levels=levels, colors="#4b5563",
                          linewidths=[0.52, 0.58, 0.65, 0.70, 0.78, 0.86], alpha=0.80)
    ax.clabel(contours, contours.levels[::2], inline=True, fontsize=7,
              fmt=lambda level: f"{level:.1f}", inline_spacing=3)
    ax.scatter(points[:, 0], points[:, 1], s=67, c="#dc2626", edgecolors="white",
               linewidths=0.85, zorder=5)
    for i, (x, y) in enumerate(points, start=1):
        # Give each light an unambiguous coordinate-table index.
        dx = -7 if x > 0.92 else 6
        dy = -8 if y > 0.94 else 5
        ax.annotate(str(i), (x, y), xytext=(dx, dy), textcoords="offset points",
                    color="#991b1b", fontsize=6.7, ha="right" if dx < 0 else "left",
                    va="top" if dy < 0 else "bottom", zorder=6,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.75})
    if len(witnesses):
        gap = max(0.0, upper - lower) if upper is not None and lower is not None else 0.0
        cutoff = reference + max(5e-4, 3.0 * gap)
        active = witnesses[values <= cutoff]
        if len(active) == 0:
            active = witnesses[[np.argmin(values)]]
        ax.scatter(active[:, 0], active[:, 1], s=39, c="#2563eb", edgecolors="white",
                   linewidths=0.6, marker="o", zorder=7, clip_on=False)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="$x$", ylabel="$y$")
    ax.set_xticks(np.arange(0, 1.01, 0.1)); ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.tick_params(labelsize=9, length=3, color="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#0f172a")
        spine.set_linewidth(1.6)
    fig.suptitle(f"{n} unit lights in the unit square", x=0.53, y=0.98,
                 fontsize=20, fontweight="semibold", color="#0f172a")
    if certified and lower is not None and upper is not None:
        subtitle = f"Verified minimum intensity: {format_bound(lower)} ≤ I ≤ {format_bound(upper, upper=True)}"
    else:
        subtitle = f"Numerical minimum intensity: {objective:.9f}  ·  certification pending"
    fig.text(0.53, 0.938, subtitle, ha="center", fontsize=11, color="#334155")
    handles = [Line2D([], [], marker="o", markersize=7.5, linestyle="", color="#dc2626",
                      label="Unit lights (numbered in coordinates.csv)"),
               Line2D([], [], marker="o", markersize=6, linestyle="", color="#2563eb",
                      label="Numerical minimum witnesses"),
               Line2D([], [], linewidth=0.8, color="#4b5563", label="Intensity contours")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.040),
               frameon=False, ncol=1, fontsize=9, handlelength=1.8, labelspacing=0.45)
    fig.text(0.53, 0.019, "Intensity at q = Σᵢ 1 / ‖q − pᵢ‖²   ·   Coordinates use x right, y up",
             ha="center", fontsize=8.5, color="#475569")
    fig.savefig(output / "diagram.svg", metadata={"Date": None, "Title": f"n={n} unit-light configuration",
                "Description": subtitle + "; exact coordinates provided in coordinates.csv"})
    fig.savefig(output / "diagram.png", dpi=dpi, metadata={"Title": f"n={n} unit-light configuration",
                "Description": subtitle + "; exact coordinates provided in coordinates.csv"})
    plt.close(fig)


def optional_float(mapping: dict, *names: str) -> float | None:
    for name in names:
        if name in mapping and mapping[name] is not None:
            value = float(mapping[name])
            return value if math.isfinite(value) else None
    return None


def render_one(n: int, checkpoints: Path, results: Path, published: dict,
               allow_unvalidated: bool = False, dpi: int = 350, grid: int = 801) -> dict:
    checkpoint_path = find_checkpoint(checkpoints, n)
    checkpoint = json.loads(checkpoint_path.read_text())
    points = np.asarray(checkpoint["points"], dtype=float)
    if checkpoint.get("n", n) != n or points.shape != (n, 2):
        raise ValueError(f"n={n}: point count/shape mismatch")
    if not np.all(np.isfinite(points)) or np.any(points < 0) or np.any(points > 1):
        raise ValueError(f"n={n}: coordinates violate the closed unit square constraint")
    output = results / f"n{n}"
    output.mkdir(parents=True, exist_ok=True)
    validation, validation_path = read_validation(checkpoint_path, output, checkpoint)
    fingerprint = validation.get("coordinates_sha256_float64_little_endian")
    if fingerprint is None and not allow_unvalidated:
        raise ValueError(f"n={n}: validation must include coordinates_sha256_float64_little_endian")
    if fingerprint is not None and fingerprint != points_hash(points):
        raise ValueError(f"n={n}: validation fingerprint does not match checkpoint coordinates")
    if validation.get("n", n) != n:
        raise ValueError(f"n={n}: validation point count does not match checkpoint")
    valid = bool(validation.get("valid", validation.get("constraints_valid", False)))
    certified = bool(validation.get("certified", False))
    lower = optional_float(validation, "objective_lower", "lower_bound")
    upper = optional_float(validation, "objective_upper", "upper_bound")
    if certified and (lower is None or upper is None or lower > upper):
        raise ValueError(f"n={n}: invalid certification interval")
    if not allow_unvalidated and not (valid and certified):
        raise ValueError(f"n={n}: valid certified validation is required; validate first")
    objective = optional_float(checkpoint, "objective", "minimum", "value")
    if objective is None:
        objective = upper
    if objective is None:
        raise ValueError(f"n={n}: no numerical objective")
    witnesses = normalize_witnesses(checkpoint, validation)
    values = intensity(points, witnesses) if len(witnesses) else np.empty(0)
    if not allow_unvalidated and not len(witnesses):
        raise ValueError(f"n={n}: no minimum witnesses supplied")
    if len(values) and lower is not None and min(values) < lower - 1e-8:
        raise ValueError(f"n={n}: a supplied witness contradicts the certified lower bound")
    record = published.get("entries", {}).get(str(n))
    improvement = lower - float(record["value"]) if record and lower is not None else None
    comparison_status = "no published entry"
    if record and lower is not None and upper is not None:
        displayed_number = record["displayed"].rstrip("+").split("=")[-1].strip()
        decimals = len(displayed_number.split(".")[1]) if "." in displayed_number else 0
        display_step = 10.0 ** -decimals
        if lower >= record["value"] + display_step:
            comparison_status = "exceeds the published display precision"
        elif lower >= record["value"] and upper < record["value"] + display_step:
            comparison_status = "matches the published value at printed precision"
        elif upper < record["value"]:
            comparison_status = "below the published displayed threshold"
        else:
            comparison_status = "overlaps the published displayed threshold"
    # Preserve validator evidence verbatim. A separate record binds artifacts to points.
    write_json(output / "validation.json", validation)
    write_json(output / "checkpoint.json", checkpoint)
    solution = {"schema_version": 1, "n": n, "domain": {"shape": "closed_unit_square", "x": [0, 1], "y": [0, 1]},
                "model": {"light_brightness": 1, "intensity": "sum_i 1 / ((x-x_i)^2 + (y-y_i)^2)",
                          "objective": "maximize the minimum intensity over every point of the unit square",
                          "at_a_light": "positive infinity"},
                "points": points.tolist(), "coordinate_encoding": "binary64 JSON numbers and 17-significant-digit CSV; x right, y up",
                "coordinates_sha256_float64_little_endian": points_hash(points),
                "objective_numerical": objective, "objective_lower": lower, "objective_upper": upper,
                "valid": valid, "certified": certified, "published": record,
                "difference_from_published_displayed_threshold": improvement,
                "comparison_status": comparison_status, "new_record_established": None,
                "comparison_caveat": "Published trailing-plus values are rounded lower thresholds; a positive difference is not proof of beating undisclosed exact coordinates. Unlisted n has no comparison baseline.",
                "checkpoint_metadata": checkpoint.get("metadata", {}),
                "source_url": published.get("source_url", SOURCE_URL),
                "source_retrieved_at_utc": published.get("retrieved_at_utc"),
                "checkpoint_sha256": file_hash(checkpoint_path),
                "witnesses": [{"point": q.tolist(), "intensity": float(v)} for q, v in zip(witnesses, values) if math.isfinite(v)]}
    write_json(output / "solution.json", solution)
    with (output / "coordinates.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["light", "x", "y", "brightness"])
        writer.writerows((i, format(x, ".17g"), format(y, ".17g"), 1) for i, (x, y) in enumerate(points, 1))
    with (output / "witnesses.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["witness", "x", "y", "intensity"])
        writer.writerows((i, format(x, ".17g"), format(y, ".17g"), format(v, ".17g"))
                         for i, ((x, y), v) in enumerate(zip(witnesses, values), 1))
    plot_solution(n, points, witnesses, values, lower, upper, objective, certified, output, dpi, grid)
    interval = f"[{format_bound(lower, 12)}, {format_bound(upper, 12, upper=True)}]" if lower is not None and upper is not None else "not certified"
    status = "valid and certified" if valid and certified else "NOT certified for submission"
    baseline_text = f"{record['displayed']} ({record['attribution']})" if record else "Not listed on the archived source page."
    text = f"""# n = {n}: unit-light configuration

Status: **{status}**. The minimum intensity of this fixed configuration is enclosed in **{interval}**.
The numerical objective recorded by the optimizer is {objective:.12f}.

Place {n} point lights, each with brightness 1, at the coordinates in `coordinates.csv`.
The domain is the closed square [0,1] × [0,1]; x increases rightward and y upward.
For q = (x,y), intensity is Σᵢ 1/((x−xᵢ)²+(y−yᵢ)²), with value +∞ at a light.
The CSV uses 17 significant decimal digits and reproduces the JSON binary64 coordinates.
Labels 1–{n} in the diagrams match the CSV row numbers.

- `solution.json`: complete geometry, objective bounds, source comparison, and metadata.
- `coordinates.csv`: light index, x, y, and brightness.
- `witnesses.csv`: numerical minimum candidates and directly evaluated intensities.
- `validation.json`: the global certification and constraint-check results.
- `checkpoint.json`: the original optimizer checkpoint.
- `diagram.svg`: editable, scalable vector diagram with selectable text.
- `diagram.png`: {10*dpi} × {round(10.65*dpi)}-pixel version of the same diagram.

Red dots are lights; blue dots are numerical witnesses near the minimum; thin curves
show selected intensity levels. The finite contour grid is for illustration only;
it is not the validation method. Certification concerns the objective of these fixed
coordinates, not global optimality over all possible light configurations.

Published comparison: {baseline_text}
Comparison at the precision printed on the page: {comparison_status}.
Source: {published.get('source_url', SOURCE_URL)}
Archived at: {published.get('retrieved_at_utc', 'unknown')}.
Published trailing-plus values are displayed lower thresholds. The displayed difference
in the summary is relative to that threshold and cannot establish improvement over
undisclosed higher-precision published coordinates. No baseline is inferred for unlisted n.
"""
    (output / "README.md").write_text(text)
    files = [path for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"]
    write_json(output / "manifest.json", {"n": n, "coordinates_sha256_float64_little_endian": points_hash(points),
              "files": [{"file": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)} for path in files]})
    return {"n": n, "published_best": record["displayed"] if record else "not listed",
            "published_threshold": record["value"] if record else None,
            "our_best": lower, "objective_upper": upper, "objective_numerical": objective,
            "improvement": improvement, "valid": valid, "certified": certified,
            "files": f"n{n}/", "comparison_status": comparison_status, "checkpoint_sha256": file_hash(checkpoint_path)}


def build_report(checkpoints: Path, results: Path, source: Path, n_min: int = 36,
                 n_max: int = 50, allow_unvalidated: bool = False,
                 dpi: int = 350, grid: int = 801) -> list[dict]:
    results.mkdir(parents=True, exist_ok=True)
    published = json.loads((source / "published.json").read_text())
    rows = []
    for n in range(n_min, n_max + 1):
        row = render_one(n, checkpoints, results, published, allow_unvalidated, dpi, grid)
        rows.append(row)
        print(f"n={n}: lower={row['our_best']} upper={row['objective_upper']} valid={row['valid']} certified={row['certified']}", flush=True)
    fields = ["n", "published_best", "our_best", "objective_upper", "improvement", "valid", "certified", "files"]
    with (results / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    write_json(results / "summary.json", {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "source": published, "results": rows})
    markdown = ["# Verified light-intensity configurations", "",
                "`our best` is the certified lower bound on the minimum intensity for the supplied coordinates. "
                "Each configuration's files give both endpoints of the verification interval. "
                "These are feasible configurations; global optimality is not claimed.", "",
                "A result in the same three-decimal display as 373.207+ is treated as matching the printed record, not as a new record.", "",
                "| n | published best | our best | improvement | valid | files |",
                "|---:|---:|---:|---:|:---:|:---|"]
    for row in rows:
        n = row["n"]
        ours = f"≥ {format_bound(row['our_best'])}" if row["our_best"] is not None else "uncertified"
        difference = f"{row['improvement']:+.9f}*" if row["improvement"] is not None else "—"
        status = "yes" if row["valid"] and row["certified"] else "pending"
        links = " · ".join(f"[{label}](n{n}/{filename})" for label, filename in
                           [("data", "solution.json"), ("CSV", "coordinates.csv"), ("SVG", "diagram.svg"),
                            ("PNG", "diagram.png"), ("validation", "validation.json")])
        markdown.append(f"| {n} | {row['published_best']} | {ours} | {difference} | {status} | {links} |")
    markdown += ["", "*Difference is relative to the printed lower threshold, not an undisclosed exact published value. "
                 "The trailing `+` must be preserved: beating a displayed threshold alone is not proof of a new record. "
                 "For n = 37–50 the archived page contains no published entry, so improvement is undefined.", "",
                 f"Published source: [{SOURCE_URL}]({SOURCE_URL}), archived {published.get('retrieved_at_utc', 'unknown')}. "
                 "The source snapshot and file hashes are in `source/`.", "",
                 "Each folder includes full-precision coordinates, independent objective validation, "
                 "witness points, vector and high-resolution raster diagrams, and SHA-256 artifact hashes. "
                 "Use the coordinate data to reproduce a result; image contours are illustrative.", ""]
    (results / "summary.md").write_text("\n".join(markdown))
    shutil.copytree(source, results / "source", dirs_exist_ok=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--source", type=Path, default=HERE / "source")
    parser.add_argument("--n-min", type=int, default=36)
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument("--allow-unvalidated", action="store_true")
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument("--grid", type=int, default=801)
    args = parser.parse_args()
    build_report(args.checkpoints, args.results, args.source, args.n_min, args.n_max,
                 args.allow_unvalidated, args.dpi, args.grid)

if __name__ == "__main__":
    main()
