#!/usr/bin/env python3
"""Independently certify checkpoint geometries, once or in a low-duty audit loop.

Final use:
  python validate_results.py --checkpoints runs/four-hour/checkpoints \
      --results results --tolerance 1e-7 --time-limit 120
Search audit:
  python validate_results.py --checkpoints runs/four-hour/checkpoints \
      --watch --interval 300 --tolerance 1e-5 --until-run-complete

Each certificate is bound to the exact binary64 coordinate hash. Historical
certificates retain the checkpoint snapshot they certify. A concurrently updated
checkpoint never receives a certificate for the older geometry.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from validation import validate

HERE = Path(__file__).resolve().parent
HASH_KEY = "coordinates_sha256_float64_little_endian"


def points_hash(points):
    return hashlib.sha256(np.asarray(points, dtype="<f8").tobytes(order="C")).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_checkpoint(path, output, *, tolerance=1e-7, time_limit=120., force=False,
                        expected_n=None):
    """Return a certificate for one immutable read snapshot, with race detection."""
    path, output = Path(path), Path(output)
    raw = path.read_bytes()
    checkpoint = json.loads(raw)
    points = np.asarray(checkpoint["points"], dtype=float)
    geometry_hash = points_hash(points)
    code_hash = hashlib.sha256((HERE / "validation.py").read_bytes()).hexdigest()
    previous_path = output / "validation.json"
    if not force and previous_path.exists():
        old = json.loads(previous_path.read_text())
        if (old.get(HASH_KEY) == geometry_hash and old.get("certified") and
                old.get("valid") and old.get("gap", float("inf")) <= tolerance and
                old.get("validator_source_sha256") == code_hash):
            return {**old, "reused_existing_certificate": True}
    witness = checkpoint.get("witness")
    if witness is None and checkpoint.get("witnesses"):
        witness = checkpoint["witnesses"][0]
    result = validate(points, tolerance=tolerance, time_limit=time_limit,
                      numerical=True, witness=witness)
    if expected_n is not None and points.shape != (expected_n, 2):
        result["valid"] = False
        result["certified"] = False
        result.setdefault("errors", []).append(f"Expected {expected_n} coordinate pairs, got shape {points.shape}.")
    result.update({
        HASH_KEY: geometry_hash,
        "points_shape": list(points.shape),
        "checkpoint_source": str(path.resolve()),
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "validator_source_sha256": code_hash,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "claimed_objective": checkpoint.get("objective"),
        "reused_existing_certificate": False,
    })
    if result.get("valid") and checkpoint.get("objective") is not None:
        discrepancy = float(checkpoint["objective"] - result["objective_upper"])
        result["claimed_objective_minus_proved_upper"] = discrepancy
        result["oracle_missed_lower_minimum"] = discrepancy > max(tolerance, 1e-6)
    historical = path.parent / "validations" / geometry_hash
    atomic_json(historical / "checkpoint.json", checkpoint)
    # Re-read only to detect a race; never modify the optimizer's checkpoint.
    current = json.loads(path.read_bytes())
    current_same = points_hash(current["points"]) == geometry_hash
    result["checkpoint_unchanged_during_validation"] = current_same
    atomic_json(historical / "validation.json", result)
    if current_same:
        atomic_json(output / "validation.json", result)
    return result


def audit_once(args):
    rows = []
    for n in range(args.n_min, args.n_max + 1):
        path = args.checkpoints / f"n{n}" / "best.json"
        if not path.exists():
            rows.append({"n": n, "status": "missing_checkpoint", "valid": False})
            if not args.watch:
                print(json.dumps(rows[-1]), flush=True)
            continue
        output = args.results / f"n{n}" if args.results else path.parent
        try:
            certificate = validate_checkpoint(path, output, tolerance=args.tolerance,
                       time_limit=args.time_limit, force=args.force, expected_n=n)
            row = {"n": n, **{key: certificate.get(key) for key in (
                "valid", "certified", "status", "objective_lower", "objective_upper", "gap",
                "elapsed_seconds", "oracle_missed_lower_minimum", "checkpoint_unchanged_during_validation",
                "reused_existing_certificate", HASH_KEY)}}
        except Exception as exc:
            row = {"n": n, "status": "validation_error", "valid": False,
                   "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    summary = {"at_utc": datetime.now(timezone.utc).isoformat(), "tolerance": args.tolerance,
               "checkpoints": str(args.checkpoints.resolve()), "results": rows}
    target = args.results if args.results else args.checkpoints.parent
    atomic_json(target / "validation_audit.json", summary)
    return all(r.get("certified") and r.get("valid") and
               r.get("checkpoint_unchanged_during_validation") and
               not r.get("oracle_missed_lower_minimum") for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--results", type=Path)
    p.add_argument("--tolerance", type=float, default=1e-7)
    p.add_argument("--time-limit", type=float, default=120.)
    p.add_argument("--n-min", type=int, default=36)
    p.add_argument("--n-max", type=int, default=50)
    p.add_argument("--force", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=300.)
    p.add_argument("--until-run-complete", action="store_true")
    args = p.parse_args()
    if args.until_run_complete and not args.watch:
        p.error("--until-run-complete requires --watch")
    while True:
        started = time.monotonic()
        success = audit_once(args)
        if not args.watch:
            return 0 if success else 1
        manifest = args.checkpoints.parent / "run.json"
        if args.until_run_complete and manifest.exists():
            state = json.loads(manifest.read_text())
            if state.get("status") not in ("running", "resuming"):
                return 0 if success else 1
        delay = max(0., args.interval - (time.monotonic() - started))
        time.sleep(delay)


if __name__ == "__main__":
    sys.exit(main())
