"""Experiment tooling for search-strategy changes (protocol and log: EXPERIMENTS.md).

    ./venv/bin/python experiments.py snapshot minima-snapshots/20260912          # freeze the minima library
    ./venv/bin/python experiments.py launch ctl --gpu-ids 0,1 --ns 13,18 --budget 15m \\
        --lib minima-snapshots/20260912 -- --face-weight 0                        # one arm, detached
    ./venv/bin/python experiments.py metrics runs/<arm1> runs/<arm2> --lib-before minima-snapshots/20260912

Every arm starts from its own copy of one frozen snapshot of the minima library, so arms are
comparable and two concurrent arms never share the writer of an n (fence/minima.py). n whose
world record is a unit-square grid (baselines status "trivial") are refused: those records are
likely optimal and not worth GPU time.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fence import evolve as E          # noqa: E402
from fence import geometry as G        # noqa: E402

DEAD_FACE = 0.9


def grid_record_ns() -> set[int]:
    """n whose published record is a grid (or simple polygon): excluded from all experiments."""
    return {n for n, b in E.load_baselines().items() if b["status"] == "trivial"}


def _parse_ns(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


# ---------------------------------------------------------------------------
# snapshot / launch
# ---------------------------------------------------------------------------

def cmd_snapshot(args):
    dst = Path(args.dst)
    if dst.exists():
        sys.exit(f"{dst} exists; snapshots are frozen, pick a new name")
    shutil.copytree(args.src, dst)
    n = sum(1 for _ in dst.rglob("*.json"))
    print(f"snapshot {dst}: {n} files from {args.src}")


def cmd_launch(args):
    ns = _parse_ns(args.ns)
    bad = sorted(set(ns) & grid_record_ns())
    if bad:
        sys.exit(f"refusing to run n={bad}: their world record is a grid (likely optimal)")
    lib = HERE / "runs" / "exp-libs" / args.arm
    if lib.exists():
        sys.exit(f"{lib} exists: arm names are single-use")
    shutil.copytree(args.lib, lib)
    log = HERE / "runs" / f"launch-exp-{args.arm}.log"
    ngpu = len(args.gpu_ids.split(","))
    cpu = args.cpu_procs or max(2, ((os.cpu_count() or 8) - 4) // args.gpus_on_host)
    cmd = [sys.executable, "run_overnight.py", "--budget", args.budget, "--gpu-ids", args.gpu_ids,
           "--n-range", ",".join(map(str, ns)), "--tag", f"exp-{args.arm}", "--minima-dir", str(lib),
           "--archive-dir", "", "--cpu-procs", str(cpu), "--seed", str(args.seed), *args.extra]
    with open(log, "w") as fh:
        fh.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    print(f"arm {args.arm}: pid {p.pid} on GPUs {args.gpu_ids} ({ngpu}), n={ns}, {cpu} CPU procs/worker, "
          f"library copy {lib}, log {log}")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def op_of(origin: str) -> str:
    """Mutation operator of an eval, from its program origin tag."""
    if origin.startswith("seed:"):
        return "seed-program"
    if origin.startswith("library:"):
        return "library"
    parts = origin.replace("+novel", "").split(":")
    if parts[0] != "random" or len(parts) < 2:
        return parts[0]
    tag = parts[1]
    if tag.startswith("edit+"):
        return "edit+"
    if tag == "face" and len(parts) > 2:
        return f"face:{parts[2]}"
    return tag


def source_area(origin: str) -> float | None:
    """Area of the elite a literal-seeded child was built from (elite / face / elite-fb ops)."""
    parts = origin.split(":")
    for i, p in enumerate(parts[:-1]):
        if p.startswith("n") and p[1:].isdigit():
            try:
                return float(parts[i + 1].split("+")[0])
            except ValueError:
                return None
    return None


def best_faces(cfg: dict) -> tuple[int, int, float]:
    """(faces, faces below DEAD_FACE, their total deficit) of a verified configuration."""
    areas = [float(a) for a in cfg["verification"]["face_areas"]]
    dead = [a for a in areas if a < DEAD_FACE]
    return len(areas), len(dead), sum(1 - a for a in dead)


def run_metrics(run_dir: Path, lib_before: Path | None) -> dict:
    base = E.load_baselines()
    info = json.loads((run_dir / "run.json").read_text())
    evals, starts, bests_ev = [], {}, defaultdict(list)
    for p in sorted((run_dir / "logs").glob("worker*.jsonl")):
        for ln in open(p):
            if '"event": "eval"' in ln:
                evals.append(json.loads(ln))
            elif '"event": "worker_start"' in ln:
                r = json.loads(ln)
                starts.setdefault(r["gpu"], r["ts"])
            elif '"event": "new_best"' in ln:
                r = json.loads(ln)
                bests_ev[r["n"]].append(r["ts"])
    t0 = min(starts.values()) if starts else (min(e["ts"] for e in evals) if evals else 0)
    ns = sorted({n for s in info["slices"] for n in s})
    per_n = {}
    for n in ns:
        ev = [e for e in evals if e["n"] == n]
        row = {"n": n, "evals": len(ev), "record": base[n]["A"], "best": None}
        f = run_dir / "best" / f"n{n:03d}.json"
        if f.exists():
            cfg = json.loads(f.read_text())
            a = float(cfg["verification"]["total_area"])
            F, nd, deficit = best_faces(cfg)
            row.update(best=a, gap=a - base[n]["A"], status=E.classify(a, base[n]), faces=F, dead=nd,
                       deficit=deficit, target_faces=int(base[n]["A"] + 1e-9) + 1)
        if lib_before is not None:
            rec = lib_before / "records" / f"N{n}.json"
            row["lib_before"] = float(json.loads(rec.read_text())["area"]) if rec.exists() else None
        row["last_improve_min"] = round((max(bests_ev[n]) - t0) / 60, 1) if bests_ev[n] else None
        per_n[n] = row
    # per operator: evals, verified share, improvements of this run's running best per n, and
    # (literal-seeded ops) how often a child beat the elite it was built from
    ops = defaultdict(lambda: Counter())
    delta = defaultdict(list)
    run_best = {}
    for e in sorted(evals, key=lambda e: e["ts"]):
        op = op_of(e["origin"])
        c = ops[op]
        c["evals"] += 1
        a = e["area"]
        if a is not None:
            c["verified"] += 1
            if e["n"] in run_best and a > run_best[e["n"]] + 1e-9:
                c["wins"] += 1
            rb = per_n.get(e["n"], {}).get("lib_before")
            if rb is not None and a > rb + 1e-9:
                c["beat_lib"] += 1
            run_best[e["n"]] = max(a, run_best.get(e["n"], a))
            sa = source_area(e["origin"])
            if sa is not None:
                delta[op].append(a - sa)
                if a > sa + 1e-4:
                    c["beat_source"] += 1
    span_h = max(1e-9, (max(e["ts"] for e in evals) - t0) / 3600) if evals else 1e-9
    ngpu = len(info["slices"])
    return {"run": run_dir.name, "commit": info.get("git_commit"), "gpus": ngpu, "span_h": span_h,
            "evals": len(evals), "evals_per_gpu_h": len(evals) / span_h / max(1, ngpu),
            "per_n": per_n, "ops": {k: dict(v) for k, v in ops.items()},
            "delta": {k: (sum(v) / len(v) if v else None) for k, v in delta.items()},
            "flags": {k: info["args"].get(k) for k in ("batch", "seed", "n_range") if k in info["args"]}}


def _fmt(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def cmd_metrics(args):
    lib = Path(args.lib_before) if args.lib_before else None
    runs = [run_metrics(Path(r), lib) for r in args.runs]
    for m in runs:
        print(f"\n## {m['run']}  (commit {m['commit']}, {m['gpus']} GPUs, {m['span_h'] * 60:.0f} min, "
              f"{m['evals']} evals, {m['evals_per_gpu_h']:.0f} evals/GPU-h)\n")
        print("| n | record | lib before | best | gap | F / target | dead (deficit) | evals | last improve (min) |")
        print("|---|---|---|---|---|---|---|---|---|")
        for n, r in m["per_n"].items():
            ft = f"{r['faces']} / {r['target_faces']}" if r.get("faces") else "-"
            dd = f"{r['dead']} ({r['deficit']:.3f})" if r.get("faces") else "-"
            print(f"| {n} | {r['record']:.5f} | {_fmt(r.get('lib_before'), 5)} | {_fmt(r['best'], 5)} | "
                  f"{_fmt(r.get('gap'), 5)} | {ft} | {dd} | {r['evals']} | {_fmt(r['last_improve_min'], 1)} |")
        print("\n| operator | evals | verified | run-best wins | beat library record | beat source | mean area - source |")
        print("|---|---|---|---|---|---|---|")
        for op, c in sorted(m["ops"].items(), key=lambda kv: -kv[1]["evals"]):
            print(f"| {op} | {c['evals']} | {c.get('verified', 0) / c['evals']:.0%} | {c.get('wins', 0)} | "
                  f"{c.get('beat_lib', 0)} | {c.get('beat_source', '-') if op in m['delta'] else '-'} | "
                  f"{_fmt(m['delta'].get(op), 4)} |")
    if len(runs) > 1:
        ns = sorted(set.intersection(*[set(m["per_n"]) for m in runs]))
        print("\n## best area per n, side by side\n")
        print("| n | " + " | ".join(m["run"] for m in runs) + " |")
        print("|---|" + "---|" * len(runs))
        for n in ns:
            print(f"| {n} | " + " | ".join(_fmt(m["per_n"][n]["best"], 5) for m in runs) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="freeze a copy of the minima library")
    s.add_argument("dst")
    s.add_argument("--src", default=str(HERE / "minima"))
    la = sub.add_parser("launch", help="start one experiment arm (detached run_overnight.py)")
    la.add_argument("arm", help="arm name, single-use (run tag exp-<arm>)")
    la.add_argument("--gpu-ids", required=True)
    la.add_argument("--ns", required=True)
    la.add_argument("--budget", default="15m")
    la.add_argument("--lib", required=True, help="frozen minima snapshot this arm starts from")
    la.add_argument("--seed", type=int, default=0)
    la.add_argument("--cpu-procs", type=int, default=0, help="per worker (0: share the host's cores over --gpus-on-host)")
    la.add_argument("--gpus-on-host", type=int, default=4, help="GPUs of all concurrent arms together")
    la.add_argument("extra", nargs="*", help="further run_overnight.py flags (after --)")
    me = sub.add_parser("metrics", help="per-n and per-operator tables of one or more runs")
    me.add_argument("runs", nargs="+")
    me.add_argument("--lib-before", default=None, help="the snapshot the arms started from (library records)")
    args = ap.parse_args(argv)
    {"snapshot": cmd_snapshot, "launch": cmd_launch, "metrics": cmd_metrics}[args.cmd](args)


if __name__ == "__main__":
    main()
