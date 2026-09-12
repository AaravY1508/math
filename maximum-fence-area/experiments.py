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
import zlib
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
    recreated, seeded = 0, {}
    for p in sorted((run_dir / "logs").glob("worker*.jsonl")):
        for ln in open(p):
            if '"event": "eval"' in ln:
                evals.append(json.loads(ln))
            elif '"event": "propose_failed"' in ln and "re-creates the library minimum" in ln:
                recreated += 1
            elif '"event": "archive_seeded"' in ln:
                r = json.loads(ln)
                seeded[r["n"]] = {(name, a) for name, a in r["elites"]}
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
    lib_evals = [e for e in evals if e.get("from_library")]
    # evaluated results that are a seeded minimum again (same topology, area within MINIMUM_EPS: the
    # library's own "same minimum"; the guards should make this 0), and results merely on a seeded
    # minimum's topology (another basin of it: allowed). Seeded minima come from the start snapshot.
    leaks = same_topology = None
    if seeded and lib_before is not None:
        from fence import minima as Mn
        L = Mn.MinimaLibrary(lib_before)
        mins = {n: [d for d in L.best(n, 100) if (d["name"], round(d["area"], 6)) in names] for n, names in seeded.items()}
        on = [(e, d) for e in evals if e["area"] is not None for d in mins.get(e["n"], []) if e.get("signature") == d["signature"]]
        same_topology = len({id(e) for e, _ in on})
        leaks = len({id(e) for e, d in on if Mn.same_minimum(d, d["signature"], e["area"])})
    return {"run": run_dir.name, "commit": info.get("git_commit"), "gpus": ngpu, "span_h": span_h,
            "library": {"evals": len(lib_evals), "dropped_recreations": recreated, "leaks": leaks,
                        "same_topology": same_topology,
                        "beat_lib": sum(1 for e in lib_evals if e["area"] is not None and
                                        (per_n.get(e["n"], {}).get("lib_before") or 1e9) < e["area"] - 1e-9)},
            "evals": len(evals), "evals_per_gpu_h": len(evals) / span_h / max(1, ngpu),
            "per_n": per_n, "ops": {k: dict(v) for k, v in ops.items()},
            "delta": {k: (sum(v) / len(v) if v else None) for k, v in delta.items()},
            "flags": {k: info["args"].get(k) for k in ("batch", "seed", "n_range") if k in info["args"]}}


# ---------------------------------------------------------------------------
# offline operator A/B: the same parents, one child per operator, the same optimizer
# ---------------------------------------------------------------------------

def library_parents(lib: Path, n: int, k: int) -> list[dict]:
    """n's all-time record plus its best other library minima: up to k configurations with
    distinct topologies, best first ({"name", "area", "config", "signature"})."""
    from fence import minima as Mn
    L = Mn.MinimaLibrary(lib)
    docs = []
    rec = L.record(n)
    if rec:
        docs.append({"name": f"record{n}", "area": float(rec["area"]), "config": rec["config"],
                     "signature": Mn.canonical_form(G.config_from_json(rec["config"])[0]).signature})
    for ms in L.minima(n).values():
        docs += [{"name": m["name"], "area": float(m["area"]), "config": m["config"], "signature": m["signature"]}
                 for m in ms]
    out, seen = [], set()
    for d in sorted(docs, key=lambda d: -d["area"]):
        if d["signature"] not in seen and d["config"]["verification"].get("valid"):
            seen.add(d["signature"])
            out.append(d)
    return out[:k]


def ab_child(arm: str, n: int, parent: dict, rng, target: int | None):
    """(sketch, what was done) of one child of `parent` under `arm`, or (None, reason)."""
    import numpy as np
    from fence import builder as Bd
    from fence import minima as Mn
    sk0 = Bd.Sketch.from_json(E._config_to_sketch(parent["config"]))
    if arm == "face":
        sk = sk0.copy()
        info = Bd.face_fill(sk, rng, target, reject=lambda t: Mn.canonical_form(t.topology()).signature ==
                            parent["signature"])
        return (sk, info) if info else (None, "not applicable")
    # today's elite operator, with at least one edit (a zero-edit child only re-optimizes the parent)
    edits = [E.EDIT_OPS[int(rng.integers(len(E.EDIT_OPS)))] for _ in range(int(rng.choice([1, 1, 2, 2, 3])))]
    try:
        sk = Bd.build_from_genome(n, rng, {"seed": {"kind": "literal", "sketch": sk0.to_json()}, "edits": edits})
    except (Bd.BuildError, G.InvalidTopology) as exc:
        return None, f"build failed: {exc}"
    return sk, {"move": "+".join(edits)}


def cmd_op_ab(args):
    import multiprocessing as mp

    import numpy as np
    from fence import minima as Mn
    from fence import optimize as O
    ns = _parse_ns(args.ns)
    bad = sorted(set(ns) & grid_record_ns())
    if bad:
        sys.exit(f"refusing n={bad}: their world record is a grid")
    base = E.load_baselines()
    tasks = [(n, p, arm, c) for n in ns for p in library_parents(Path(args.lib), n, args.parents)
             for c in range(args.children) for arm in ("face", "elite")]
    k, m = map(int, args.share.split("/"))
    tasks = tasks[k::m]
    if args.only:
        tasks = [t for t in tasks if f"{t[0]}:{t[1]['name']}:{t[3]}:{t[2]}" in args.only.split(",")]
    best_parent = {n: max((p["area"] for p in library_parents(Path(args.lib), n, args.parents)), default=0.0) for n in ns}
    pool = mp.get_context("spawn").Pool(args.cpu_procs, initializer=O.pool_init)
    out = open(args.out, "a")
    for n, parent, arm, c in tasks:
        seed = zlib.crc32(f"{n}:{parent['name']}:{c}:{args.seed}".encode()) % (2 ** 31)   # same for both arms
        rng = np.random.default_rng(seed)
        t0 = time.time()
        rec = {"n": n, "parent": parent["name"], "parent_area": parent["area"], "arm": arm, "child": c}
        sk, info = ab_child(arm, n, parent, rng, E.target_faces(base[n]))
        if sk is None:
            out.write(json.dumps({**rec, "built": False, "why": info}) + "\n")
            out.flush()
            continue
        topo = sk.topology()
        cfg = O.OptConfig(batch=O.default_batch(n))
        res = O.optimize_topology(topo, sk.coords(), n, cfg, device="cuda", seed=seed, pool=pool)
        best = res["best"]
        row = {**rec, "built": True, "move": info["move"], "where": info.get("where"),
               "same_topology": Mn.canonical_form(topo).signature == parent["signature"],
               "faces_child": len(topo.faces), "area": best["area"] if best else None, "t": round(time.time() - t0, 1)}
        if best:
            F, nd, deficit = best_faces(best["config"])
            row.update(faces_best=F, dead_best=nd, deficit_best=round(deficit, 4), delta=best["area"] - parent["area"])
            if args.save_dir and best["area"] > best_parent[n] + 1e-9:
                # better than every parent of n, i.e. than its library record: keep the configuration
                d = Path(args.save_dir)
                d.mkdir(parents=True, exist_ok=True)
                f = d / f"n{n:03d}_{best['area']:.8f}_{arm}_{parent['name']}_{c}.json"
                f.write_text(json.dumps({**best["config"], "found_by": {**rec, "move": info["move"], "seed": seed}}, indent=1))
                row["saved"] = str(f)
        out.write(json.dumps(row) + "\n")
        out.flush()
    pool.terminate()


def cmd_op_ab_report(args):
    rows = [json.loads(ln) for f in args.files for ln in open(f)]
    def summary(rs):
        built = [r for r in rs if r["built"]]
        ver = [r for r in built if r.get("area") is not None]
        beat = [r for r in ver if r["delta"] > 1e-4]
        d = [r["delta"] for r in ver]
        return (f"{len(rs)} | {len(built)} | {sum(r['same_topology'] for r in built)} | {len(ver)} | "
                f"{len(beat)} ({len(beat) / max(1, len(ver)):.0%}) | {_fmt(sum(d) / len(d) if d else None, 3)} | "
                f"{_fmt(max(d) if d else None, 4)} | "
                f"{_fmt(sum(r['dead_best'] for r in ver) / len(ver) if ver else None, 2)}")
    print("| n | arm | children | built | same topology as parent | verified | beat parent | mean delta | best delta | dead faces / result |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n in sorted({r["n"] for r in rows}):
        for arm in ("face", "elite"):
            print(f"| {n} | {arm} | " + summary([r for r in rows if r["n"] == n and r["arm"] == arm]) + " |")
    for arm in ("face", "elite"):
        print(f"| all | {arm} | " + summary([r for r in rows if r["arm"] == arm]) + " |")
    moves = defaultdict(list)
    for r in rows:
        if r["arm"] == "face" and r["built"] and r.get("area") is not None:
            moves[f"{r['move']}-{r['where']}"].append(r["delta"])
    print("\nface moves: " + ", ".join(f"{k}: {len(v)} children, {sum(x > 1e-4 for x in v)} beat parent, "
                                       f"mean delta {sum(v) / len(v):.3f}" for k, v in sorted(moves.items())))


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
        lb = m["library"]
        print(f"\nchildren of seeded library minima: {lb['evals']} evals, {lb['beat_lib']} beat the library record, "
              f"{lb['dropped_recreations']} dropped for re-creating their own minimum's topology; "
              f"verified results that are a seeded minimum again: {'-' if lb['leaks'] is None else lb['leaks']} "
              f"(on a seeded minimum's topology, other basins allowed: {'-' if lb['same_topology'] is None else lb['same_topology']})")
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
    me = sub.add_parser("metrics", help="per-n and per-operator tables of one or more runs")
    me.add_argument("runs", nargs="+")
    me.add_argument("--lib-before", default=None, help="the snapshot the arms started from (library records)")
    ab = sub.add_parser("op-ab", help="offline A/B of the face-aware vs the elite operator on library parents")
    ab.add_argument("--ns", required=True)
    ab.add_argument("--lib", required=True)
    ab.add_argument("--parents", type=int, default=3, help="per n: the record plus the best other minima")
    ab.add_argument("--children", type=int, default=5, help="per parent and arm")
    ab.add_argument("--share", default="0/1", help="k/m: this process runs every m-th task from the k-th")
    ab.add_argument("--cpu-procs", type=int, default=15)
    ab.add_argument("--seed", type=int, default=0)
    ab.add_argument("--out", required=True)
    ab.add_argument("--save-dir", default=None, help="keep verified configurations that beat n's library record")
    ab.add_argument("--only", default=None, help="comma-separated n:parent:child:arm tasks to run")
    abr = sub.add_parser("op-ab-report", help="tables of op-ab result files")
    abr.add_argument("files", nargs="+")
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []     # launch: flags for run_overnight.py
    args = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    args.extra = extra
    {"snapshot": cmd_snapshot, "launch": cmd_launch, "metrics": cmd_metrics, "op-ab": cmd_op_ab,
     "op-ab-report": cmd_op_ab_report}[args.cmd](args)


if __name__ == "__main__":
    main()
