#!/usr/bin/env python
"""Two-stage net -> geometry pipeline for the Maximum Fence Area problem.

Stage 1 (`fence.nets`)      enumerate candidate nets for each n, reject any that
                            break an enabled hard rule (`fence.rules`), rank the
                            survivors with the enabled heuristics (`fence.heuristics`).
Stage 2 (`fence.netevolve`) for each top-ranked net, evolve the continuous geometry
                            (vertex positions, T-junction positions along their hosts)
                            with the existing local optimizer refining every candidate,
                            and with mutations that can also change the net locally.

Every rule, heuristic, generator and mutation move is separately toggleable, and
every rejection is logged with the id of the check that fired, so switching one off
and re-running says what it was worth.

  ./venv/bin/python pipeline.py --list
  ./venv/bin/python pipeline.py --n-range 12-16 --stage1-only
  ./venv/bin/python pipeline.py --n-range 13,16 --nets-per-n 3 --evals-per-net 12 --device cuda
  ./venv/bin/python pipeline.py --n-range 22 --no-rule record_faces      # what does rule 14 cost?
  ./venv/bin/python pipeline.py --calibrate
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fence import heuristics as H            # noqa: E402
from fence import geometry as G              # noqa: E402
from fence import minima as Mn               # noqa: E402
from fence import nets as N                  # noqa: E402
from fence import netevolve as NE            # noqa: E402
from fence import optimize as O              # noqa: E402
from fence import rules as R                 # noqa: E402
from fence.evolve import classify, load_baselines   # noqa: E402
from fence.minima import MinimaLibrary       # noqa: E402

HERE = Path(__file__).resolve().parent


def parse_ns(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# ---------------------------------------------------------------------------
# the two stages for one n
# ---------------------------------------------------------------------------

def stage1(n: int, args, rng, ruleset, hset, lib, baselines) -> N.NetStageResult:
    base = baselines.get(n)
    rec = base["A"] if base else None
    ctx = {
        "record_area": rec,
        "min_faces": R.min_faces_for(rec),
        "library": lib,
        "shapes_per_c": args.shapes_per_c,
    }
    return N.enumerate_nets(n, rng, ruleset, hset, ctx,
                            generators=parse_generators(args.generators),
                            max_candidates=args.max_candidates, keep=args.keep_nets)


def index_minimum(lib, n: int, config: dict, meta: dict) -> bool:
    """Index one verified configuration as a minimum of the pipeline's lineage at n.

    The library keeps distinct minima per lineage (same topology and area within 1e-6 is the
    same minimum), so every net's best goes in: that is what gives later runs, and the
    pipeline's own `library` generator, the topologies this run found and not only its record.
    """
    if lib is None or not config or not config.get("verification", {}).get("valid"):
        return False
    try:
        topo, X = G.config_from_json(config)
        form = Mn.canonical_form(topo)
    except Exception:  # noqa: BLE001 -- a config the signature code cannot read is simply not indexed
        return False
    new, _ = lib.add_minimum(Mn.PIPELINE_PROPOSER, n, config, form.signature, X[form.orders[0]], meta)
    return new


def stage2(n: int, s1: N.NetStageResult, args, rng, ruleset, hset, moves, baselines,
           pool, log, deadline, lib=None, run_name: str = "") -> dict:
    base = baselines.get(n)
    rec = base["A"] if base else None
    ctx = {"record_area": rec, "min_faces": R.min_faces_for(rec),
           "jitter_sigma": args.jitter, "dead_face": args.dead_face}
    cfg = O.OptConfig(batch=args.batch, steps=args.steps, topk=args.topk,
                      polish_iters=args.polish_iters, verify_top=args.verify_top,
                      mem_cap_gb=args.mem_cap_gb)
    best, per_net, ledger = None, [], R.Ledger()
    ledger.merge(s1.ledger)
    for k, net in enumerate(s1.ranked[: args.nets_per_n]):
        if deadline and time.time() > deadline:
            break
        ev = NE.NetEvolver(n, net.sketch, ruleset=ruleset, hset=hset, moves=moves, cfg=cfg,
                           ctx=ctx, device=args.device, rng=rng, pool=pool,
                           pop=args.pop, elite=args.elite, log=log)
        res = ev.run(max_evals=args.evals_per_net, generations=args.generations,
                     deadline=deadline)
        ledger.merge(res.ledger)
        row = {"rank": k, "net_score": round(net.score.total, 4), "family": net.cand.family,
               "origin": net.cand.origin, "F": net.cand.stats.F, "T": net.cand.stats.T,
               "b": net.cand.stats.b, **{q: v for q, v in res.to_json().items()
                                         if q in ("best_area", "evaluations", "seconds")}}
        if res.best is not None and res.best.config:
            row["indexed"] = index_minimum(lib, n, res.best.config,
                                           {"run": run_name, "via": "pipeline", "net_rank": k,
                                            "net_family": net.cand.family, "origin": res.best.origin})
        per_net.append(row)
        log({"event": "net_done", "n": n, **row})
        if res.best is not None and res.best.area is not None and \
                (best is None or res.best.area > best["area"]):
            best = {"area": res.best.area, "config": res.best.config, "origin": res.best.origin,
                    "net_rank": k, "net_family": net.cand.family}
    return {"n": n, "best": best, "per_net": per_net, "ledger": ledger,
            "stage1": s1.to_json(), "coverage": N.coverage(s1)}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report(results: list[dict], baselines: dict) -> str:
    lines = [
        "  n |  best known |       ours |      delta | status | F/target | nets | evals",
        "----+-------------+------------+------------+--------+----------+------+------",
    ]
    tot_gap = 0.0
    for r in results:
        n = r["n"]
        base = baselines.get(n)
        rec = base["A"] if base else None
        best = r.get("best")
        area = best["area"] if best else None
        F = "-"
        if best and best.get("config"):
            F = str(len(best["config"]["topology"]["faces"]))
        target = R.min_faces_for(rec)
        status = classify(area, base)
        delta = (area - rec) if (area is not None and rec is not None) else None
        if delta is not None and delta < 0:
            tot_gap += -delta
        evals = sum(p.get("evaluations", 0) for p in r.get("per_net", []))
        lines.append(
            f"{n:>3} | {(f'{rec:.5f}' if rec else '-'):>11} | "
            f"{(f'{area:.5f}' if area is not None else '-'):>10} | "
            f"{(f'{delta:+.5f}' if delta is not None else '-'):>10} | "
            f"{status:<6} | {F:>3}/{(target if target else '-'):<4} | "
            f"{len(r.get('per_net', [])):>4} | {evals:>5}")
    lines.append(f"total shortfall against the published records: {tot_gap:.5f}")
    return "\n".join(lines)


def calibration_report(lib_dir: str, baselines: dict) -> str:
    """Check heuristics 1 and 2 against the configurations we actually have."""
    from fence import geometry as G

    recs = sorted(Path(lib_dir, "records").glob("N*.json"),
                  key=lambda p: int(p.stem[1:]))
    cfgs = []
    for p in recs:
        d = json.loads(p.read_text())
        if d["n"] < 12:
            continue      # the coefficients are for n >= 12; at n = 3 the record is a triangle, b = 3
        topo, X = G.config_from_json(d["config"])
        cfgs.append((d["n"], topo, X, d["area"]))
    cal = H.calibrate(cfgs)
    if not cal:
        return "no configurations in the library to calibrate against"
    out = [f"calibrated against {cal['n_configs']} referee-verified configurations in {lib_dir}/records",
           "",
           "heuristic 1 (boundary size b):",
           f"  square-grid model   b = {cal['boundary_c_model']} * sqrt(n)",
           f"  measured            b = {cal['boundary_c_measured']} * sqrt(n)  "
           f"(relative spread +-{cal['boundary_spread']:.0%})",
           f"  T-junction-free     b = {cal['boundary_c_t_free']} * sqrt(n)",
           f"  T-junction-rich     b = {cal['boundary_c_t_rich']} * sqrt(n)",
           "  -> the population is bimodal, so neither the model nor a single pooled coefficient works:",
           "     pooled, the n=12 record (a 2x2 grid, b=8) scores 0.21 and ranks below four worse layouts.",
           f"     The default target interpolates on the candidate's own T, from {H.BOUNDARY_C_GRID} at T=0",
           f"     to {H.BOUNDARY_C_RICH} at T >= {H.T_RICH}; that scores the 39 records 0.946 on average against 0.663",
           "     pooled.",
           "",
           "heuristic 2 (face count F):",
           f"  square-grid model   F = {cal['face_slope_model']} * n",
           f"  measured            F = {cal['face_slope_measured']} * n",
           f"  F == ceil(total area) on {cal['face_equals_ceil_area']} of {cal['n_configs']} configurations",
           "  -> the face count is the record-based bound of hard rule 14 itself, not n/2, which",
           "     overshoots by ~25%; the target is ceil(A_record)."]
    return "\n".join(out)


def combine(dirs: list[str], baselines: dict) -> str:
    """Merge the per-n results of several pipeline runs (e.g. one per GPU shard) into one table.

    Each n is kept from whichever run found the largest verified area for it, and the
    rejection ledgers are summed, so a sharded run reports exactly as a single one would.
    """
    best: dict[int, dict] = {}
    ledger = R.Ledger()
    for d in dirs:
        root = Path(d)
        for f in sorted(root.glob("n[0-9][0-9][0-9].json")):
            rec = json.loads(f.read_text())
            n = int(rec["n"])
            cur = best.get(n)
            a = (rec.get("best") or {}).get("area")
            if cur is None or (a is not None and (cur.get("best") or {}).get("area", -1) < a):
                best[n] = rec
        summary = root / "summary.json"
        if summary.exists():
            led = json.loads(summary.read_text()).get("ledger", {})
            ledger.accepted += led.get("accepted", 0)
            for k, v in led.get("rejected", {}).items():
                ledger.counts[k] += v
            for k, v in led.get("samples", {}).items():
                ledger.samples.setdefault(k, []).extend(v[:1])
    results = [best[n] for n in sorted(best)]
    return report(results, baselines) + "\n\n" + ledger.table()


def ablate(kind: str, ns: list[int], args, ruleset, hset, lib, baselines) -> str:
    """Switch each check off in turn and report what changed, over stage 1 only.

    For a *rule* the cost is what it rejects: how many nets survive with it off, and what
    the extra ones look like. For a *heuristic* nothing is rejected, so the cost is what it
    moves: how much of the top-5 ranking it is responsible for, and whether the best net
    changes at all.
    """
    def generate(rs):
        """Stage 1 once per n with this rule set; returns the surviving nets per n."""
        out = []
        for n in ns:
            rng = np.random.default_rng(args.seed + 1000 * n)
            out.append(stage1(n, args, rng, rs, hset, lib, baselines))
        return out

    # `distinct` counts every net that passed the rules; `ranked` is truncated to --keep-nets,
    # so counting it would cap every ablation at keep_nets * len(ns) and hide what a rule rejects
    base = generate(ruleset)
    base_kept = sum(r.distinct for r in base)
    lines = [f"ablation over n = {ns}, stage 1 only, {args.max_candidates} candidates per n",
             f"baseline: {base_kept} distinct nets pass the rules"]
    if kind == "rules":
        lines += ["", f"{'rule off':<16} {'kept':>6} {'delta':>7}   what it was rejecting"]
        for rid in sorted(R.RULES, key=lambda k: R.RULES[k].number):
            if rid not in ruleset.enabled:
                continue
            kept = sum(r.distinct for r in generate(R.RuleSet(ruleset.enabled - {rid},
                                                               ruleset.ctx)))
            lines.append(f"{rid:<16} {kept:>6} {kept - base_kept:>+7}   {R.RULES[rid].summary[:52]}")
        return "\n".join(lines)

    # a heuristic rejects nothing, so the same nets are re-scored rather than regenerated:
    # what it is worth is how much of the ranking it is responsible for
    def rank(hs, res):
        rec = baselines.get(res.n, {}).get("A")
        ctx = {"record_area": rec, "min_faces": R.min_faces_for(rec)}
        order = sorted(res.ranked, key=lambda x: -hs.score(x.cand, ctx).total)
        return [x.cand.origin for x in order[:5]]

    base_top = [rank(hset, r) for r in base]
    lines += ["", f"{'heuristic off':<16} {'top-5 kept':>12} {'best net changed':>18}"]
    for hid in hset.enabled():
        hs2 = H.HeuristicSet({**hset.weights, hid: 0.0}, hset.ctx)
        top = [rank(hs2, r) for r in base]
        same = sum(len(set(a) & set(b)) for a, b in zip(base_top, top))
        total = sum(len(a) for a in base_top)
        changed = sum(1 for a, b in zip(base_top, top) if a and b and a[0] != b[0])
        lines.append(f"{hid:<16} {same:>7}/{total:<4} {f'{changed}/{len(ns)} n':>18}")
    return "\n".join(lines)


def list_checks() -> str:
    return "\n".join([
        "hard rules (--no-rule <id>, --only-rules <id,...>):", R.describe_rules(), "",
        "heuristics (--no-heuristic <id>, --weight <id>=<w>):", H.describe_heuristics(), "",
        "net generators (--generators <id,...>):", N.describe_generators(), "",
        "mutation moves (--no-move <id>, --move-weight <id>=<w>, --move-kinds geometry,topology):",
        NE.describe_moves(),
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-range", default="12-20", help="n values, e.g. 12-20 or 13,16,22-25")
    p.add_argument("--list", action="store_true", help="list every rule, heuristic, generator and move")
    p.add_argument("--calibrate", action="store_true",
                   help="check heuristics 1 and 2 against the verified configurations we have")
    p.add_argument("--combine", nargs="+", metavar="RUN_DIR",
                   help="merge the per-n results of several runs (e.g. one per GPU shard) into one table")
    p.add_argument("--ablate", choices=["rules", "heuristics"],
                   help="switch each check off in turn over --n-range and report what it was worth")
    p.add_argument("--stage1-only", action="store_true", help="enumerate and rank nets, do not search them")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="directory for the JSON results (default runs/pipeline-<ts>)")
    p.add_argument("--budget", default=None, help="wall-clock budget, e.g. 30m or 2h")
    # toggles
    p.add_argument("--no-rule", action="append", default=[], metavar="ID")
    p.add_argument("--only-rules", default=None, metavar="ID,...")
    p.add_argument("--no-heuristic", action="append", default=[], metavar="ID")
    p.add_argument("--weight", action="append", default=[], metavar="ID=W")
    p.add_argument("--no-move", action="append", default=[], metavar="ID")
    p.add_argument("--move-weight", action="append", default=[], metavar="ID=W")
    p.add_argument("--move-kinds", default=None, help="geometry, topology, or both (default both)")
    p.add_argument("--generators", default=None, help="which net generators to run (default all)")
    # stage 1
    p.add_argument("--max-candidates", type=int, default=800)
    p.add_argument("--keep-nets", type=int, default=25)
    p.add_argument("--shapes-per-c", type=int, default=8)
    # stage 2
    p.add_argument("--nets-per-n", type=int, default=4)
    p.add_argument("--evals-per-net", type=int, default=16)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--pop", type=int, default=6)
    p.add_argument("--elite", type=int, default=3)
    p.add_argument("--jitter", type=float, default=0.05)
    p.add_argument("--dead-face", type=float, default=0.9)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--polish-iters", type=int, default=800)
    p.add_argument("--verify-top", type=int, default=3)
    p.add_argument("--mem-cap-gb", type=float, default=14.0,
                   help="hard cap on the GPU working set of one batch; lower it to share a GPU")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--torch-threads", type=int, default=4,
                   help="CPU threads for the torch stage. Left unbounded on a many-core box torch "
                        "oversubscribes every core and the Adam stage runs slower than on 4.")
    p.add_argument("--minima-dir", default=str(HERE / "minima"))
    p.add_argument("--no-library", action="store_true", help="do not seed stage 1 from the minima library")
    p.add_argument("--library-writes", choices=["on", "off"], default="on",
                   help="on (default): each n's verified best becomes the library record if it is larger, and "
                        "every net's best is indexed as a minimum of the 'pipeline' lineage. Writes take the "
                        "library's per-n lock, so a host sharing the library cannot lose one.")
    p.add_argument("--per-n-budget", default=None,
                   help="time box per n: a duration (e.g. 25m), or 'auto' = the remaining --budget split evenly "
                        "over the remaining n, so time an early n leaves unused rolls forward (default: none)")
    p.add_argument("--grace", type=float, default=120.0,
                   help="seconds past --budget before the watchdog stops the run (then 90 s more to a hard exit)")
    return p


def parse_generators(spec: str | None) -> list[str] | None:
    """`--generators a,b` -> ["a", "b"]; an unknown id raises. (The flag used to be passed on as
    the raw string, so any use of it iterated characters and failed on generator 'p'.)"""
    if not spec:
        return None
    names = [g.strip() for g in spec.split(",") if g.strip()]
    unknown = [g for g in names if g not in N.GENERATORS]
    if unknown:
        raise ValueError(f"unknown generators: {unknown} (known: {sorted(N.GENERATORS)})")
    return names


def parse_budget(s: str | None) -> float | None:
    if not s:
        return None
    unit = s[-1].lower()
    mul = {"s": 1, "m": 60, "h": 3600}.get(unit)
    return float(s[:-1]) * mul if mul else float(s)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print(list_checks())
        return 0
    if args.torch_threads > 0:
        try:
            import torch

            torch.set_num_threads(args.torch_threads)
        except ImportError:
            pass
    baselines = load_baselines()
    if args.calibrate:
        print(calibration_report(args.minima_dir, baselines))
        return 0
    if args.combine:
        print(combine(args.combine, baselines))
        return 0

    try:
        ruleset = R.RuleSet.from_flags(args.no_rule,
                                       args.only_rules.split(",") if args.only_rules else None)
        hset = H.HeuristicSet.from_flags(args.no_heuristic, args.weight)
        moves = NE.MoveSet.from_flags(args.no_move, args.move_weight,
                                      args.move_kinds.split(",") if args.move_kinds else None)
        parse_generators(args.generators)
    except ValueError as exc:
        print(f"error: {exc}\n\nrun with --list to see every rule, heuristic, generator and move",
              file=sys.stderr)
        return 2
    lib = None if args.no_library else MinimaLibrary(args.minima_dir)
    writes = lib is not None and args.library_writes == "on"
    ns = parse_ns(args.n_range)
    out_dir = Path(args.out or HERE / "runs" / f"pipeline-{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = parse_budget(args.budget)
    deadline = (time.time() + budget) if budget else None

    log_fh = (out_dir / "log.jsonl").open("a")
    log_lock = threading.Lock()

    def log(rec: dict):
        rec.setdefault("t", round(time.time(), 3))
        with log_lock:
            log_fh.write(json.dumps(rec, default=str) + "\n")
            log_fh.flush()

    if args.ablate:
        print(ablate(args.ablate, ns, args, ruleset, hset, lib, baselines))
        return 0
    print(f"n = {ns}")
    print(f"rules off: {ruleset.disabled() or 'none'}; heuristics off: {hset.disabled() or 'none'}; "
          f"moves off: {sorted(set(NE.MOVES) - set(moves.enabled())) or 'none'}")
    print(f"results -> {out_dir}" + ("" if writes else "  (library writes off)"), flush=True)
    log({"event": "start", "argv": sys.argv[1:], "ns": ns, "rules_off": ruleset.disabled(),
         "heuristics_off": hset.disabled(), "moves_off": sorted(set(NE.MOVES) - set(moves.enabled())),
         "pid": os.getpid(), "host": os.uname().nodename, "library_writes": writes})

    # A stop request (SIGTERM from a supervisor or the watchdog, SIGINT from a terminal) unwinds
    # through `finally`, so the report and summary of every finished n are always written.
    class Stop(BaseException):
        """BaseException, not Exception: the stage-2 code treats `except Exception` around the
        optimizer as a failed individual, and a stop request must not be swallowed there."""

    def on_signal(signum, frame):
        raise Stop(f"signal {signum}")

    prev = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    finished = threading.Event()        # set when main returns, so the watchdog never outlives it

    if deadline:
        def watchdog():
            if finished.wait(max(0.0, deadline + args.grace - time.time())):
                return
            log({"event": "watchdog_stop", "over_budget_s": round(time.time() - deadline, 1)})
            os.kill(os.getpid(), signal.SIGTERM)
            if finished.wait(90):
                return
            log({"event": "watchdog_hard_exit"})
            os._exit(3)
        threading.Thread(target=watchdog, daemon=True).start()

    pool = None
    if not args.stage1_only and args.workers > 1:
        import multiprocessing as mp
        # single-threaded BLAS in the polish pool, as run_overnight.py gives its workers: the
        # pool already runs one process per core, and pool_init sets these only after the child
        # has imported numpy, when OpenBLAS has already sized its thread pool
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        pool = mp.get_context("spawn").Pool(args.workers, initializer=O.pool_init)
    results, total_ledger = [], R.Ledger()
    stopped = None
    try:
        for idx, n in enumerate(ns):
            if deadline and time.time() > deadline:
                print(f"budget reached, stopping before n={n}")
                break
            done_file = out_dir / f"n{n:03d}.json"
            if done_file.exists() and not args.stage1_only:
                # resume: the rng of every n is seeded from (seed, n) alone, so a finished n is
                # exactly what re-running it would give, and skipping it loses nothing
                r = json.loads(done_file.read_text())
                r["ledger"] = R.Ledger.from_json(r.get("ledger", {}))
                results.append(r)
                total_ledger.merge(r["ledger"])
                if writes and r.get("best") and r["best"].get("config"):
                    lib.update_record(n, r["best"]["config"], {"run": out_dir.name, "via": "pipeline",
                                                               "proposer": Mn.PIPELINE_PROPOSER})
                log({"event": "resumed_skip", "n": n})
                print(f"n={n:<3} already done in {done_file.name}, skipped", flush=True)
                continue
            rng = np.random.default_rng(args.seed + 1000 * n)
            t0 = time.time()
            n_deadline = deadline
            if args.per_n_budget == "auto" and deadline:
                todo = sum(1 for m in ns[idx:] if not (out_dir / f"n{m:03d}.json").exists())
                n_deadline = t0 + (deadline - t0) / max(1, todo)
            elif args.per_n_budget and args.per_n_budget != "auto":
                n_deadline = min(deadline or float("inf"), t0 + parse_budget(args.per_n_budget))
            s1 = stage1(n, args, rng, ruleset, hset, lib, baselines)
            log({"event": "stage1", "n": n, **s1.to_json()})
            print(f"n={n:<3} stage 1: {s1.ledger.accepted}/{s1.ledger.seen} candidates passed "
                  f"the rules, {s1.distinct} distinct nets ({s1.duplicates} repeats) "
                  f"in {time.time() - t0:.0f}s"
                  + (f", best score {s1.ranked[0].score.total:.3f} {s1.ranked[0].cand.family}"
                     if s1.ranked else ", NONE SURVIVED"), flush=True)
            for line in s1.ledger.table().splitlines()[1:]:
                print("      " + line.strip())
            if args.stage1_only:
                results.append({"n": n, "best": None, "per_net": [], "ledger": s1.ledger,
                                "stage1": s1.to_json(), "coverage": N.coverage(s1)})
                total_ledger.merge(s1.ledger)
                continue
            log({"event": "n_start", "n": n, "time_box_s": round(n_deadline - t0, 1) if n_deadline else None})
            r = stage2(n, s1, args, rng, ruleset, hset, moves, baselines, pool, log, n_deadline,
                       lib=lib if writes else None, run_name=out_dir.name)
            results.append(r)
            total_ledger.merge(r["ledger"])
            base = baselines.get(n)
            a = r["best"]["area"] if r["best"] else None
            if writes and r["best"] and r["best"].get("config"):
                # the record goes in before n's result file: a result file on disk means the library has it
                r["record_updated"] = lib.update_record(
                    n, r["best"]["config"], {"run": out_dir.name, "via": "pipeline", "proposer": Mn.PIPELINE_PROPOSER,
                                             "net_family": r["best"].get("net_family"),
                                             "origin": r["best"].get("origin")})
                if r["record_updated"]:
                    log({"event": "library_record", "n": n, "area": a})
            print(f"n={n:<3} stage 2: best {a if a is None else round(a, 6)} "
                  f"(record {base['A'] if base else '-'}, {classify(a, base)})"
                  + (" -> new library record" if r.get("record_updated") else "")
                  + f" in {time.time() - t0:.0f}s", flush=True)
            if r["best"] and r["best"].get("config"):
                (out_dir / f"best-n{n:03d}.json").write_text(json.dumps(r["best"]["config"], indent=1))
            tmp = done_file.with_name(done_file.name + ".tmp")
            tmp.write_text(json.dumps({k: (v.to_json() if isinstance(v, R.Ledger) else v) for k, v in r.items()},
                                      indent=1, default=str))
            os.replace(tmp, done_file)
    except Stop as exc:
        stopped = str(exc)
        print(f"stopping on {stopped}; writing the report for the {len(results)} finished n", flush=True)
        log({"event": "stopped", "why": stopped, "finished": [r["n"] for r in results]})
    finally:
        for sig, h in prev.items():
            signal.signal(sig, h)
        if pool is not None:
            pool.terminate()
            pool.join()
        table = report(results, baselines)
        print()
        print(table)
        print()
        print(total_ledger.table())
        (out_dir / "report.txt").write_text(table + "\n\n" + total_ledger.table() + "\n")
        (out_dir / "summary.json").write_text(json.dumps(
            {"ns": ns, "rules_off": ruleset.disabled(), "heuristics_off": hset.disabled(),
             "stopped": stopped, "ledger": total_ledger.to_json(),
             "results": [{"n": r["n"], "best_area": (r["best"] or {}).get("area"),
                          "coverage": r.get("coverage"),
                          "per_net": r.get("per_net", [])} for r in results]},
            indent=1, default=str))
        log({"event": "end", "stopped": stopped})
        finished.set()
        log_fh.close()
    return 0 if stopped is None else 4

if __name__ == "__main__":
    raise SystemExit(main())
