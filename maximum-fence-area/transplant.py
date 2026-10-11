"""Transplant records across n (EXPERIMENTS.md change 17).

The record layouts at n and n +- 1 are close relatives (one ring cell more or less). For each n this
builds seeds from the library records at n-1 grown by one fence ("up": `fit_count`, or `add_face`
then pruned back) and at n+1 pruned by one ("down": `fit_count`, or two `remove_face` then regrown),
over many rng seeds, deduplicated by canonical signature and gated by the hard rules (rule 14's face
floor from the page). Every distinct seed is screened (Adam + polish + referee, cheap settings), the
best `--full-top` get full settings, and a referee-verified (1e-9 and 1e-12) result above the library
is written with `update_record` and indexed under the `transplant` lineage.

usage: transplant.py --queue 35,44,... --tag C1 --budget-min 6 --out runs/x
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fence import builder as Bd, geometry as G, minima as Mn, netevolve as NE, optimize as O, rules as R  # noqa: E402
from fence.referee import verify  # noqa: E402
import neighbourhood as NB  # noqa: E402


def transplants(ev: NE.NetEvolver, src: Bd.Sketch, n: int, direction: str, seeds: int, seen: set,
                grow: dict | None = None) -> list[tuple]:
    """Distinct legal n-fence layouts one fence away from `src` (n-1 fences: up, n+1: down)."""
    grow = {**Bd.DEFAULT_GROW, **(grow or {})}
    out = []
    for k in range(seeds):
        rng = np.random.default_rng(10007 * k + (1 if direction == "up" else 2))
        sk = src.copy()
        variant = k % 2
        try:
            if direction == "up":
                if variant == 1:
                    Bd.add_face(sk, rng, grow)          # a 1-3 fence cell, then prune back to n
                Bd.fit_count(sk, n, rng, grow, 1)
            else:
                if variant == 1:
                    Bd.remove_face(sk, rng, grow)       # two fences out, then regrow to n
                    Bd.remove_face(sk, rng, grow)
                Bd.fit_count(sk, n, rng, grow, 1)
        except (Bd.BuildError, G.InvalidTopology, ValueError):
            continue
        if sk.n != n:
            continue
        cand = ev._candidate(sk, f"transplant-{direction}")
        if cand is None:
            continue
        sig = ev._signature(cand.topo)
        if sig is None or sig in seen:
            continue
        seen.add(sig)
        out.append((f"{direction}{'+' if variant else ''}", sig, sk))
    return out


def run_one(n: int, a, lib: Mn.MinimaLibrary, pool, stop: NB.Stop) -> dict:
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    base = json.load(open(ROOT / "baselines.json"))["baselines"]
    rec = base[str(n)]["A"]
    cur = lib.record(n)
    libA = float(cur["area"]) if cur else 0.0
    ctx = {"record_area": rec, "min_faces": R.min_faces_for(rec)}
    screen = O.OptConfig(batch=a.screen_batch, steps=a.screen_steps, topk=3, verify_top=1)
    t0 = time.time()
    t_screen_end = t0 + 60 * a.budget_min * (1 - a.full_frac)
    t_end = t0 + 60 * a.budget_min
    logp = out / f"tp-n{n:03d}-{a.tag}.jsonl"
    k = 1
    while logp.exists():
        k += 1
        logp = out / f"tp-n{n:03d}-{a.tag}.{k}.jsonl"
    logf = open(logp, "a")

    def log(d):
        d["t"] = time.strftime("%H:%M:%S")
        logf.write(json.dumps(d) + "\n")
        logf.flush()

    seen = set()
    if cur:
        t_cur, _ = G.config_from_json(cur["config"])
        seen.add(Mn.canonical_form(t_cur).signature)
    cands = []
    ev = None
    for direction, m in (("up", n - 1), ("down", n + 1)):
        srec = lib.record(m) if 3 <= m <= 50 else None
        if srec is None:
            continue
        topo_s, X_s = G.config_from_json(srec["config"])
        src = Bd.Sketch.from_config(topo_s, X_s)
        ev = ev or NE.NetEvolver(n, src, ctx=ctx, cfg=screen, device=a.device, pool=pool)
        got = transplants(ev, src, n, direction, a.seeds, seen)
        log({"event": "seeds", "n": n, "direction": direction, "from": m, "from_area": srec["area"],
             "distinct": len(got)})
        cands += got
    # interleave directions so a short budget still screens both
    ups = [c for c in cands if c[0].startswith("up")]
    downs = [c for c in cands if c[0].startswith("down")]
    order = [c for pair in zip(ups, downs) for c in pair] + ups[len(downs):] + downs[len(ups):]
    scored = []
    for how, sig, sk in order:
        if stop(t_screen_end):
            break
        ind = NE.Individual(sk, how, 1)
        ev.evaluate(ind)
        scored.append((ind.area if ind.area is not None else -1, how, sig, ind))
        log({"event": "eval", "how": how, "sig": sig[:12], "area": ind.area, "features": sk.topology().features()})
    scored.sort(key=lambda x: -x[0])
    final = []
    full = O.OptConfig(batch=a.full_batch, steps=a.full_steps)
    for area_s, how, sig, ind in scored[: a.full_top]:
        if area_s < 0 or stop(t_end):
            break
        if ind.config is not None:
            final.append((area_s, how, sig, ind.config))
        cand = ev._candidate(ind.sketch, how)
        if cand is None:
            continue
        for seed in range(a.full_seeds):
            if stop(t_end):
                break
            r = O.optimize_topology(cand.topo, cand.coords, n, full, device=a.device, seed=seed, pool=pool)
            if r["best"]:
                final.append((r["best"]["area"], how, sig, r["best"]["config"]))
                log({"event": "final", "how": how, "sig": sig[:12], "screen": area_s, "full": r["best"]["area"]})
    final.sort(key=lambda x: -x[0])
    res = {"n": n, "tag": a.tag, "record": rec, "library": libA, "candidates": len(cands),
           "up": len(ups), "down": len(downs), "screened": len(scored), "seconds": round(time.time() - t0, 1),
           "top": [(s[0], s[1], s[2][:12]) for s in scored[:8]]}
    if final:
        area, how, sig, cfg = final[0]
        v9, v12 = verify(cfg, 1e-9), verify(cfg, 1e-12)
        res["best"] = {"area": area, "how": how, "sig": sig[:12], "ref_1e9": v9.valid, "ref_1e12": v12.valid,
                       "vs_record": area - rec, "vs_library": area - libA}
        json.dump(cfg, open(out / "results" / f"{logp.stem}-best.json", "w"))
        if v9.valid and v12.valid and area > libA + 1e-9:
            meta = {"run": a.run or out.name, "via": "transplant", "how": how, "tag": a.tag,
                    "proposer": Mn.TRANSPLANT_PROPOSER}
            res["record_updated"] = lib.update_record(n, cfg, meta)
            res["indexed"] = NB.index_minimum(lib, Mn.TRANSPLANT_PROPOSER, n, cfg, meta)
    json.dump(res, open(out / "results" / f"{logp.stem}.json", "w"), indent=1)
    log({"event": "done", **{k: v for k, v in res.items() if k != "top"}})
    logf.close()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queue", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budget-min", type=float, default=6.0, help="per n")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--seeds", type=int, default=40, help="rng seeds per direction")
    ap.add_argument("--screen-batch", type=int, default=1024)
    ap.add_argument("--screen-steps", type=int, default=600)
    ap.add_argument("--full-batch", type=int, default=2048)
    ap.add_argument("--full-steps", type=int, default=1000)
    ap.add_argument("--full-seeds", type=int, default=2)
    ap.add_argument("--full-top", type=int, default=3)
    ap.add_argument("--full-frac", type=float, default=0.3)
    ap.add_argument("--library", default=str(ROOT / "minima"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stop-file", default=None)
    a = ap.parse_args(argv)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    stop = NB.Stop(a.stop_file or str(Path(a.out) / "STOP"))
    signal.signal(signal.SIGTERM, lambda s, f: setattr(stop, "signalled", True))
    pool = None
    if a.workers > 0:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, initializer=O.pool_init)
    lib = Mn.MinimaLibrary(a.library)
    results = []
    try:
        for n in [int(x) for x in a.queue.split(",") if x.strip()]:
            if stop():
                break
            r = run_one(n, a, lib, pool, stop)
            results.append(r)
            b = r.get("best", {})
            print(f"n={n} tag={a.tag} cands={r['candidates']} best={b.get('area')} how={b.get('how')} "
                  f"vs_lib={b.get('vs_library')} vs_page={b.get('vs_record')} updated={r.get('record_updated')}",
                  flush=True)
    finally:
        if pool is not None:
            pool.close()
    return results


if __name__ == "__main__":
    main()
