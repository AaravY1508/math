"""Digitize the page's record pictures, optimize every distinct digitized topology, referee-verify,
and write library records (under the per-n lock) when larger. EXPERIMENTS.md change 11.

  ./venv/bin/python digitize_records.py --fetch --ns 17,24,50 --out runs/digitize-<ts>   # CUDA_VISIBLE_DEVICES picks the GPU
  ./venv/bin/python digitize_records.py --ns 44,46 --v2 --seeds 1 --out ...               # the wider v2 grid

Each picture is read with a grid of parameter settings (`GRID`, or `GRID_V2`), every distinct legal
topology gets `optimize_topology` + the referee at 1e-9 and 1e-12; a result above the library goes
in with `MinimaLibrary.update_record`, meta {"via": "digitized", "run": <out dir name>}.
"""
import argparse, itertools, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PAGE = "https://erich-friedman.github.io/packing/fence/{n}.gif"
import numpy as np
from fence import digitize as D, geometry as G, minima as Mn, optimize as O
from fence.referee import verify

GRID = [dict(min_edge=me, rdp_eps=re, straight_deg=sd)
        for me, re, sd in itertools.product([0.10, 0.12, 0.15, 0.08, 0.06], [2.0, 3.0], [16.0, 24.0])]
# v2 (after the 46 / 49 diagnosis): ring-branch junctions, T straightening, close-node merge, fence price
GRID_V2 = [dict(min_edge=me, rdp_eps=2.0, straight_deg=sd, lams=(0.5, 1, 2, 4, -0.5, -1), tee_shift=ts,
                merge_close=me, ring_r=rr)
           for rr, ts, me, sd in itertools.product([4, 5], [0.0, 0.08], [0.06, 0.1, 0.15], [16.0, 24.0, 32.0])]


# finer reads (scale 5): lower min_edge, so short stubs (small faces, close junctions) survive
GRID_FINE = [dict(min_edge=me, rdp_eps=re, straight_deg=sd, lams=(0.5, 1, -0.5))
             for me, re, sd in itertools.product([0.03, 0.05, 0.08, 0.10], [2.0, 3.5], [16.0, 24.0])]


def distinct_topologies(n, log, gif, grid=None, scale=3):
    out = {}
    for kw in (grid or GRID):
        try:
            topo, P, info = D.digitize(str(gif), n, scale=scale, **kw)
        except Exception as e:  # noqa: BLE001
            log({"n": n, "event": "digitize_fail", "kw": kw, "error": str(e)[:200]})
            continue
        sig = Mn.canonical_form(topo).signature
        if sig not in out:
            out[sig] = (topo, P, kw, topo.features())
            log({"n": n, "event": "topology", "kw": kw, "features": topo.features(), "sig": sig[:12]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    ap.add_argument("--gpu", default="0", help="label for the log file name only")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out", required=True, help="run directory (results/, topos/, log)")
    ap.add_argument("--gif-dir", default=None, help="where <n>.gif live (default <out>/gif)")
    ap.add_argument("--fetch", action="store_true", help="download missing <n>.gif from the page")
    ap.add_argument("--stop-file", default=None, help="stop before the next n/topology once this file exists")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--v2", action="store_true", help="wide v2 grid; results as results/v2-n<n>.json")
    ap.add_argument("--scale", type=int, default=3, help="picture upscale before thinning (3: the 09-24 reads)")
    ap.add_argument("--fine", action="store_true", help="GRID_FINE (low min_edge); results as results/s<scale>-n<n>.json")
    a = ap.parse_args()
    HERE = Path(a.out)
    RUN = HERE.name
    gifs = Path(a.gif_dir) if a.gif_dir else HERE / "gif"
    for d in (HERE / "results", HERE / "topos", gifs):
        d.mkdir(parents=True, exist_ok=True)
    a.stop_file = a.stop_file or str(HERE / "STOP")
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(8, initializer=O.pool_init)
    base = json.load(open(ROOT / "baselines.json"))["baselines"]
    lib = Mn.MinimaLibrary(ROOT / "minima")
    logf = open(HERE / f"log-gpu{a.gpu}.jsonl", "a")

    def log(d):
        d["t"] = time.strftime("%H:%M:%S")
        logf.write(json.dumps(d) + "\n"); logf.flush()
    for n in [int(x) for x in a.ns.split(",")]:
        if os.path.exists(a.stop_file):
            log({"event": "stop_file", "n": n}); break
        pre = "v2-" if a.v2 else (f"s{a.scale}-" if a.fine or a.scale != 3 else "")
        if (HERE / "results" / f"{pre}n{n:03d}.json").exists():
            log({"event": "skip_done", "n": n}); continue
        t0 = time.time()
        gif = gifs / f"{n}.gif"
        if a.fetch and not gif.exists():
            import urllib.request
            urllib.request.urlretrieve(PAGE.format(n=n), gif)
        tops = distinct_topologies(n, log, gif, GRID_V2 if a.v2 else (GRID_FINE if a.fine else None), scale=a.scale)
        rec = base[str(n)]["A"]
        libA = float(lib.record(n)["area"])
        res = {"n": n, "record": rec, "library": libA, "topologies": []}
        best = None
        for sig, (topo, P, kw, feat) in tops.items():
            json.dump({"topology": topo.to_json(), "coords": P.tolist(), "kw": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}},
                      open(HERE / "topos" / f"{pre}n{n:03d}-{sig[:10]}.json", "w"))
            for seed in range(a.seeds):
                if os.path.exists(a.stop_file):
                    break
                cfg = O.OptConfig(batch=a.batch, steps=a.steps)
                ts = time.time()
                try:
                    r = O.optimize_topology(topo, P, n, cfg, device="cuda", seed=seed, pool=pool)
                except Exception as e:  # noqa: BLE001
                    log({"n": n, "event": "opt_error", "sig": sig[:12], "error": repr(e)[:300]})
                    continue
                area = r["best"]["area"] if r["best"] else None
                ent = {"sig": sig[:12], "kw": kw, "features": feat, "seed": seed, "area": area,
                       "adam": r["adam_areas"][:3], "polished": r["polished_areas"][:3], "s": round(time.time() - ts, 1)}
                res["topologies"].append(ent)
                log({"n": n, "event": "opt", **ent})
                if r["best"] and (best is None or area > best["area"]):
                    best = {"area": area, "config": r["best"]["config"], "sig": sig[:12], "features": feat}
        if best:
            cfg = best["config"]
            v9, v12 = verify(cfg, 1e-9), verify(cfg, 1e-12)
            res["best"] = {"area": best["area"], "sig": best["sig"], "features": best["features"],
                           "ref_1e9": v9.valid, "ref_1e12": v12.valid, "area_ref": float(v9.total_area) if v9.valid else None,
                           "vs_record": best["area"] - rec, "vs_library": best["area"] - libA}
            json.dump(cfg, open(HERE / "results" / f"{pre}n{n:03d}-best.json", "w"))
            if v9.valid and v12.valid and not a.no_write:
                meta = {"run": RUN, "via": "digitized-v2" if a.v2 else ("digitized-" + pre.rstrip("-") if pre else "digitized"),
                        "proposer": Mn.DIGITIZED_PROPOSER}
                # index it as a minimum of the digitized lineage, so MinimaLibrary.best() -- which
                # seeds later runs -- can see it (a record with no lineage is left out there)
                topo_b, X_b = G.config_from_json(cfg)
                form = Mn.canonical_form(topo_b)
                lib.add_minimum(Mn.DIGITIZED_PROPOSER, n, cfg, form.signature, X_b[form.orders[0]], meta)
                if best["area"] > libA + 1e-9:
                    res["record_updated"] = lib.update_record(n, cfg, meta)
        res["s"] = round(time.time() - t0, 1)
        json.dump(res, open(HERE / "results" / f"{pre}n{n:03d}.json", "w"), indent=1)
        log({"event": "n_done", "n": n, "best": res.get("best"), "record_updated": res.get("record_updated"), "s": res["s"]})
    pool.close()


if __name__ == "__main__":
    main()
