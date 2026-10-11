"""Move-neighbourhood search around a seed topology (EXPERIMENTS.md change 12; promoted from the
2026-09-24 run's `runs/4h-20260924/neighbours.py`).

Level 1: every topology move of fence/netevolve.py is applied to the seed under many rng seeds --
the moves are randomized, so sampling with dedupe by canonical signature is how their applications
are enumerated. Every distinct neighbour that passes the hard rules is screened (Adam + polish +
referee, cheap settings). Level 2: the same from the best `--beam` level-1 neighbours. The best few
are re-optimized at full settings; a referee-verified (1e-9 and 1e-12) result above the library goes
in with `update_record` and is indexed as a minimum of `--lineage`, so `MinimaLibrary.best()` (which
seeds later runs) sees it.

`--focus` restricts attention to a region: level-1 neighbours whose change touches one of the focus
points (within its radius, in the seed configuration's own coordinates) are screened first, the rest
only after them (or never, with `--focus-only`).

Stopping (the `--stop-file` appears, SIGTERM, or the per-n budget runs out) keeps the best screened
result, which the screen already referee-verified.

usage:
  neighbourhood.py --n 41 [--seed library|file.json] --tag r7 --budget-min 15 --out runs/x
  neighbourhood.py --queue 35,44,41 --tag r7 --budget-min 15 --out runs/x
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

TOPO_MOVES = ["join_to_t", "t_to_join", "slide_t_past", "repin", "add_face", "remove_face", "face_fill"]


# ---------------------------------------------------------------------------
# neighbour enumeration
# ---------------------------------------------------------------------------

def graph_edges_xy(sk: Bd.Sketch) -> np.ndarray:
    """[m, 4] endpoint coordinates of every graph edge (fence pieces between consecutive vertices)."""
    X = sk.coords()
    E = sk.topology().edges()
    return np.array([[*X[a], *X[b]] for a, b in E], dtype=float).reshape(-1, 4)


def changed_points(sk0: Bd.Sketch, sk: Bd.Sketch, tol: float = 1e-4) -> np.ndarray:
    """Endpoints of the graph edges present in one of the two layouts and not in the other: where a
    move changed the net. Layouts are compared in their own coordinates (a move keeps the frame)."""
    from scipy.spatial import cKDTree
    A, B = graph_edges_xy(sk0), graph_edges_xy(sk)
    out = []
    for P, Q in ((A, B), (B, A)):
        if len(Q) == 0:
            out.append(P)
            continue
        Qs = np.vstack([Q, Q[:, [2, 3, 0, 1]]])
        d, _ = cKDTree(Qs).query(P)
        out.append(P[d > tol])
    D = np.vstack(out) if out else np.zeros((0, 4))
    return D.reshape(-1, 2)


def touches(points: np.ndarray, focus: list[tuple[float, float, float]]) -> bool:
    """True when any changed point lies within the radius of a focus point (x, y, r)."""
    if len(points) == 0 or not focus:
        return False
    for x, y, r in focus:
        if np.any(np.hypot(points[:, 0] - x, points[:, 1] - y) <= r):
            return True
    return False


def neighbours(ev: NE.NetEvolver, sk0: Bd.Sketch, seen: set, samples: int, rng_base: int,
               moves=TOPO_MOVES, focus=None) -> list[tuple]:
    """Distinct legal neighbours of sk0, as (move, signature, sketch, focused). New signatures are
    added to `seen`. With a focus, focused neighbours come first."""
    out = []
    for mv in moves:
        for k in range(samples):
            ev.rng = np.random.default_rng(rng_base + 7919 * k + sum(map(ord, mv)) % 1000)
            sk = sk0.copy()
            if ev.apply_move(sk, mv) is None:
                continue
            cand = ev._candidate(sk, mv)
            if cand is None:
                continue
            sig = ev._signature(cand.topo)
            if sig is None or sig in seen:
                continue
            seen.add(sig)
            foc = False
            if focus:
                try:
                    foc = touches(changed_points(sk0, sk), focus)
                except (G.InvalidTopology, Bd.BuildError):
                    foc = False
            out.append((mv, sig, sk, foc))
    if focus:
        out.sort(key=lambda x: not x[3])      # stable: focused first, move order kept within each
    return out


def unique_log(out: Path, n: int, tag: str) -> Path:
    """nb-n<NNN>-<tag>.jsonl, never an existing file: a second run with the same (n, tag) gets a
    suffix instead of interleaving its lines with the first (as happened once on 09-24)."""
    p = out / f"nb-n{n:03d}-{tag}.jsonl"
    k = 1
    while p.exists():
        k += 1
        p = out / f"nb-n{n:03d}-{tag}.{k}.jsonl"
    return p


def load_focus(spec: str | None, n: int, X0: np.ndarray) -> list[tuple[float, float, float]] | None:
    """--focus file: {"<n>": [[x, y, r], ...]} (or {"<n>": {"points": [[x, y], ...], "radius": r}})
    in the seed configuration's coordinates; returned in the sketch frame (Sketch.from_config
    centres the layout on its vertex mean)."""
    if not spec:
        return None
    d = json.load(open(spec)).get(str(n))
    if not d:
        return None
    if isinstance(d, dict):
        d = [[x, y, d.get("radius", 0.5)] for x, y in d["points"]]
    c = np.asarray(X0).mean(axis=0)
    return [(float(x) - c[0], float(y) - c[1], float(r)) for x, y, r in d]


# ---------------------------------------------------------------------------
# one n
# ---------------------------------------------------------------------------

class Stop:
    """Stop request: the stop file, SIGTERM, or the deadline."""

    def __init__(self, stop_file: str | None):
        self.stop_file = stop_file
        self.signalled = False

    def __call__(self, deadline: float | None = None) -> bool:
        return (self.signalled or (self.stop_file is not None and os.path.exists(self.stop_file))
                or (deadline is not None and time.time() > deadline))


def run_one(n: int, a, lib: Mn.MinimaLibrary, pool, stop: Stop) -> dict:
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    base = json.load(open(ROOT / "baselines.json"))["baselines"]
    rec = base[str(n)]["A"]
    cur = lib.record(n)
    libA = float(cur["area"]) if cur else 0.0
    if a.seed == "library":
        cfg0 = cur["config"]
    else:
        cfg0 = json.load(open(a.seed.format(n=n)))
        cfg0 = cfg0.get("config", cfg0)
    topo0, X0 = G.config_from_json(cfg0)
    sk0 = Bd.Sketch.from_config(topo0, X0)
    focus = load_focus(a.focus, n, X0)
    moves = [m for m in a.moves.split(",") if m] + [m for m in (a.extra_moves or "").split(",") if m]
    moves2 = [m for m in a.moves2.split(",") if m] if a.moves2 else moves
    ctx = {"record_area": rec, "min_faces": R.min_faces_for(rec)}
    screen = O.OptConfig(batch=a.screen_batch, steps=a.screen_steps, topk=3, verify_top=1)
    ev = NE.NetEvolver(n, sk0, ctx=ctx, cfg=screen, device=a.device, pool=pool)
    t_start = time.time()
    t_screen_end = t_start + 60 * a.budget_min * (1 - a.full_frac)
    t_end = t_start + 60 * a.budget_min
    logp = unique_log(out, n, a.tag)
    logf = open(logp, "a")

    def log(d):
        d["t"] = time.strftime("%H:%M:%S")
        logf.write(json.dumps(d) + "\n")
        logf.flush()

    seen = {ev._signature(topo0)}
    log({"event": "start", "n": n, "tag": a.tag, "record": rec, "library": libA, "seed": a.seed,
         "moves": moves, "moves2": moves2, "focus": focus, "seed_features": topo0.features(), "min_faces": ctx["min_faces"]})
    scored, L1n, L2n, sweep = [], 0, 0, 0
    # recentring bar: the library (default), or -inf with --climb, so that a seed below the library
    # (another read of the picture, say) is hill-climbed from rather than only probed around
    cur_sk, cur_area, cur_focus = sk0, (-float("inf") if a.climb else libA), focus
    while not stop(t_screen_end) and sweep < a.max_sweeps:
        # one sweep: level 1 around the current centre, then level 2 from its best `beam`
        base_rng = 1000 + a.rng_offset + 100003 * sweep
        L1 = neighbours(ev, cur_sk, seen, a.samples, base_rng, moves, cur_focus)
        if cur_focus and a.focus_only:
            L1 = [x for x in L1 if x[3]]
        L1n += len(L1)
        log({"event": "level1", "n": n, "sweep": sweep, "distinct": len(L1), "focused": sum(1 for x in L1 if x[3]),
             "by_move": {m: sum(1 for x in L1 if x[0] == m) for m in moves}})
        sweep_scored = []
        for mv, sig, sk, foc in L1:
            if stop(t_screen_end):
                break
            ind = NE.Individual(sk, mv, 1)
            ev.evaluate(ind)
            sweep_scored.append((ind.area if ind.area is not None else -1, mv, sig, ind))
            log({"event": "eval", "level": 1, "sweep": sweep, "move": mv, "sig": sig[:12], "area": ind.area,
                 "focused": foc})
        sweep_scored.sort(key=lambda x: -x[0])
        new2 = 0
        for area1, mv1, sig1, ind1 in list(sweep_scored[: a.beam]):
            if area1 < 0 or stop(t_screen_end):
                break
            # mutate the optimized geometry of the level-1 neighbour
            L2 = neighbours(ev, ind1.sketch, seen, max(8, a.samples // 3), base_rng + 5000 + L2n, moves2)
            new2 += len(L2)
            log({"event": "level2", "sweep": sweep, "from": sig1[:12], "from_move": mv1, "from_area": area1,
                 "distinct": len(L2)})
            for mv2, sig2, sk2, _ in L2:
                if stop(t_screen_end):
                    break
                ind = NE.Individual(sk2, f"{mv1}+{mv2}", 2)
                ev.evaluate(ind)
                L2n += 1
                sweep_scored.append((ind.area if ind.area is not None else -1, f"{mv1}+{mv2}", sig2, ind))
                log({"event": "eval", "level": 2, "sweep": sweep, "move": f"{mv1}+{mv2}", "sig": sig2[:12],
                     "area": ind.area})
        scored += sweep_scored
        if not L1 and not new2:
            break                          # fresh rng seeds found nothing new: the neighbourhood is exhausted
        # iterate: the next sweep starts from the best layout screened so far when it beats the centre
        scored.sort(key=lambda x: -x[0])
        if scored and scored[0][0] > cur_area + 1e-9 and scored[0][3].config is not None:
            cur_area, cur_sk, cur_focus = scored[0][0], scored[0][3].sketch, None
            log({"event": "recentre", "sweep": sweep, "area": cur_area, "move": scored[0][1]})
        sweep += 1
    scored.sort(key=lambda x: -x[0])
    # full settings on the top distinct candidates, inside the budget
    final = []
    full = O.OptConfig(batch=a.full_batch, steps=a.full_steps)
    for area_s, mv, sig, ind in scored[: a.full_top]:
        if area_s < 0 or stop(t_end):
            break
        if ind.config is not None:
            final.append((area_s, mv, sig, ind.config))      # the screened result is verified already
        cand = ev._candidate(ind.sketch, mv)
        if cand is None:
            continue
        best = None
        for seed in range(a.full_seeds):
            if stop(t_end):
                break
            r = O.optimize_topology(cand.topo, cand.coords, n, full, device=a.device, seed=seed, pool=pool)
            if r["best"] and (best is None or r["best"]["area"] > best["area"]):
                best = r["best"]
        if best:
            final.append((best["area"], mv, sig, best["config"]))
            log({"event": "final", "move": mv, "sig": sig[:12], "screen": area_s, "full": best["area"]})
    if not final:
        # stopped before the full-settings stage: keep the best screened result
        for area_s, mv, sig, ind in scored[:1]:
            if area_s > 0 and ind.config is not None:
                final.append((area_s, mv, sig, ind.config))
    final.sort(key=lambda x: -x[0])
    res = {"n": n, "tag": a.tag, "record": rec, "library": libA, "seed": a.seed, "moves": moves,
           "focused": bool(focus), "sweeps": sweep, "level1": L1n, "level2_evals": L2n, "screened": len(scored),
           "seconds": round(time.time() - t_start, 1),
           "top": [(s[0], s[1], s[2][:12]) for s in scored[:8]]}
    if final:
        area, mv, sig, cfg = final[0]
        v9, v12 = verify(cfg, 1e-9), verify(cfg, 1e-12)
        res["best"] = {"area": area, "move": mv, "sig": sig[:12], "ref_1e9": v9.valid, "ref_1e12": v12.valid,
                       "vs_record": area - rec, "vs_library": area - libA}
        json.dump(cfg, open(out / "results" / f"{logp.stem}-best.json", "w"))
        if v9.valid and v12.valid and area > libA + 1e-9:
            meta = {"run": a.run or out.name, "via": "neighbourhood", "move": mv, "tag": a.tag,
                    "proposer": a.lineage}
            res["record_updated"] = lib.update_record(n, cfg, meta)
            res["indexed"] = index_minimum(lib, a.lineage, n, cfg, meta)
    json.dump(res, open(out / "results" / f"{logp.stem}.json", "w"), indent=1)
    log({"event": "done", **{k: v for k, v in res.items() if k != "top"}})
    logf.close()
    return res


def index_minimum(lib: Mn.MinimaLibrary, lineage: str, n: int, cfg: dict, meta: dict) -> str:
    """Index cfg as a minimum of `lineage` (best() leaves lineage-less records out)."""
    topo, X = G.config_from_json(cfg)
    form = Mn.canonical_form(topo)
    new, doc = lib.add_minimum(lineage, n, cfg, form.signature, X[form.orders[0]], meta)
    return doc["name"] if new else f"already:{doc['name']}"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int)
    g.add_argument("--queue", help="n1,n2,...: run each n in turn, re-reading the library before each")
    ap.add_argument("--seed", default="library", help="'library' (the current record) or a config file; "
                                                      "'{n}' in the path is replaced by n")
    ap.add_argument("--tag", required=True, help="log tag; unique per (n, round)")
    ap.add_argument("--budget-min", type=float, default=15.0, help="per n")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None, help="run name recorded with a new record (default: --out's name)")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--moves", default=",".join(TOPO_MOVES))
    ap.add_argument("--extra-moves", default="", help="moves added to --moves (e.g. pinwheel)")
    ap.add_argument("--moves2", default=None, help="moves of level 2 (default: the level-1 moves)")
    ap.add_argument("--focus", default=None, help="json {n: [[x, y, r], ...]} in the seed's coordinates")
    ap.add_argument("--focus-only", action="store_true")
    ap.add_argument("--screen-batch", type=int, default=1024)
    ap.add_argument("--screen-steps", type=int, default=600)
    ap.add_argument("--full-batch", type=int, default=2048)
    ap.add_argument("--full-steps", type=int, default=1000)
    ap.add_argument("--full-seeds", type=int, default=2)
    ap.add_argument("--full-top", type=int, default=3)
    ap.add_argument("--full-frac", type=float, default=0.2, help="share of the budget kept for full settings")
    ap.add_argument("--climb", action="store_true",
                    help="recentre on the best screened layout even below the library (hill-climb from any seed)")
    ap.add_argument("--max-sweeps", type=int, default=1000, help="iterated sweeps per n inside the budget")
    ap.add_argument("--rng-offset", type=int, default=0, help="shifts the move rng seeds (a new round's samples)")
    ap.add_argument("--lineage", default=Mn.DIGITIZED_PROPOSER)
    ap.add_argument("--library", default=str(ROOT / "minima"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8, help="polish pool size (0: no pool)")
    ap.add_argument("--stop-file", default=None, help="default: <out>/STOP")
    return ap.parse_args(argv)


def main(argv=None) -> list[dict]:
    a = parse_args(argv)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    stop = Stop(a.stop_file or str(Path(a.out) / "STOP"))

    def on_term(signum, frame):
        stop.signalled = True
    signal.signal(signal.SIGTERM, on_term)
    ns = [a.n] if a.n is not None else [int(x) for x in a.queue.split(",") if x.strip()]
    pool = None
    if a.workers > 0:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, initializer=O.pool_init)
    lib = Mn.MinimaLibrary(a.library)
    results = []
    try:
        for n in ns:
            if stop():
                break
            r = run_one(n, a, lib, pool, stop)
            results.append(r)
            b = r.get("best", {})
            print(f"n={n} tag={a.tag} best={b.get('area')} vs_lib={b.get('vs_library')} "
                  f"vs_page={b.get('vs_record')} updated={r.get('record_updated')}", flush=True)
    finally:
        if pool is not None:
            pool.close()
    return results


if __name__ == "__main__":
    main()
