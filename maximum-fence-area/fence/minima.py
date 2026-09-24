"""Per-proposer library of distinct local minima, kept across runs.

    ./venv/bin/python -m fence.minima import-run runs/<timestamp>   # seed records from an old run
    ./venv/bin/python -m fence.minima summary                       # what the library holds

Terms
-----
proposer   the lineage an evolved program belongs to: the hand-written seed program it
           descends from (fence.evolve.Program.proposer). The id is a content hash of that
           seed program's source, so the same proposer has the same id in every run.
minimum    a referee-verified configuration on which one proposer's search at one n
           stabilized (fence.evolve decides when). Two minima are the same when their
           layout topology matches (same canonical signature, below) and their areas agree
           within MINIMUM_EPS. Only distinct minima are ever indexed.

Library directory
-----------------
    proposers/<proposer>/proposer.json      the lineage's root program (genome + source)
    proposers/<proposer>/N<n>.<k>.json      k-th distinct minimum of this proposer at n (k = 0, 1, ...)
    proposers/<proposer>/N<n>.trivial.json  this proposer's search at n stabilized on the grid-trivial answer
    records/N<n>.json                       best verified configuration for n over all runs

    locks/N<n>.lock                         held while a writer reads-modifies-writes anything at n

Every write is atomic (temp file + rename), and every read-modify-write at n (a record, a
minimum's index k, a trivial marker's count) runs under n's lock file, created with O_EXCL
so it also excludes writers on other hosts sharing the library over NFS. Runs still give
each host a disjoint n-range, but a mistake there now costs time rather than a record.

Topology signature
------------------
A layout topology is a connected plane graph (vertices: joints and T-junction pins;
edges: the pieces of the fences between them) together with the partition of its edges
into fences, which is what tells a T-junction from a joint. Its canonical code is the
lexicographically smallest code of a breadth-first walk of the rotation system, started
from every dart and in both orientations (Weinberg's method for planar maps). The code
does not change under relabeling of vertices or fences, rigid motions or reflection, and
two topologies get the same code only if they are the same layout. The rotation system
is read off the face cycles, so the signature never depends on floating-point angles:
a sketch and the optimized configuration built on it always share one signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fence import geometry as G

MINIMUM_EPS = 1e-6       # same topology and |area difference| <= this: the same minimum
LOCK_STALE_S = 120.0     # a lock older than this was left by a dead writer and may be taken over
LOCK_POLL_S = 0.02
PIPELINE_PROPOSER = "pipeline"   # lineage id of configurations found by pipeline.py (no seed program)


# ---------------------------------------------------------------------------
# topology signature
# ---------------------------------------------------------------------------

@dataclass
class CanonicalForm:
    signature: str             # short hash of the canonical code
    orders: list[list[int]]    # one per automorphism: vertex ids listed in canonical label order


def _darts(topo: G.Topology):
    """Dart 2k runs along edge k, dart 2k+1 = 2k ^ 1 is its twin; each carries its fence id."""
    ids, src, dst, fen = {}, [], [], []
    for f, c in enumerate(topo.chains):
        for k in range(len(c) - 1):
            for a, b in ((c[k], c[k + 1]), (c[k + 1], c[k])):
                ids[(a, b)] = len(src)
                src.append(a)
                dst.append(b)
                fen.append(f)
    return ids, src, dst, fen


def _walk(start, rot, src, dst, fen, deg, best):
    """Code of the breadth-first walk from dart `start`, following rotation `rot`.

    Returns (code, order, smaller) where `order` lists the vertices by canonical label and
    `smaller` says the code is strictly below `best`; returns None as soon as the code is
    known to exceed `best` (all codes of one topology have the same length).
    """
    label = {src[start]: 0}
    flabel = {}
    first = [start]            # first[i] is the dart through which vertex i is walked
    code = []
    equal = best is not None   # still equal to `best` on the prefix written so far
    i = 0
    while i < len(first):
        d0 = first[i]
        i += 1
        items = [-deg[src[d0]]]
        d = d0
        while True:
            w = dst[d]
            if w not in label:
                label[w] = len(first)
                first.append(d ^ 1)
            f = fen[d]
            if f not in flabel:
                flabel[f] = len(flabel)
            items.append(label[w])
            items.append(flabel[f])
            d = rot[d]
            if d == d0:
                break
        if equal:
            p = len(code)
            for j, x in enumerate(items):
                b = best[p + j]
                if x != b:
                    if x > b:
                        return None
                    equal = False
                    break
        code.extend(items)
    return code, [src[d] for d in first], best is None or not equal


def canonical_form(topo: G.Topology, max_orders: int = 16) -> CanonicalForm:
    """Canonical signature of a layout topology plus the vertex orders that realize it."""
    ids, src, dst, fen = _darts(topo)
    D = len(src)
    ccw = [-1] * D
    for cyc in list(topo.faces) + [topo.outer]:
        m = len(cyc)
        for i in range(m):
            u, w, x = cyc[i - 1], cyc[i], cyc[(i + 1) % m]
            # a face walk u -> w -> x leaves w along the dart just clockwise of w -> u
            # (fence.geometry.trace_faces), i.e. w -> u is the next dart counter-clockwise
            ccw[ids[(w, x)]] = ids[(w, u)]
    if -1 in ccw:
        raise G.InvalidTopology("face cycles do not cover every dart")
    cw = [0] * D
    for d, e in enumerate(ccw):
        cw[e] = d
    deg = {}
    for v in src:
        deg[v] = deg.get(v, 0) + 1
    top = max(deg.values())
    starts = [d for d in range(D) if deg[src[d]] == top]   # isomorphism-invariant filter
    best, orders = None, []
    for rot in (ccw, cw):
        for s in starts:
            res = _walk(s, rot, src, dst, fen, deg, best)
            if res is None:
                continue
            code, order, smaller = res
            if smaller:
                best, orders = code, [order]
            elif len(orders) < max_orders:
                orders.append(order)
    if len(orders[0]) != topo.n_vertices:
        raise G.InvalidTopology("topology is not connected")
    return CanonicalForm(signature=hashlib.sha1(repr(best).encode()).hexdigest()[:16], orders=orders)


# ---------------------------------------------------------------------------
# rigid comparison of configurations of one topology
# ---------------------------------------------------------------------------

def rigid_rms(X: np.ndarray, Y: np.ndarray) -> float:
    """RMS per-vertex distance between [V, 2] point sets after the best rotation,
    translation and reflection of Y onto X (closed form in 2D)."""
    Xc = np.asarray(X, float) - np.mean(X, axis=0)
    Yc = np.asarray(Y, float) - np.mean(Y, axis=0)
    sxx, syy = float((Xc ** 2).sum()), float((Yc ** 2).sum())
    best = math.inf
    for Z in (Yc, Yc * np.array([1.0, -1.0])):
        a = float((Xc * Z).sum())
        b = float((Xc[:, 0] * Z[:, 1] - Xc[:, 1] * Z[:, 0]).sum())
        best = min(best, sxx + syy - 2.0 * math.hypot(a, b))
    return math.sqrt(max(best, 0.0) / len(Xc))


def avoid_images(form: CanonicalForm, minima: list[dict], per_minimum: int = 8) -> np.ndarray | None:
    """Known minima of this topology in its own vertex labeling: [M, V, 2].

    A minimum is stored with its coordinates in canonical label order; every automorphism
    of the topology gives one image, since the optimizer may approach the minimum under
    any relabeling that preserves the layout. Images that coincide up to a rigid motion
    are kept once (1e-6 sits above rigid_rms's cancellation floor of ~sqrt(machine eps)).
    """
    out = []
    for m in minima:
        C = np.asarray(m["canon_coords"], dtype=float)
        imgs = []
        for order in form.orders:
            Y = np.empty_like(C)
            Y[order] = C
            if all(rigid_rms(Y, Z) > 1e-6 for Z in imgs):
                imgs.append(Y)
            if len(imgs) >= per_minimum:
                break
        out.extend(imgs)
    return np.stack(out) if out else None


def same_minimum(m: dict, signature: str, area: float) -> bool:
    return m["signature"] == signature and abs(float(m["area"]) - float(area)) <= MINIMUM_EPS


# ---------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, obj) -> None:
    # the library may sit on NFS shared by several hosts: the temp name must be unique across them
    tmp = path.with_name(f"{path.name}.tmp.{socket.gethostname()}.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class NLock:
    """Exclusive per-n lock: a file created with O_EXCL, which NFSv3 honours across hosts.

    Held for milliseconds (one read and one atomic write). A lock whose file is older than
    LOCK_STALE_S belongs to a writer that died holding it; it is moved aside with a rename,
    which only one contender can win, and the contender that wins retries the create.
    """

    def __init__(self, path: Path, stale_s: float = LOCK_STALE_S):
        self.path = path
        self.stale_s = stale_s
        self.token = f"{socket.gethostname()}:{os.getpid()}:{time.time():.6f}"

    def __enter__(self):
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue                     # released between the create and the stat
                if age > self.stale_s:
                    aside = self.path.with_name(f"{self.path.name}.stale.{socket.gethostname()}.{os.getpid()}")
                    try:
                        os.rename(self.path, aside)
                        aside.unlink(missing_ok=True)
                    except FileNotFoundError:
                        pass                     # another contender took it over first
                    continue
                time.sleep(LOCK_POLL_S)
                continue
            with os.fdopen(fd, "w") as fh:
                fh.write(self.token)
            return self

    def __exit__(self, *exc):
        try:
            if self.path.read_text() == self.token:
                self.path.unlink()
        except FileNotFoundError:
            pass
        return False


class MinimaLibrary:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "proposers").mkdir(parents=True, exist_ok=True)
        (self.root / "records").mkdir(parents=True, exist_ok=True)
        (self.root / "locks").mkdir(parents=True, exist_ok=True)

    def lock(self, n: int) -> NLock:
        return NLock(self.root / "locks" / f"N{n}.lock")

    def _pdir(self, proposer: str) -> Path:
        return self.root / "proposers" / proposer

    # -- proposers -------------------------------------------------------------
    def register_proposer(self, proposer: str, genome=None, source=None) -> None:
        p = self._pdir(proposer) / "proposer.json"
        if p.exists():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(p, {"proposer": proposer, "genome": genome, "source": source,
                          "registered": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def proposers(self) -> list[str]:
        return sorted(p.name for p in (self.root / "proposers").iterdir() if p.is_dir())

    # -- minima ------------------------------------------------------------------
    @staticmethod
    def _index_of(path: Path, n: int) -> int | None:
        k = path.name[len(f"N{n}."):-len(".json")]
        return int(k) if k.isdigit() else None

    def minima_of(self, proposer: str, n: int) -> list[dict]:
        out = []
        for f in self._pdir(proposer).glob(f"N{n}.*.json"):
            if self._index_of(f, n) is not None and (m := _read(f)) is not None:
                out.append(m)
        return sorted(out, key=lambda m: m["k"])

    def minima(self, n: int) -> dict[str, list[dict]]:
        """proposer -> its indexed minima at n, in index order."""
        return {p: ms for p in self.proposers() if (ms := self.minima_of(p, n))}

    def add_minimum(self, proposer: str, n: int, config: dict, signature: str, canon_coords, meta=None):
        """Index a stabilized minimum as N<n>.<k> unless the proposer already has it.

        Returns (True, doc) for a new minimum, (False, existing doc) for a duplicate.
        """
        area = float(config["verification"]["total_area"])
        with self.lock(n):
            existing = self.minima_of(proposer, n)
            for m in existing:
                if same_minimum(m, signature, area):
                    return False, m
            k = max((m["k"] for m in existing), default=-1) + 1
            doc = {"name": f"N{n}.{k}", "proposer": proposer, "n": n, "k": k, "area": area,
                   "area_50": config["verification"]["total_area"], "signature": signature,
                   "epsilon": MINIMUM_EPS, "canon_coords": np.asarray(canon_coords, dtype=float).tolist(),
                   "indexed": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {}), "config": config}
            self._pdir(proposer).mkdir(parents=True, exist_ok=True)
            _atomic_write(self._pdir(proposer) / f"N{n}.{k}.json", doc)
        return True, doc

    def best(self, n: int, k: int) -> list[dict]:
        """Up to k verified configurations for n with distinct topologies, largest area first:
        the all-time record and the proposers' indexed minima. Each is {"name", "area",
        "proposer", "signature", "config"}; a record whose lineage is unknown (runs before
        lineages were tracked) takes it from an indexed minimum or a grid-trivial marker of the
        same topology, or is left out."""
        docs = [{"name": m["name"], "area": float(m["area"]), "proposer": p, "signature": m["signature"],
                 "config": m["config"]} for p, ms in self.minima(n).items() for m in ms]
        rec = self.record(n)
        if rec is not None:
            sig = canonical_form(G.config_from_json(rec["config"])[0]).signature
            prop = rec.get("proposer") or rec["config"].get("proposer") or \
                next((d["proposer"] for d in docs if d["signature"] == sig), None) or \
                next((p for p in sorted(self.trivial_proposers(n)) if self._trivial_signature(p, n) == sig), None)
            if prop:
                docs.insert(0, {"name": f"record:N{n}", "area": float(rec["area"]), "proposer": prop,
                                "signature": sig, "config": rec["config"]})
        out, seen = [], set()
        for d in sorted(docs, key=lambda d: -d["area"]):       # stable: the record first on a tie
            if d["signature"] not in seen and d["config"].get("verification", {}).get("valid"):
                seen.add(d["signature"])
                out.append(d)
        return out[:k]

    # -- grid-trivial markers -------------------------------------------------------
    def mark_trivial(self, proposer: str, n: int, config: dict, meta=None) -> None:
        p = self._pdir(proposer) / f"N{n}.trivial.json"
        with self.lock(n):
            old = _read(p) or {}
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            p.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(p, {"proposer": proposer, "n": n, "area": float(config["verification"]["total_area"]),
                              "count": int(old.get("count", 0)) + 1, "first": old.get("first", now), "last": now,
                              **(meta or {}), "config": config})

    def trivial_proposers(self, n: int) -> set[str]:
        return {f.parent.name for f in (self.root / "proposers").glob(f"*/N{n}.trivial.json")}

    def _trivial_signature(self, proposer: str, n: int) -> str | None:
        """Topology signature of the grid-trivial answer a proposer's search at n stabilized on."""
        d = _read(self._pdir(proposer) / f"N{n}.trivial.json")
        try:
            return canonical_form(G.config_from_json(d["config"])[0]).signature
        except (TypeError, KeyError, ValueError, G.InvalidTopology):
            return None

    # -- all-time records -------------------------------------------------------------
    def record(self, n: int) -> dict | None:
        return _read(self.root / "records" / f"N{n}.json")

    def update_record(self, n: int, config: dict, meta=None) -> bool:
        """Keep the best verified configuration for n: only a strictly larger area replaces the
        saved record. The read and the write happen under n's lock, so of two writers racing on
        the same n the larger area always survives, whichever host each is on."""
        area = float(config["verification"]["total_area"])
        with self.lock(n):
            cur = self.record(n)
            if cur is not None and area <= float(cur["area"]) + 1e-12:
                return False
            _atomic_write(self.root / "records" / f"N{n}.json",
                          {"n": n, "area": area, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {}),
                           "config": config})
        return True


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def import_run(lib: MinimaLibrary, run_dir) -> dict:
    """Seed the library from an earlier run's best/ configurations.

    Every verified best becomes (or improves) the record for its n. A grid-trivial best
    that came straight from a seed program also marks that seed's lineage as trivial at
    n: the saved program source is the seed program itself, so its content hash is the
    proposer id. Other bests carry no lineage (runs before this module did not track it).
    """
    from fence import evolve as E

    run_dir = Path(run_dir)
    summary = {"records": [], "trivial": [], "skipped": []}
    for f in sorted((run_dir / "best").glob("n*.json")):
        cfg = _read(f)
        if not cfg or not cfg.get("verification", {}).get("valid"):
            summary["skipped"].append(f.name)
            continue
        n = int(cfg["n"])
        area = float(cfg["verification"]["total_area"])
        meta = {"run": run_dir.name, "proposer": cfg.get("proposer"), "via": "import-run"}
        if lib.update_record(n, cfg, meta):
            summary["records"].append(n)
        if not (E.is_grid_trivial(area, n) and str(cfg.get("program_origin", "")).startswith("seed:")):
            continue
        srcs = list(run_dir.glob(f"worker*/programs/n{n:03d}_best.py"))
        if len(srcs) != 1:
            continue
        source = srcs[0].read_text()
        pid = E.proposer_id(source)
        genome = next((g for g in E.seed_genomes(n) if E.render_program(g) == source), None)
        lib.register_proposer(pid, genome, source)
        lib.mark_trivial(pid, n, cfg, {"run": run_dir.name, "program": cfg.get("found_by_program"),
                                       "origin": cfg.get("program_origin"), "via": "import-run"})
        summary["trivial"].append((n, pid))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minima-dir", default=str(Path(__file__).resolve().parent.parent / "minima"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import-run", help="seed records (and grid-trivial seed lineages) from a run directory")
    imp.add_argument("run_dir")
    sub.add_parser("summary", help="minima, trivial lineages and records per n")
    args = ap.parse_args(argv)
    lib = MinimaLibrary(args.minima_dir)
    if args.cmd == "import-run":
        s = import_run(lib, args.run_dir)
        print(f"records written for n = {s['records']}")
        print("grid-trivial seed lineages: " + (", ".join(f"n={n}:{p}" for n, p in s["trivial"]) or "none"))
        if s["skipped"]:
            print(f"skipped (not verified): {s['skipped']}")
        return
    ns = sorted({int(f.name[1:].split(".")[0]) for f in (lib.root / "proposers").glob("*/N*.json")} |
                {int(f.stem[1:]) for f in (lib.root / "records").glob("N*.json")})
    print(f"library {lib.root}: {len(lib.proposers())} proposers")
    print(f"{'n':>3} | {'record':>12} | {'minima':>6} | {'proposers':>9} | trivial lineages")
    for n in ns:
        rec = lib.record(n)
        mins = lib.minima(n)
        print(f"{n:>3} | {rec['area'] if rec else float('nan'):>12.8f} | {sum(map(len, mins.values())):>6} | "
              f"{len(mins):>9} | {len(lib.trivial_proposers(n))}")


if __name__ == "__main__":
    main()
