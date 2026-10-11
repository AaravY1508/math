"""neighbourhood.py: neighbour enumeration, the focus filter, unique log tags, and one tiny
end-to-end run on the CPU that writes and indexes a better record in a scratch library."""
import json

import numpy as np

import neighbourhood as NB
from fence import builder as Bd
from fence import geometry as G
from fence import minima as Mn
from fence import netevolve as NE
from fence import optimize as O
from fence import rules as R


def grid_sketch(rows=2, cols=3):
    sk = Bd.grid(rows, cols)
    return Bd.Sketch.from_config(sk.topology(), sk.coords())


def test_neighbours_are_distinct_legal_and_exclude_the_seed():
    sk0 = grid_sketch()
    ev = NE.NetEvolver(sk0.n, sk0, ctx={}, cfg=O.OptConfig(batch=8, steps=5), device="cpu")
    seen = {ev._signature(sk0.topology())}
    L = NB.neighbours(ev, sk0, seen, 6, 0)
    sigs = [s for _, s, _, _ in L]
    assert L and len(sigs) == len(set(sigs))
    assert ev._signature(sk0.topology()) not in sigs
    for mv, sig, sk, foc in L:
        assert sk.n == sk0.n and mv in NB.TOPO_MOVES and foc is False
        assert ev._signature(sk.topology()) == sig


def test_changed_points_locate_a_move():
    sk0 = grid_sketch()
    assert len(NB.changed_points(sk0, sk0.copy())) == 0
    sk = sk0.copy()
    X = sk.coords()
    v = int(np.argmax(X[:, 0] + 0.01 * X[:, 1]))          # the top-right corner
    sk.P[v] = [X[v, 0] + 0.3, X[v, 1] + 0.3]
    sk._dirty()
    pts = NB.changed_points(sk0, sk)
    assert len(pts) and np.all(np.hypot(*(pts - X[v]).T) <= 1.0 + 0.5)
    right, left = (X[v, 0], X[v, 1], 0.4), (X[:, 0].min(), X[:, 1].min(), 0.4)
    assert NB.touches(pts, [right]) and not NB.touches(pts, [left])


def test_focus_puts_touching_neighbours_first():
    sk0 = grid_sketch()
    ev = NE.NetEvolver(sk0.n, sk0, ctx={}, cfg=O.OptConfig(batch=8, steps=5), device="cpu")
    X = sk0.coords()
    focus = [(float(X[:, 0].max()), float(X[:, 1].max()), 0.5)]
    L = NB.neighbours(ev, sk0, {ev._signature(sk0.topology())}, 6, 0, focus=focus)
    flags = [foc for *_, foc in L]
    assert any(flags) and flags == sorted(flags, reverse=True)


def test_load_focus_moves_points_into_the_sketch_frame(tmp_path):
    X0 = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"17": [[2.0, 2.0, 0.4]], "18": {"points": [[0, 0]], "radius": 0.7}}))
    assert NB.load_focus(str(p), 17, X0) == [(1.0, 1.0, 0.4)]
    assert NB.load_focus(str(p), 18, X0) == [(-1.0, -1.0, 0.7)]
    assert NB.load_focus(str(p), 19, X0) is None and NB.load_focus(None, 17, X0) is None


def test_log_tags_never_reuse_a_file(tmp_path):
    p1 = NB.unique_log(tmp_path, 41, "r7")
    p1.write_text("x")
    p2 = NB.unique_log(tmp_path, 41, "r7")
    assert p1.name == "nb-n041-r7.jsonl" and p2 != p1 and not p2.exists()


def test_end_to_end_writes_and_indexes_a_better_record(tmp_path):
    """n=7 from a poor 7-fence layout (one unit square with a wasted fence inside it): a neighbour
    that the moves reach optimizes above it, and the result is written and indexed."""
    sk = Bd.grid(1, 2)           # the domino: 7 fences, A = 2 (the optimum); start from a worse copy
    topo, X = sk.topology(), sk.coords()
    cfg = G.config_to_json(topo, X * np.array([1.0, 0.8]), n=7)
    cfg["verification"] = {"total_area": 1.6, "valid": True, "n_faces": 2, "n_tjunctions": 0}
    lib = Mn.MinimaLibrary(tmp_path / "lib")
    (tmp_path / "lib" / "records").mkdir(exist_ok=True)
    (tmp_path / "lib" / "records" / "N7.json").write_text(json.dumps({"n": 7, "area": 1.6, "config": cfg}))
    res = NB.main(["--n", "7", "--tag", "t", "--budget-min", "1.5", "--out", str(tmp_path / "out"),
                   "--library", str(tmp_path / "lib"), "--device", "cpu", "--workers", "0",
                   "--samples", "2", "--beam", "0", "--screen-batch", "32", "--screen-steps", "200",
                   "--full-top", "0"])
    [r] = res
    assert r["best"]["ref_1e9"] and r["best"]["ref_1e12"] and r["best"]["area"] > 1.6
    assert r["record_updated"] is True and r["indexed"].startswith("N7.")
    assert lib.record(7)["area"] == r["best"]["area"] and lib.record(7)["proposer"] == Mn.DIGITIZED_PROPOSER
    [top] = lib.best(7, 1)
    assert top["name"] == "record:N7" and top["proposer"] == Mn.DIGITIZED_PROPOSER
    assert (tmp_path / "out" / "nb-n007-t.jsonl").exists()
    assert R.min_faces_for(2.0) >= 1
