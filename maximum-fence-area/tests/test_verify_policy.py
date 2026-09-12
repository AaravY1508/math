"""--verify-policy worthy: which polished candidates are worth the 50-digit referee."""
import numpy as np

from fence import evolve as E
from fence import geometry as G
from fence import optimize as O
from tests.test_evolve import make_worker
from tests.test_face_edits import RECORDS


def job_and_candidate(w, n, area):
    cfg = RECORDS[str(n)]
    topo, X = G.config_from_json(cfg)
    prog = E.Program.from_genome(E.seed_genomes(n)[2])
    job = {"n": n, "island": 0, "topo": topo, "program": prog}
    return job, {"area": area, "z": np.concatenate([X.ravel(), np.full(len(topo.pins), 0.5)])}


def test_defaults_are_the_validated_settings():
    a = E.worker_parser().parse_args(["--gpu", "0", "--ns", "20", "--run-dir", "r", "--deadline", "0"])
    assert (a.verify_policy, a.face_weight, a.seed_elites, a.batch) == ("worthy", 0.25, 6, 0)
    assert O.default_batch(14) == O.default_batch(45) == 2048


def test_worthy_candidates(tmp_path):
    w = make_worker(tmp_path, "20", extra=["--verify-policy", "worthy"])
    st = w.state[20]
    job, p = job_and_candidate(w, 20, 6.9)
    assert w._worthy(job, p)                                  # nothing verified yet at n
    st.best = {"area": 7.0}
    assert w._worthy(job, p)                                  # below n's best, but its archive cell is empty
    topo, X = G.config_from_json(RECORDS["20"])
    feats = {"faces": len(topo.faces), "tjunctions": len(topo.pins),
             "sym": G.rotational_symmetry_order(G.segments_of(X, topo.fences), tol=1e-6)}
    st.islands[0].try_insert(E.Elite(job["program"], 6.95 / st.base_A, 6.95, feats, {}))
    assert w._worthy(job, p)                                  # no pending minimum of its proposer yet
    st.epochs[job["program"].proposer] = {"best": {"area": 6.92}, "stale": 1}
    assert not w._worthy(job, p)                              # beats nothing: skip the referee
    assert w._worthy(job, {**p, "area": 6.96})                # would take over the archive cell
    assert w._worthy(job, {**p, "area": 7.01})                # would be n's new best
    st.islands[0].cells.clear()
    assert w._worthy(job, p)
