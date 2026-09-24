"""pipeline.py persistence: resume at n granularity, a report on every stop, library write-back.

Before these, a killed pipeline shard wrote no summary and restarted from its first n, and no
result it found ever reached the minima library (8 of 11 launches on 2026-09-17 left nothing
usable; the run's best n=39 layout stayed in a run directory for four days).
"""
import json
import os
import signal

import numpy as np
import pytest

import pipeline as P
from fence import geometry as G
from fence import minima as M
from tests import configs as C

BASE12 = {12: {"A": 4.0, "upper": 4.0, "status": "trivial", "display": "4"}}


def verified_cfg(segs, n):
    s = np.array([[float(v) for v in q] for q in segs])
    coords, fences, pins = G.topology_from_segments(s, tol=1e-9)
    topo = G.make_topology(coords, fences, pins)
    cfg = G.config_to_json(topo, coords, n=n)
    cfg["verification"] = G.verify(cfg, 1e-9).to_json()
    assert cfg["verification"]["valid"]
    return cfg


def events(out):
    return [json.loads(ln) for ln in (out / "log.jsonl").read_text().splitlines()]


def test_resume_skips_a_finished_n_and_still_reports_it(tmp_path):
    done = {"n": 12, "best": None, "per_net": [{"evaluations": 3}],
            "ledger": {"accepted": 4, "rejected": {"bridgeless": 2}, "samples": {}}}
    (tmp_path / "n012.json").write_text(json.dumps(done))
    rc = P.main(["--n-range", "12", "--no-library", "--workers", "1", "--out", str(tmp_path)])
    assert rc == 0
    assert any(e["event"] == "resumed_skip" and e["n"] == 12 for e in events(tmp_path))
    assert not any(e["event"] == "stage1" for e in events(tmp_path)), "a finished n is not searched again"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert [r["n"] for r in summary["results"]] == [12]
    assert summary["ledger"]["rejected"] == {"bridgeless": 2}


def test_a_stop_signal_still_writes_the_report(tmp_path, monkeypatch):
    real = P.stage1
    calls = []

    def stage1(n, *a, **k):
        calls.append(n)
        if len(calls) == 2:
            os.kill(os.getpid(), signal.SIGTERM)      # what a supervisor or the watchdog sends
        return real(n, *a, **k)

    monkeypatch.setattr(P, "stage1", stage1)
    before = signal.getsignal(signal.SIGTERM)
    rc = P.main(["--n-range", "12,13,14", "--stage1-only", "--max-candidates", "30",
                 "--keep-nets", "3", "--no-library", "--out", str(tmp_path)])
    assert rc == 4
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["stopped"] and [r["n"] for r in summary["results"]] == [12]
    assert (tmp_path / "report.txt").exists()
    assert signal.getsignal(signal.SIGTERM) is before, "main restores the caller's handlers"


def test_index_minimum_writes_a_pipeline_lineage_that_best_returns(tmp_path):
    lib = M.MinimaLibrary(tmp_path)
    cfg = verified_cfg(C.GRID_2x2, 12)
    assert P.index_minimum(lib, 12, cfg, {"run": "t"})
    assert not P.index_minimum(lib, 12, cfg, {"run": "t"}), "the same minimum is indexed once"
    lib.update_record(12, cfg, {"proposer": M.PIPELINE_PROPOSER, "via": "pipeline"})
    best = lib.best(12, 5)
    assert best and best[0]["proposer"] == M.PIPELINE_PROPOSER
    assert (tmp_path / "proposers" / M.PIPELINE_PROPOSER / "N12.0.json").exists()


def test_index_minimum_ignores_unverified_or_missing_configs(tmp_path):
    lib = M.MinimaLibrary(tmp_path)
    cfg = verified_cfg(C.GRID_2x2, 12)
    assert not P.index_minimum(lib, 12, {**cfg, "verification": {"valid": False}}, {})
    assert not P.index_minimum(None, 12, cfg, {})
    assert not P.index_minimum(lib, 12, None, {})


def test_ablate_counts_every_distinct_net_not_the_truncated_list():
    args = P.build_parser().parse_args(["--n-range", "22", "--max-candidates", "80",
                                        "--keep-nets", "2", "--no-library"])
    rs = P.R.RuleSet.from_flags(["no_crossings", "t_straight", "embedding", "face_area_max", "unit_fences",
                                 "objective", "incidence", "t_bound", "min_degree", "face_size", "euler",
                                 "no_free_ends", "bridgeless"])
    base = {22: {"A": 8.07384, "upper": 8.07385, "status": "best_known", "display": "8.07384+"}}
    text = P.ablate("rules", [22], args, rs, P.H.HeuristicSet(), None, base)
    kept = int(text.split("baseline: ")[1].split()[0])
    assert kept > 2, "with --keep-nets 2 a truncated count could never exceed 2"


@pytest.mark.slow
def test_full_run_writes_its_best_back_to_the_library(tmp_path):
    lib_dir = tmp_path / "minima"
    rc = P.main(["--n-range", "12", "--nets-per-n", "1", "--evals-per-net", "2",
                 "--generations", "1", "--pop", "2", "--elite", "1", "--batch", "32",
                 "--steps", "40", "--topk", "1", "--polish-iters", "80", "--verify-top", "1",
                 "--max-candidates", "60", "--keep-nets", "3", "--workers", "1",
                 "--minima-dir", str(lib_dir), "--out", str(tmp_path / "run")])
    assert rc == 0
    lib = M.MinimaLibrary(lib_dir)
    rec = lib.record(12)
    assert rec is not None and rec["proposer"] == M.PIPELINE_PROPOSER and rec["via"] == "pipeline"
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert abs(rec["area"] - summary["results"][0]["best_area"]) < 1e-9
    assert list((lib_dir / "proposers" / M.PIPELINE_PROPOSER).glob("N12.*.json"))
    # re-running into the same --out resumes: nothing is searched and the record is untouched
    P.main(["--n-range", "12", "--workers", "1", "--minima-dir", str(lib_dir), "--out", str(tmp_path / "run")])
    assert lib.record(12)["updated"] == rec["updated"]
