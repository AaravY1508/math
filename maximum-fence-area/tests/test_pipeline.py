"""The pipeline driver: flag parsing, the toggles reaching both stages, and the report."""
import json
from pathlib import Path

import pytest

import pipeline as P
from fence import heuristics as H
from fence import netevolve as NE
from fence import rules as R

HERE = Path(__file__).resolve().parent.parent


def test_parse_ns_handles_ranges_and_lists():
    assert P.parse_ns("12-15") == [12, 13, 14, 15]
    assert P.parse_ns("13,16,20-22") == [13, 16, 20, 21, 22]
    assert P.parse_ns("20-22,21") == [20, 21, 22]
    assert P.parse_ns("") == []


def test_parse_budget():
    assert P.parse_budget(None) is None
    assert P.parse_budget("90s") == 90
    assert P.parse_budget("30m") == 1800
    assert P.parse_budget("2h") == 7200


def test_list_checks_names_every_rule_heuristic_generator_and_move():
    text = P.list_checks()
    for k in R.RULES:
        assert k in text
    for k in H.HEURISTICS:
        assert k in text
    for k in NE.MOVES:
        assert k in text


def test_every_toggle_flag_is_wired_to_its_registry():
    args = P.build_parser().parse_args([
        "--no-rule", "record_faces", "--no-heuristic", "boundary",
        "--no-move", "add_face", "--weight", "face_count=2", "--move-weight", "jitter=0.1"])
    rs = R.RuleSet.from_flags(args.no_rule, None)
    hs = H.HeuristicSet.from_flags(args.no_heuristic, args.weight)
    ms = NE.MoveSet.from_flags(args.no_move, args.move_weight)
    assert rs.disabled() == ["record_faces"]
    assert hs.disabled() == ["boundary"] and hs.weights["face_count"] == 2.0
    assert "add_face" not in ms.enabled() and ms.weights["jitter"] == 0.1


def test_a_typo_in_a_toggle_is_an_error_not_a_silent_no_op():
    for fn, flag in ((R.RuleSet.from_flags, "rule"), (H.HeuristicSet.from_flags, "heuristic"),
                     (NE.MoveSet.from_flags, "move")):
        with pytest.raises(ValueError):
            fn(["definitely_not_a_" + flag])


def test_report_table_puts_ours_next_to_the_record():
    baselines = {12: {"A": 4.0, "upper": 4.0, "status": "trivial", "display": "4"},
                 13: {"A": 4.16199, "upper": 4.16200, "status": "best_known", "display": "4.16199+"}}
    results = [
        {"n": 12, "best": {"area": 4.0, "config": {"topology": {"faces": [[0], [1], [2], [3]]}}},
         "per_net": [{"evaluations": 5}]},
        {"n": 13, "best": None, "per_net": []},
    ]
    text = P.report(results, baselines)
    assert "4.00000" in text and "4.16199" in text
    assert "TIES" in text                       # 12 matches its record
    assert "+0.00000" in text
    assert "total shortfall" in text


def test_report_counts_only_shortfalls_in_the_total():
    baselines = {12: {"A": 4.0, "upper": 4.0, "status": "trivial", "display": "4"}}
    over = P.report([{"n": 12, "best": {"area": 4.5, "config": None}, "per_net": []}], baselines)
    assert "total shortfall against the published records: 0.00000" in over


@pytest.mark.skipif(not (HERE / "minima" / "records").exists(), reason="no minima library")
def test_calibration_report_states_both_mismatches():
    text = P.calibration_report(str(HERE / "minima"), {})
    assert "heuristic 1" in text and "heuristic 2" in text
    assert "square-grid model" in text and "measured" in text


def test_stage1_only_run_writes_a_report(tmp_path):
    rc = P.main(["--n-range", "12", "--stage1-only", "--max-candidates", "60",
                 "--keep-nets", "5", "--no-library", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "report.txt").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["ns"] == [12]
    assert summary["results"][0]["n"] == 12
    assert summary["ledger"]["seen"] > 0


def test_switching_a_rule_off_shows_up_in_the_summary(tmp_path):
    P.main(["--n-range", "22", "--stage1-only", "--max-candidates", "60", "--keep-nets", "5",
            "--no-library", "--no-rule", "record_faces", "--out", str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["rules_off"] == ["record_faces"]
    assert "record_faces" not in summary["ledger"]["rejected"]


@pytest.mark.slow
def test_full_two_stage_run_reports_an_area_against_the_record(tmp_path):
    rc = P.main(["--n-range", "12", "--nets-per-n", "1", "--evals-per-net", "2",
                 "--generations", "1", "--pop", "2", "--elite", "1", "--batch", "32",
                 "--steps", "40", "--topk", "1", "--polish-iters", "80", "--verify-top", "1",
                 "--max-candidates", "60", "--keep-nets", "3", "--workers", "1",
                 "--no-library", "--out", str(tmp_path)])
    assert rc == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    area = summary["results"][0]["best_area"]
    assert area is not None and area > 3.0        # n=12's record is the 2x2 grid, area 4
    assert (tmp_path / "best-n012.json").exists()
    cfg = json.loads((tmp_path / "best-n012.json").read_text())
    assert cfg["n"] == 12 and len(cfg["segments"]) == 12


def test_combine_merges_shards_and_keeps_the_best_area(tmp_path):
    a, b = tmp_path / "shard0", tmp_path / "shard1"
    for d in (a, b):
        d.mkdir()
    (a / "n012.json").write_text(json.dumps({"n": 12, "best": {"area": 3.5, "config": None}, "per_net": []}))
    (b / "n012.json").write_text(json.dumps({"n": 12, "best": {"area": 3.9, "config": None}, "per_net": []}))
    (b / "n013.json").write_text(json.dumps({"n": 13, "best": {"area": 4.0, "config": None}, "per_net": []}))
    (a / "summary.json").write_text(json.dumps(
        {"ledger": {"accepted": 5, "rejected": {"euler": 2}, "samples": {"euler": ["x"]}}}))
    (b / "summary.json").write_text(json.dumps(
        {"ledger": {"accepted": 7, "rejected": {"euler": 1, "bridgeless": 4}, "samples": {}}}))
    baselines = {12: {"A": 4.0, "upper": 4.0, "status": "trivial", "display": "4"},
                 13: {"A": 4.16199, "upper": 4.162, "status": "best_known", "display": "4.16199+"}}
    text = P.combine([str(a), str(b)], baselines)
    assert "3.90000" in text and "3.50000" not in text      # the better of the two shards
    assert "4.00000" in text
    assert "euler" in text and "12" in text                 # 5 + 7 accepted, 3 euler rejections
    assert "bridgeless" in text


@pytest.mark.slow
def test_ablate_rules_reports_a_row_per_enabled_rule(capsys):
    args = P.build_parser().parse_args(["--n-range", "12", "--max-candidates", "40",
                                        "--keep-nets", "99", "--no-library"])
    rs = R.RuleSet.from_flags(["no_crossings", "t_straight", "embedding", "face_area_max",
                               "unit_fences", "objective", "incidence", "t_bound",
                               "min_degree", "face_size"])
    text = P.ablate("rules", [12], args, rs, H.HeuristicSet(), None,
                    {12: {"A": 4.0, "upper": 4.0, "status": "trivial", "display": "4"}})
    for rid in rs.enabled:
        assert rid in text
    assert "baseline:" in text


def test_a_typo_in_a_toggle_exits_cleanly_rather_than_tracebacking(capsys):
    assert P.main(["--n-range", "12", "--no-rule", "eluer"]) == 2
    err = capsys.readouterr().err
    assert "unknown rules" in err and "--list" in err
