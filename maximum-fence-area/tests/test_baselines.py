"""baselines.json must be a real scrape of the Friedman page, never a placeholder."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "baselines.json"


def load():
    return json.loads(BASE.read_text())


def test_no_todo_placeholder():
    doc = load()
    assert "TODO" not in doc, f"baselines.json is a placeholder: {doc.get('reason')}"
    assert doc["source"].startswith("https://erich-friedman.github.io/packing/fence")


def test_complete_table():
    b = load()["baselines"]
    assert sorted(map(int, b)) == list(range(3, 51))


def test_spot_values_from_page():
    # values as printed on the page at the last scrape (2026-09-14); update them only with a re-scrape
    b = load()["baselines"]
    assert abs(b["3"]["A"] - 3 ** 0.5 / 4) < 1e-15
    assert b["6"]["A"] == 1.47585 and b["6"]["truncated"]
    assert b["15"]["A"] == 5.06345 and b["15"]["decimals"] == 5
    assert b["12"]["A"] == 4.0 and b["12"]["status"] == "trivial"
    assert b["50"]["A"] == 20.42937


def test_parser_reads_every_printed_form():
    """The page has printed a 6-decimal record (n=15 was 5.049067+ until September 2026); it prints
    none today, so that case is pinned on the parser directly rather than on the live table."""
    import scrape_baselines as SB
    assert SB.parse_area("5.049067+") == {"A": 5.049067, "display": "5.049067+", "truncated": True,
                                          "decimals": 6, "exact_expr": None}
    assert SB.parse_area("4") == {"A": 4.0, "display": "4", "truncated": False, "decimals": 0, "exact_expr": None}
    r = SB.parse_area("&radic;3/4 = .43301+")
    assert abs(r["A"] - 3 ** 0.5 / 4) < 1e-15 and r["exact_expr"] == "\u221a3/4" and r["decimals"] == 5


def test_values_increase_with_n():
    b = load()["baselines"]
    vals = [b[str(n)]["A"] for n in range(3, 51)]
    assert all(v2 >= v1 for v1, v2 in zip(vals, vals[1:]))
