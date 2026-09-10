"""Scrape the table of best-known areas from Erich Friedman's Maximum Fence Area page.

Writes baselines.json next to this file. If the page cannot be fetched or parsed
into a complete n=3..50 table, a placeholder file containing a TODO marker is
written instead and the script exits non-zero. The test-suite (and therefore the
overnight runner) refuses to proceed while that marker is present, so no search
ever runs against invented numbers.

Usage:  ./venv/bin/python scrape_baselines.py [--html saved_page.html]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html as _html
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

URL = "https://erich-friedman.github.io/packing/fence/"
OUT = Path(__file__).resolve().parent / "baselines.json"
EXPECTED_N = list(range(3, 51))

# One table cell looks like:
#   <font size=+3>6.<br><img src="6.gif" ...><br><font size=+1>A = 1.47585+<br>Found by Antoine Lagache<br>in August 2026.
#   <font size=+3>3.<br><img src="3.gif" ...><br><font size=+1>A = &radic;3/4 = .43301+<br>Trivial.
CELL_RE = re.compile(
    r"<font size=\+3>\s*(?P<n>\d+)\.\s*<br>\s*<img[^>]*>\s*<br>\s*<font size=\+1>\s*"
    r"A\s*=\s*(?P<area>[^<]+?)\s*<br>\s*(?P<credit>[^<]+?)(?:<br>\s*(?P<date>[^<]+?))?\s*(?=<td|<TD|</TR|</tr|\n|$)",
    re.IGNORECASE,
)

# Closed forms that appear on the page (checked against the printed decimals).
CLOSED_FORMS = {"√3/4": math.sqrt(3) / 4}


def parse_area(raw: str) -> dict:
    """'1.47585+' -> value 1.47585, truncated; '4' -> exact 4; '√3/4 = .43301+' -> exact sqrt(3)/4."""
    text = _html.unescape(raw).strip()
    exact_expr = None
    if "=" in text:  # e.g. "√3/4 = .43301+"
        exact_expr, text = [s.strip() for s in text.split("=", 1)]
    truncated = text.endswith("+")
    num = text.rstrip("+").strip()
    if not re.fullmatch(r"\d*\.?\d+", num):
        raise ValueError(f"unparseable area {raw!r}")
    decimals = len(num.split(".", 1)[1]) if "." in num else 0
    value = float(num)
    exact_value = None
    if exact_expr is not None:
        if exact_expr not in CLOSED_FORMS:
            raise ValueError(f"unknown closed form {exact_expr!r}")
        exact_value = CLOSED_FORMS[exact_expr]
        if not (value <= exact_value < value + 10.0 ** (-decimals)):
            raise ValueError(f"closed form {exact_expr} disagrees with printed {num}")
    return {
        "A": exact_value if exact_value is not None else value,
        "display": _html.unescape(raw).strip(),
        "truncated": truncated,
        "decimals": decimals,
        "exact_expr": exact_expr,
    }


def scrape(html_text: str) -> dict:
    rows = {}
    for m in CELL_RE.finditer(html_text):
        n = int(m.group("n"))
        area = parse_area(m.group("area"))
        credit = _html.unescape(m.group("credit")).strip().rstrip(".")
        date = (m.group("date") or "").strip().rstrip(".")
        status = "trivial" if credit.lower().startswith("trivial") else "best_known"
        found_by = None
        if status == "best_known":
            found_by = re.sub(r"^Found by\s+", "", credit)
            date = re.sub(r"^in\s+", "", date)
        rows[str(n)] = {
            **area,
            "status": status,
            "found_by": found_by,
            "date": date or None,
        }
    missing = [n for n in EXPECTED_N if str(n) not in rows]
    if missing:
        raise ValueError(f"parse incomplete: missing n={missing}")
    return rows


def write_placeholder(reason: str) -> None:
    OUT.write_text(json.dumps({
        "TODO": "baselines could not be scraped automatically -- fill in by hand from " + URL,
        "reason": reason,
        "baselines": {},
    }, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", help="parse a saved copy of the page instead of fetching it")
    args = ap.parse_args()
    try:
        if args.html:
            raw = Path(args.html).read_bytes()
        else:
            with urllib.request.urlopen(URL, timeout=30) as r:
                raw = r.read()
        rows = scrape(raw.decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 -- any failure must be loud and leave the TODO marker
        write_placeholder(repr(exc))
        print(f"FATAL: baseline scrape failed ({exc!r}); wrote TODO placeholder to {OUT}", file=sys.stderr)
        return 1
    doc = {
        "source": URL,
        "scraped_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "page_sha256": hashlib.sha256(raw).hexdigest(),
        "notes": [
            "A is the best known total area. 'truncated': true means the page prints the value with a trailing '+',",
            "i.e. the true record lies in [A, A + 10^-decimals). 'trivial' entries are unit-square grids; the page",
            "does not claim they are optimal.",
        ],
        "baselines": rows,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {len(rows)} baselines (n={min(map(int, rows))}..{max(map(int, rows))}) to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
