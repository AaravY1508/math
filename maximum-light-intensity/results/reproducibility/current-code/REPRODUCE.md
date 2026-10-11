# Reproduce the delivered geometry and certificates

Run from this directory with Python 3.12. The exact light coordinates are in
`../../n36/solution.json` through `../../n50/solution.json`, with decimal and
hexadecimal binary64 encodings. No stochastic search replay is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m unittest discover -p 'tests_*.py' -v
.venv/bin/python validate_results.py --checkpoints ../../final-checkpoints --results ../.. --tolerance 1e-7 --time-limit 180
.venv/bin/python report.py --checkpoints ../../final-checkpoints --results ../.. --source source
.venv/bin/python verify_package.py ../..
```

`validation.py` independently covers the whole square with outward-rounded
bounds. Each certificate is bound to the exact coordinate bytes by SHA-256.
The `original-search-source` sibling folder contains the timed search core;
this directory includes subsequent correctness repairs and finalization tools.
