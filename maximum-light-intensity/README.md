# Maximum minimum light intensity

A from-scratch search and independent validator for [Erich Friedman's light problem](https://erich-friedman.github.io/packing/light/), for **n = 36–50**.

The requested four-hour run **completed**, using 20 parallel workers from **2026-10-02 12:37:44.985 UTC to 16:37:45.292 UTC**. It processed **204,263 attempts**, of which **204,257 returned normally**. Six failed local solves were caught and the workers continued. Elapsed search time was **14,400.307 seconds**; the earlier throughput benchmark ran separately for 90.199 seconds. Aggregate task wall time was 79.841 worker-hours, which is not a measurement of CPU time.

Start with [the verified results table](results/summary.md). Every n has full-precision coordinates, an independent objective enclosure, CSV/JSON data, and SVG/350-dpi PNG diagrams and full-resolution static GIFs. [Run evidence](results/reproducibility/) preserves the timing, per-n allocation, checkpoint provenance, and search implementation. Original mutable logs and checkpoints remain in `runs/four-hour/`.

## Exact problem

Place n point lights with unit brightness at positions pᵢ in the closed square [0,1]². For any location q in that same square, define

```text
F(q; p) = sum_i 1 / ((q_x - p_ix)^2 + (q_y - p_iy)^2)
I(p)    = min_{q in [0,1]^2} F(q; p).
```

The optimization maximizes I(p). Intensity is positive infinity at a light. The page imposes no separation constraint and does not prohibit coincident lights. The square's boundary and corners are part of the minimization domain. A certified value for a fixed placement is a feasible lower bound for the placement problem; it is not proof that the arrangement is globally optimal.

The archived source lists **373.207+ for n=36**, attributed to Mishari Alhussain, and **no entries for n=37–50**. The plus sign does not supply an upper bound on the original result. Differences in the summary are against the displayed threshold only; no improvement over undisclosed higher-precision published coordinates is asserted. The source HTML, images, retrieval metadata, hashes, and latest comparison are archived under `source/` and copied into the results package. A digitized n=36 image provides one seed family; all other seed families and optimization code were built here.

## Search and validation

`solver.py` searches for weakly illuminated points in the interior, on each edge, and at all four corners. It supplies analytic light-position gradients to minimax SLSQP. `experiments.py` tracks moving local minima using damped Newton steps and the envelope derivative, with Numba acceleration. Multi-start perturbations, alternative boundary rings and row grids, insertion/deletion from neighboring n, and a small diverse pool explore different arrangements. `run_search.py` allocates workers using coverage, accumulated task time, and recent improvements. Atomic write/flush/rename checkpoints retain the incumbent for each n and archive each improvement. A failed or timed-out worker is replaced.

A 90-second, 20-worker benchmark completed 925 attempts, exposed an array-conversion issue, and supplied starting solutions for the timed search. The issue was fixed before that run. JIT-compiled local-minimum tracking and an approximate fast oracle made CPU restarts effective for these small continuous optimization problems. A separate robust oracle checks candidates; a spatial validator checks the complete square independently.

`validation.py` performs outward-rounded branch and bound on rectangles covering the entire square. It combines:

- A sum of reciprocal squared farthest distances to each light.
- A cancellation-aware quadratic lower bound derived from `1/t >= 2/s - t/s²`.
- Away from light poles, a cubic Taylor remainder with `|D³(1/r²)| <= 24/r⁵`.

Point evaluation supplies an upper bound; rectangle coverage supplies the lower bound. Light singularities do not invalidate the reciprocal bound. The validator checks geometry, reports both endpoints and their gap, and binds the certificate to the exact binary64 coordinate bytes with SHA-256. Final certificates target an absolute gap of **1e-7**. Decimal displays of bounds are rounded outward.

### Corrections found during independent validation

Periodic audits correctly detected five overly optimistic numerical objectives in the timed run (n=39,40,43,47,50). Narrow interior minima near corners were missed because coarse seed deduplication removed better starts and large L-BFGS steps could leave their basins. The original logs and coordinates were preserved. The finalization stage reranks saved candidates using the repaired oracle, feeds the independently discovered minima into local refinement, and certifies the selected geometry. The completed repair compared 94 distinct geometries and polished 54 candidates in 50.77 seconds of wall time, recorded separately from the original four-hour run.

The repaired oracle preserves supplied witnesses, uses fine deduplication, and limits Newton steps. Regression fixtures reproduce all five failures. Future searches additionally validate prospective improvements before promotion. Six exchange-solver failures caused by pruning all contact constraints after rejecting a step were also fixed. Reports use the independently evaluated/certified values, not the optimistic historical claims.

## Reproduce

Python 3.12 was used. The tested environment is recorded in `requirements.lock.txt`.

```bash
cd maximum-light-intensity
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m unittest discover -p 'tests_*.py' -v
.venv/bin/python verify_package.py results

# A new run; use a fresh directory to keep the completed run intact.
.venv/bin/python run_search.py --hours 4 --workers 20 --run-dir runs/new-search

# Resume an interrupted run against its recorded deadline.
.venv/bin/python run_search.py --resume --workers 20 --run-dir runs/new-search

# Independently certify and render existing final checkpoints.
.venv/bin/python validate_results.py --checkpoints results/final-checkpoints \
    --results results --tolerance 1e-7 --time-limit 180
.venv/bin/python report.py --checkpoints results/final-checkpoints --results results
```

Parallel scheduling can change subsequent seeds and floating-point optimizers can vary between library versions; the stored coordinate files reproduce the actual delivered placements without replaying the stochastic search. Each `solution.json` contains hexadecimal binary64 encodings as well as JSON decimal coordinates. Parse the CSV's 17-significant-digit coordinates to nearest binary64 to reproduce the certified points.

`results/reproducibility/original-search-source/` preserves the search core matching the original run hashes. The top-level program contains the subsequent correctness fixes. All scientific diagrams use Matplotlib; contour sampling is illustrative and has no role in certification.
