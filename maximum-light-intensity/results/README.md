# Verified light configurations, n=36–50

Open **[summary.md](summary.md)** for the requested table and links to every result. A spreadsheet-friendly version is in [summary.csv](summary.csv).

Each `n36/` through `n50/` folder contains:

- `solution.json`: exact geometry, hexadecimal binary64 coordinates, numerical minimum, certified bounds, and published comparison.
- `coordinates.csv`: all n light coordinates and their unit brightness, with diagram indices.
- `witnesses.csv`: locations and values of numerical minimum candidates.
- `validation.json`: independent full-square validation with coordinate hash and absolute interval width at most 1e-7.
- `diagram.svg`, `diagram.png`, and `diagram.gif`: a scalable diagram and 3500×3727-pixel PNG/GIF versions.
- `README.md`, `checkpoint.json`, and `manifest.json`: interpretation, provenance, and artifact hashes.

The square is [0,1]², x rightward and y upward. Intensity at q is the sum of inverse squared distances to all unit lights. Red points are indexed lights, blue points are numerical minimum witnesses, and curves are intensity contours. Contour pixels are illustrative; the certificate covers every point of the square.

The timed search completed **14,400.307 seconds (four hours)** with 20 workers and 204,263 attempts. A separate 50.77-second repair/finalization pass compared 94 archived geometries and refined 54 candidates after independent validation identified five missed numerical minima. All delivered placements were then independently certified to an absolute gap of at most 1e-7. The repaired minimum finder passes regression fixtures for all five cases.

The source lists 373.207+ for n=36; our n=36 result matches that printed precision and does not establish a new record. It lists no entries for n=37–50. Missing baselines are shown as “not listed,” and the n=36 table difference is against the printed lower threshold only. No placement is claimed globally optimal.

[SUBMISSION.md](SUBMISSION.md) explains what can be submitted. [reproducibility/](reproducibility/) contains original run evidence, source snapshots, dependency versions, and runnable verification code. Historical uncorrected search claims are explicitly separated there and are not the submission values. The exact delivered configurations are always the top-level `nXX/solution.json` files.

To reproduce all certificates and diagrams, follow [REPRODUCE.md](reproducibility/current-code/REPRODUCE.md). The archived source page, images, and retrieval/hash metadata are in [source/](source/). The package-wide [manifest.json](manifest.json) hashes every delivered file except itself.
