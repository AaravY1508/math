# Candidate configurations for the light-intensity problem

This package supplies n=36–50 placements of n unit-brightness point lights in the closed unit square, together with numerical data and an independent enclosure of the minimum inverse-square intensity over the entire square.

For **n=37–50**, the referenced page contains no published entry in the archived source and its later source check. These configurations can be submitted as candidates for those entries. Use the certified lower endpoints in `summary.md` as guaranteed objective values; the matching upper endpoints and numerical estimates are included in each `solution.json`.

For **n=36**, the package reproduces the published configuration's displayed value 373.207+. The source attributes that result to Mishari Alhussain in September 2026. Its digitized image was one search seed. No new n=36 record is claimed, and a positive difference from the printed threshold is not proof of improving the original higher-precision result.

For each n, submit `solution.json`, `coordinates.csv`, `validation.json`, and `diagram.svg`, `diagram.png`, or `diagram.gif`; the per-n README explains the model and coordinate convention. Coordinates have 17-significant-digit decimal encodings plus exact hexadecimal binary64 encodings. Every light has brightness one. Independent validation checks the complete square, including all boundaries and singular light locations, with absolute enclosure width at most 1e-7.

The configurations are feasible numerical results. Their objective values are certified for the supplied fixed coordinates; global optimality over all placements is not claimed. The original four-hour run, subsequent correctness repair, executable verifier, and source comparison are preserved in `reproducibility/`.

Problem and published source: https://erich-friedman.github.io/packing/light/
