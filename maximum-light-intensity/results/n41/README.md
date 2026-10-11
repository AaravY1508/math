# n = 41: unit-light configuration

Status: **valid and certified**. The minimum intensity of this fixed configuration is enclosed in **[443.973887839505, 443.973887939007]**.
The minimum numerical witness intensity is 443.973887939002.
The original optimizer checkpoint reports 443.973887939002.

Place 41 point lights, each with brightness 1, at the coordinates in `coordinates.csv`.
The domain is the closed square [0,1] × [0,1]; x increases rightward and y upward.
For q = (x,y), intensity is Σᵢ 1/((x−xᵢ)²+(y−yᵢ)²), with value +∞ at a light.
The CSV uses 17 significant decimal digits. Interpret these decimal strings and
the JSON point numbers by rounding to the nearest IEEE-754 binary64 value.
The `points_hex` field in `solution.json` gives each exact rational coordinate in
hexadecimal floating-point notation; Python `float.fromhex()` reconstructs it
without ambiguity. The independent certificate applies to these exact values.
Labels 1–41 in the diagrams match the CSV row numbers.

- `solution.json`: complete geometry, objective bounds, source comparison, and metadata.
- `coordinates.csv`: light index, x, y, and brightness.
- `witnesses.csv`: numerical minimum candidates and directly evaluated intensities.
- `validation.json`: the global certification and constraint-check results.
- `checkpoint.json`: the original optimizer checkpoint.
- `diagram.svg`: editable, scalable vector diagram with selectable text.
- `diagram.png`: 3500 × 3727-pixel version of the same diagram.
- `diagram.gif`: static 256-color GIF at the same 3500 × 3727-pixel resolution.

Red dots are lights; blue dots are numerical witnesses near the minimum; thin curves
show selected intensity levels. The finite contour grid is for illustration only;
it is not the validation method. Certification concerns the objective of these fixed
coordinates, not global optimality over all possible light configurations.

Published comparison: Not listed on the archived source page.
Comparison at the precision printed on the page: no published entry.
Source: https://erich-friedman.github.io/packing/light/
Archived at: 2026-10-02T12:27:31.681864+00:00.
Published trailing-plus values are displayed lower thresholds; the `+` alone supplies
no upper bound on the unpublished objective. The displayed difference
in the summary is relative to that threshold and cannot establish improvement over
undisclosed higher-precision published coordinates. No baseline is inferred for unlisted n.
