# Search-strategy experiments

Short log of changes to the search (not the optimizer's correctness), so their effects can be
compared and each change reverted on its own.

## Protocol

- Every change sits behind a flag whose off value reproduces the previous behaviour; the control
  arm runs the same commit with the flag off.
- Arms run concurrently on disjoint GPUs of one host (ynez, 4x RTX 3090, 64 cores), same n, same
  budget, same `--seed`, each from its own copy of one frozen minima snapshot
  (`minima-snapshots/20260912`, 1574 files), started with `experiments.py launch`.
- n whose world record is a unit-square grid (12, 17, 24, 31, 40, 49) are never run: those
  records are likely optimal. `experiments.py launch` refuses them.
- Tables come from `experiments.py metrics`, so every row is computed the same way.
  - *F / target*: faces of the best result / floor(record) + 1, the fewest faces that could beat
    the record (every face holds at most area 1).
  - *dead (deficit)*: faces of the best below area 0.9, and the area they miss.
  - *run-best wins*: evaluations that raised this run's best for their n.
  - *beat source*: children of a literal-seeded operator (elite, face) that beat the elite they
    were built from.
- 15-minute arms give ~40-60 evaluations per n. Area differences at that length are mostly noise,
  so operator-level numbers (win rates, face counts, validity) are the primary evidence; area
  claims need longer runs.

Face-count identity used throughout: a connected layout with n fences and J joints (vertices that
are not T-junction pins) has exactly F = n + 1 - J bounded faces (Euler; holds on all 1541
library configurations). Adding an m-fence cell adds m - 1 joints, removing a pin-free fence
removes none, so only an edit that removes a joint can raise F at fixed n.

## Reference: last 6 h run before these changes (20260911-052929-ynez, commit ccb38dd)

23 n on 4 GPUs, 12990 evals (549 evals/GPU-h). Per operator:

| operator | evals | verified | run-best wins | beat source | mean area - source |
|---|---|---|---|---|---|
| elite | 4154 | 98% | 106 | 440 (10.6%) | -0.480 |
| param | 3104 | 96% | 42 | - | - |
| edit+ | 1578 | 95% | 12 | - | - |
| seed | 1015 | 89% | 7 | - | - |
| sym | 980 | 95% | 5 | - | - |
| fresh | 955 | 90% | 0 | - | - |
| seed-program | 857 | 91% | 17 | - | - |
| edit- | 347 | 98% | 2 | - | - |

## Log

### 1. Face-aware mutation operator (`--face-weight`)

**Offline operator A/B**, 2026-09-12 (`experiments.py op-ab`, 4 GPUs, ~12 min, results in
`runs/exp-op-ab-1/`). Parents: each n's library record plus its two best other minima with
distinct topologies (snapshot 20260912). 5 children per parent and operator, production
optimizer settings (default batch, 1000 Adam steps, SLSQP polish, top-3 referee). Control:
today's elite operator with at least one random edit (1-3). Both operators get the same seed
per (parent, child).

| n | op | built | same topology as parent | beat parent | mean area - parent | best child - n's record | dead faces / result |
|---|---|---|---|---|---|---|---|
| 13 | face | 0/15 (no move applies) | - | - | - | - | - |
| 13 | elite | 15/15 | 8 | 0 | -0.116 | +0.0000 | 1.33 |
| 18 | face | 15/15 | 0 | 0 | -0.000 | +0.0000 | 3.87 |
| 18 | elite | 15/15 | 0 | 0 | -0.199 | +0.0000 | 2.33 |
| 20 | face | 15/15 | 0 | 0 | -0.306 | -0.0525 | 2.40 |
| 20 | elite | 15/15 | 2 | 0 | -0.554 | +0.0000 | 0.27 |
| 23 | face | 12/15 | 0 | 1 | +0.013 | **+0.1181** | 1.92 |
| 23 | elite | 15/15 | 1 | 3 | -0.304 | -0.0000 | 1.13 |
| 25 | face | 15/15 | 0 | 0 | -0.010 | -0.0127 | 2.33 |
| 25 | elite | 15/15 | 1 | 1 | -0.237 | +0.0005 | 1.71 |
| 34 | face | 15/15 | 0 | 0 | -0.490 | -0.3124 | 2.53 |
| 34 | elite | 15/15 | 1 | 0 | -0.721 | -0.0005 | 0.60 |
| all | face | 72/90 | 0 | 1 | **-0.166** | | 2.64 |
| all | elite | 90/90 | 13 | 4 | -0.356 | | 1.22 |

By move: relocate (40 children) mean -0.004, none beat the parent; dissolve into a face split
(23) mean -0.375, one beat the parent; dissolve into an outer notch (9) mean -0.352.

- Every face child is a legal layout with n fences and the predicted face count (relocate F,
  dissolve F + 1), and none repeats its parent's topology (13/90 elite children did).
- Face children lose half as much area (-0.17 vs -0.36); relocate is nearly area-neutral.
- The one substantive improvement: a dissolve at n=23 took N23.2 (7.885, 8 faces) to 8.118
  (9 faces), above n=23's library record 8.0. Re-run with the same seed and saved:
  **8.12331131592952** (referee-valid at 50 digits), now the library's n=23 record
  (`minima/records/N23.json`; world record 8.53463). The elite operator's best was +0.0005 over
  a record (n=25).
- Cost: dissolve results carry more dead faces (the added face often does not fill), and the
  improvement rate is not higher at this sample size (1/72 vs 4/89, the elite wins all < 0.015).
- No move applies on 5 of the 33 non-grid records: 13 and 15 (the dead face is a pinwheel's
  eye, every boundary vertex a pin), 26, 37, 44 (every boundary fence hosts a pin or ends in a
  second corner). There the operator falls back to the elite operator (`elite-fb`).
- Where the best of n is grid-trivial, the grid lineage is blocked from mutation; the face
  operator may still edit its elites (a dissolve adds a face, so it cannot re-derive the grid
  answer), and such children start their own lineage.
