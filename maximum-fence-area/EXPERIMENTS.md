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
