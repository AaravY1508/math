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

**Integrated A/B #1**, 2026-09-12 06:30, commit 01efbd6 (change 1 only): control
`--face-weight 0` on GPUs 0-1 vs `--face-weight 0.25` on GPUs 2-3, n = 13, 18, 20, 23, 25, 34,
`--budget 15m` (~11 min of search), `--seed 0`, empty archives (no library seeding in this
commit). Pre-registered pass criteria: (1) no worker deaths or hangs and throughput within 15%;
(2) summed best area over the 6 n not below control by more than 0.1, or at least as many n
at or above the snapshot record.

| n | lib record | control best | face best |
|---|---|---|---|
| 13 | 4.04795 | 4.00000 | 4.00000 |
| 18 | 6.18182 | 6.00000 | 6.00346 |
| 20 | 7.00000 | 6.61474 | 6.65482 |
| 23 | 8.00000 | 8.00000 | 7.38308 |
| 25 | 9.01763 | 9.00000 | 9.00000 |
| 34 | 12.99948 | 12.63601 | 12.62038 |
| sum | | 46.25075 | 45.66174 (-0.589) |

- (1) passed: 725 vs 723 evals/GPU-h, no deaths or hangs.
- (2) **failed**: -0.589, all of it at n=23, where the control found the grid-plus-sliver 8.0 at
  minute 9.5; at ~30 evals per n that is one lucky find. The other five n are within +-0.04.
- Operator level (face arm): the face op made 4 of the arm's 8 run-best improvements with 42 of
  255 evals; relocate children beat their source 8/32 times, mean +0.014 area. The elite op
  made 0 improvements with 68 evals (mean -0.145); in the control, elite 0/58 (mean -0.096).
- Decision: not a pass. One replication, in the regime the final run uses (library seeding on in
  both arms), with its criterion fixed before it runs: face arm summed best area >= control -
  0.1 and at least as many n beating the snapshot record. Change 1 passes only if that holds.

**Integrated A/B #2 (replication)**, 2026-09-12 06:43, commit 8fe5de7 (changes 1 + 2), both arms
`--seed-elites 6` (library seeding, the regime of the final run), `--seed 1`, otherwise as #1.
Pre-registered: face arm summed best area >= control - 0.1, and at least as many n beating the
snapshot record.

| n | snapshot record | control best | face best |
|---|---|---|---|
| 13 | 4.04795 | 4.04686 | 4.04680 |
| 18 | 6.18182 | 6.18182 | 6.18182 |
| 20 | 7.00000 | 7.00000 | 7.00000 |
| 23 | 8.00000 | 7.72655 | **8.12279** |
| 25 | 9.01763 | **9.01773** | 9.01650 |
| 34 | 12.99948 | 12.99911 | 12.99902 |
| sum | | 46.97207 | 47.36693 (+0.395) |

- Passed both: +0.395, and one n above the snapshot record in each arm (face: n=23 by a
  dissolve, control: n=25 by +0.0001). The face arm found the same n=23 layout (8.1228) the
  offline A/B had found (8.1233), independently.
- Throughput 740 vs 731 evals/GPU-h.
- **Change 1 validated; setting: `--face-weight 0.25`.**

### 2. Library-seeded archive and focused runs (`--seed-elites`, `--focus`)

- While analysing the replication (which also ran change 2's seeding), the leak check found
  children that re-converged to a *different* seeded minimum than the one they descend from
  (another lineage's; 2 verified results in the control arm were the same minimum again: same
  topology, area within 1e-6). The first change 2 A/B (06:57, `--seed 2`) was stopped after 2
  minutes, since it would have failed its criterion on this gap.
- Fix (folded into the change 2 commit, now 42b9279): the seeded minima are known minima of
  every lineage at n (`Worker._known_minima`, checkpointed): restarts on their topology are
  repelled from them, polished results matching them never reach the referee, and a result equal
  to one is a duplicate. Other basins of the same topology stay open. Also: records imported
  without a lineage now take it from a grid-trivial marker of the same topology, so the grid
  records of face-short n (22, 23, 32, 36, 38, 39, 41, 50) are seeded and the face operator's
  dissolve can reach them.

### 3. Adam batch vs topologies per hour (`--batch`, `--verify-policy`)

Pre-registered before any change 3 run:
- Offline benchmark (`experiments.py batch-bench`, n = 14, 21, 30, 45; 6 topologies per n, half
  edited library minima, half seed programs; 2 seeds; batches 16384 / 4096 / 2048): a batch B
  qualifies if its best verified area is on average no more than 0.01 below 16384's on the same
  topology and seed, with Adam time at most 40% of 16384's. The smallest qualifying B is taken.
- Integrated A/B (30 min, 2 GPUs per arm, changes 1 and 2 at their validated settings,
  `--focus 8`): control default batch and `--verify-policy top`; treatment `--batch B
  --verify-policy worthy`. Passes if (1) no worker deaths or hangs and treatment evals/GPU-h
  >= 1.5x control, and (2) summed best area >= control - 0.1 with at least as many n beating the
  snapshot record.

**Integrated A/B**, 2026-09-12 07:00, commit 42b9279, both arms `--face-weight 0.25 --focus 6`
over the 33 non-grid n (both picked n = 14, 15, 16, 18, 21, 22 from the snapshot), `--seed 3`:
control `--seed-elites 0` on GPUs 0-1 vs `--seed-elites 6` on GPUs 2-3. Pre-registered: (1) no
worker deaths or hangs, no verified result that is a seeded minimum again, the same focus set in
both arms and no grid-record n; (2) summed best area >= control - 0.1 and at least as many n
beating the snapshot record.

| n | snapshot record | control best | seeded best |
|---|---|---|---|
| 14 | 4.73264 | 4.60107 | 4.72659 |
| 15 | 5.00650 | 4.82185 | 5.00000 |
| 16 | 5.46372 | 5.40883 | 5.46323 |
| 18 | 6.18182 | 6.09988 | 6.18182 |
| 21 | 7.67171 | 7.43526 | **7.69571** |
| 22 | 8.00000 | 7.24433 | 7.99877 |
| sum | | 35.61122 | 37.06612 (+1.455) |

- (1) passed: no deaths or hangs; 231 evaluations descended from seeded minima, 2 children were
  dropped before GPU time for re-creating their own minimum's topology, 0 verified results were a
  seeded minimum again (2 landed on a seeded topology in another basin, which is allowed).
- (2) passed: +1.455; the seeded arm beat the snapshot record at n=21 (7.69571 vs 7.67171, world
  record 7.72489) with a face-operator relocate of a seeded minimum; the control beat none.
- Throughput 952 vs 981 evals/GPU-h. The unseeded arm spends its time re-deriving what the
  library already holds: it ends 0.05-0.8 below the snapshot record at every n.
- **Change 2 validated; settings: `--seed-elites 6`; `--focus` available, off for full runs.**

**Offline batch benchmark**, 2026-09-12 07:13, commit f0b65ea, 4 GPUs, ~9 min, 144 runs (48 per
batch size), all verified. Delta = best verified area minus 16384's on the same topology and seed.

| n | batch | mean delta | lost / gained > 1e-4 | Adam s | polish CPU s |
|---|---|---|---|---|---|
| 14 | 4096 / 2048 | 0.0000 / 0.0000 | 0/0, 0/0 | 1.46 / 1.24 (16384: 3.75) | ~1.6 |
| 21 | 4096 / 2048 | 0.0000 / -0.0000 | 0/0, 0/0 | 1.39 / 0.89 (4.84) | ~3.2 |
| 30 | 4096 / 2048 | -0.0228 / -0.0229 | 2/0, 2/0 | 2.10 / 1.21 (7.66) | ~7 |
| 45 | 4096 / 2048 | -0.0255 / +0.0002 | 2/2, 1/2 | 3.36 / 1.82 (12.49) | ~12 |
| all | 4096 | **-0.0121** | 4 / 2 | 2.08 (29% of 7.19) | 5.9 |
| all | 2048 | **-0.0057** | 3 / 2 | 1.29 (18%) | 6.2 |

- By the pre-registered rule, 4096 just misses (-0.0121 < -0.01) and 2048 qualifies: **B = 2048**.
  The two differ by noise: most topologies reach exactly the same area at every batch size, and
  the mean is set by a few n=30 and n=45 topologies that lose 0.2-0.3 either way.
- Polish costs ~6 CPU-s per job at any batch size (8 candidates), well within a 15-process pool
  per GPU even at 1.3 s of Adam per job.

**Integrated A/B**, 2026-09-12 07:15, commit f0b65ea, `--budget 30m` (~25 min of search), both
arms `--face-weight 0.25 --seed-elites 6 --focus 8` (both picked n = 14, 15, 16, 18, 21, 22, 29,
38), `--seed 4`: control default batch (16384) and `--verify-policy top` on GPUs 0-1 vs
`--batch 2048 --verify-policy worthy` on GPUs 2-3.

| n | snapshot record | control best | lean best |
|---|---|---|---|
| 14 | 4.73264 | 4.72096 | 4.73243 |
| 15 | 5.00650 | 4.99950 | 5.00000 |
| 16 | 5.46372 | **5.46425** | **5.46986** |
| 18 | 6.18182 | 6.18182 | 6.18182 |
| 21 | 7.67171 | **7.69571** | **7.72264** |
| 22 | 8.00000 | 7.99920 | 8.00000 |
| 29 | 11.00000 | 10.99951 | 11.00000 |
| 38 | 15.00000 | 15.00000 | 15.00000 |
| sum | | 63.06095 | 63.10675 (+0.046) |

- (1) passed: no deaths or hangs; **3507 vs 824 evals/GPU-h (4.3x)**; median GPU time per job
  0.87 s vs 3.92 s. The GPU loop waited 3 s in 25 min on the CPU pool: polish and referee keep
  up. The worthy policy skipped 2209 referee calls.
- (2) passed: +0.046, and both arms beat the snapshot record at n=16 and n=21; the lean arm got
  closer at both (n=21: 7.72264, 0.0023 below the world record 7.72489).
- **Change 3 validated; settings: `--batch 2048 --verify-policy worthy`.**

## Summary (2026-09-12)

| change | commit | validated setting | evidence |
|---|---|---|---|
| 1. face-aware operator | 4c17910 | `--face-weight 0.25` | offline: children lose half as much area, the only real improvement (n=23); integrated #1 failed on one n (noise), replication passed +0.395 |
| 2. library seeding, focus | 42b9279 | `--seed-elites 6` (`--focus` off for full runs) | +1.455 summed area, 0 re-converged seeded minima |
| 3. batch 2048, worthy referee | f0b65ea | `--batch 2048 --verify-policy worthy` | 4.3x evals/GPU-h, same or better areas |
| defaults | 065fb04 | all of the above on by default | |

Library records improved by these experiments (all re-verified at 50 digits):
n=16 5.46372 -> 5.46986, n=21 7.67171 -> 7.72264, n=23 8.00000 -> 8.12331, n=25 9.01763 -> 9.01773.

## 6 h run with all three changes (2026-09-12 07:43-13:38, commit a177c21)

`run_overnight.py --budget 6h --gpus 4 --archive-dir ~/fence-results --n-range
13-16,18-23,25-30,32-39,41-48,50` (33 n, grid-record n left out), run dir `runs/20260912-074322`,
table in `runs/combined-20260912.txt`.

- No worker deaths, hangs or restarts. 58469 evals, 2474 evals/GPU-h (the 6 h run of 2026-09-11 on
  the same 4 GPUs: 12990, 549).
- **All-time library record improved at 24 of 33 n**; the summed record over the 33 n went
  from 394.651 to 397.477 (+2.826; +0.64 of it from the experiments earlier in the day). All 24
  re-verified at 50 digits.
- **n=16 ties the printed record**: 5.53706377675388740289 vs 5.53706+. No n beats its record.
- Closest remaining gaps: n=16 0, n=21 0.0022, n=14 0.012, n=19 0.022, n=48 0.029, n=47 0.039,
  n=15 0.042, n=23 0.052.
- Seeding: 24986 evals descended from seeded minima, 79 children dropped for re-creating their own
  minimum, 0 verified results that were a seeded minimum again.
- Run-best improvements by operator: elite 221, face relocate 102, param 54, edit+ 16, sym 15,
  face dissolve 9, elite-fb 7. Face relocate children stay closest to their parent (-0.046 mean)
  vs elite children (-0.371).

## Day 2 (2026-09-13)

Diagnostics first, on the 6 h run's 58469 evaluations and the library as it stands
(`minima-snapshots/20260913`, 6005 files). Three of the leads did not survive contact with the
data, and are recorded here so they are not re-run:

- **The local optimizer is saturated, not stale.** Re-optimizing each n's library-record topology
  at `batch 16384, steps 4000, topk 16, verify_top 4` against production `2048 / 1000 / 8`:
  identical verified areas to 1e-15 at **24 of 24 n**. An offline sweep on search-like topologies
  (`bench_topologies`, n = 14, 21, 30, 45, 6 topologies x 2 seeds) also finds no cheap throughput:
  600 steps instead of 1000 saves **4%** of Adam time (about 90% of a job's GPU time is fixed
  overhead -- TorchTopology build, CUDA-graph capture, the eager multiplier updates -- not the
  steps), and batch 1024 saves 27% but loses area at more topologies than it gains (5 lost / 3
  gained > 1e-4). The +-0.004 spread across 600 / 1500 / 4096 steps and batches is basin-flip
  noise, not a trend. No change made.
- **The GPU is the bottleneck, not the CPU pool.** Over the 6 h run `t_gpu` summed to 20.85 h of
  23.66 GPU-h (88% of wall) and the GPU loop waited 4752 s on the CPU pool (5.6%), not the 0.2%
  the 25-minute change-3 A/B measured. `verify_top = 3` is nearly free to cut -- the verified area
  came from the best polished candidate in 21825 of 21832 verified evaluations -- but it buys only
  a few percent, so it is not worth a change slot.
- **"12.41 of area lost to dead faces" is arithmetic, not opportunity.** Deficit is `F - area` by
  definition, so a record of 4.16199 on 5 faces carries 0.838 of deficit itself. Summed over the
  33 n the records must carry **>= 19.90**; ours is 12.41, and at 20 of 33 n our deficit is
  already below the record's lower bound. The only recoverable quantity is the 5.63 gap.

Confirmed: 58.8% of evaluations end with every polished candidate skipped by `--verify-policy
worthy` (they would change nothing), 37.3% verify, 2.3% polish to nothing. The pipeline is healthy.

### 4. Growth paths of three or more fences (`--m3-growth`)

`arc_points(Q, P, m, ell)` orders the m - 1 interior points of a path from Q to P. Two places
used that order backwards, and both are reachable only when there are at least two interior
points, i.e. m >= 3: `sample_growth` closed the proposed cell as `chain + inner[::-1]` (a bowtie
of signed area ~0, so the candidate lost the cost comparison or was dropped by `area <= 0.05`),
and `grow_once` built it with `add_path_points(P, Q, inner)` (the path crosses itself and
`topology()` rejects the step). So no growth step has ever added a 3-fence cell, although three
unit fences on a unit-long piece of boundary are exactly a unit square -- the densest cell there
is, and the only one-step move that can add a near-full face.

**Offline**, 2026-09-13, over the library records of n = 13, 16, 19, 21, 23, 26, 30, 35, 44, 48,
1200 outer growth steps restricted to m = 3:

| | built | area added |
|---|---|---|
| today (M3_GROWTH off) | **0 / 1200** | - |
| fixed | 1196 / 1200 | mean **1.170**, 1169 of them >= 0.9 |
| either, default m_weights | 1200 / 1200 | mean 0.512, max 0.872, **none >= 0.9** |

The default-mix rows are identical because the bowtie area makes m = 3 lose the cost comparison
before it can be built: both mistakes have to go together. On the same records, outer growth
today proposes m = 2 paths almost exclusively (13861 of 15398 surviving candidates, mean area
0.318); 8309 of 9969 m = 3 candidates are discarded for "area <= 0.05".

**Integrated A/B**, 2026-09-13 07:0x, commit 04a3bb9, worktree `erich-friedman-exp/m3`, from
snapshot `minima-snapshots/20260913`. Control `--m3-growth off` on GPUs 0-1 vs treatment
`--m3-growth on` on GPUs 2-3; n = 13, 20, 23, 32, 34, 39, 44, 50 (small, face-short and
large-gap); `--budget 30m`, `--seed 10`, changes 1-3 at their validated defaults in both arms.

Pre-registered pass criteria, all three must hold:
1. no worker deaths or hangs, and treatment evals/GPU-h >= 0.85 x control;
2. **mechanism**: mean fill (verified area / faces) over all verified evaluations of the arm,
   treatment >= control + 0.02 (thousands of evaluations, so not noise-dominated);
3. summed best area over the 8 n >= control - 0.10, and at least as many n at or above the
   snapshot record.

At 30 minutes criterion 3 is a non-inferiority check; the case for the change is the offline
result above.

| n | snapshot record | control (off) | m3 (on) |
|---|---|---|---|
| 13 | 4.10184 | 4.10058 | 4.10029 |
| 20 | 7.00000 | 6.99900 | **7.00000** |
| 23 | 8.48292 | 8.48257 | **8.48512** |
| 32 | 12.01256 | **12.01633** | 12.01245 |
| 34 | 12.99952 | **12.99972** | **13.00000** |
| 39 | 15.01127 | **15.01222** | **15.01137** |
| 44 | 17.44699 | **17.44745** | **17.44810** |
| 50 | 20.00000 | 20.00000 | 20.00000 |
| sum | | 97.05786 | 97.05734 (-0.00052) |

- (1) passed: no worker deaths, hangs, restarts or GPU errors in either arm; 2525 vs 2423
  evals/GPU-h (1.04x).
- (2) passed: mean fill (verified area / faces) **0.9108 vs 0.8677, +0.0431**, over 1006 and 935
  verified evaluations. Mean faces per verified result fell 12.27 -> 11.62 at the same n: fewer
  faces, fuller ones, which is what a 3-fence cell trades.
- (3) passed: -0.00052 summed over the 8 n, and 6 n at or above the snapshot record against 5.
- Behaviour worth recording: three unit fences on a unit-long piece of boundary are a unit
  square, so the fixed growth walks into grid answers more readily. The treatment arm marked
  13 lineages grid-trivial against the control's 5, and landed exactly on the grid answer at
  n = 20 (7.00000) and n = 34 (13.00000, above the snapshot's 12.99952 but still the lattice
  maximum). The existing machinery handles that -- a lineage that stabilizes on the grid answer
  is blocked from mutation at n while n's best is grid-trivial -- but it is the cost side of
  this change, and the face-short n are where it will show.
- **Change 4 validated; setting: `--m3-growth on`.**

### 5. Operator mix tuned against measured win rates (`--op-mix`)

Run-best wins per 1000 evaluations over the 58469 evaluations of the 6 h run of 2026-09-12:

| operator | evals | wins | wins / 1000 |
|---|---|---|---|
| elite-fb | 462 | 7 | 15.2 |
| face:relocate-split | 7546 | 107 | 14.2 |
| elite | 19984 | 233 | 11.7 |
| param (jitter) | 8577 | 57 | 6.7 |
| sym | 2789 | 16 | 5.7 |
| edit+ | 4079 | 19 | 4.7 |
| edit- | 1036 | 3 | 2.9 |
| face:dissolve-* | 5054 | 12 | 2.4 |
| seed-program | 2270 | 5 | 2.2 |
| seed | 2700 | 4 | 1.5 |
| fresh | 2728 | 2 | 0.7 |
| face:relocate-outer | 1244 | 1 | 0.8 |

`fresh`, `seed` and the seed-program injection spent 13% of the run's evaluations on 8 of its
466 run-best improvements. `--op-mix tuned` moves that share to the operators that earn it
(elite .36 -> .56, jitter .25 -> .20, add_edit .12 -> .09, symmetry .08 -> .06, remove_edit
.05 -> .04, seed .08 -> .03, fresh .06 -> .02, seed-program injection 0.04 -> 0.015; none goes
to zero) and fixes the `remove_edit` fall-through: `elif op == "remove_edit" and g.get("edits")`
misses on an edit-free genome, drops to the chain's `else` branch and silently becomes `fresh`.
Face-operator children always carry `"edits": []`, so they were its main source; an estimated
600-750 of the run's evaluations went that way. The face operator's own internal split is left
alone -- `dissolve` is the only move that can add a face, which 12 n still need.

**Integrated A/B**, 2026-09-13 15:4x, commit b2e213e, worktree `erich-friedman-exp/opmix`, from
snapshot `minima-snapshots/20260913`. Control `--op-mix legacy` on GPUs 0-1 vs treatment
`--op-mix tuned` on GPUs 2-3, both arms `--m3-growth on` (change 4 at its validated setting);
n = 13, 20, 23, 32, 34, 39, 44, 50; `--budget 30m`, `--seed 11`.

Pre-registered pass criteria, all three must hold:
1. no worker deaths or hangs, and throughput within 15%;
2. **mechanism**: evaluations spent on operators measured below 3 wins/1000 (`fresh`, `seed`,
   `seed-program`, `edit-`) fall from about 15% of the arm (15.6% in the change-4 control) to
   **<= 7%**, with `fresh` **<= 2.5%** -- which also shows the `remove_edit` fall-through is gone;
3. summed best area over the 8 n >= control - 0.10, and at least as many n at or above the
   snapshot record.

At 30 minutes criterion 3 is a non-inferiority check: an arm sees ~250 evaluations per n and the
whole measured effect is worth about +1.4 run-best wins per 1000 evaluations, which is inside
the noise at that length. The case for the change is the 58469-evaluation table above.

| n | snapshot record | legacy | tuned |
|---|---|---|---|
| 13 | 4.10184 | 4.10119 | 4.10056 |
| 20 | 7.00000 | **7.00000** | 6.88292 |
| 23 | 8.48292 | 8.48248 | **8.49695** |
| 32 | 12.01256 | **12.01264** | **12.02565** |
| 34 | 12.99952 | **13.11026** | **13.00000** |
| 39 | 15.01127 | **15.01588** | **15.01184** |
| 44 | 17.44699 | **17.44713** | 17.44680 |
| 50 | 20.00000 | 20.00000 | 20.00000 |
| sum | | 97.16957 | 96.96471 (-0.20486) |

- (1) passed: no deaths, hangs, restarts or GPU errors; 2520 vs 2515 evals/GPU-h (1.002x).
- (2) passed: low-yield operators 16.1% -> **4.6%** of the arm and `fresh` 5.9% -> **0.8%**, so the
  weights land where they were meant to and the `remove_edit` fall-through is gone. `elite` took
  54.3% of the tuned arm's evaluations against 30.3%.
- (3) **failed on both halves**: -0.20486 summed over the 8 n (limit -0.10), and 5 n at or above
  the snapshot record against 6.
- The whole deficit is two n, and tracing them shows it is not the mechanism the change is about:
  - n=20: the legacy arm's 7.00000 came from `random:seed:grid`, an operator this change cuts --
    but it only re-derives the grid answer the library already holds as n=20's record, so it adds
    nothing a long run would keep.
  - n=34: the legacy arm's 13.11026 (genuinely above the grid, and a new library record) came from
    `face:relocate-split`, whose share this change does **not** touch (15.1% legacy vs 14.6%
    tuned). That is luck, not mechanism.
- The outcome metric shows no benefit either: run-best wins per 1000 evaluations were **19.6
  (legacy) vs 20.0 (tuned)**. The +1.4/1000 predicted from the 6 h run's per-operator rates is
  inside the noise at this length and did not appear. Recomputing on what each arm's library
  would actually keep, max(run best, snapshot record), legacy is still ahead (+0.0875), again
  entirely because of the one n=34 find.
- **Change 5 not validated.** `--op-mix` stays at `legacy` by default; the flag, the tuned weights
  and the tests stay in the tree for a longer experiment. Note this leaves the `remove_edit`
  fall-through in place for the 6 h run: it is a real bug, worth about 1.2% of evaluations going
  to the worst-scoring operator, but it is entangled with the reweighting under one flag and
  separating it needs its own A/B. Not rescued by a replication: a replication is for a result
  that looks like noise around a real effect, and here the effect itself did not show.

## Summary (2026-09-13)

| change | commit | outcome | evidence |
|---|---|---|---|
| 4. 3-fence growth paths | 04a3bb9 | **validated, `--m3-growth on`** | offline 0/1200 -> 1196/1200 cells built, mean area added 1.170; A/B mean fill 0.8677 -> 0.9108 at equal throughput, area within -0.0005 |
| 5. tuned operator mix | b2e213e | **not validated**, stays `legacy` | weights land as intended (low-yield ops 16.1% -> 4.6%) but area -0.205 and wins/1000 19.6 vs 20.0 |
| no change | - | Adam budget left alone | optimizer saturated at 24/24 n; 600 steps saves 4%, batch 1024 loses more than it gains |

## 6 h run with 3-fence growth (2026-09-13 16:07-22:03, commit 925b888)

`run_overnight.py --budget 6h --gpus 4 --archive-dir ~/fence-results --n-range
13-16,18-23,25-30,32-39,41-48,50` (33 n, grid-record n left out), run dir `runs/20260913-160737`,
table in `runs/combined-20260913.txt`. `--m3-growth on`; `--op-mix legacy` (change 5 did not pass).

- No worker deaths, hangs, restarts, GPU errors, job timeouts or pool recreations. 59008 evals,
  2497 evals/GPU-h (2026-09-12: 58469, 2474). 4009 minima indexed.
- **All-time library record improved at 19 of 33 n**; the summed record over the 33 n went from
  397.477 to 398.384 (+0.907, of which +0.144 came from the day's two A/B arms). All 19
  re-verified at 50 digits by `fence/referee.py`.
- **Total gap to the records over the 33 n: 5.627 -> 4.720 (-0.907)**, the largest one-day fall so
  far. Still no n beats its record; n=16 keeps its tie (5.53706377675388772275 vs 5.53706+).
- Largest gains: n=42 +0.247, n=34 +0.178, n=35 +0.128, n=26 +0.072, n=37 +0.051, n=46 +0.040.
- Closest remaining gaps: n=16 0 (tie), 21 0.0022, 19 0.0041, 18 0.0073, 14 0.0123, 48 0.0158,
  26 0.0197, 23 0.0377, 47 0.0383, 28 0.0417, 15 0.0423, 22 0.0501. n=18 and n=19 moved into that
  group this run (from 0.0361 and 0.0223).
- **Change 4 held up at scale.** Mean fill of a verified result (area / faces) over the whole run:
  **0.9217** against **0.8911** for the 2026-09-12 run on the same 33 n, with mean faces 12.13 vs
  12.32 -- the same trade the A/B measured (0.9108 vs 0.8677 on its 8 n), at 1.01x the throughput.
- **n=34 reached its target face count**: 13 faces -> 14 (`F / target` 14 / 14), at 13.17793 from
  12.99952, i.e. it left the grid answer it had been sitting on. That leaves 11 of the 33 n a face
  short of `floor(record) + 1`: 20, 22, 27, 29, 33, 36, 38, 42, 43, 45, 47.
- Eight n still sit exactly on an integer (20, 22, 27, 29, 36, 38, 43, 45, plus 41 and 50 at the
  lattice maximum with a sliver face). Those need a structurally different layout, and neither
  change this day addressed them.
- Operators: elite 117 run-best wins in 17977 evals, face relocate-split 54 in 7721, param 48 in
  9034, edit+ 21, sym 16, seed 8, face dissolve-split 7. `fresh` 3 in 3054 and `seed-program` 4 in
  2282 again earned little, which is what change 5 tried and failed to act on.

## Day 3 (2026-09-14)

### Note: the clock was never wrong

On day 2 I reported that the session's early `date` reads (~06:30) were "stale" against a real
time of ~15:10. That was wrong. File mtimes settle it: the diagnostics and benchmarks were
written 06:22-06:45, and the first code commit (04a3bb9) is stamped 15:08:29. `date`, file
mtimes and git all agree at every point; ynez has been up since 2026-09-08 with NTP
synchronized. What happened is that the session blocked on two interactive questions right after
06:45 and was answered about 8.4 hours later. The cost was real but it was not a clock fault:
the GPUs sat idle for those 8.4 hours (4 GPUs on ynez; serrano's 5 were idle all day anyway).

### 6. `remove_edit` fall-through fix on its own flag (`--remove-edit-fix`)

`RandomMutator.mutate` walks an if/elif chain after choosing an operator by weight; its
`remove_edit` arm is `elif op == "remove_edit" and g.get("edits")`, so a draw on an edit-free
genome falls through to the `else` branch and silently becomes `fresh`. Change 5 fixed this only
under `--op-mix tuned`, and that reweighting failed its A/B. This change moves the fix onto its
own flag, independent of `--op-mix`: a bug, not a weight.

**Integrated A/B**, 2026-09-14 00:4x, commit 8e4f62b, worktree `erich-friedman-exp/rmfix`, run
on **serrano** (5x RTX 3090, 48 cores; both arms on the same host, disjoint GPUs), from snapshot
`minima-snapshots/20260914` (10192 files). Control `--remove-edit-fix off` on serrano GPUs 0-1 vs
treatment `--remove-edit-fix on` on GPUs 2-3; both `--op-mix legacy --m3-growth on`; n = 13, 20,
23, 32, 34, 39, 44, 50; `--budget 30m`, `--seed 12`.

Pre-registered pass criteria, all three must hold:
1. **no eval-count regression** (the bar asked for): no worker deaths or hangs, and treatment
   evals/GPU-h >= 0.95 x control;
2. **mechanism**: the treatment arm's `fresh` share is at least 1.0 percentage point below the
   control's (expected ~5.9% -> ~3.7%: `fresh`'s own weight, 0.06 scaled by the face operator's
   share, is 3.65%);
3. summed best area over the 8 n >= control - 0.10, and at least as many n at or above the
   snapshot record.

At 30 minutes criterion 3 is a non-inferiority check; this change moves about 1-2% of
evaluations from the worst operator to jitter, and its area effect is expected to be small.

### 7. Face-short n: breed from target-face-count layouts (`--target-faces`) -- exploratory

Asked: take the eleven face-short n (20, 22, 27, 29, 33, 36, 38, 42, 43, 45, 47) and try to
resolve them the way n=34 was resolved, 13 -> 14 faces and 12.99952 -> 13.17793.

What n=34 actually shows, from the library's minima metadata: its target face count was always
14 (floor(13.27290) + 1; it is derived from the record, not a setting). It had 14-face layouts
from 2026-09-11 on -- 113 of them -- but all of them filled below the 13-face grid's 13.0 until
one passed it in the day-2 A/B with 3-fence growth on. The same holds for all eleven now:

| n | record | ours | F / target | target-F layouts in library | best of them | vs ours |
|---|---|---|---|---|---|---|
| 20 | 7.14732 | 7.00000 | 7 / 8 | 158 | 6.93786 | -0.062 |
| 22 | 8.05008 | 8.00000 | 8 / 9 | 38 | 7.56615 | -0.434 |
| 27 | 10.12802 | 10.00000 | 10 / 11 | 129 | 9.90640 | -0.094 |
| 29 | 11.08218 | 11.00000 | 11 / 12 | 113 | 10.76012 | -0.240 |
| 33 | 13.00524 | 12.77986 | 13 / 14 | 26 | 12.13355 | -0.646 |
| 36 | 14.13555 | 14.00000 | 14 / 15 | 159 | 13.86746 | -0.133 |
| 38 | 15.07607 | 15.00000 | 15 / 16 | 96 | 14.49626 | -0.504 |
| 42 | 17.00233 | 16.78036 | 17 / 18 | 20 | 16.01212 | -0.768 |
| 43 | 17.31048 | 17.00000 | 17 / 18 | 74 | 16.68022 | -0.320 |
| 45 | 18.07909 | 18.00000 | 18 / 19 | 101 | 17.84848 | -0.152 |
| 47 | 19.03829 | 19.00000 | 19 / 20 | 112 | 18.61991 | -0.380 |

And the search never works on those layouts: seeding by area fills each face-short archive with
the grid basin. On the 6 h run of 2026-09-13 all eleven were seeded with six layouts at F - 1 (n=20:
7.0, 7.0, 7.0, 7.0, 7.0, 6.999; n=38: six at 15.0) and none at the target. `--target-faces on`
(commit 18afe45) seeds and breeds face-short n from target-F layouts only.

**Run**, 2026-09-14 00:4x, worktree `erich-friedman-exp/tface` at 18afe45, **ynez GPU 1** (the
only ynez GPU free; another user holds 0, 2, 3), private library from `minima-snapshots/20260914`,
the eleven n, `--budget 8h`, `--seed 13`, `--target-faces on --m3-growth on --op-mix legacy`.
It runs alongside the production 8 h run, which works on the same eleven n with normal seeding.

Pre-registered outcome definitions (this is exploratory, not a validation of a default):
- **resolved**: the run finds a referee-verified layout with at least the target face count whose
  area exceeds that n's library record at the start of the day (the snapshot). That is the n=34
  outcome. Reported per n, with the count of the eleven that resolve.
- Also per n: the best target-F area at the start (table above) and at the end.
- Compared with the production run's best target-F area on the same n. That comparison is
  observational -- the two differ in GPU count, host and n count -- and is not an A/B.
- Expectation, stated before the run: the four closest (20, 27, 36, 45, 0.06-0.15 short of the
  current best) are the plausible ones; 33 and 42 need 0.65-0.77 and are unlikely to resolve.

**Result** (A/B ran 00:38-01:03 on serrano; scored at 05:58 -- see the note on the session gap below).

| n | snapshot record | fix off | fix on |
|---|---|---|---|
| 13 | 4.10184 | 4.09998 | 4.09888 |
| 20 | 7.00000 | 7.00000 | 7.00000 |
| 23 | 8.49695 | 8.49498 | 8.49015 |
| 32 | 12.02919 | **12.03759** | 12.02847 |
| 34 | 13.17793 | 13.17530 | **13.17835** |
| 39 | 15.01645 | 15.01644 | **15.01645** |
| 44 | 17.45272 | **17.56262** | 17.45129 |
| 50 | 20.00000 | - (0 evals) | - (0 evals) |
| sum (7 n) | | 77.38691 | 77.26358 (-0.12333) |

- (1) passed -- **no eval-count regression**: no deaths, hangs, restarts or GPU errors in either arm;
  1917 vs 1799 evals/GPU-h (**1.066x**).
- (2) passed: `fresh` share **5.7% -> 2.6%** (-3.1 points; needed >= 1.0). Low-yield operators
  15.3% -> 12.7%.
- (3) **failed**: -0.12333 summed (limit -0.10); 3 n at or above the snapshot record in each arm.
- Almost all of the deficit (-0.111) is n=44, where the control found 17.56262, well above the
  snapshot record. That layout came from `elite` (move_fence + remove_face), an operator this change
  does not touch. On the outcome metric the fix is ahead: run-best wins per 1000 evaluations
  **28.3 (on) vs 25.3 (off)**.
- **Change 6 not validated by its pre-registered criteria**, and it stays **off** for the 8 h run:
  change 5 was held to the same criterion, and turning this one on anyway would be restating a
  criterion after the fact. Unlike change 5, the effect this change is about did show (the `fresh`
  share fell as predicted, throughput rose, wins per 1000 rose), so the result looks like noise
  around a real effect -- which is what a replication with fresh criteria is for. Not run today: it
  would have held back an already-late 8 h run on 8 GPUs.
- n=50 got **zero evaluations in both arms** (74-75 `n_exhausted` events each); see below.

### Session gap (2026-09-14 01:03-05:58)

The A/B finished at 01:03 and the session carrying this work ended before it was scored; it was
picked up again at 05:58. serrano's 5 GPUs sat idle for those ~4.9 hours (task 5 kept ynez GPU 1
busy throughout; ynez GPUs 0, 2, 3 were held by another user until some time in that window). The
8 h production run therefore started at 06:02 instead of ~01:15.

**Phase 1** (00:47-08:47, 8 h, ynez GPU 1, commit 18afe45): 15608 evaluations, no restarts, one
CUDA out-of-memory at n=47 recovered in place. **n=43 resolved**: an 18-face layout at 17.08181,
past its 17.00000 grid -- the n=34 outcome. Target-F best improved at 9 of the 10 n it could
reach (n=43 +0.402, n=33 +0.224, n=22 +0.217, n=42 +0.161, n=27 +0.093 to 9.99983). **n=38 got no
evaluation at all** (287 `n_exhausted`): all its target-F layouts belong to lineages that once
stabilized on the 15.0 grid, and the grid-trivial block is per lineage, so it barred them too.

**Phase 2** (from 08:49, ~5 h 10 m, ynez GPU 1, worktree `erich-friedman-exp/tface2` at 4b16eb1 on
a detached HEAD -- the main tree is not touched under the live production run): the follow-up fix
exempts target-F layouts with a non-grid-trivial area from the lineage block (they can never be
the grid answer, which is all the block exists to prevent). Continues from phase 1's private
library, `--seed 14`, otherwise as phase 1; same outcome definitions. n=38 is evaluated from its
first minutes.

**Outcome** (task 5, both phases, scored against the pre-registered definitions):

| n | target F | best target-F: start | phase 1 | phase 1+2 | production (same n) | resolved |
|---|---|---|---|---|---|---|
| 20 | 8 | 6.93786 | 6.93389 | **7.03935** | 7.00799 | **yes** (7.0 grid -> 7.03935) |
| 22 | 9 | 7.56615 | 7.78332 | 7.85192 | 7.44177 | no |
| 27 | 11 | 9.90640 | 9.99983 | 9.99983 | 9.75862 | no (0.00017 short of 10.0) |
| 29 | 12 | 10.76012 | 10.78420 | 10.78420 | 10.72700 | no |
| 33 | 14 | 12.13355 | 12.35759 | 12.35875 | 12.13590 | no |
| 36 | 15 | 13.86746 | 13.89824 | 13.90539 | 13.76130 | no |
| 38 | 16 | 14.49626 | - (blocked) | 14.68939 | 14.39067 | no |
| 42 | 18 | 16.01212 | 16.17295 | 16.17410 | 16.00409 | no |
| 43 | 18 | 16.68022 | 17.08181 | **17.16777** | 16.82203 | **yes** (17.0 grid -> 17.16777) |
| 45 | 19 | 17.84848 | 17.89507 | 17.89877 | 17.61185 | no |
| 47 | 20 | 18.61991 | 18.62900 | 18.65072 | 18.61679 | no |

- **2 of 11 resolved**, both into the library record at their target face count: face-short n
  11 -> 9. Against the expectation stated before the run: n=20 was among the four named, n=43 was
  not (it started 0.32 short); n=27, 36, 45 were named and did not resolve; 33 and 42 did not, as
  expected.
- The production comparison is confounded: serrano (below) held 8 of these 11 n and searched them
  for 1.4 h, not 8. The three on ynez are the fair part -- production had the full 8 h there, with
  3 GPUs over 12 n against task 5's 1 GPU over 11 -- and task 5's best target-F layout was still
  ahead at all three (n=33 12.359 vs 12.136, n=42 16.174 vs 16.004, n=43 17.168 vs 16.822). Three n
  is suggestive, not a validation: `--target-faces` stays off by default until it has its own A/B.

## 8 h run (2026-09-14 06:02-14:02, commit 6bbc889, two hosts)

`run_overnight.py --budget 8h` on ynez GPUs 0, 2, 3 (12 n; GPU 1 ran task 5) and serrano GPUs 0-4
(21 n), disjoint `--n-range`s split by schedule weight (5.00 vs 5.30 per GPU), one shared library.
`--m3-growth on`, `--op-mix legacy`, `--remove-edit-fix off`. Table: `runs/combined-20260914.txt`.

- **ynez: 62048 evals, 2615 evals/GPU-h, no restarts, hangs or errors**; CPU wait 4.3%.
- **serrano went down at 07:24:39**, 1 h 22 m in: all five workers and heartbeats stopped in the
  same second, and the host answered neither ping ("no route to host") nor ssh for the rest of the
  day. 11054 evals before that. Nothing in the library was damaged: records are written atomically
  and only ever replaced by a larger area; all 21 of its records and the 744 minima files it wrote
  parse. Its n were not moved to ynez: without knowing whether serrano had crashed or was only cut
  off, a second writer on the same n was the larger risk, and ynez had no free GPU.
- **Records, against the current page** (re-scraped this day; 19 entries rose since 2026-09-10):
  improved at 17 of 33 n, all re-verified at 50 digits at tol 1e-9 and 1e-12; total gap
  **5.15948 -> 4.54024 (-0.61924)**. Largest gains n=44 +0.199, n=43 +0.168 (task 5), n=42 +0.086,
  n=35 +0.070, n=20 +0.039 (task 5), n=26 +0.021. n=16 keeps its tie; nothing beats a record.
- A run logged two IMPROVEMENT events at n=26 (9.88939406 against 9.88822+). They were against the
  stale table; the page says 9.89324+. The referee passed the layout, so it is a real library
  record, 0.00356 short of the published one.
- **n=50 got no evaluation at all**: every seed program and every seeded elite at n=50 belongs to a
  lineage marked grid-trivial there (27 of them), so make_job declares n exhausted before the face
  operator -- which is meant to keep editing blocked grid layouts -- is ever called. n=38 and n=41
  are close to the same state (no open seed program; 1 and 4 unblocked elites). Not fixed today:
  it is a change to the default search and needs its own A/B.
