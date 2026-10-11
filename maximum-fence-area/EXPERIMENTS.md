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

## Day 4 (2026-09-17)

### 8. Two-stage net pipeline (`pipeline.py`) -- new search, not yet A/B'd against the old one

The standing diagnosis is that the local optimizer is saturated (0 gain at 24 of 24 n from 8x
batch and 4x steps) and that **topology discovery is the bottleneck**. `run_overnight.py`
attacks it indirectly, by evolving programs that draw layouts. This change adds a second,
independent entry point that attacks it directly: enumerate the combinatorial structures first,
reject the ones that provably cannot reach the record, rank what is left, and only then spend
optimizer time.

New modules, all additive -- `run_overnight.py`, `fence/evolve.py` and the optimizer are
untouched, so nothing here can change a production run's behaviour:

| module | what it holds |
|---|---|
| `fence/rules.py` | the 14 hard rules, one registered check each, plus the rejection `Ledger` |
| `fence/heuristics.py` | the 9 ranking heuristics, one registered scorer each, with weights |
| `fence/nets.py` | the counting envelope, 6 net generators, the filter-and-rank stage |
| `fence/netevolve.py` | a population over one net's geometry, with 12 mutation moves |
| `pipeline.py` | the driver, `--list`, `--calibrate`, and the per-n report |

Every rule, heuristic, generator and move is a separate switch (`--no-rule`, `--no-heuristic`,
`--weight`, `--no-move`, `--move-weight`, `--generators`), so the cost of each can be measured
one at a time, and every rejection is counted against the id of the check that fired. An unknown
id in any of those flags raises rather than being ignored: a typo would otherwise look exactly
like the check being free.

#### The rules check out against every verified configuration we hold

Running all 14 rules over the 39 referee-verified records in `minima/records`:

- **13 of 14 rules pass on all 39.** That is a real cross-check of the counting rules against
  independent data: Euler (V - (n+T) + F = 1) and the incidence identity
  (sum of face sizes = 2(n+T) - b) hold exactly on every one, with T counted as distinct
  T-junction *points*.
- **rule 14 (F >= ceil(A_record)) rejects exactly 9**: n = 22, 27, 29, 33, 36, 38, 42, 45, 47.
  That is precisely the face-short list, re-derived from scratch by a module that knows nothing
  about the search. Rule 14 rejecting our own current best at those n *is* the reason they are
  stuck: they cannot reach the record at their face count, whatever the optimizer does.

A derived bound worth recording, from rule 11 applied to the outer boundary rather than to a
face: the bounded area is what the outer cycle encloses and each of its b graph-edges is at most
1 long, so **total area <= b^2 / (4 pi)**, i.e. b >= 2 sqrt(pi A_record). It holds on all 39 and
it prunes the counting envelope (at n = 22 it removes boundary sizes the face rules allow).

#### Heuristics 1 and 2 disagree with the records, so the defaults follow the records

Both were stated as coming from a square-grid model and as unverified. Measured over the same
39 configurations (`pipeline.py --calibrate`):

| | square-grid model | measured | verdict |
|---|---|---|---|
| boundary size b | 2 sqrt(2n) = 2.83 sqrt(n) | **3.31 sqrt(n)**, spread +-17% | model too small |
| face count F | n/2 | **0.402 n** | model too large by ~25% |

The boundary figure splits sharply by structure: **2.59 sqrt(n)** for the T-junction-free grid
layouts and **3.65 sqrt(n)** for the T-rich ones, so a single coefficient is a weak prior either
way and the heuristic is weighted accordingly.

The face-count result is the sharper one: **F = ceil(total area) on 37 of 39** configurations
(all 39 under exact rounding). The face count is not an independent target at all -- it is the
record-based bound of hard rule 14, reached exactly. So the default face target is ceil(A_record),
not n/2, which would rank the right nets last at every n. `--boundary-model square-grid` and
`--face-model half-n` restore the model values for comparison.

#### What stage 1 says about the face-short n

At n = 22 the generators that came with the old search produce almost nothing legal: rule 14
rejects 132 of 141 candidates, and the 2 survivors are pinwheels. That is the supply problem,
visible before a second of GPU time is spent. A generator that drives the face count up to the
bound directly (repeated `face_fill` dissolves, each trading a corner joint for a T-junction and
a returned one-fence cell) fixes the supply: at n = 22, 42 of 184 candidates now reach F >= 9,
and at n = 27, 56 of 223 reach F >= 11.

#### What each check is actually worth

`pipeline.py --ablate rules --n-range 12,22,30 --max-candidates 300` (stage 1 only, 151 nets
kept with everything on):

| rule off | nets kept | delta |
|---|---|---|
| `record_faces` (14) | 374 | **+223** |
| `bridgeless` (10) | 157 | **+6** |
| the other twelve | 151 | +0 |

Counting rejections across a whole run rather than one ablation gives the same answer, and
says where each rule earns its keep:

| | stage 1 (net generators) | stage 2 (mutation moves) |
|---|---|---|
| `record_faces` (14) | 2256 | 80 |
| `bridgeless` (10) | 248 | 29 |
| `t_straight` (12) | 0 | 2 |
| everything else | 0 | 0 |

So as *filters*, only rules 10, 12 and 14 do any work today, and rule 14 does almost all of
it. The other eleven never fire because `fence.builder` already refuses to return an illegal
embedding, and because `apply_move` restores the fence count before a mutant is judged. That
is not an argument for deleting them: they are the specification, they are what makes a new
generator or move safe to add, and they pass on all 39 verified records, which is what says
the counting identities are right. It is an argument against expecting them to find anything.

The heuristics were ablated the same way over n = 12, 16, 22, 30, 38, by re-scoring the same
nets rather than regenerating them (a heuristic rejects nothing):

| heuristic off | top-5 nets kept | best net changed |
|---|---|---|
| `boundary` | 11/25 | 5/5 n |
| `convex_outer` | 14/25 | 3/5 n |
| `lattice_snap` | 14/25 | 1/5 n |
| `t_at_boundary` | 15/25 | 2/5 n |
| `face_quality` | 16/25 | 1/5 n |
| `seed_family` | 17/25 | 1/5 n |
| `symmetry` | 18/25 | 0/5 n |
| `face_count` | 19/25 | 0/5 n |
| `one_cluster` | 19/25 | 0/5 n |

`one_cluster` scores 1.000 on every net that survives the rules -- rules 6 and 13 already
require one component -- and `face_count` takes four distinct values with mean 0.98, because
rule 14 already filtered on it. Both are only live when those rules are switched off.

#### Bugs this work turned up

- `NetEvolver` offspring loop could spin forever when every enabled move declined on a net
  (the attempt counter only counted successes). Fixed with an attempt budget.
- `RuleSet.from_flags` silently ignored unknown ids, which in an experiment harness would make a
  mistyped `--no-rule` look like a free check. Now raises, and the same for heuristics and moves.
- Not a bug in the search, but a real cost: **torch and the BLAS behind the SLSQP polish default
  to one thread per core**, and on a 64-core box the oversubscription makes the small-batch CPU
  path far slower than four threads. The two end-to-end optimizer tests took 107 s of wall and
  **40 minutes of CPU**; with `tests/conftest.py` capping threads at 4 they take 5 s. The whole
  suite (268 tests) now runs in 99 s. `pipeline.py --torch-threads` defaults to 4 for the same
  reason.

### Cold-start run of the pipeline (2026-09-17 07:20-08:04, ynez, 4 shared GPUs)

`pipeline.py` over **all 48 n (3..50)**, one process per GPU on disjoint n (n mod 4), 44 minutes
wall, `--nets-per-n 4 --evals-per-net 14 --generations 5 --pop 5 --elite 3 --batch 640 --steps 450
--mem-cap-gb 1.5 --seed 20260917`. No shard hit its budget. Table:
`runs/combined-pipeline-20260917.txt`. This is a **cold start**: no evolved programs, no
checkpoints, nothing carried over except the library's topologies as one of six net generators.

ynez's four GPUs were held by another user's job throughout (about 17 GB each, 18-34% utilization),
so the run took 1.5 GB per GPU and the CPU path was capped at 2 torch threads. serrano was
unreachable all session ("no route to host"), so this is a one-host run.

- **11 of 48 n reach the published record**: n = 3, 4, 5, 6, 7, 12, 17, 24, 31, 40, 49. Ten of
  them re-verify at tol 1e-9 **and** 1e-12. n=40 holds only at 1e-9: its area is
  16.000000000000056 with one fence 6.5e-11 off unit length, so the polish did not fully tighten
  at `--polish-iters 800` (production uses 1200). Reported as a tie at the project's certification
  tolerance, not at 1e-12.
- **n=6 is the interesting tie**: 1.47585295983, the Antoine Lagache record, found from scratch as
  a 2-face layout with 2 T-junctions. The other ten ties are the unit-square grids and the tiny n.
- **n=39: 15.28190316, beating the library's own record of 15.01647 by +0.26543**, verified at both
  1e-9 and 1e-12 with max|len-1| = 5.2e-16. Still 0.253 below the published 15.53498. The structures
  are not close: the pipeline's layout is F=16, **T=3**, V=27, b=17, where the library's is F=16,
  **T=23**, V=47, b=23. A much simpler net at the same face count does better -- which is the kind
  of thing the net stage exists to find, and the kind of thing a program-mutation search reaches
  only by accident.
- **Every n came back at exactly its target face count** (F = ceil(A_record) at all 48), because
  rule 14 rejects anything short before it costs GPU time.

Honest comparison, since these are not comparable budgets:

| | n covered | total shortfall vs the page |
|---|---|---|
| pipeline, one 44-minute cold start | 48 (3..50) | **18.563** |
| minima library, many GPU-hours over 4 days | 39 (12..50) | **4.540** |
| pipeline, restricted to those same 39 n | 39 | 17.966 |

The library is four times closer, as it should be after days of GPU time against 44 minutes. The
pipeline is ahead at exactly one n (39) and behind at 30. What the run establishes is that the
two-stage pipeline works end to end at every n, produces only referee-verified results, and lands
in the right structural region immediately -- not that it is a better search. **It has had no A/B
against `run_overnight.py` and is not a default.** The comparison worth running next is equal GPU
time from the same frozen library snapshot, on the face-short n, where rule 14 says the existing
search is structurally stuck.

n = 3..11 are new: `run_overnight.py` defaults to n >= 12, so the harness had never searched them.
Five of the nine tie their record.

## Day 5 (2026-09-21)

### Baselines re-scraped: 23 records rose in four days

`baselines.json` re-scraped 2026-09-21 12:16 UTC (page sha `78504fff`, was `75166e3e`). **23 of 48
records rose**, including all five grid records the protocol above calls "likely optimal": n = 17,
24, 31, 40, 49 are now 6.01086, 9.02394, 12.03575, 16.04172, 20.07383 (Yanis Gomes, September
2026). The grid answer is no longer the record at any n >= 17, so the protocol's exclusion of those
n no longer applies. The library's total gap against the page is **5.398** today, not 4.540 --
the page moved, the library did not. `tests/test_baselines.py` spot values updated with the scrape.

### Stranded pipeline results promoted into the library

`pipeline.py` had no write-back path, so everything the 2026-09-17 pipeline run found stayed in its
run directory. Promoted (`runs/exp-20260921/promote.py`), each re-verified by the referee at 1e-9
**and** 1e-12 first -- all ten pass both: **n = 39 at 15.28190316 (+0.26543 over the library's
15.01647)**, and n = 3..11, which had no library records at all. `pipeline.py` now writes every n's
verified best to `records/` (if larger) and indexes every net's best as a minimum of a `pipeline`
lineage (`--library-writes`, on by default), under the new per-n library lock.

### Rule 14, resolved

Rule 14 (F >= ceil(A_record)) rejects our own records at 22, 27, 29, 33, 36, 38, 42, 45, 47 (against
the 09-17 page) because each is **exactly one face short**, and at seven of them every face is
already at area 1.0 -- saturated grid answers with no geometric headroom. The rule is right; the
rejection is the diagnosis. What was wrong is where it bites: the `library` generator yields the
record, the rule rejects it, and the strongest known geometry never reached stage 2 at exactly the
n that needed it.

Rule 14 is also looser than it could be. A face with m graph-edges holds at most
`isoperimetric_cap(m)` (a triangle sqrt(3)/4, not 1), so **rule 15 `area_cap`: sum of face caps >=
A_record**. No record violates it against its own area (39 of 39), and against today's page it
rejects rule 14's set plus n = 39, 41, 50, whose layouts carry a triangle (n = 41: 16 unit squares +
a triangle, cap 16.433 < 16.448).

### Measured before changing anything: the stage-1 ranking is anti-correlated with capacity

Of the distinct nets that pass the rules, how many have a cap sum >= the record, and how many of the
top 4 (`--nets-per-n 4`: all that ever get optimizer time): n=22 32 of 44 but **1 of 4** (the most
capacious ranked 42nd of 44); n=45 24 of 49 but **0 of 4**; n=39 7 of 69, **0 of 4**. With rule 15 on,
every survivor can hold the record by construction.

### Refuted before implementing: re-fitting the boundary heuristic

The records' b/sqrt(n) splits most sharply at T <= 2 / T > 2 (2.56 vs 3.65), and the per-regime
spreads (5% / 7%) are much tighter than the pooled 0.17 in use. Scored on the 39 records, the
current interpolation (T_RICH = 8, spread 0.17) is still the best model: mean 0.946, worst 0.56.
Tight per-regime spreads give 0.728 (0.707 leave-one-out) with 8-9 records below 0.5 -- they would
rank real record layouts down. Not changed.

### 9. Collapsed faces: open them before re-optimizing (`inflate_face`) -- FAILED, off

Five records carry a face at area <= 0.036 (n = 13, 15, 39, 41, 50; 4.9e-7 at 41 and 50) whose
isoperimetric caps add up to 3.72 units of headroom. Hypothesis: an optimizer failure, recoverable by
pushing the face's vertices along the gradient of its own area and re-optimizing.
Pre-registered: same seeds and settings, arms differ only in the inflation; PASS = strictly higher
best area at >= 2 of the 5 and no fewer polish successes. **Result: equal at 5 of 5** -- all 30
runs returned to the record's area, the inflated ones to the same collapsed face (n=41: 4.9e-7 again).
The collapsed face is a genuine local optimum of those topologies; the headroom is capacity the
bound allows and the geometry cannot realize. Those n need a topology change as well. The move stays
registered with weight 0. (`runs/exp-20260921/inflate-n*.json`)

### 10. Net stage: rule 15 + `area_capacity` + `library_face_target` -- pre-registered

Treatment: defaults as of this commit (rule 15 on; heuristic 10 `area_capacity`, weight 1.5; generator
`library_face_target`, which dissolves corners of the library's best layouts until rules 14 and 15
both pass -- at n = 45 all four top-ranked nets come from it). Control: `--no-rule area_cap
--no-heuristic area_capacity` and the six previous generators. Everything else equal: n = 22, 33, 38,
41, 45 (stuck, their library bests fail rule 15), one process per (arm, n), both arms of an n on the
same serrano GPU, `--budget 20m --per-n-budget auto --nets-per-n 8 --evals-per-net 40 --generations 8
--pop 6 --elite 3 --batch 1024 --steps 450 --workers 4 --seed 20260921`, each arm from its own copy
of the post-promotion library (`minima-snapshots/20260921-ab`).

PASS requires both: (i) summed best verified area over the 5 n, treatment >= control - 0.02;
(ii) treatment >= control at >= 3 of the 5. On a FAIL the night's net-stage run uses the control
settings (the plumbing fixes stay: they do not change what is searched).

**Result: FAIL** (arms 05:40-05:57 PDT; `runs/exp-20260921/score_ab.txt`). Every arm spent its
caps (8 nets x 40 evaluations, 307-320 evaluations each) before the 20 min budget.

| n | library | control | treatment |
|---|---|---|---|
| 22 | 8.00000 | **7.56168** | 7.47370 |
| 33 | 12.79735 | 12.00000 | 12.00000 |
| 38 | 15.00000 | 14.32450 | **14.43234** |
| 41 | 16.00000 | 16.00000 | 16.00000 |
| 45 | 18.00000 | **17.55649** | 17.47820 |
| sum | | 67.44267 | 67.38424 (-0.05843) |

(i) failed (-0.058 < -0.02), (ii) failed (2 of 5). The differences are within what one lucky net
makes, and neither arm reached the library at any of the five n: a layout with the extra face the
record needs starts 0.4-0.8 below the saturated grid answer, whichever stage-1 ranking picks it
(the same gap the day-3 table found for target-F layouts). The ranking was measurably wrong and
rule 15 fixes it by construction, but a better-ranked candidate list did not turn into area here.
As pre-registered: rule 15, heuristic 10 and `library_face_target` go **off by default**
(`--rule area_cap`, `--weight area_capacity=1.5`, `--generators ...,library_face_target` turn them
on), and the night's net stage runs with the control settings.

Also found while setting the arms up: **`--generators` had never worked** -- the value was passed on
as a string and iterated character by character ("unknown net generator 'p'"), so no generator could
be switched off from the command line. Fixed (`parse_generators`), with a test.

## Night run (2026-09-21 05:57-~12:00 PDT, serrano 5 GPUs, `runs/night-20260921-125751`)

Capacity: serrano's five GPUs only. temescal lists GPUs 0, 2, 3 as idle but its driver wants a reboot
(GPU 1 fell off the bus) and refuses cuInit on every device -- the new preflight CUDA probe rejects
all three; ynez's four are another user's (llama-server, load 65 on 64 cores) and were left alone.

- **Explore** (GPUs 0-2): `pipeline.py`, control net-stage settings, `--budget 6h --per-n-budget auto
  --nets-per-n 30 --evals-per-net 40`, live library writes, over the 23 n where the library's best
  layout cannot reach the record by rule 15 or needs a new topology anyway: 8-11, 13, 15, 17, 22,
  24, 27, 29, 31, 33, 36, 38, 39, 40, 41, 42, 45, 47, 49, 50 (gap 3.438).
- **Exploit** (GPUs 3-4): `run_overnight.py --tag serrano-exploit`, production defaults, over the
  18 geometry-limited n: 14, 18-21, 23, 25, 26, 28, 30, 32, 34, 35, 37, 43, 44, 46, 48 (gap 2.278).
- Excluded, already at their record: 3-7, 12, 16.
- Launch library = `minima-snapshots/20260921-ab` (post-promotion; nothing wrote to the live library
  between that copy and the launch). The run was launched from an uncommitted tree; the diff and the
  new files are saved in the run dir (`working-tree.diff`, `untracked-files.tgz`, `base-commit`).
- **07:07 PDT, added**: the three explore GPUs measured **21-28% busy** over 40 s (run_overnight's
  two: 93-100%) -- pipeline.py polishes and verifies synchronously, the deferred throughput change.
  A second pipeline process per explore GPU (`explore-b-shard*`, `launch_explore_b.sh`, `--budget
  285m --seed 20260923`) uses the idle cycles with experiment 10's treatment settings: failed as a
  default, but they generate different nets (library-layout dissolves) from the control's. Labelled
  as an extra explorer; both processes of a GPU share its n, which the library lock makes safe.
- **09:39 PDT**: `explore-shard1` finished all 8 of its n (nets exhausted, not budget). **09:45, added**:
  a second `run_overnight.py` on the freed GPU 1 (`runs/20260921-164559-serrano-exploit-b`, `--budget
  135m --seed 20260921`) over the six largest-gap exploit n, 32, 37, 43, 44, 46, 48 -- where the night's
  steady gains were coming from -- alongside the first instance, safe under the library lock.
- **11:12 PDT, added**: `explore-shard2` finished its 8 n at ~11:10; a third `run_overnight.py` on GPU 2 (`runs/*-serrano-exploit-c`, `--budget 46m`) over exploit n 20, 25, 30, 34, 35 until the end.

### Night run result (ended 12:00 PDT; `runs/night-20260921-125751/final-comparison.txt`)

Every process ended cleanly: all six explore shards finished their full n lists, all three
run_overnight supervisors exited normally with 0 deaths, hangs or restarts.

**13 library records improved, every one re-verified by the referee at 1e-9 and, separately, at
1e-12.** None reaches its published record; the page's ties stay n = 3-7, 12, 16.

| | n = 12..50 | n = 3..11 |
|---|---|---|
| shortfall, library after the 2026-09-14 run | 5.39832 | no records |
| at tonight's launch (after promoting the stranded n=39 and n=3..11) | 5.13288 | 0.58330 |
| **now** | **4.93865** | **0.48865** |
| tonight | -0.19423 | -0.09465 |

All shortfalls are against the page as scraped 2026-09-21; the 4.540 quoted on 09-14 was against a page
that has since moved (5.398 is the same library against today's page).

| source | GPU time | gain | where |
|---|---|---|---|
| explore (pipeline, 6 processes on 3 GPUs) | 3 GPUs x 6 h | +0.13394 | n=10 +0.06067, n=39 +0.03929, n=8 +0.03396, n=9 +0.00002 |
| exploit (run_overnight, GPUs 3-4) | 2 x 6 h | +0.07263 | n=37 +0.07263 (n=28 +3e-9, noise) |
| exploit-b (GPU 1, from 09:45) | 2.25 h | +0.06390 | n=46 +0.03222, n=44 +0.02296, n=48 +0.00454, n=43 +0.00261, n=32 +0.00157 |
| exploit-c (GPU 2, from 11:13) | 0.77 h | +0.01842 | n=30 +0.01635, n=35 +0.00207 |

What the night says:
- **The stuck n stayed stuck.** At the 16 capacity-capped n other than 39, no layout with the extra
  face the record needs reached the saturated grid answer: the best were 0.2-0.8 below it (n=33 12.297
  vs 12.797, n=45 17.645 vs 18.0, n=47 18.564 vs 19.0). They are now indexed as `pipeline` minima, so a
  later run starts from them rather than from the grid. Filling those layouts, not finding or ranking
  them, is the open problem.
- The exploit gains came mostly from the two instances added on GPUs freed mid-run (+0.082 in 3 GPU-h
  against +0.073 in 12 GPU-h for the main instance), at n the main instance also held: a fresh seed and
  a fresh archive on a few high-gap n paid better than a sixth hour on a long-running archive. Observational.
- explore vs explore-b (control vs experiment 10's treatment settings, different seeds, shared GPUs,
  explore-b started 70 min later): control higher at 7 of 18 n, treatment at 8, equal at 3 -- no sign of
  a systematic difference, consistent with the A/B.
- Utilization: the pipeline GPUs ran at 21-28% with one process and 44-46% with two; run_overnight's at
  89-100%. Asynchronous polish in `netevolve` (deferred today) is the pipeline's largest throughput lever.
- 3750 build errors across the explore shards; the ledger's sample is `symmetrize` leaving a zero-length
  edge, but which moves account for the count is not logged (stage 2 keeps move stats in memory only).

## Day 6 (2026-09-24) -- 4 h run: chase the record shapes (`PROMPT-4h-record-shapes.md`)

Clock: the run's first start (04:51 PDT) was lost -- the session sat 4 h 07 min on the dirty-tree
question before any work began. The watchdog was restarted at 08:59 PDT when work could begin
(3:45 stop at 12:44, 4 h at 12:59). Snapshot ref `refs/backup/pre-4h-run-2026-09-24` (tracked
changes only; the untracked test files stay on disk). `pytest -q` green before any change.
Baselines re-scraped once at 08:59 PDT (15:59 UTC): **no record moved since 2026-09-21**; the gap is
still 4.93865 over n = 12..50 and 0.48865 over 3..11. GPUs: serrano 0-4 (CUDA probe OK); temescal's
GPU 1 fell off the bus again and cuInit fails on all its devices; ynez is another user's (~9 GB per GPU).

### 11. Digitize the page's record pictures (`fence/digitize.py`) -- pre-registered

Hypothesis: the September records at grid-tied n are not grids but rounded ring/onion layouts our
seed families never build; loading the record's own topology from its picture and optimizing it
reaches the record.

Method: `<n>.gif` -> threshold -> Zhang-Suen skeleton -> planar graph (crossing-number junctions,
Douglas-Peucker corners, stubs < `min_edge` fence lengths contracted) -> lines of straight edges ->
each line cut by a small DP into pieces of near-integer length (a whole number of unit fences) ->
`make_topology`. Each picture is read with a grid of 20 parameter settings; every distinct legal
topology gets `optimize_topology` (batch 2048, 1000 steps, 2 seeds) on a serrano GPU, then the
referee at 1e-9 and 1e-12. A verified result above the library goes in with `update_record`, meta
`{"via": "digitized", "run": "4h-20260924"}`.

Pre-registered PASS: at least 3 of n = 17, 24, 41, 50 give a referee-verified layout within 0.01
of the page's figure. Fewer than 3 in the time box is a FAIL, and the entry says which failure
it is -- bad digitization (no legal topology with the record's face count) or optimizer shortfall
(the right topology, not filled).

### 12. Move-neighbourhood search around record-shaped topologies -- pre-registered

Hypothesis: a topology one or two moves from a record (or from a digitized near-record) is more
likely to beat it than anything a generic search finds cold.

Method (`runs/4h-20260924/neighbours.py`): from a seed configuration, apply each of netevolve's seven
topology moves (`join_to_t`, `t_to_join`, `slide_t_past`, `repin`, `add_face`, `remove_face`,
`face_fill`) under 40 rng seeds each; the moves are randomized, so this sampling, deduplicated by
canonical signature, is how their applications are enumerated. Every distinct neighbour that passes
the default hard rules (with rule 14's `min_faces` from today's page) is screened (batch 1024, 600
steps, referee). Level 2 applies the moves again to the best 6 level-1 neighbours (every pair from
the whole first set does not fit the time box -- a documented deviation from the brief). The top 3
get full settings (batch 2048, 1000 steps, 2 seeds). Seeds: the Phase 1 digitized bests that sit
below their record, and the library records of the non-grid highest-gap n where no digitization
succeeded. One GPU per n, 25 min per n.

Pre-registered: no success bar. The result is, per n, whether any referee-verified candidate beats
(a) the page's record and (b) the library record, and by how much -- "no improvement found" is a
result, not a reason to extend the box.

### 13. Ring/onion net generator (`--generators ...,onion`) -- pre-registered, off by default

Hypothesis: the page's September shape -- a core polygon inside concentric rings of cells with a
rounded boundary -- is a parametric family, and building it directly gives stage 2 record-shaped
nets at any n, without a picture.

Generator `onion` (`fence/nets.py`, `default=False`): a core c-gon (c = 3..6), then 1-3 rings; ring
j's outer boundary is a regular s_j-gon of unit chords about one unit outside the polygon inside it,
with m_j >= s_j/3 spokes running inward; a spoke's outer end is a vertex joint or (variant) a
T-junction mid-fence, its inner end a joint near a vertex and a T-junction otherwise. n = c +
sum(s_j + m_j), F = 1 + sum(m_j); every (c, rings) that hits n exactly, most faces first, 60 specs x 4
variants per n.

Pre-registered PASS requires both: (i) a net passing every default hard rule (rule 14 against
today's page) for >= 80% of n in 12..50; (ii) `pipeline.py --generators onion --nets-per-n 4
--evals-per-net 20 --no-library --library-writes off` reaches within 5% of the library record as it
stood at launch (after Phase 1) at each of n = 17, 24, 33, 41, 50. Structural check, reported but not
part of the bar: for 17, 24, 41, 50, does the family contain a net with the digitized record's
(F, T, joints)? On a FAIL the generator stays in the tree, off, with these numbers.

Already measured before the stage-2 arm (generation only, no optimizer): (i) **35 of 39 n (90%)**
-- none at n = 12, 13 (too small for a core and a ring), rule 14 rejects all at 14, 15. Structural
check: the closest onion net has (F, T, joints) = (7, 11, 11) vs the digitized record's (7, 12, 11) at
17, identical (10, 14, 15) at 24, (17, 25, 25) vs (17, 24, 25) at 41, identical (21, 34, 30) at 50 --
with outer-T spokes; with vertex spokes only, T was 2-3x too low.

**Result of 13: FAIL** on (ii) at all five n (arms 09:27-09:40 PDT, serrano GPUs 1, 3, 4, `runs/4h-20260924/onion-n*`):

| n | library at launch (after Phase 1) | onion arm, best verified | vs library | distinct onion nets |
|---|---|---|---|---|
| 17 | 6.01087 | 5.13452 | -14.6% | 2 |
| 24 | 9.02159 | 8.54861 | -5.2% | 10 |
| 33 | 13.01867 | 11.16437 | -14.2% | 4 |
| 41 | 16.43200 | 15.22380 | -7.4% | 30 |
| 50 | 20.55748 | 18.83024 | -8.4% | 106 |

(i) passed (35 of 39 n); the structural check passed (the family contains nets with the records'
(F, T, joints)); (ii) failed everywhere -- n = 24 came closest, 0.2 points outside the bar. Matching
the counts is not matching the net: the digitized record topologies optimize straight to within
0.01 of the page (change 11), while the family's nets with the same F, T and joint count fill 5-15%
lower, so what matters is which spoke lands on which fence, and where -- the detail the regular
construction does not get right. As pre-registered, `onion` stays in the tree **off by default**
(`--generators ...,onion` turns it on). A family worth retrying would be fitted to the digitized
records' spoke placement rather than spreading spokes evenly.

**Result of 11: PASS** (08:59-09:50 PDT, serrano GPUs 0-4; `runs/4h-20260924/`). At the time the
criterion was scored: n = 17 **6.0108687** (page 6.01086+, a tie at the page's precision), n = 24
**9.0215924** (-0.00235), n = 50 **20.5574787** (-0.00081) -- three of four within 0.01, every one
referee-valid at 1e-9 and at 1e-12. n = 41 is the miss: its picture never read cleanly (a bend 0.25
fence lengths above a 4-way crossing), so its graph was patched by hand at that one spot (the
brief's fallback) and the result, 16.43200, is 0.0164 short -- a local topology misread or an
optimizer shortfall, not separable in the time box (Phase 2 took it to 16.43382, still 0.0146 short).

What the digitizer needed, in the order the pictures demanded it (all in `fence/digitize.py`):
crossing-number junctions with the 3x3 neighbourhood cut out (a pixel-count test marks every
diagonal staircase pixel as a junction); one point per BFS level before Douglas-Peucker (sorted
pixels zig-zag); stubs contracted at 0.1-0.15 fence lengths (n = 17's record carries a pinwheel
triangle with sides ~0.2 -- at 0.15 it disappears and the read has 6 faces, which cannot exceed 6.0);
a DP that cuts each straight line into near-integer pieces instead of cutting at every bend (pixel
noise bends 3-5 degrees); forced cuts where two lines would pass through one node (fences never
cross); noise-bend nodes dropped instead of pinned. That read 38 of the 41 pictures tried (n = 8..50
where the library was short). v2 (after n = 44, 46, 49 failed): a per-fence price in the DP searched
until the count is exactly n; unconnected nodes merged; a skeleton T pulled towards its stem put
back on its through-line; and a ring-branch junction test -- Zhang-Suen leaves two-pixel blobs at
some junctions that hide them from the crossing number (n = 46 read 18 faces for a record that needs
19 until this). v2 read all three.

| n | library before | digitized v1 | v2 | page | best digitized vs page | F |
|---|---|---|---|---|---|---|
| 8 | 2.03396 | 2.10306 |  | 2.10306 | -0.00000 | 3 |
| 9 | 2.61097 | 2.63629 |  | 2.63630 | -0.00001 | 3 |
| 10 | 2.72698 | 3.04687 |  | 3.04687 | +0.00000 | 4 |
| 11 | 3.46288 | 3.53722 |  | 3.53721 | +0.00001 | 4 |
| 13 | 4.10184 | 4.16199 |  | 4.16199 | +0.00000 | 5 |
| 14 | 4.73264 | 4.74494 |  | 4.74494 | +0.00000 | 5 |
| 15 | 5.00680 | 5.06175 |  | 5.06345 | -0.00170 | 6 |
| 17 | 6.00000 | 6.01087 |  | 6.01086 | +0.00001 | 7 |
| 18 | 6.21109 | 6.24792 |  | 6.24791 | +0.00001 | 7 |
| 19 | 6.87088 | 6.86085 |  | 6.87498 | -0.01413 | 7 |
| 20 | 7.03935 | 7.12479 | 7.12479 | 7.14732 | -0.02253 | 8 |
| 21 | 7.72265 | 7.72127 |  | 7.72489 | -0.00362 | 8 |
| 22 | 8.00000 | 8.07101 |  | 8.07384 | -0.00283 | 9 |
| 23 | 8.49774 | 8.52784 | 8.52784 | 8.53463 | -0.00679 | 9 |
| 24 | 9.00000 | 9.02159 |  | 9.02394 | -0.00235 | 10 |
| 25 | 9.18668 | 9.29876 |  | 9.30077 | -0.00201 | 10 |
| 26 | 9.88968 | 9.82517 |  | 9.89324 | -0.06807 | 10 |
| 27 | 10.00000 | 10.14176 | 10.14476 | 10.16136 | -0.01660 | 11 |
| 28 | 10.69154 | 10.72129 | 10.72129 | 10.73270 | -0.01141 | 11 |
| 29 | 11.00000 | 11.07503 | 11.07630 | 11.08218 | -0.00588 | 12 |
| 30 | 11.47735 | 11.49534 | 11.49534 | 11.53492 | -0.03958 | 12 |
| 31 | 12.00000 | 12.03423 |  | 12.03575 | -0.00152 | 13 |
| 32 | 12.03916 | 12.42961 |  | 12.43064 | -0.00103 | 13 |
| 33 | 12.79735 | 13.01867 |  | 13.01887 | -0.00020 | 14 |
| 34 | 13.17973 | 13.28159 | 13.28159 | 13.28913 | -0.00754 | 14 |
| 35 | 13.80858 | 13.83892 | 13.83892 | 13.90296 | -0.06404 | 14 |
| 36 | 14.00000 | 14.18128 |  | 14.18217 | -0.00089 | 15 |
| 37 | 14.68952 | 14.74906 | 14.76530 | 14.76529 | +0.00001 | 15 |
| 38 | 15.00000 | 15.07965 |  | 15.08152 | -0.00187 | 16 |
| 39 | 15.32119 | 15.47781 | 15.47781 | 15.53725 | -0.05944 | 16 |
| 40 | 16.00000 | 16.03664 | 16.03664 | 16.04172 | -0.00508 | 17 |
| 41 (hand-patched) | 16.00000 | 16.43200 | 16.43376 | 16.44841 | -0.01465 | 17 |
| 42 | 16.86606 | 17.01970 |  | 17.02275 | -0.00305 | 18 |
| 43 | 17.17038 | 17.32798 | 17.32798 | 17.34095 | -0.01297 | 18 |
| 44 | 17.67467 | no legal read; 17.82217 with the fence price (λ) | 16.99950 | 17.91952 | -0.09735 | 18 |
| 45 | 18.00000 | 18.17878 |  | 18.17998 | -0.00120 | 19 |
| 46 | 18.49536 | no legal read | 18.80409 | 18.82057 | -0.01648 | 19 |
| 47 | 19.00000 | 19.14890 |  | 19.15056 | -0.00166 | 20 |
| 48 | 19.45591 | 19.73724 | 19.00000 | 19.75072 | -0.01348 | 20 |
| 49 | 20.00000 | no legal read | 20.06049 | 20.07383 | -0.01334 | 21 |
| 50 | 20.00000 | 20.55748 |  | 20.55829 | -0.00081 | 21 |

Every row's best is referee-valid at 1e-9 and 1e-12 and went into the library via
`update_record` when larger (meta `via: digitized` / `digitized-v2`). Where the digitized layout
is below the library (19, 21, 26) the library kept its own; those reads are wrong somewhere, not the
record's topology (26 is 0.068 short). Two seeds on one read agreed to 1e-9 wherever both ran, so
the optimizer is repeatable on a read: a shortfall points at the read (or at a basin both seeds
share), not at seed luck.

### 14. Optimizer effort on near-record topologies -- pre-registered

Hypothesis: where the library now sits < 0.002 below the page on a digitized (or neighbourhood)
topology, the rest is optimizer shortfall on the right topology, which more restarts and steps close.
Arms: the library record's own topology and geometry as the sketch, `optimize_topology` with batch
8192 (4x), 3000 Adam steps (3x), topk 16, 3 seeds, vs the library value it started from. n = 8, 9,
22, 24, 25, 31, 32, 33, 38, 45, 47, 50 (every n with 0 < gap < 0.0025 at launch, 10:08 PDT).
PASS: at least 3 of the 12 reach a tie with the page at its 5 printed decimals. Every result is
referee-checked at 1e-9 and 1e-12; a larger one goes in with `update_record` (`via: heavy-opt`).

**Result of 14: FAIL, 0 of 12** (10:08-10:22 PDT, serrano GPUs 3-4, `runs/4h-20260924/heavy-*`). Every
arm returned its starting area to within 1e-14 -- n = 8 2.1030583, 9 2.6362898, 22 8.0734532, 24
9.0228529, 25 9.2987613, 31 12.0342277, 32 12.4296086, 33 13.0187327, 38 15.0796486, 45
18.1787789, 47 19.1488965, 50 20.5577469 -- all referee-valid at 1e-9 and 1e-12, none changed the
library. Four times the restarts and three times the steps land in the same optimum from three seeds:
the gaps of 1e-6 to 2e-3 that remain are topological (a joint that should be a T, a fence end on the
wrong edge), not optimizer shortfall. Useful for what it rules out: more optimizer effort on these n
is wasted; neighbourhood moves (change 12) are what moved them.

**Result of 12** (09:21-11:50 PDT, serrano; `runs/4h-20260924/nb-n*.jsonl`, `results/nb-*`). The
pre-registered first round (seeds: the digitized 41 and 50; the old library records of 44 and 46,
where digitization had failed at launch): 41 +0.00182 (16.43382), 50 +0.00027 (20.55775); 44 and 46
gained +0.00293 / +0.00338 over their old seeds, but digitization v2 had meanwhile put both
libraries far above those seeds, so neither wrote anything. **No arm beat the page anywhere.**

Added after the first round, as the GPUs freed (not part of the pre-registration, same script and
settings, 15 min per n, each seeded from the library as it stood when the arm started, i.e. from
the previous round's best -- an iterated local search): rounds r2-r5 over every n still below the
page. Every row below is referee-valid at 1e-9 and 1e-12; "written" means `update_record` took it.

| n | round | seed (library) | level-1 neighbours | evaluated | best verified | vs seed | vs page | move | written |
|---|---|---|---|---|---|---|---|---|---|
| 15 | r2 | 5.06175 | 27 | 88 | 5.06175 | +0.00000 | -0.00170 | remove_face | yes |
| 19 | r3 | 6.87137 | 36 | 155 | 6.87154 | +0.00018 | -0.00344 | join_to_t | yes |
| 20 | r2 | 7.12479 | 39 | 156 | 7.14422 | +0.01943 | -0.00310 | join_to_t+remove_face | yes |
| 20 | r5 | 7.14422 | 45 | 146 | 7.14422 | -0.00000 | -0.00310 | remove_face | no |
| 21 | r3 | 7.72265 | 45 | 175 | 7.72264 | -0.00002 | -0.00225 | join_to_t | no |
| 23 | r3 | 8.52784 | 54 | 192 | 8.52987 | +0.00204 | -0.00476 | join_to_t+join_to_t | yes |
| 23 | r5 | 8.52987 | 49 | 192 | 8.53437 | +0.00450 | -0.00026 | remove_face+remove_face | yes |
| 24 | r2 | 9.02285 | 39 | 149 | 9.02388 | +0.00103 | -0.00006 | remove_face+remove_face | yes |
| 24 | r5 | 9.02388 | 38 | 151 | 9.02388 | -0.00000 | -0.00006 | join_to_t | no |
| 25 | r2 | 9.29876 | 51 | 216 | 9.30022 | +0.00146 | -0.00055 | remove_face+remove_face | yes |
| 26 | r3 | 9.88968 | 56 | 218 | 9.88967 | -0.00001 | -0.00357 | join_to_t | no |
| 27 | r2 | 10.14476 | 53 | 186 | 10.15292 | +0.00816 | -0.00844 | remove_face | yes |
| 27 | r4 | 10.15292 | 52 | 197 | 10.15654 | +0.00361 | -0.00482 | remove_face+remove_face | yes |
| 27 | r6 | 10.15654 | 39 | 171 | 10.15640 | -0.00014 | -0.00496 | join_to_t | no |
| 28 | r2 | 10.72129 | 52 | 226 | 10.72363 | +0.00234 | -0.00907 | remove_face | yes |
| 28 | r5 | 10.72363 | 59 | 223 | 10.72361 | -0.00002 | -0.00909 | join_to_t | no |
| 29 | r3 | 11.07630 | 54 | 210 | 11.08040 | +0.00410 | -0.00178 | remove_face+join_to_t | yes |
| 29 | r5 | 11.08040 | 49 | 204 | 11.08114 | +0.00074 | -0.00104 | remove_face | yes |
| 30 | r2 | 11.49534 | 64 | 229 | 11.53279 | +0.03745 | -0.00213 | remove_face+remove_face | yes |
| 30 | r5 | 11.53279 | 61 | 238 | 11.53425 | +0.00146 | -0.00067 | remove_face+remove_face | yes |
| 32 | r2 | 12.42961 | 62 | 225 | 12.42963 | +0.00002 | -0.00101 | join_to_t+remove_face | yes |
| 33 | r2 | 13.01867 | 56 | 214 | 13.01873 | +0.00007 | -0.00014 | remove_face+join_to_t | yes |
| 34 | r3 | 13.28159 | 71 | 183 | 13.28911 | +0.00752 | -0.00002 | remove_face+remove_face | yes |
| 35 | r2 | 13.83892 | 67 | 256 | 13.85895 | +0.02002 | -0.04401 | join_to_t+remove_face | yes |
| 35 | r3 | 13.85895 | 76 | 229 | 13.85918 | +0.00023 | -0.04378 | remove_face+join_to_t | yes |
| 35 | r5 | 13.85918 | 64 | 223 | 13.86803 | +0.00886 | -0.03493 | join_to_t+remove_face | yes |
| 36 | r2 | 14.18128 | 67 | 261 | 14.18218 | +0.00090 | +0.00001 | join_to_t+remove_face | yes |
| 38 | r2 | 15.07965 | 51 | 185 | 15.07962 | -0.00002 | -0.00190 | remove_face | no |
| 39 | r2 | 15.47781 | 79 | 193 | 15.51077 | +0.03296 | -0.02648 | remove_face+remove_face | yes |
| 39 | r4 | 15.51077 | 77 | 222 | 15.52419 | +0.01342 | -0.01306 | join_to_t+join_to_t | yes |
| 39 | r5 | 15.52419 | 66 | 212 | 15.53201 | +0.00782 | -0.00524 | join_to_t+remove_face | yes |
| 40 | r3 | 16.03664 | 74 | 160 | 16.03899 | +0.00235 | -0.00273 | remove_face+remove_face | yes |
| 40 | r5 | 16.03899 | 71 | 182 | 16.03931 | +0.00032 | -0.00241 | remove_face+remove_face | yes |
| 41 | lib | 16.43200 | 72 | 309 | 16.43382 | +0.00182 | -0.01459 | remove_face+remove_face | yes |
| 41 | r3 | 16.43382 | 72 | 149 | 16.43484 | +0.00103 | -0.01357 | remove_face+slide_t_past | yes |
| 41 | r4 | 16.43484 | 73 | 129 | 16.43468 | -0.00016 | -0.01373 | remove_face+t_to_join | no |
| 41 | r5 | 16.43484 | 81 | 145 | 16.43488 | +0.00004 | -0.01353 | join_to_t | yes |
| 42 | r3 | 17.01970 | 69 | 160 | 17.02099 | +0.00130 | -0.00176 | join_to_t+join_to_t | yes |
| 43 | r2 | 17.32798 | 69 | 111 | 17.33372 | +0.00574 | -0.00723 | remove_face+remove_face | yes |
| 43 | r5 | 17.33372 | 61 | 108 | 17.33370 | -0.00001 | -0.00725 | remove_face | no |
| 44 | lib | 17.67467 | 89 | 229 | 17.67760 | +0.00293 | -0.24192 | remove_face+t_to_join | no |
| 44 | r2 | 17.82217 | 93 | 223 | 17.85675 | +0.03458 | -0.06277 | face_fill+join_to_t | yes |
| 44 | r3 | 17.85675 | 77 | 167 | 17.86767 | +0.01092 | -0.05185 | remove_face+join_to_t | yes |
| 44 | r4 | 17.86767 | 82 | 147 | 17.87313 | +0.00547 | -0.04639 | remove_face+repin | yes |
| 44 | r5 | 17.87313 | 77 | 125 | 17.88397 | +0.01084 | -0.03555 | remove_face+join_to_t | yes |
| 44 | r6 | 17.88397 | 74 | 144 | 17.88471 | +0.00074 | -0.03481 | remove_face+remove_face | yes |
| 45 | r2 | 18.17878 | 65 | 131 | 18.17878 | -0.00000 | -0.00120 | join_to_t | no |
| 46 | lib | 18.49536 | 80 | 227 | 18.49873 | +0.00338 | -0.32184 | remove_face | no |
| 46 | r2 | 18.80409 | 72 | 155 | 18.81581 | +0.01172 | -0.00476 | join_to_t+join_to_t | yes |
| 46 | r3 | 18.81581 | 65 | 125 | 18.81902 | +0.00321 | -0.00155 | remove_face+join_to_t | yes |
| 46 | r4 | 18.81902 | 72 | 115 | 18.82035 | +0.00133 | -0.00022 | remove_face | yes |
| 47 | r2 | 19.14890 | 72 | 154 | 19.15037 | +0.00147 | -0.00019 | join_to_t | yes |
| 47 | r5 | 19.15037 | 74 | 158 | 19.15042 | +0.00005 | -0.00014 | remove_face | yes |
| 48 | r2 | 19.73724 | 64 | 114 | 19.74120 | +0.00396 | -0.00952 | remove_face+remove_face | yes |
| 48 | r4 | 19.74120 | 73 | 99 | 19.74692 | +0.00571 | -0.00380 | remove_face | yes |
| 49 | r2 | 20.06049 | 69 | 135 | 20.06157 | +0.00108 | -0.01226 | remove_face+remove_face | yes |
| 49 | r4 | 20.06157 | 81 | 116 | 20.06196 | +0.00039 | -0.01187 | remove_face | yes |
| 49 | r5 | 20.06196 | 81 | 121 | 20.06322 | +0.00127 | -0.01061 | remove_face | yes |
| 50 | lib | 20.55748 | 82 | 168 | 20.55775 | +0.00027 | -0.00054 | remove_face | yes |
| 50 | r4 | 20.55775 | 85 | 87 | 20.55828 | +0.00053 | -0.00001 | remove_face | yes |

Across the 48 written wins, `remove_face` (drop a fence and merge its two faces; `fit_count` then
puts a fence back elsewhere -- in effect relocating one fence) appears 55 times, often as both moves
of a pair, and `join_to_t` 24 times; `face_fill`, `repin` and `slide_t_past` once each,
`t_to_join` never; `add_face` produced no distinct legal neighbour at these n. The moves that pay
are small and local -- consistent with repairing local misreads of the pictures (a joint that should
be a T, a fence on the wrong edge), though which win repaired a misread and which found a different
topology is not separated. Iterating keeps paying where the read was far off (44: +0.035, +0.011,
+0.005, +0.011, +0.001 over five rounds) and stalls within ~1e-4 of the page (20, 24, 43, 45 gained
nothing in their last round).

## Day 6 result (4 h run of 2026-09-24, 08:59-11:50 PDT of GPU work; `runs/4h-20260924/final-comparison.txt`)

All GPU work stopped at 11:50, 2 h 51 min into the box. Every record the run wrote (39 n) was
re-verified by the referee at 1e-9 **and** at 1e-12 after the last writer exited, against the page as
scraped once at 08:59 (unchanged since 2026-09-21). Every one is indexed as a minimum of the new
`digitized` lineage (`MinimaLibrary.best()` leaves a record with no lineage out, so without this no
later run would have seeded from any of them).

| | n = 12..50 | n = 3..11 |
|---|---|---|
| shortfall at the start of the run | 4.93865 | 0.48865 |
| **now** | **0.15130** | **0.00000** |

- **39 n improved**; every n in 8..50 except 12 and 16 (already ties), 21 and 26 (their digitized
  reads came in 0.0014 and 0.0056 below the library, and one neighbourhood round from the library
  record came within 2e-5 of it, not above). Largest gains:
  50 +0.55828, 41 +0.43488, 32 +0.39047, 46 +0.32499, 10 +0.31989, 48 +0.29101, 33 +0.22138,
  39 +0.21082, 44 +0.21004.
- **Ties with the page at its printed precision: n = 3-7, 10-14, 16-18, 36, 37** (new today: 10, 11,
  13, 14, 17, 18, 36, 37). **No n beats its record.** Closest below: 8 (1.7e-6), 9 (1.0e-5), 50
  (1.1e-5), 34 (2e-5), 24 (6e-5), 33 and 47 (1.4e-4), 46 (2.2e-4), 23 (2.6e-4).
- Largest remaining gaps: 35 (0.0349), 44 (0.0348), 41 (0.0135), 49 (0.0106), 28 (0.0091), 43 (0.0072).

What the day says:
- The records were the missing ingredient, not the optimizer. Given the record's own topology,
  `optimize_topology` fills it to the page's figure or within ~1e-3 at most n in one call; on the
  topology it already has, 4x restarts and 3x steps change nothing (change 14). The 09-21 diagnosis --
  the stuck n need a different topology with more faces -- was right; the topologies were on the
  page all along, as pictures.
- Reading a picture is where the errors are: every remaining gap sits on a digitized topology, and
  small local moves (change 12) kept repairing them round after round.
- A regular ring/onion construction with the right counts is not the right net (change 13, failed).
- Operational: spawned polish workers inherit thread settings -- five processes at 4 BLAS threads
  each drove serrano to load 146 with every GPU idle until relaunched with `OMP_NUM_THREADS=1`.

## Day 7 (2026-09-25) -- 6 h run: close the last gaps, try to beat a record (`PROMPT-6h-close-and-beat.md`)

Clock: started 03:09 PDT (background watchdog `runs/6h-20260925/watchdog.sh`: WARN 08:24, STOP 08:49,
6 h at 09:09). Tree clean except the untracked brief, so no question asked; snapshot ref
`refs/backup/pre-6h-run-20260925` = HEAD c628c67. `pytest` green (295). Baselines re-scraped once at
03:09 PDT: page sha256 unchanged since 2026-09-21, **no record moved**. GPUs: serrano 0-4 (CUDA matmul
OK on each); temescal's GPU 1 is off the bus again (cuInit fails on every device); ynez is another
user's (~9 GB + ~6 GB per GPU, load 67). Start table (`runs/6h-20260925/gap-start.txt`): shortfall
0.15130 over n = 12..50, ties as listed in the brief, largest gaps 35 0.0349, 44 0.0348, 41 0.0135,
49 0.0106, 28 0.0091, 43 0.0072.

Tooling: the 09-24 neighbourhood driver is now `neighbourhood.py` (`--n`/`--queue`, `--seed`,
`--tag`, `--budget-min`, `--out`, `--focus`, `--extra-moves`; it iterates sweeps -- recentring on the
best screened layout -- until the per-n budget runs out, keeps the best screened result when
stopped, and indexes every record it writes under `--lineage`, default `digitized`). Tests:
`tests/test_neighbourhood.py`. Background queue (`runs/6h-20260925/bgqueue.py`): the five serrano
GPUs run change 12's iterated rounds r7, r8, ... (15 min per n, largest gap first, seeded from the
library) whenever a phase does not need them; Phase A's six n are held out of the queue while
Phase A runs so its result is its own.

### 15. Overlay the optimized layout on the record picture; fix the misreads there -- pre-registered

Hypothesis: the gaps at 35, 44, 41, 49, 28, 43 are local misreads of the digitized topology;
overlaying the optimized layout on the picture locates them, and moves applied at that spot close
most of each gap.

Method: `fence/overlay.py` fits a similarity (reflection allowed) of the layout onto the picture's
skeleton by two-way ICP, then scores each fence by the mean distance of its samples to the nearest
skeleton pixel, each skeleton branch by its distance to the nearest fence, and each face by IoU with
the picture face it overlaps most. Flagged places become `--focus` points (radius 0.5) for
`neighbourhood.py`, which screens the neighbours whose change touches them first. Also: re-read the
picture at a finer scale / lower `min_edge` and optimize the re-reads.

Pre-registered PASS: at least 3 of {35, 44, 41, 49, 28, 43} end Phase A with gap <= half their
Phase-0 gap (35 <= 0.01746, 44 <= 0.01740, 41 <= 0.00677, 49 <= 0.00530, 28 <= 0.00454,
43 <= 0.00362), in the 90-min box.

Sanity check before any search (03:30 PDT): the fit is tight on ties -- fit cost 0.0046 (n=17),
0.0057 (37), 0.0056 (36), 0.0060 (24) units, per-fence median 0.002-0.003 -- and loose on exactly the
gap n: 35 0.041, 44 0.034, 49 0.020, 41 0.018, 28 0.018, 43 0.015. The face counts of layout and
picture agree at all six. At n=41 the overlay does **not** single out the 0.72 / 0.76 faces the brief
expected: the picture has faces of the same shape there (IoU 0.915 and similar picture-face area),
and the misfit is spread thinly (worst fence 0.039). At 35 and 44 it is local and visible: whole cells
in the lower corners of 44 and the left column of 35 are attached differently from the picture.

### 16. Pinwheel insertion move (`pinwheel`, netevolve, weight 0 = off) -- pre-registered

Hypothesis: a pinwheel -- three (or more) fences each stopping just past the next around a tiny face
-- turns a joint into T-junctions at no fence cost and adds a face (F = n + 1 - J); n=17's record beats
6.0 by 0.01086 exactly this way. No existing move makes it, so it may add up to ~0.01 at any joint
where the geometry leaves room, including at n that already tie -- the most plausible way to *beat* a
record.

Move (`fence.builder.pinwheel_at` / `pinwheel`; `netevolve` move `pinwheel`, weight 0): at a joint v
with k >= 3 fence ends sorted by angle, fence i keeps its far end and ends where its line (shifted
sideways by d = 0.05) meets fence i+1's, pinned on fence i+1; alternating projections settle every new
pin exactly onto its host; one sign of d gives a legal layout when every angular gap is below 180
degrees (a joint on a straight boundary admits none). A T-junction cannot host one (its host does not
end there), so only joints are candidates -- a deviation from the brief's "or a T".
Check (`tests/test_pinwheel.py`): the unit-square 2x3 grid is *not* one pinwheel from the record --
its boundary joints admit none and its interior 4-way joints give a square pinwheel that optimizes to
5.97998 (serrano, 2 seeds x 2 sizes); but collapsing the record's own triangle to a joint gives a
6-face layout that optimizes to 6.0 (grid value, T = 9), and `pinwheel` at its one eligible joint
rebuilds the record topology, which optimizes to 6.0108686594 (the record to 10 digits; also >= 6.0108
on the CPU in the slow test).

Runs: `neighbourhood.py --moves pinwheel --moves2 <the 7 usual moves>,pinwheel` (every eligible joint
singly at level 1, then paired with the usual moves at level 2) from the library records at every tie
(10, 11, 13, 14, 17, 18, 36, 37) and every n within 3e-4 of the page (8, 9, 23, 24, 33, 34, 46, 47, 50),
12 min per n.

Pre-registered PASS: a referee-verified area at or above the next 5th-decimal step of the page at >= 1 n
(a genuine beat); failing that, a gain > 1e-4 over the library at >= 3 n. Reported either way: how
many joints admitted a pinwheel, and the mean gain (pinwheel result minus library, per n).

### 17. Transplant records across n (`transplant.py`) -- pre-registered

Hypothesis: the record layouts at n and n +- 1 are close relatives (one ring cell more or less), so
growing the n-1 record by one fence or pruning the n+1 record by one seeds a better start than n's own
(possibly misread) layout.

Method (`transplant.py`, new lineage `transplant`): for every n in 9..50, "up" seeds are the library
record at n-1 grown to n by `fit_count` (and `add_face` then pruned back), "down" seeds the record at
n+1 pruned by `fit_count` (and two `remove_face` then regrown), 40 rng seeds per direction,
deduplicated by canonical signature and gated by the hard rules (rule 14 against today's page). Every
distinct seed is screened (batch 1024, 600 steps, referee), best 2 at full settings (batch 2048, 1000
steps); 4.5 min per n, three queues. Order: n below the page by gap, then ties, then Phase A's six last
(so the two phases do not overlap on an n while Phase A runs).

Pre-registered PASS: a transplant beats the library at >= 5 n, or closes >= 0.02 of the summed gap over
the six largest-gap n (35, 44, 41, 49, 28, 43). Reported: direction (up / down) of every win.

### 18. `onion_fitted` generator -- measured, then skipped (not run)

The brief's Phase D (an onion family whose spoke placement is sampled from the digitized reads, bar:
within 5% of the library at 5 sample n) was measured first (`runs/6h-20260925/rings.py`: faces
layered by dual-graph distance from the outer face, radial edges classified by their end kinds) on the
records at 17, 24, 33, 36, 37, 41, 46, 50:

| n | cells per ring (outermost first) | radial walls of ring 1: T-T / T-J / J-T |
|---|---|---|
| 17 | 5, 2 | 5 / 0 / 0 |
| 24 | 8, 2 | 6 / 6 / 1 |
| 33 | 10, 4 | 8 / 2 / 1 |
| 36 | 11, 4 | 6 / 6 / 4 |
| 41 | 12, 5 | 8 / 2 / 2 |
| 46 | 13, 5, 1 | 9 / 2 / 3 |
| 50 | 13, 7, 1 | 10 / 3 / 1 |

Two things change 13's family gets wrong, not one: most radial walls are pinned at *both* ends (T-T,
brick-wall offsets; change 13's spokes start at a vertex or at the middle of an outer fence), and the
inside of the outer ring is a cluster of 2-7 cells that all touch ring 1 -- not a single core polygon
inside further rings. Re-sampling spoke placement alone would keep the second error, and a faithful
construction of the inner cluster did not fit a 45-min box. With serrano's GPUs and CPUs saturated
by Phases A-C and the background rounds (which were paying), Phase D was **skipped** at 04:06 PDT
without a run; nothing was added to `fence/nets.py`.

**Result of 17: FAIL** (03:51-04:30 PDT, three queues on serrano GPUs 1 and 3, `runs/6h-20260925/phaseC/`).
All 42 n in 9..50 ran; **no transplant beat the library at any n** (0 of 42; bar: 5), so nothing
closed any of the six largest gaps (bar: 0.02). The best transplant per n sat 0.06-0.51 below the
library record (median about -0.3; closest 23 -0.057, 16 -0.074, 11 -0.126). Direction of the best
per n: "up" at 35 n, "down" at 7. The "down" side was mostly empty: at 25 of 42 n rule 14 rejected every
pruned layout -- removing a fence merges two faces, and the n+1 record minus a face is below the page's
face floor for n. "Up" seeds (8-15 distinct per n) add a cell somewhere the n-1 layout has room,
which is never where n's record has its extra cell: the records at n and n +- 1 are not one cell
apart in any way `fit_count` / `remove_face` can find. `transplant.py` stays in the tree (no default
changed); lineage `transplant` exists but has no minima.

**Result of 16: FAIL** (03:48-04:44 PDT, `runs/6h-20260925/phaseB/`, tag B2; a first launch, B1, was a
no-op: `pinwheel_at` did not yet re-project the pins other fences already had on the rotated fences, so
no joint of any record admitted a pinwheel -- fixed with `builder.project_pins`, covered by a test).
- Joints that admitted a pinwheel: 14 of the 17 n had at least one (38 distinct pinwheel topologies at
  level 1); 8, 14 and 17 have no joint of three or more fence ends at all (their records use T's) --
  n=17's own pinwheel is already in its record.
- **No gain and no beat at any n.** Best per n, pinwheel alone or paired with a usual move at level 2,
  minus the library: 9 -0.0064, 10 -0.0021, 11 -0.0173, 13 -0.0045, 18 -0.0063, 23 -0.0211,
  24 -0.00043, 33 -0.0015, 34 -0.0054, 36 -0.0040, 37 -0.0130, 46 -0.0211, 47 -0.0046, 50 -0.0159;
  mean -0.0088. A pinwheel buys its extra face with geometry the record already uses: at an optimized
  record's Y-joint the three cells around it are full, and the tiny face costs more than it frees.
  Its one success is the 17 record itself, which is what the move was built from.
- `pinwheel` stays in `netevolve` with weight 0 (off by default).

**Result of 15: FAIL -- 1 of 6** (03:30-04:52 PDT, inside the 90-min box; `runs/6h-20260925/phaseA/`,
`overlay/`, `tsplit/`, `digit-s4*/`). Every gain below is referee-verified at 1e-9 and 1e-12 and indexed
under `digitized`.

| n | gap at start | gap at 04:52 | closed | bar (half) | what moved it (time, gain) |
|---|---|---|---|---|---|
| 41 | 0.013530 | **0.001281** | 90% | PASS | re-read at scale 4, v2 grid (03:40, +0.010917: a new topology, F 17 / T 25 / J 25), then a focused `join_to_t` from the picture-fitted seed (04:25, +0.001333) |
| 43 | 0.007232 | 0.005180 | 28% | no | overlay-focused `remove_face` (04:08, +0.002053) |
| 44 | 0.034808 | 0.032511 | 7% | no | `remove_face+repin` (+0.000606), t-split pair: fence 16 made a T at both ends (+0.000756), t-split of the stem at v11 (+0.000072), face-focused `join_to_t+join_to_t` (+0.000936) |
| 35 | 0.034927 | 0.033341 | 5% | no | `remove_face+slide_t_past` (+0.000975), `repin+repin` (+0.000321), t-split at v9: the left horizontal made a T on the fence below the break (+0.000115), `slide_t_past+remove_face` (+0.000176) |
| 49 | 0.010606 | 0.010606 | 0 | no | nothing: t-split had one candidate joint (-1.8e-5), the re-reads came in 0.0027 below |
| 28 | 0.009071 | 0.009071 | 0 | no | nothing: t-split best -1.7e-5, re-reads -0.001 / -0.002 |

What the overlay found, per step:
1. **Overlay (as pre-registered).** Fit cost is tie-level (0.005) on ties and 0.015-0.041 on the six;
   face counts always agree with the picture. At 35 and 44 whole cells sit visibly differently (44's
   lower corners, 35's left column); at 41, 49, 28, 43 the misfit is thin and spread. It did not single
   out 41's 0.72 / 0.76 faces: the picture has faces of that shape there too.
2. **Picture-fit start.** Bending the library topology onto the picture (`overlay.picture_fit`: Adam on
   the vertices against the skeleton's distance transform, weak unit-length pull) brings the fit to
   0.0067-0.0088 at all six -- *the drawn graph is right* -- and optimizing from that geometry (2 noise
   levels x 2 seeds) returns exactly the library value at all six (`phaseA/picfit-*.jsonl`). The gap is
   in the topology, not the basin.
3. **Fitted lengths locate it.** The picture is drawn to scale, so after the fit the page's fences are
   unit length. On ties every fitted fence is within 0.0045 of 1; on the gap n some are 2-7% off, in
   long/short pairs along a line (44: right boundary 1.065 then 0.982, 0.983, 0.984; 35: left column
   1.060, 0.973, 0.969). Zoomed overlays show why: where the page has a T-junction 0.03-0.07 from a
   fence break, the digitizer merged the two into one three-way joint (the stub-contraction pitfall of
   09-24).
4. **Repairing it gave little.** Exhaustive t-splits at the flagged joints (every stem/host choice,
   singly and in pairs, `tsplit.py`), overlay- / length- / face-focused neighbourhood rounds
   (`--focus`, tags A1-A5), a re-decomposition of the fitted graph with the digitizer's unit-length DP
   (`regraph.py`: 0.0016-0.0043 below the library), and a CPU beam search over incidence moves scored by
   fitted-length consistency (`picsearch.py`: its most consistent candidates optimized 1.4-2.2 *below* --
   consistency under a soft length pull is not a proxy for area) -- together +0.0016 at 35 and +0.0023
   at 44. Making the fitted lengths right locally does not recover the page's value there: the page
   fills a face that is 0.936 (35) / 0.977 (44) in ours and leaves its single short face elsewhere, a
   global difference the local repairs have not found.
5. **Re-reading the picture** at scale 4 with the v2 grid (`digitize_records.py --v2 --scale 4`, new
   `--scale` flag, default 3) gave 41's jump; scale 5 (v2) and scale 4 (v1 grid) re-reads were lower at
   the other five. The same scale-4 v2 re-read over the other n below the page (not part of this bar) paid
   at 15, 22, 26, 38, 42 (see the day's table).

### 19. Hill-climb from other reads of the picture (`neighbourhood.py --climb`) -- pre-registered late

Hypothesis: where local repairs of the record stall (35, 44 and the other larger gaps), a *different*
read of the picture -- a topology the record's neighbourhood does not contain -- climbed with the same
moves can end above the record, as 41's scale-4 read did today (+0.0109, then +0.0013).

Method: `--climb` makes `neighbourhood.py` recentre each sweep on the best layout screened so far even
when it is below the library (default: only above it). Seeds (`runs/6h-20260925/readseeds.py`): the best
distinct read topologies of each n from every digitize log of 09-24 and today, excluding the record's
own topology. Honest timing: the first five climbs (35: three reads, 44: two; 20 min each) started at
04:55 before this entry; their first sweeps were seen before it was written (35 best read 13.8493 ->
13.8593, 44 17.8497 -> 17.8717, both still below the library). Everything after 05:11 was launched
after it: the two best continued 30 min (tag R0b), and the two best reads of 49, 28, 43, 39, 27, 48,
26, 20 at 15 min per n (tags R0, R1).

Pre-registered PASS: a climb ends above the library record at >= 2 of the climbed n (35, 44, 49, 28,
43, 39, 27, 48, 26, 20). Reported per n: seed read, best climbed value, vs library at launch.

**Result of 19: PASS** (04:55-08:49 PDT, `runs/6h-20260925/climb/`). Of the pre-registered n, climbs from
other reads ended above the library at **28** (read `684378f3834e` of the scale-5 v2 re-read, 10.72267 ->
10.72979, **+0.006162**, gap 0.00907 -> 0.00291) and **26** (+0.000047, now a tie); from the second list,
launched after the entry, **25** (+0.000556, now a tie). Everywhere else the climbs ended at or below the
library: at 19, 20, 21, 22, 26 (R0), 27, 28 (R1), 29, 32, 43 a read climbed *exactly* onto the library value --
the record is the attractor of those reads' neighbourhoods -- and at 35 (best of five climbs 13.86801,
-0.0016 after 1 h 20 min), 44 (17.87896, -0.0081), 39 (-0.027), 48 (-0.0074), 49 (-0.0021), 40 (-0.000013)
they stayed below. Unregistered variant, no bar: escape climbs from the record itself (`--seed library
--climb`: the first sweep may recentre below the record) gave +0.000012 at 39 and nothing at 20, 27, 35, 43,
44, 48, 49.

### 12 (continued). Background neighbourhood rounds r7-r12 (change 12's method, library-seeded)

The five serrano GPUs ran `bgqueue.py` the whole run (03:20-08:49): rounds r7..r12, 15 min per n, largest
gap first, Phase A's six n held out 03:20-04:52. Plus two unregistered side queues with change 12's
method: tie rounds (T1, the 8 ties + 33 with `--extra-moves pinwheel`, 12 min per n: no beat, closest an
alternative 2.3e-6 below at 37) and near-tie rounds (N1, 41 46 23 30 42 45 50 9, 12 min per n: 42 +3.1e-5
to a tie, 41 +1.6e-6).
Result: 27 + 32 + 27 + 25 + 25 + 5 n in rounds r7-r12 (r12 cut by the STOP); library gains at 19 n totalling **+0.01672**
(9, 15, 19, 23, 26, 29, 30, 31, 33, 34, 39, 41, 42, 44, 45, 46, 47, 48, 50), the largest 19 +0.0028,
26 +0.0035 (with the climb), 15 +0.0017 (to a tie), 31 +0.0015 (to a tie), 39 +0.0017, 42 +0.0017, 41
+0.0012 (after Phase A's +0.0122). Moves of the winning neighbours, as on 09-24: `remove_face` and
`join_to_t`, singly or paired. Gains came in the first four hours; from 06:50 to 08:49 only 39 moved.

## Day 7 result (6 h run of 2026-09-25, 03:09-08:49 PDT of GPU work; `runs/combined-20260925.txt`)

STOP at 08:49:26 (5:40 into the box), every GPU process gone at 08:49:57. All 25 records the run wrote
were re-verified by the referee at 1e-9 **and** 1e-12 after the last writer exited (all valid), against
the page as scraped once at 03:09 (unchanged since 2026-09-21), and each is `MinimaLibrary.best()`'s
first entry under the `digitized` lineage.

| | n = 12..50 | n = 3..11 |
|---|---|---|
| shortfall at the start of the run | 0.15130 | 0.00000 |
| **now** | **0.10725** | **0.00000** |

- **Ties with the page at its printed precision: n = 3-7, 10-18, 25, 26, 31, 33, 34, 36, 37, 42, 47**
  (new today: 15, 25, 26, 31, 33, 34, 42, 47). **No n beats its record**; the nearest ties are 1.1e-6 (18)
  and 1.3e-6 (17) below the page's next printed step, at the optimum of their topologies, and the tie
  rounds with the pinwheel move found nothing above them. Nearest below: 8 (1.7e-6), 9 (2e-6), 50 (5e-6),
  45 (2.9e-5), 46 (5.6e-5), 41 (7.1e-5).
- Largest remaining gaps: 35 0.0333, 44 0.0325, 49 0.0106, 43 0.0052, 27 0.0048, 39 0.0032, 20 0.0031.
- Where the 0.04405 came from (`runs/6h-20260925/attribution.py`): background rounds (12) +0.01672 over
  19 n; Phase A (15) +0.01828 over 35, 41, 43, 44 -- 41's scale-4 re-read alone +0.0109; scale-4 re-reads
  of other n +0.00233 over 15, 22, 26, 38, 42; climbs from other reads (19) +0.00676 over 25, 26, 28;
  near-tie and escape-climb rounds +0.00004.
- Verdicts: 15 overlay repair FAIL (1 of 6), 16 pinwheel FAIL (0 gains, mean -0.0088), 17 transplant FAIL
  (0 of 42), 18 onion_fitted not run (measured, skipped), 19 climb from other reads PASS (28, 26; and 25).

What the day says:
- The pictures still hold the answers, but a single read is not the answer: another read of the same
  picture (scale 4 or 5) plus a climb found 41's and 28's topologies, which no local repair of the
  record reached. The drawn graph was right at every gap n; the errors are incidences -- a T-junction
  0.03-0.07 from a fence break read as one joint -- that a to-scale fit of the picture locates.
- Locating them was not enough at 35 and 44: fixing the flagged spots one or two at a time never paid
  more than ~0.002. There the page fills a face ours leaves at 0.93-0.96 and keeps one short face
  elsewhere -- a different arrangement, about 60% of the remaining shortfall.
- Operational: at ~15 GPU processes serrano's CPU (eight polish workers each) was the bottleneck, load
  40-44 of 48 with GPU SM use 0-47%.
