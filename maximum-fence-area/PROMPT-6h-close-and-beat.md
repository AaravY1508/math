# 6-hour task: close the last gaps and try to beat a record

Paste this whole file as the prompt to a fresh Claude Code session in this repo
(`maximum-fence-area/`, branch `search-strategy`). It follows the 2026-09-24 run
(EXPERIMENTS.md day 6, commit `c628c67`), which read the page's record pictures
(`fence/digitize.py`) and took the n = 12..50 shortfall from 4.939 to 0.151, with no record
beaten. Hard budget: **6 hours wall clock from the start of Phase 0**, tracked by a background
watchdog (write a STOP file at 5:40, warn at 5:15), not by estimate. Check `date` whenever you
need the time. The last run over-estimated elapsed time badly.

## Where things stand (verify, don't trust: recompute in Phase 0)

- Ties with the page at its printed precision: n = 3-7, 10-14, 16-18, 36, 37. No beats.
- Within 3e-4 of the page: 8, 9, 23, 24, 33, 34, 46, 47, 50.
- Largest gaps: 35 (0.035), 44 (0.035), 41 (0.014), 49 (0.011), 28 (0.009), 43 (0.007),
  39 (0.005), 27 (0.005), 48 (0.004), 26 (0.004), 19 (0.003), 20 (0.003).
- Established on 09-24, don't re-test: more optimizer effort on a fixed topology changes nothing
  (change 14, 0 of 12). The regular `onion` generator fills 5-15% low (change 13). Iterated
  move-neighbourhood search (change 12) kept paying round after round, mostly through
  `remove_face` and `join_to_t`.
- The page truncates: "14.18217+" means [14.18217, 14.18218). **A beat means our area is at or
  above the next 5th-decimal step**, re-verified by the referee at 1e-9 and 1e-12. Anything less
  is a tie. Never call a tie a beat.

## Ground rules (keep them)

- Every change is flag-gated; the default reproduces current behaviour. A failed idea stays in
  the tree, off, with its numbers in EXPERIMENTS.md. Don't delete what this task didn't add.
- Start with `git status`. The tree should be clean. If it isn't, ask the user **once** whether to
  keep or stash; if there's no answer within 10 minutes, keep the changes in place (the
  non-destructive default) and go on. Last time a blocking question cost 4 hours. Take a snapshot
  ref regardless: `git stash create` -> `git update-ref refs/backup/pre-6h-run-<date> <sha>`
  (or the HEAD sha if the tree is clean).
- Re-scrape `baselines.json` once at the start (`./venv/bin/python scrape_baselines.py`), never
  mid-run. If any record moved, say so and use the fresh numbers.
- GPUs: probe every host (temescal, serrano, ynez) with `nvidia-smi` and a real CUDA matmul per
  GPU. Skip any GPU that errors or already holds > 1 GB of someone else's memory. On 09-24 only
  serrano's 5 were usable: temescal's GPU 1 had fallen off the bus, which kills cuInit on the
  whole host, and ynez was another user's. serrano shares this repo over NFS.
- Launch every GPU job with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
  Spawned polish workers inherit threads, and without this serrano hit load 146 with its GPUs
  idle. Launch detached: `ssh -f serrano "cd ... && setsid nohup ... > log 2>&1 < /dev/null &"`.
  Never `pkill -f <pattern>` over ssh where the pattern also matches the ssh command line.
- Library: write only through `MinimaLibrary.update_record` / `add_minimum` (per-n lock).
  **Index every new record as a minimum of a lineage** (`Mn.DIGITIZED_PROPOSER`, or a new
  constant for a new source). `best()` silently skips lineage-less records, and later runs then
  never seed from them.
- Only referee-verified areas count; re-verify every changed record at 1e-9 and 1e-12 at the end.
- EXPERIMENTS.md: a new numbered change per idea, with the hypothesis and pass/fail written
  **before** running and the honest result after. Log tags must be unique per (n, round). Last
  time a duplicate tag interleaved two runs in one log.
- Commits: commit on `search-strategy` only if the user authorizes it in this session; otherwise
  leave everything staged and described. `pytest -q` must be green at the end (note: `pytest.ini`
  adds `-q`, so `-q -q` hides the summary; check the exit code).

## Phase 0 -- setup (20 min)

1. Snapshot, re-scrape, GPU probe and watchdog as above. Run `pytest` and make sure it's green.
2. Promote the 09-24 run's neighbourhood driver into the repo: `runs/4h-20260924/neighbours.py`
   and `nb_queue.sh` are gitignored. Make them a top-level `neighbourhood.py` with `--n`, `--seed`
   (library | file), `--tag`, `--budget-min`, `--out`. It must index its best under a lineage,
   keep its best screened result when stopped, and take a `--queue n1,n2,...` so no shell loop is
   needed. Add a small test. This is the workhorse for everything below.
3. Recompute the per-n gap table from `minima/records` and the fresh baselines
   (`runs/4h-20260924/final_table.py` shows how). Use the fresh numbers from here on.
4. **Background GPU queue, all run long:** keep every GPU that no phase below is using busy with
   iterated neighbourhood rounds (15 min per n, seeded from the current library), largest gap
   first. It needs no pre-registration beyond change 12's; label the rounds r7, r8, ... and report
   them in change 12's style.

## Phase A -- find the misreads by comparing against the picture (90 min, highest priority)

**Hypothesis:** the remaining gaps at 35, 44, 41, 49, 28 and 43 come from local misreads in the
digitized topology. Overlaying the optimized layout on the record's picture locates them, and a
move applied at that spot closes most of the gap.

1. Write `fence/overlay.py`. Fit a similarity transform (scale, rotation, translation, reflection
   allowed) that maps the library layout onto the picture's skeleton, by ICP on the skeleton pixels
   against points sampled along the fences. Then score each fence by the mean distance from its
   samples to the nearest skeleton pixel (a distance transform via `scipy.ndimage`), and each
   skeleton branch by its distance to the nearest fence. High scores mark where the two disagree:
   a missing fence, an extra fence, a T on the wrong side.
2. Sanity-check it on a tie (n=17 or 37): the whole layout should score low. On n=41, it should
   flag the two faces at area ~0.72 and 0.76.
3. For each flagged region, restrict the neighbourhood moves to fences and vertices within ~0.5
   of it (add a `focus` option to the move sampler: prefer moves touching flagged elements),
   and run focused rounds. Also try re-digitizing that sub-window of the picture at a finer
   scale (`-scale 500%`) with a lower `min_edge`, and splice the re-read into the topology.
4. **Pass (pre-register):** at least 3 of {35, 44, 41, 49, 28, 43} get at least 50% closer to the
   page (gap now <= half its Phase-0 value) in the time box. Report per n: the flagged regions,
   what the fix was, before and after.

## Phase B -- pinwheel insertion move (60 min)

**Hypothesis:** n=17's record beats the 2x3 grid by 0.01086 with a pinwheel: three fences each
running slightly past the next around a tiny triangle, which costs no extra fence. Turning a
degree-3 joint or T-junction into a pinwheel is a topology move none of `netevolve`'s moves makes.
It's worth up to ~0.01 at each junction where the local geometry leaves room, including at n that
already tie the page, so it is the most plausible way to **beat** a record.

1. Add a topology move `pinwheel` in `fence/netevolve.py` (weight 0, so it's off by default): pick
   a vertex where three fence ends meet (or a T), shorten the incidences into a cyclic chain of
   three pins (fence a's end on b, b's end on c, c's end on a) around a triangle with ~0.05 sides.
   The check is `make_topology` plus the hard rules.
2. Test that it turns the n=17 grid (library minimum `pipeline`/grid at 6.0) into a 7-face
   topology that optimizes to >= 6.0108.
3. Run it as an extra move, both singly and paired with the usual moves, from the library records
   at every tie (10, 11, 13, 14, 17, 18, 36, 37) and every n within 3e-4 of the page.
4. **Pass (pre-register):** a referee-verified area at or above the next 5th-decimal step of the
   page at >= 1 n (a genuine beat), or, failing that, a gain > 1e-4 over the library at >= 3 n.
   Either way, report the count of junctions that admitted a pinwheel and the mean gain.

## Phase C -- transplant records across n (60 min)

**Hypothesis:** the record layouts at n and n±1 are close relatives (one ring cell more or less).
Growing the n-1 record by one fence (`add_face` / `fit_count`) or pruning the n+1 record by one
(`remove_face`) seeds a better start than the n's own misread picture where that misread is bad.

1. For every n in 9..50, build candidate seeds from the library records at n-1 (+1 fence) and n+1
   (-1 fence): every distinct legal result of `fit_count` / `remove_face` from ~40 rng seeds,
   deduplicated by canonical signature. Screen them (batch 1024, 600 steps), then give the best 3
   per n full settings.
2. **Pass (pre-register):** a transplant beats the library at >= 5 n, or closes >= 0.02 of the
   summed gap over the six largest-gap n. Report the transplant direction (up or down) for each win.

## Phase D -- onion family fitted to the reads (45 min, only with runway left)

Change 13 failed because spokes were spread evenly. Measure the digitized records instead. For
each ring, which inner edge each spoke lands on, whether it lands at a vertex or a T, and the
offset between rings. Add an `onion_fitted` generator (off by default) that samples those
statistics. Pre-register the same bar as change 13: within 5% of the library at 5 sample n.
Skip it cleanly if Phase A-C used the time, and say so.

## Phase E -- wrap-up (20 min, non-negotiable)

1. At the 5:40 STOP, let running arms finish their current n; start nothing new. All GPU work
   done by 5:45.
2. Index every new record under a lineage; re-verify all changed records at 1e-9 and 1e-12;
   recompute the full before/after table against the once-scraped baselines. Save it to
   `runs/<run>/final-comparison.txt`.
3. EXPERIMENTS.md: a new dated "Day" section, each phase with its pre-registered criterion and
   the honest result (pass / fail / not reached). README "Status": shortfall, ties, **beats
   (only if genuine)**, open problem.
4. `pytest` green (exit code 0). Update the memory notes if something durable was learned.
5. Final message: per improved n, before/after and referee confirmation; any beat, with the
   exact value and the page's interval; which ideas failed, with numbers; what was not reached;
   and exactly what is committed or uncommitted.

## Non-goals

- No operator-mix retuning or other large default changes that would need the full concurrent
  A/B protocol.
- Don't touch `fence/referee.py` or existing tests except to add tests for new code.
- Don't reopen verdicts already recorded (changes 5, 9, 10, 13, 14, the 09-23 cleanup).
