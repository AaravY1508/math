# 4-hour task: chase the record shapes

Paste this whole file as the prompt to a fresh Claude Code session in this repo
(`maximum-fence-area/`, branch `search-strategy`). It implements and tests, in priority
order, the ideas from the 2026-09-24 review of where the gap to the page's records sits.
Hard budget: **4 hours wall clock from when you start Phase 0**, tracked for real (a
background watchdog, not a promise) so partial progress is never lost.

## Ground rules (from this project's own conventions -- keep them)

- Every change is flag-gated; the default reproduces current behaviour.
- Before touching anything: `git status`, and if the tree is dirty, confirm with the user
  whether to keep those changes staged or stash them -- do not discard uncommitted work.
  Take a snapshot ref first regardless: `git stash create "pre-4h-run $(date -Iseconds)"`
  then `git update-ref refs/backup/pre-4h-run-<date> <sha>`.
- Re-scrape `baselines.json` once, at the very start (`scrape_baselines.py`), and not again
  mid-run -- a mid-run rescrape corrupts any "beat" claim (see memory: fence-baselines-go-stale).
- GPU list is not fixed -- re-probe with `nvidia-smi --query-gpu=index,name,memory.used
  --format=csv,noheader` at launch. Skip any GPU that errors (a driver wanting a reboot
  shows as "Unable to determine the device handle") or holds > 1 GB already. Never share a
  GPU with another user's process.
- Every read-modify-write on the minima library already takes a per-n lock
  (`fence/minima.py: MinimaLibrary.lock`); use `MinimaLibrary.update_record` /
  `add_minimum`, never hand-write into `minima/`.
- Every verified area must pass `fence/referee.py:verify()` before it counts as anything.
  Report only referee-verified numbers.
- Log every attempt to `EXPERIMENTS.md` in this project's existing style: a numbered
  change, a one-line hypothesis, pre-registered pass/fail criteria stated *before* running,
  then the result honestly, including failures. Do not retroactively soften a criterion.
- At the end, whatever state things are in, write a status summary (win or not) and leave
  the tree in a state that `pytest` passes on. Commit only if the user has authorized it in
  this session; otherwise stop with everything staged and described.

## Phase 0 -- setup (15 min)

1. `git status`; snapshot per the rule above.
2. `./venv/bin/python -m pytest -q` -- must be green before changing anything. If it is
   not, stop and report; do not build on a broken tree.
3. `./venv/bin/python scrape_baselines.py` (or whatever the current invocation is -- check
   `--help`) to refresh `baselines.json`, once.
4. Probe GPUs as above; record which indices are usable.
5. Read `EXPERIMENTS.md`'s last "Day" section and the current README's "Removed" and
   "Status" sections so the numbers you compare against are today's, not stale ones.
6. Compute the current per-n gap table (record area minus `minima/records/N<n>.json`
   area) and sort by gap descending. Confirm it roughly matches: largest gaps at n = 50
   (~0.56), 41 (~0.45), 32 (~0.39), 46 (~0.33), 10 (~0.32), 48 (~0.29), 44 (~0.24). If the
   library has moved since, use the fresh numbers for the rest of this run.

## Phase 1 -- digitize the record pictures (70 min, highest priority)

**Hypothesis:** the page's September-2026 records at grid-tied n (17, 22, 24, 27, 29, 31,
36, 38, 40, 41, 45, 47, 49, 50) are not grids -- they are rounded ring/onion layouts our
seed families never build. Loading the record's own topology and optimizing it should
close most of the grid-tied group's gap.

1. Fetch `https://erich-friedman.github.io/packing/fence/<n>.gif` for every n in that
   list (and any other n whose record picture looks non-trivial). Convert to PNG at 3x
   with `convert <n>.gif -scale 300% <n>.png` (ImageMagick is present) so line-following
   is easier, whether done visually or programmatically.
2. Write a small digitizer (numpy/scipy only -- no `cv2`/`skimage` in this venv):
   - Threshold the image to a black/white mask of the drawn lines.
   - Skeletonize or trace line segments (a corner/junction-detection approach is fine;
     scipy.ndimage has what you need for connected-component and local-maxima work).
   - Snap endpoints to a common vertex set within a pixel tolerance, then rescale the
     whole picture so the *shortest* edge (or the modal edge length, if T-junctions make
     some edges shorter) maps to length 1.
   - Feed the resulting segment list to `fence.geometry.topology_from_segments` to get a
     `Topology`, and `fence.builder.Sketch.from_config` / whatever the existing loader
     path is (check how `pipeline.py`'s `index_minimum` and library `config_from_json`
     round-trip) to get something `fence.optimize` can polish.
   - This is fiddly -- if full automation eats more than ~30 min on the first n (n=17,
     the simplest picture, 186x173px, 7 faces), fall back to reading the endpoints off the
     picture by eye (coordinates from the image, unit-rescaled) for the handful of n that
     matter most (17, 24, 41, 50), and automate only if there's time left.
   - Do NOT worry about exact lengths/angles from the picture -- only the topology
     (which vertex touches which fence, where) needs to be right. The optimizer fixes
     geometry from any topologically-correct sketch.
3. For each digitized topology: run it through `fence/optimize.py`'s batched GPU stage,
   then `fence/referee.py:verify()`. If it verifies and its area is >= 99% of the record,
   that is strong evidence the topology read is correct; if it's far off, re-check the
   digitization before concluding the idea failed.
4. **Pass criterion (pre-register before running):** at least 3 of {17, 24, 41, 50} produce
   a referee-verified layout within 0.01 of the page's own figure. If fewer than 3 succeed
   in the time box, downgrade the rest of this phase's budget to Phase 2 and note why
   (bad digitization vs. genuine optimizer shortfall -- these are different failures).
5. Every verified result that beats the current library record goes in via
   `MinimaLibrary.update_record` (respecting the lock), tagged `{"via": "digitized",
   "run": "<this run's name>"}` in its meta.

## Phase 2 -- neighbourhood search around record/digitized topologies (60 min)

**Hypothesis:** a topology one or two moves away from a record (or from a digitized
near-record) is more likely to beat it than anything a generic search finds cold.

1. For every topology from Phase 1 (digitized or record-tied grid) and every topology
   already in `minima/records/` for the non-grid highest-gap n (32, 46, 48, 44, 43, 33),
   enumerate every single application of each topology move in `fence/netevolve.py`
   (`join_to_t`, `t_to_join`, `slide_t_past`, `repin`, `add_face`, `remove_face`,
   `face_fill`), then every legal pair of two moves from that first set. Keep only
   candidates that pass the hard rule set.
2. Batch-optimize and referee-verify every survivor (GPU stage as elsewhere in the
   pipeline; use `fence.optimize`'s existing batched path, do not write a new one).
3. **Pass criterion (pre-register before running):** report, per n, whether any verified
   candidate beats the current library record, and by how much. There is no bar for
   "success" here beyond "did it beat the record" -- record the true outcome, including a
   clean "no improvement found" if that's what happens; that is a legitimate result worth
   keeping in EXPERIMENTS.md, not a reason to keep searching past the time box.
4. Anything that beats a record goes into the library the same way as Phase 1.

## Phase 3 -- parametric ring/onion seed family (40 min)

Only if Phase 1 leaves enough runway; skip cleanly and say so if not.

1. Add a new net generator in `fence/nets.py` (own id, `default=True` once it earns its
   place, but keep it flag-off while unproven, matching how other exploratory generators
   were added -- see `library_face_target`'s removed pattern in EXPERIMENTS.md change 10
   for how *not* to leave a failed one on by default) that builds: a small core polygon,
   surrounded by k rings of near-quadrilateral cells, boundary rounded by treating the
   outermost ring's outer edges as chords instead of straight sides. Parametrize by
   (core sides, cells per ring, number of rings) and enumerate combinations that can hit
   exactly n fences for a spread of n (at least covering 12..50).
2. Sanity-test: for n = 17, 24, 41, 50 does this family produce a net structurally similar
   (same F, similar T, similar ring pattern) to the digitized record from Phase 1? That is
   the check that it is the right family, independent of whether the optimizer reaches the
   record area from it.
3. **Pass criterion:** the family builds a legal net (passes all hard rules) for at least
   80% of n in 12..50, and stage-2 (netevolve, short budget e.g. `--evals-per-net 20`) on 5
   sample n reaches within 5% of the current library record on each. If it doesn't clear
   this bar, leave it in the tree off by default with the numbers recorded, exactly like
   the other failed generators -- do not delete a generator this task itself just added
   without giving it its numbers in EXPERIMENTS.md first.

## Phase 4 -- wrap-up (15 min, non-negotiable even if running over)

1. Stop all GPU work at the 3:45 mark regardless of phase.
2. Recompute the full per-n gap table against the (once) re-scraped baselines.
3. Write an EXPERIMENTS.md entry (new "Day" section, dated) covering every phase run,
   with the pre-registered criteria and the honest result of each -- pass, fail, or "ran
   out of time before reaching a verdict" are all acceptable outcomes to write down.
4. Update README's "Status" section numbers (shortfall, ties, open problem) to match.
5. Run `./venv/bin/python -m pytest -q` again; it must be green.
6. Final message to the user: which n improved (with before/after areas and referee
   confirmation), which ideas were tested and failed (with numbers), which were not
   reached, and exactly what is uncommitted and why.

## Explicit non-goals for this 4-hour box

- Do not attempt `--op-mix`-style large weight retuning; that needs the project's full
  A/B protocol (concurrent arms, pre-registered non-inferiority bars) and does not fit
  here.
- Do not modify `fence/referee.py` or anything in the validation gate (`tests/`) except to
  add tests for genuinely new code (e.g. the digitizer, the ring generator).
- Do not delete anything this task didn't itself add, and don't re-open the 2026-09-23
  cleanup (op-mix tuned / inflate_face / rule 15 bundle / one_cluster) -- those already
  have their verdicts recorded.
