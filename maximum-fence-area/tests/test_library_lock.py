"""The minima library's per-n lock: concurrent writers on one n never lose the better record.

Before the lock, update_record was an unguarded read-then-write, and two writers on the same n
could interleave read/read/write/write so the smaller area landed last. Runs only avoid that by
giving each host a disjoint n-range; these tests pin the guarantee on the library itself.
"""
import multiprocessing as mp
import os
import time

from fence import minima as M


def _cfg(area: float) -> dict:
    return {"n": 7, "verification": {"total_area": repr(area), "valid": True}}


def _writer(root, area, barrier):
    lib = M.MinimaLibrary(root)
    read = lib.record

    def slow_read(n):
        # widen the read-to-write window from microseconds to 50 ms, so an unguarded
        # read-modify-write loses an update almost surely instead of occasionally
        cur = read(n)
        time.sleep(0.05)
        return cur

    lib.record = slow_read
    barrier.wait()
    lib.update_record(7, _cfg(area))


def race(root, areas, lock=True):
    if not lock:
        M.NLock.__enter__ = lambda self: self          # in the forked children only
        M.NLock.__exit__ = lambda self, *e: False
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(len(areas))
    procs = [ctx.Process(target=_writer, args=(str(root), a, barrier)) for a in areas]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    return M.MinimaLibrary(root).record(7)["area"]


AREAS = [1.0, 1.5, 2.0, 1.25, 1.75, 0.5]


def test_racing_writers_keep_the_largest_area(tmp_path):
    for t in range(8):
        assert race(tmp_path / str(t), AREAS) == max(AREAS)
        assert not list((tmp_path / str(t) / "locks").iterdir()), "every lock was released"


def test_the_race_is_real_without_the_lock(tmp_path):
    """Guards the test above: with the lock disabled the same race loses the largest area."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def trial():
        lost = sum(race(tmp_path / str(t), AREAS, lock=False) != max(AREAS) for t in range(8))
        q.put(lost)

    p = ctx.Process(target=trial)        # the monkeypatch must not leak into this process
    p.start()
    p.join(120)
    assert q.get(timeout=5) >= 1


def test_lock_excludes_a_second_holder(tmp_path):
    import threading

    lib = M.MinimaLibrary(tmp_path)
    got = threading.Event()

    def contender():
        with lib.lock(5):
            got.set()

    with lib.lock(5):
        t = threading.Thread(target=contender)
        t.start()
        time.sleep(0.3)
        assert not got.is_set(), "a second writer entered while the lock was held"
    t.join(5)
    assert got.is_set()
    with lib.lock(6):                     # a different n is independent
        pass
    assert not (tmp_path / "locks" / "N5.lock").exists()


def test_stale_lock_left_by_a_dead_writer_is_taken_over(tmp_path):
    lib = M.MinimaLibrary(tmp_path)
    stale = tmp_path / "locks" / "N9.lock"
    stale.write_text("deadhost:1:0")
    old = time.time() - 10 * M.LOCK_STALE_S
    os.utime(stale, (old, old))
    t0 = time.time()
    assert lib.update_record(9, _cfg(3.0))
    assert time.time() - t0 < 5.0
    assert lib.record(9)["area"] == 3.0
    assert not stale.exists()


def test_release_never_removes_someone_elses_lock(tmp_path):
    lk = M.NLock(tmp_path / "N1.lock")
    with lk:
        lk.path.write_text("another-writer")        # taken over while we held it (we were presumed dead)
    assert lk.path.read_text() == "another-writer"
