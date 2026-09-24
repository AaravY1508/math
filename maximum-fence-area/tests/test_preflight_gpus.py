"""run_overnight's GPU preflight: which requested GPUs are usable.

On 2026-09-20 temescal's GPU 1 fell off the bus. nvidia-smi then listed indices 0, 2, 3, and
the old check (`index >= number of rows`) rejected `--gpu-ids 0,2,3` as "missing [3]". The same
host also showed that a GPU listed as idle is not necessarily usable: the driver asked for a
reboot and refused cuInit on every device, so the preflight probes CUDA per GPU as well.
"""
import sys

import run_overnight as RO

GAPPED = ["0, NVIDIA GeForce RTX 3090, 1 MiB",
          "2, NVIDIA GeForce RTX 3090, 1 MiB",
          "3, NVIDIA GeForce RTX 3090, 1 MiB"]


def test_a_gap_in_the_indices_is_not_a_missing_gpu():
    usable, missing, busy = RO.check_gpus(GAPPED, [0, 2, 3])
    assert usable == [0, 2, 3] and missing == [] and busy == []


def test_the_gpu_that_fell_off_the_bus_is_missing():
    usable, missing, busy = RO.check_gpus(GAPPED, [0, 1, 2, 3])
    assert missing == [1] and usable == [0, 2, 3]


def test_another_users_gpu_is_left_out_not_shared():
    rows = ["0, RTX 3090, 8752 MiB", "1, RTX 3090, 4 MiB", "2, RTX 3090, 9056 MiB", "3, RTX 3090, 36 MiB"]
    usable, missing, busy = RO.check_gpus(rows, [0, 1, 2, 3])
    assert busy == [0, 2] and usable == [1, 3] and missing == []


def test_unparseable_rows_are_ignored():
    rows = ["Unable to determine the device handle for GPU1: 0000:21:00.0: Unknown Error"] + GAPPED
    assert RO.check_gpus(rows, [0, 2]) == ([0, 2], [], [])


def test_cuda_probe_reports_failure_with_a_reason(monkeypatch):
    # a probe interpreter that cannot even import the probe fails cleanly with a message
    monkeypatch.setattr(RO, "CUDA_PROBE", "raise SystemExit('cuda unavailable')")
    ok, why = RO.cuda_works(0, timeout=60)
    assert not ok and "cuda unavailable" in why


def test_cuda_probe_passes_when_the_probe_succeeds(monkeypatch):
    monkeypatch.setattr(RO, "CUDA_PROBE", "import sys; sys.exit(0)")
    assert RO.cuda_works(0, timeout=60)[0]
    assert sys.executable
