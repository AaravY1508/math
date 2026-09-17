"""Shared test setup.

The validation gate runs on boxes with many cores (64 here) that other people are
also using. Torch and the BLAS behind the SLSQP polish both default to one thread
per core, and on the tiny batches these tests use the oversubscription costs far
more in synchronization than it wins in parallelism -- the two end-to-end optimizer
tests took 107 s of wall time and 40 minutes of CPU unthrottled, and 5 s on four
threads. Production runs set their own thread count; this only affects the tests.
"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

try:
    import torch

    torch.set_num_threads(4)
except ImportError:      # pragma: no cover - the rule/heuristic tests do not need torch
    pass
