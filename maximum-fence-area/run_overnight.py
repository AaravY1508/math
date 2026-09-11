"""Overnight search supervisor for the Maximum Fence Area problem.

    ./venv/bin/python run_overnight.py --budget 6h --gpus 4 --archive-dir ~/fence-results
    ./venv/bin/python run_overnight.py --smoke                      # 60 s end-to-end check at small n
    ./venv/bin/python run_overnight.py --resume runs/<timestamp>    # continue a crashed run

What it does
  1. Hard precondition: runs the pytest suite. If anything fails, it writes
     ~/erich-friedman-BLOCKED.txt and exits without searching.
  2. Loads baselines.json (refuses a TODO placeholder) and checks the GPUs.
  3. Splits the n-range across GPUs, balanced by fence.evolve.schedule_weight ("best
     known" n above the trivial grid cases and small n, n >= 32 highest, and an n whose
     all-time record in the minima library is a grid-trivial tie barely at all), and
     starts one worker process per GPU with CUDA_VISIBLE_DEVICES=<gpu>. Each worker runs
     fence.evolve (island MAP-Elites over proposer programs) + fence.optimize (GPU Adam,
     CPU SLSQP) + the 50-digit referee, checkpointing every 5 minutes to
     runs/<ts>/worker<i>/, and records stabilized minima per proposer in the minima
     library (--minima-dir, shared by all runs; see fence/minima.py).
  4. Supervises: a worker that dies or stops heart-beating is logged loudly and
     restarted from its last checkpoint; the other workers keep running.
  5. Exits hard at the wall-clock budget: workers stop taking new work near the end,
     then the final report (table + SVGs) is written and copied to --archive-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BLOCKED = Path.home() / "erich-friedman-BLOCKED.txt"


def parse_duration(s: str) -> float:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", str(s))
    if not m:
        raise argparse.ArgumentTypeError(f"bad duration {s!r} (use e.g. 6h, 90m, 60s)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


def write_blocked(reason: str, detail: str = ""):
    BLOCKED.write_text(
        f"BLOCKED at {time.strftime('%Y-%m-%d %H:%M:%S')} -- the overnight fence search was NOT launched.\n\n"
        f"Reason: {reason}\n\n{detail}\n")
    print(f"!!! BLOCKED: {reason} (details in {BLOCKED})", flush=True)


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.log_fh = None

    # -- logging ------------------------------------------------------------
    def log(self, rec: dict, echo: str | None = None):
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "ts": round(time.time(), 3), **rec}
        if self.log_fh:
            self.log_fh.write(json.dumps(rec, default=str) + "\n")
            self.log_fh.flush()
        if echo:
            print(f"[{time.strftime('%H:%M:%S')}] {echo}", flush=True)

    # -- preflight ------------------------------------------------------------
    def preflight(self) -> bool:
        print(f"[{time.strftime('%H:%M:%S')}] preflight: running pytest (hard precondition) ...", flush=True)
        try:
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=HERE, capture_output=True,
                               text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            write_blocked("pytest timed out after 30 minutes")
            return False
        tail = (p.stdout + p.stderr)[-6000:]
        print(tail.strip().splitlines()[-1] if tail.strip() else "(no pytest output)", flush=True)
        if p.returncode != 0:
            write_blocked(f"pytest failed (exit code {p.returncode})", tail)
            return False
        try:
            from fence.evolve import load_baselines
            self.baselines = load_baselines()
        except Exception as exc:  # noqa: BLE001
            write_blocked("baselines.json unusable", repr(exc))
            return False
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=60)
            gpus = [ln for ln in q.stdout.strip().splitlines() if ln.strip()]
        except Exception as exc:  # noqa: BLE001
            write_blocked("nvidia-smi failed", repr(exc))
            return False
        missing = [g for g in self.args.gpu_ids if g >= len(gpus)]
        if missing:
            write_blocked(f"asked for GPUs {missing} but nvidia-smi lists {len(gpus)}", "\n".join(gpus))
            return False
        busy = [ln for ln in gpus if int(ln.split(",")[0]) in self.args.gpu_ids
                and int(ln.split(",")[2].split()[0]) > 1024]
        if busy:
            print("!!! warning: requested GPUs already hold memory of other processes: " + "; ".join(busy), flush=True)
        self.gpu_info = gpus
        return True

    # -- assignment -----------------------------------------------------------
    def weight(self, n: int) -> float:
        from fence.evolve import schedule_weight
        return schedule_weight(n, self.baselines.get(n), self.records.get(n))

    def assign(self, ns: list[int]) -> list[list[int]]:
        load = [0.0] * self.args.gpus
        slices = [[] for _ in range(self.args.gpus)]
        for n in sorted(ns, key=lambda n: (-self.weight(n), -n)):
            g = min(range(self.args.gpus), key=lambda i: (load[i], len(slices[i])))
            slices[g].append(n)
            load[g] += self.weight(n)
        return [sorted(s) for s in slices]

    def _minima_dir(self) -> Path | None:
        a = self.args
        if a.no_minima:
            return None
        if a.minima_dir:
            return Path(os.path.expanduser(a.minima_dir)).resolve()
        # a smoke run must not leave small-n minima in the library the real runs use
        return self.run_dir / "minima" if a.smoke else HERE / "minima"

    def _load_records(self, ns: list[int]) -> dict[int, float]:
        if not self.minima_dir:
            return {}
        from fence.minima import MinimaLibrary
        lib = MinimaLibrary(self.minima_dir)
        return {n: float(rec["area"]) for n in ns if (rec := lib.record(n))}

    # -- workers ----------------------------------------------------------------
    def spawn(self, i: int, resume: bool):
        a = self.args
        cmd = [sys.executable, "-m", "fence.evolve", "--gpu", str(i), "--ns", ",".join(map(str, self.slices[i])),
               "--run-dir", str(self.run_dir), "--deadline", str(self.worker_deadline), "--seed", str(a.seed),
               "--mutator", a.mutator, "--steps", str(a.steps), "--batch", str(a.batch),
               "--cpu-procs", str(a.cpu_procs), "--drain-seconds", str(self.drain),
               "--checkpoint-every", str(a.checkpoint_every), "--polish-iters", str(a.polish_iters),
               "--job-timeout", str(a.job_timeout), "--tol", str(a.tol),
               "--minima-dir", str(self.minima_dir or ""), "--stable-after", str(a.stable_after),
               "--topology-cap", str(a.topology_cap)]
        if resume:
            cmd.append("--resume")
        # single-threaded BLAS in the worker and (inherited) in its CPU polish pool: the pool
        # already uses one process per core, extra BLAS threads would only oversubscribe
        dev = a.gpu_ids[i]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(dev), "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(HERE),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        out = open(self.run_dir / "logs" / f"worker{i}.out", "a")
        p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        self.procs[i] = p
        self.started[i] = time.time()
        self.log({"event": "worker_spawn", "gpu": i, "device": dev, "pid": p.pid, "resume": resume, "ns": self.slices[i]},
                 echo=f"worker {i} started (pid {p.pid}) on GPU {dev} for n={self.slices[i]}" + (" [resumed from checkpoint]" if resume else ""))

    def heartbeat_age(self, i: int) -> float:
        hb = self.run_dir / "logs" / f"worker{i}.heartbeat"
        try:
            return time.time() - float(hb.read_text())
        except Exception:  # noqa: BLE001
            return time.time() - self.started[i]

    def kill(self, i: int, sig=signal.SIGKILL):
        p = self.procs.get(i)
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                pass

    # -- status -----------------------------------------------------------------
    def status_line(self) -> str:
        from fence.evolve import classify
        bests = {}
        for f in (self.run_dir / "best").glob("n*.json"):
            try:
                cfg = json.loads(f.read_text())
                bests[int(cfg["n"])] = float(cfg["verification"]["total_area"])
            except Exception:  # noqa: BLE001
                pass
        counts = {}
        for n, a in bests.items():
            s = classify(a, self.baselines.get(n))
            counts[s] = counts.get(s, 0) + 1
        evals = minima = 0
        for i in range(self.args.gpus):
            lp = self.run_dir / "logs" / f"worker{i}.jsonl"
            if lp.exists():
                with open(lp, "rb") as fh:
                    for ln in fh:
                        if b'"event": "eval"' in ln:
                            evals += 1
                        elif b'"event": "minimum_indexed"' in ln:
                            minima += 1
        alive = sum(1 for p in self.procs.values() if p.poll() is None)
        el = time.time() - self.T0
        return (f"elapsed {int(el // 3600)}:{int(el % 3600 // 60):02d}  workers alive {alive}/{self.args.gpus}  "
                f"evals {evals}  minima indexed {minima}  n with results {len(bests)}  status counts {counts}  "
                f"restarts {sum(self.restarts.values())}")

    def sync_archive(self, final=False):
        if not self.archive:
            return
        dst = self.archive / self.run_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for sub in ("best", "improvements", "svg"):
            s = self.run_dir / sub
            if s.exists():
                shutil.copytree(s, dst / sub, dirs_exist_ok=True)
        for f in ("run.json", "report.txt", "report.json"):
            if (self.run_dir / f).exists():
                shutil.copy2(self.run_dir / f, dst / f)
        if final and self.minima_dir and self.minima_dir.exists():
            shutil.copytree(self.minima_dir, dst / "minima", dirs_exist_ok=True)   # snapshot of the library
        self.log({"event": "archive_sync", "dst": str(dst), "final": final})

    def save_runinfo(self):
        info = {**self.runinfo, "elapsed_s": round(time.time() - self.T0 + self.prev_elapsed, 1),
                "restarts": self.restarts}
        (self.run_dir / "run.json").write_text(json.dumps(info, indent=1))

    # -- main ---------------------------------------------------------------------
    def run(self) -> int:
        a = self.args
        if not self.preflight():
            return 2
        ns = parse_range(a.n_range)
        self.prev_elapsed = 0.0
        if a.resume:
            self.run_dir = Path(a.resume).resolve()
            old = json.loads((self.run_dir / "run.json").read_text())
            budget = parse_duration(a.budget) if a.budget_given else max(60.0, old["budget_s"] - old.get("elapsed_s", 0))
            self.prev_elapsed = old.get("elapsed_s", 0)
            resume = True
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if a.smoke else "") + (f"-{a.tag}" if a.tag else "")
            self.run_dir = (HERE / "runs" / stamp).resolve()
            budget = parse_duration(a.budget)
            resume = False
        self.minima_dir = self._minima_dir()
        self.records = self._load_records(ns)
        self.slices = old["slices"] if a.resume else self.assign(ns)
        if not all(self.slices):
            # fewer n than GPUs: a worker without any n would only crash and be restarted
            keep = [i for i, s in enumerate(self.slices) if s]
            print(f"  only {len(keep)} GPUs have work; leaving GPUs "
                  f"{[g for i, g in enumerate(a.gpu_ids) if i not in keep]} idle", flush=True)
            a.gpu_ids = [a.gpu_ids[i] for i in keep]
            self.slices = [self.slices[i] for i in keep]
            a.gpus = len(keep)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.log_fh = open(self.run_dir / "logs" / "supervisor.jsonl", "a")
        self.archive = Path(os.path.expanduser(a.archive_dir)).resolve() if a.archive_dir else None
        self.T0 = time.time()
        self.T_end = self.T0 + budget
        self.reserve = min(240.0, max(12.0, 0.2 * budget))
        self.drain = min(90.0, max(8.0, 0.15 * budget))
        self.worker_deadline = self.T_end - self.reserve
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                                    text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            commit = None
        self.runinfo = {"run_dir": str(self.run_dir), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "budget_s": budget if not a.resume else old["budget_s"], "session_budget_s": budget,
                        "deadline": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.T_end)),
                        "gpus": a.gpus, "slices": self.slices, "args": vars(a), "git_commit": commit,
                        "supervisor_pid": os.getpid(), "gpu_info": self.gpu_info,
                        "minima_dir": str(self.minima_dir) if self.minima_dir else None}
        self.restarts = {i: 0 for i in range(a.gpus)}
        self.save_runinfo()
        self.log({"event": "supervisor_start", **self.runinfo},
                 echo=f"run dir {self.run_dir}\n  budget {budget / 3600:.2f} h -> hard stop at "
                      f"{self.runinfo['deadline']}; GPUs {a.gpus}; mutator={a.mutator}")
        for i, s in enumerate(self.slices):
            print(f"  GPU {a.gpu_ids[i]}: n = {s}", flush=True)
        if self.minima_dir:
            from fence.evolve import trivial_tie
            ties = [n for n in ns if trivial_tie(n, self.baselines.get(n), self.records.get(n))]
            print(f"  minima library {self.minima_dir}: all-time records for {len(self.records)} n; "
                  f"grid-trivial ties kept to a small share: {ties}", flush=True)

        self.procs, self.started, self.next_restart = {}, {}, {}

        # watchdog: never outlive the budget by more than a minute, whatever happens
        def watchdog():
            while True:
                time.sleep(5)
                if time.time() > self.T_end + 60:
                    for i in list(self.procs):
                        self.kill(i)
                    print("!!! watchdog: budget exceeded, hard exit", flush=True)
                    os._exit(3)
        threading.Thread(target=watchdog, daemon=True).start()

        def on_term(signum, frame):
            self.log({"event": "supervisor_signal", "signum": signum}, echo=f"supervisor got signal {signum}; stopping workers")
            for i in list(self.procs):
                self.kill(i, signal.SIGTERM)
            time.sleep(3)
            for i in list(self.procs):
                self.kill(i)
            os._exit(4)
        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGINT, on_term)

        for i in range(a.gpus):
            self.spawn(i, resume=resume)
        last_status = last_sync = last_info = time.time()
        finished = set()
        while True:
            now = time.time()
            if now >= self.worker_deadline + 10:
                break
            for i in range(a.gpus):
                if i in finished:
                    continue
                p = self.procs[i]
                rc = p.poll()
                if rc is not None:
                    if rc == 0 and now >= self.worker_deadline - self.drain - 5:
                        finished.add(i)
                        self.log({"event": "worker_finished", "gpu": i}, echo=f"worker {i} finished normally")
                        continue
                    if i not in self.next_restart:
                        tail = ""
                        try:
                            tail = (self.run_dir / "logs" / f"worker{i}.out").read_text()[-1500:]
                        except Exception:  # noqa: BLE001
                            pass
                        self.log({"event": "WORKER_DIED", "gpu": i, "returncode": rc, "tail": tail},
                                 echo=f"!!!!!!!! WORKER {i} DIED (exit code {rc}) -- restarting from its last checkpoint; "
                                      f"other workers unaffected. Last output:\n{tail[-600:]}")
                        self.restarts[i] += 1
                        self.next_restart[i] = now + min(60.0, 5.0 * self.restarts[i])
                    if now >= self.next_restart[i]:
                        del self.next_restart[i]
                        if self.restarts[i] > a.max_restarts:
                            finished.add(i)
                            self.log({"event": "worker_abandoned", "gpu": i}, echo=f"!!! worker {i} restarted too often; giving up on GPU {i}")
                        elif self.worker_deadline - now > 90:
                            self.spawn(i, resume=True)
                        else:
                            finished.add(i)
                elif self.heartbeat_age(i) > a.hang_timeout:
                    self.log({"event": "WORKER_HUNG", "gpu": i, "heartbeat_age": self.heartbeat_age(i)},
                             echo=f"!!!!!!!! WORKER {i} HUNG (no heartbeat for {self.heartbeat_age(i):.0f}s) -- killing and restarting")
                    self.kill(i)
            if len(finished) == a.gpus:
                break
            if now - last_status > a.status_every:
                self.log({"event": "status"}, echo=self.status_line())
                last_status = now
            if now - last_info > 60:
                self.save_runinfo()
                last_info = now
            if now - last_sync > a.archive_every:
                self.sync_archive()
                last_sync = now
            time.sleep(2)

        # -- shutdown ---------------------------------------------------------------
        for i in range(a.gpus):
            self.kill(i, signal.SIGTERM)
        t_kill = time.time() + 15
        while time.time() < t_kill and any(p.poll() is None for p in self.procs.values()):
            time.sleep(0.5)
        for i in range(a.gpus):
            self.kill(i)
        self.save_runinfo()
        self.log({"event": "workers_stopped"}, echo="all workers stopped; " + self.status_line())
        import report
        left = self.T_end - time.time() - 20
        try:
            rep = report.build_report(self.run_dir, reverify=True, svg_all=True, tol=a.tol, time_limit=max(10.0, left * 0.7))
            txt = report.format_table(rep) + "\n\n" + report.loud_summary(rep) + "\n"
            (self.run_dir / "report.txt").write_text(txt)
            (self.run_dir / "report.json").write_text(json.dumps(rep, indent=1, default=str))
            print("\n" + txt, flush=True)
            self.log({"event": "final_report", "improvements": rep["improvements"], "ties": [r["n"] for r in rep["ties"]]})
        except Exception as exc:  # noqa: BLE001
            self.log({"event": "report_failed", "error": repr(exc)}, echo=f"!!! final report failed: {exc!r}")
        if self.archive and not a.smoke:
            self.sync_archive(final=True)
            print(f"[{time.strftime('%H:%M:%S')}] archived results to {self.archive / self.run_dir.name}", flush=True)
        self.log({"event": "supervisor_exit"}, echo=f"done. run dir: {self.run_dir}")
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", default=None, help="wall-clock budget, e.g. 6h, 90m, 60s (default 6h; 60s with --smoke)")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-ids", default=None,
                    help="physical GPUs to use, e.g. 1,2,3 (overrides --gpus; default 0..gpus-1)")
    ap.add_argument("--tag", default="", help="suffix for the run directory name, e.g. the host name")
    ap.add_argument("--n-range", default=None, help="e.g. 12-50 or 13,15,20-30 (default 12-50; 6-9 with --smoke)")
    ap.add_argument("--archive-dir", default="~/fence-results", help="copy of final results OUTSIDE the repo")
    ap.add_argument("--resume", default=None, help="continue an existing runs/<timestamp> directory")
    ap.add_argument("--smoke", action="store_true", help="60-second end-to-end check at small n")
    ap.add_argument("--mutator", choices=["random", "llm"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0, help="restarts per topology on the GPU (0 = by n: 32k/16k/10k)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--polish-iters", type=int, default=1200)
    ap.add_argument("--tol", type=float, default=1e-9, help="referee tolerance (absolute, unit scale)")
    ap.add_argument("--cpu-procs", type=int, default=0, help="CPU polish processes per GPU worker (0 = auto)")
    ap.add_argument("--checkpoint-every", type=float, default=300.0)
    ap.add_argument("--job-timeout", type=float, default=600.0)
    ap.add_argument("--hang-timeout", type=float, default=1200.0, help="restart a worker after this long without heartbeat")
    ap.add_argument("--max-restarts", type=int, default=25)
    ap.add_argument("--status-every", type=float, default=300.0)
    ap.add_argument("--archive-every", type=float, default=1800.0)
    ap.add_argument("--minima-dir", default=None,
                    help="minima library shared across runs (default: ./minima; a private one in the run dir with --smoke)")
    ap.add_argument("--no-minima", action="store_true", help="neither record nor avoid minima")
    ap.add_argument("--stable-after", type=int, default=6,
                    help="evaluations of a proposer at n without improvement before its best new result "
                         "is recorded as a minimum")
    ap.add_argument("--topology-cap", type=int, default=3,
                    help="re-propose a topology once its proposer has this many minima on it (0: never)")
    args = ap.parse_args(argv)
    args.budget_given = args.budget is not None
    args.gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else list(range(args.gpus))
    args.gpus = len(args.gpu_ids)
    if args.smoke:
        args.budget = args.budget or "60s"
        args.n_range = args.n_range or "6-9"
        args.batch = args.batch or 4096
        args.steps = min(args.steps, 1000)
        args.polish_iters = min(args.polish_iters, 600)
        args.checkpoint_every = min(args.checkpoint_every, 20.0)
        args.status_every = min(args.status_every, 20.0)
        args.job_timeout = min(args.job_timeout, 60.0)
    args.budget = args.budget or "6h"
    args.n_range = args.n_range or "12-50"
    if not args.cpu_procs:
        args.cpu_procs = max(2, ((os.cpu_count() or 8) - 4) // max(1, args.gpus))
    return Supervisor(args).run()


if __name__ == "__main__":
    sys.exit(main())
