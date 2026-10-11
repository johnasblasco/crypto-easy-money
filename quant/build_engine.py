"""Build the signal engine in one command: history, the studies it depends on, then the engine.

Usage:
    python -m quant.build_engine              # every step that isn't done yet
    python -m quant.build_engine --dry-run    # list the steps and which are already done
    python -m quant.build_engine --retrain    # rerun only the final step (quant.train_engine)
    python -m quant.build_engine --refresh    # later: update every coin to today, then retrain

In Docker (the dashboard's engine card picks the result up by itself):
    docker compose run --rm web python -m quant.build_engine

Steps, each skipped when its output already exists, so rerunning resumes:

1. ``quant.data download``: 1-minute Binance history for the 16 coins (~3 GB). Missing
   coins are retried. Coins already on disk are not topped up here: the feature caches
   the next steps build would then go stale. The live engine syncs recent bars itself.
2. ``quant.studies.execution_realism``: research-period statistics of the event rule.
   ``quant.train_engine`` cannot run without them.
3. ``quant.studies.intraday_conversion``, ``daily_panel 1``, ``daily_panel 3``: these
   record every research trial in the ledger (data/ledger). ``quant.train_engine``
   deflates each model candidate's Sharpe ratio over those trials; without them the
   models can never pass and are marked NOT_VALIDATED.
4. ``quant.train_engine``: evaluates each frozen candidate once on the holdout period
   and writes models/engine/.

The other research studies (trend, event_hypotheses, ...) feed the research report,
not the engine, so they are not run here.

Each step runs in its own Python process, so its memory is freed before the next one.
Everything printed is also appended to data/logs/build_engine.log, which survives the
Docker container. Budget about 8 GB of memory and 5 GB of disk.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import data as D
from . import engine as E
from . import experiment as X
from . import ledger as L

DOWNLOAD_ATTEMPTS = 3
RETRY_WAIT_S = 60          # x attempt number: lets a short network outage pass
MIN_MEMORY_GB = 9.5        # decimal GB: a default 8 GiB Docker VM warns, a 10 GB one doesn't
MIN_FREE_DISK_GB = 5.0
MEMORY_ADVICE = ("   Docker Desktop: give it at least 10 GB (Mac / Hyper-V: Settings > Resources; Windows with WSL 2: "
                 "memory=10GB under [wsl2] in %UserProfile%\\.wslconfig, then wsl --shutdown and restart Docker "
                 "Desktop; never lower a larger existing value), and stop the dashboard while building: "
                 "docker compose stop web")


@dataclass
class Step:
    name: str
    args: Callable[[], list[str]]      # arguments after `python -m`
    done: Callable[[], bool]
    why: str
    attempts: int = 1
    force: bool = False                 # run even if done() already holds


def _readable(path: Path) -> bool:
    import pyarrow.parquet as pq

    try:
        pq.read_metadata(path)
        return True
    except Exception:          # missing, or cut off by an interrupted write
        return False


def missing_coins() -> list[str]:
    return [s for s in D.UNIVERSE if not _readable(D.KLINE_DIR / f"{s}.parquet")]


def _study_done(study: str, result: str, h: int | None = None) -> bool:
    """The study wrote its results file (last thing it does) and its trials are in the ledger."""
    if not (X.RESULTS_DIR / f"{result}.json").exists():
        return False
    return any(t["config"].get("study") == study and (h is None or t["config"].get("h") == h)
               for t in L.trials(L.LEDGER_DIR))


def engine_artifacts() -> list[str]:
    from .train_engine import CANDIDATES, EVENT_CANDIDATES, INFORMATIONAL

    return ["trend_evidence.json"] + [f"{c['name']}.joblib" for c in CANDIDATES + EVENT_CANDIDATES + INFORMATIONAL]


def _engine_done() -> bool:
    return all((E.ENGINE_DIR / name).exists() for name in engine_artifacts())


def steps(retrain: bool = False) -> list[Step]:
    return [
        Step("download 1-minute history", lambda: ["quant.data", "download", "--symbols", *missing_coins()],
             lambda: not missing_coins(), "16 coins since 2020 (~3 GB)", attempts=DOWNLOAD_ATTEMPTS),
        Step("execution_realism study", lambda: ["quant.studies.execution_realism"],
             lambda: (X.RESULTS_DIR / "execution_realism.json").exists(), "event-rule statistics train_engine reads"),
        Step("intraday_conversion study", lambda: ["quant.studies.intraday_conversion"],
             lambda: _study_done("intraday_conversion", "intraday_conversion"), "trial ledger for the ETH model"),
        Step("daily_panel 1 study", lambda: ["quant.studies.daily_panel", "1"],
             lambda: _study_done("daily_panel", "daily_panel_h1", 1440), "trial ledger for the daily panel model"),
        Step("daily_panel 3 study", lambda: ["quant.studies.daily_panel", "3"],
             lambda: _study_done("daily_panel", "daily_panel_h3", 4320), "trial ledger for the daily panel model"),
        Step("train the engine", lambda: ["quant.train_engine"], _engine_done,
             "holdout evaluation, writes models/engine/", force=retrain),
    ]


# ------------------------------------------------------------------ running

def _log_path() -> Path:
    return D.ROOT / "data" / "logs" / "build_engine.log"


class Tee:
    """Print and append to the build log."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()


class BuildLock:
    """One build at a time, also across Docker containers that share ./data.

    Process ids mean nothing across containers, so the holder touches the lock file
    every minute and a lock untouched for 5 minutes counts as left over from a killed build.
    """

    STALE_S = 300
    BEAT_S = 60

    def __init__(self, path: Path):
        self.path = path
        self._stop = threading.Event()

    def acquire(self) -> bool:
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime < self.STALE_S:
                        return False
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            os.write(fd, f"{socket.gethostname()} pid {os.getpid()}\n".encode())
            os.close(fd)
            threading.Thread(target=self._beat, daemon=True).start()
            return True
        return False

    def describe(self) -> str:
        try:
            holder = self.path.read_text().strip()
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return "unknown holder"
        return f"{holder}, last heartbeat {age:.0f} s ago"

    def _beat(self) -> None:
        while not self._stop.wait(self.BEAT_S):
            try:
                os.utime(self.path)
            except OSError:
                pass

    def release(self) -> None:
        self._stop.set()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def run_module(args: list[str], out: Callable[[str], None]) -> int:
    """``python -m <args>`` in a child process, streaming its output."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, "-m", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env, cwd=D.ROOT)
    for line in proc.stdout:
        out("    " + line.rstrip("\n"))
    return proc.wait()


def memory_gb() -> float | None:
    """Memory available to this process: the container limit if there is one, else the machine's."""
    total = None
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, ValueError, OSError):
        pass
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit.isdigit():
            total = min(total or float("inf"), int(limit) / 1e9)
    except OSError:
        pass
    return total


def preflight(out: Callable[[str], None], need_download: bool) -> None:
    mem = memory_gb()
    if mem is not None and mem < MIN_MEMORY_GB:
        out(f"!! Only {mem:.1f} GB of memory is available; the studies and training need about 8 GB (give Docker 10 GB) "
            "and may be killed.")
        out(MEMORY_ADVICE)
    if need_download:
        D.KLINE_DIR.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(D.KLINE_DIR).free / 1e9
        if free < MIN_FREE_DISK_GB:
            out(f"!! Only {free:.1f} GB of free disk space; the history needs about 3 GB plus about 1 GB of caches.")


def to_do(plan: list[Step]) -> list[bool]:
    """Which steps a build would run: not done yet, forced, or after a step that reruns."""
    out, ran = [], False
    for step in plan:
        run = step.force or ran or not step.done()
        ran = ran or run
        out.append(run)
    return out


def build(plan: list[Step], out: Callable[[str], None], runner=run_module) -> int:
    t_all = time.time()
    ran = False                 # once a step reruns, everything after it depends on new inputs
    for i, step in enumerate(plan, 1):
        head = f"[{i}/{len(plan)}] {step.name}"
        if not (step.force or ran) and step.done():
            out(f"{head}: already done, skipping")
            continue
        ok = False
        for attempt in range(1, step.attempts + 1):
            if attempt > 1:
                out(f"{head}: waiting {RETRY_WAIT_S * (attempt - 1)} s before retrying")
                time.sleep(RETRY_WAIT_S * (attempt - 1))
            args = step.args()
            note = f" (attempt {attempt}/{step.attempts})" if step.attempts > 1 else ""
            out(f"{head}{note}: python -m {' '.join(args)}")
            t0 = time.time()
            rc = runner(args, out)
            out(f"{head}: exit code {rc} after {(time.time() - t0) / 60:.1f} min")
            if rc < 0 or rc == 137:
                out(f"!! The process was killed (signal {-rc if rc < 0 else 9}), most likely because it ran out of memory.")
                out(MEMORY_ADVICE)
            ok = rc == 0 and step.done()
            if ok:
                break
        ran = True
        if not ok:
            out(f"!! Step failed: {step.name}. Fix the problem above, then rerun this command: finished steps are skipped.")
            if step.name.startswith("download"):
                out(f"   Still missing: {' '.join(missing_coins())}. Check the network (data-api.binance.vision) "
                    "and that the data/ folder is writable (on Linux: HOST_UID/HOST_GID in .env).")
            return 1
    out(f"Engine built in {(time.time() - t_all) / 3600:.1f} h.")
    summarize(out)
    return 0


def refresh_history(out: Callable[[str], None], runner=run_module) -> int:
    """Top every coin up to today and drop the feature caches, which would otherwise still end
    at the old date. The research studies use data before the holdout only, so they stay valid."""
    out("updating every coin's 1-minute history to today: python -m quant.data download")
    if runner(["quant.data", "download"], out) != 0:
        out("!! Updating the history failed (see above). Rerun the same command to retry.")
        return 1
    try:                                   # the daily-panel model's sentiment input, also stale otherwise
        D.fetch_fng()
    except Exception as exc:
        out(f"!! Updating the Fear & Greed index failed ({exc}). Check the network (api.alternative.me) and rerun.")
        return 1
    out(f"updated {D.FNG_PATH}")
    shutil.rmtree(X.FEATURE_DIR, ignore_errors=True)
    out(f"cleared the feature caches in {X.FEATURE_DIR}")
    return 0


def summarize(out: Callable[[str], None]) -> None:
    try:
        specs = E.load_specialists(E.ENGINE_DIR)
    except Exception as exc:
        out(f"(could not read the artifacts back: {exc})")
        return
    for sp in specs:
        out(f"  {sp.name}: {sp.evidence.status}")
    out("The dashboard's engine card loads the new engine on its next refresh (every 5 minutes, or reload the page).")
    out("If you stopped the dashboard for memory: docker compose up -d")
    if all(sp.evidence.status != "VALIDATED" for sp in specs):
        out("No trading signal is VALIDATED, so the card will show NO TRADE for every coin. A RISK_OVERLAY "
            "specialist only shows its exposure as position-size guidance, never a trade.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="list the steps and which are already done")
    p.add_argument("--retrain", action="store_true", help="rerun quant.train_engine even if models/engine exists")
    p.add_argument("--refresh", action="store_true",
                   help="update every coin's history to today, then retrain (the research studies are unaffected)")
    args = p.parse_args(argv)
    from .train_engine import recover

    recover(E.ENGINE_DIR)                  # an engine swap that was interrupted: put the previous engine back
    plan = steps(retrain=args.retrain or args.refresh)
    if args.dry_run:
        if args.refresh:
            print("first: update every coin's 1-minute history and the Fear & Greed index, clear the feature caches")
        for i, (step, run) in enumerate(zip(plan, to_do(plan)), 1):
            print(f"[{i}/{len(plan)}] {'to do' if run else 'done '}  {step.name}: {step.why}")
        return 0
    try:
        out = Tee(_log_path())
    except OSError as exc:
        print(f"!! Cannot write to {_log_path().parent}: {exc}")
        print("   The data/ folder must be writable. Docker on Linux: set HOST_UID and HOST_GID in .env to your "
              "`id -u` and `id -g` (0 and 0 for rootless Docker), then docker compose build.")
        return 1
    lock = BuildLock(_log_path().with_name("build_engine.lock"))
    if not lock.acquire():
        out(f"!! Another build is already running: {lock.describe()}. Follow it in {_log_path()}.")
        out(f"   A build that was killed (e.g. docker stop) leaves this lock behind; it expires "
            f"{BuildLock.STALE_S // 60} minutes after its last heartbeat. If `docker ps` shows no build, "
            f"you can delete {lock.path}.")
        return 1
    rc = 1
    try:
        out(f"=== build_engine started {time.strftime('%Y-%m-%d %H:%M:%S')} (log: {_log_path()})")
        preflight(out, need_download=bool(missing_coins()))
        rc = 1 if args.refresh and refresh_history(out) != 0 else build(plan, out)
        return rc
    finally:
        lock.release()
        out(f"=== build_engine finished: {'OK' if rc == 0 else 'FAILED'} {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    raise SystemExit(main())
