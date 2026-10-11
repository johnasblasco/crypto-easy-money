"""One-command engine build, cold-start data, and the dashboard picking up a newly built engine."""
import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quant import build_engine as B
from quant import data as D
from quant import engine as E
from quant import experiment as X
from quant import ledger as L
from quant import live


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """An empty data/ and models/ layout."""
    monkeypatch.setattr(B, "RETRY_WAIT_S", 0)
    monkeypatch.setattr(D, "KLINE_DIR", tmp_path / "klines_1m")
    monkeypatch.setattr(X, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(L, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(E, "ENGINE_DIR", tmp_path / "engine")
    return tmp_path


def fake_runner(fail_coins=(), fail_times=0, rc_override=None, rc_for=None):
    """Stands in for `python -m ...`: creates each step's outputs, records the calls."""
    calls, failures = [], {"left": fail_times}

    def run(args, out):
        calls.append(args)
        mod = args[0]
        if mod == "quant.data":
            D.KLINE_DIR.mkdir(parents=True, exist_ok=True)
            for s in args[3:]:
                if s in fail_coins and failures["left"] > 0:
                    continue
                pd.DataFrame({"open_time": [0]}).to_parquet(D.KLINE_DIR / f"{s}.parquet")
            failures["left"] -= 1
            return 1 if B.missing_coins() else 0
        if mod == "quant.studies.execution_realism":
            X.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            (X.RESULTS_DIR / "execution_realism.json").write_text("{}")
        elif mod in ("quant.studies.intraday_conversion", "quant.studies.daily_panel"):
            study = mod.rsplit(".", 1)[1]
            h = {"1": 1440, "3": 4320}.get(args[1] if len(args) > 1 else "", 240)
            name = study if study == "intraday_conversion" else f"daily_panel_h{args[1]}"
            X.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            (X.RESULTS_DIR / f"{name}.json").write_text("{}")
            L.record({"study": study, "h": h}, {}, pd.Series([0.0]), ledger_dir=L.LEDGER_DIR)
        elif mod == "quant.train_engine":
            E.ENGINE_DIR.mkdir(parents=True, exist_ok=True)
            for name in B.engine_artifacts():
                (E.ENGINE_DIR / name).touch()
        if rc_for and mod in rc_for:
            return rc_for[mod]
        return 0 if rc_override is None else rc_override

    return run, calls


def test_build_runs_every_step_in_order_then_resumes(layout):
    run, calls = fake_runner()
    log = []
    assert B.build(B.steps(), log.append, runner=run) == 0
    assert [c[0] for c in calls] == ["quant.data", "quant.studies.execution_realism",
                                     "quant.studies.intraday_conversion", "quant.studies.daily_panel",
                                     "quant.studies.daily_panel", "quant.train_engine"]
    assert calls[0][3:] == D.UNIVERSE
    assert calls[3][1:] == ["1"] and calls[4][1:] == ["3"]

    run2, calls2 = fake_runner()
    assert B.build(B.steps(), log.append, runner=run2) == 0
    assert calls2 == []                                   # everything already done


def test_download_retries_only_missing_coins(layout):
    run, calls = fake_runner(fail_coins={"SOLUSDT"}, fail_times=1)
    assert B.build(B.steps(), [].append, runner=run) == 0
    downloads = [c for c in calls if c[0] == "quant.data"]
    assert len(downloads) == 2 and downloads[1][3:] == ["SOLUSDT"]


def test_download_that_keeps_failing_stops_the_build(layout):
    run, calls = fake_runner(fail_coins={"ETHUSDT"}, fail_times=99)
    log = []
    assert B.build(B.steps(), log.append, runner=run) == 1
    assert len(calls) == B.DOWNLOAD_ATTEMPTS                 # never reaches the studies
    assert any("Still missing: ETHUSDT" in line for line in log)


def test_failed_step_is_not_trusted_even_if_old_outputs_exist(layout):
    run, _ = fake_runner()
    assert B.build(B.steps(), [].append, runner=run) == 0
    crash, calls = fake_runner(rc_override=1)
    assert B.build(B.steps(retrain=True), [].append, runner=crash) == 1    # stale artifacts don't count
    assert [c[0] for c in calls] == ["quant.train_engine"]


def test_retrain_reruns_only_the_engine(layout):
    run, _ = fake_runner()
    B.build(B.steps(), [].append, runner=run)
    run2, calls = fake_runner()
    assert B.build(B.steps(retrain=True), [].append, runner=run2) == 0
    assert calls == [["quant.train_engine"]]


def test_a_rerun_step_reruns_everything_after_it(layout):
    run, _ = fake_runner()
    B.build(B.steps(), [].append, runner=run)
    (X.RESULTS_DIR / "daily_panel_h1.json").unlink()          # e.g. an older setup that never ran it
    run2, calls = fake_runner()
    assert B.build(B.steps(), [].append, runner=run2) == 0
    assert calls == [["quant.studies.daily_panel", "1"], ["quant.studies.daily_panel", "3"], ["quant.train_engine"]]


def test_cut_off_history_file_counts_as_missing(layout):
    D.KLINE_DIR.mkdir(parents=True)
    for s in D.UNIVERSE:
        pd.DataFrame({"open_time": [0]}).to_parquet(D.KLINE_DIR / f"{s}.parquet")
    assert B.missing_coins() == []
    (D.KLINE_DIR / "SOLUSDT.parquet").write_bytes(b"PAR1 cut off")
    assert B.missing_coins() == ["SOLUSDT"]


def test_killed_step_explains_memory(layout):
    run, _ = fake_runner(rc_for={"quant.studies.execution_realism": -9})
    log = []
    assert B.build(B.steps(), log.append, runner=run) == 1
    assert any("ran out of memory" in line for line in log)
    assert any("docker compose stop web" in line for line in log)


def test_only_one_build_at_a_time(tmp_path):
    first, second = B.BuildLock(tmp_path / "lock"), B.BuildLock(tmp_path / "lock")
    assert first.acquire() and not second.acquire()
    first.release()
    assert second.acquire()
    second.release()
    stale = tmp_path / "lock"
    stale.write_text("other-container pid 7")
    old = time.time() - B.BuildLock.STALE_S - 10
    os.utime(stale, (old, old))                                 # left over from a killed build
    assert first.acquire()
    first.release()


def test_unwritable_data_folder_gives_guidance(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "data"
    blocker.write_text("a file where the folder should be")
    monkeypatch.setattr(B, "_log_path", lambda: blocker / "logs" / "build_engine.log")
    assert B.main([]) == 1
    assert "HOST_UID" in capsys.readouterr().out


def test_ledger_study_needs_its_trials_not_just_the_json(layout):
    X.RESULTS_DIR.mkdir(parents=True)
    (X.RESULTS_DIR / "daily_panel_h3.json").write_text("{}")
    assert not B._study_done("daily_panel", "daily_panel_h3", 4320)
    L.record({"study": "daily_panel", "h": 1440}, {}, pd.Series([0.0]), ledger_dir=L.LEDGER_DIR)
    assert not B._study_done("daily_panel", "daily_panel_h3", 4320)      # wrong horizon
    L.record({"study": "daily_panel", "h": 4320}, {}, pd.Series([0.0]), ledger_dir=L.LEDGER_DIR)
    assert B._study_done("daily_panel", "daily_panel_h3", 4320)


def test_run_module_streams_child_output():
    lines = []
    assert B.run_module(["json.tool", "--help"], lines.append) == 0
    assert any("json" in line for line in lines)


def test_download_replaces_a_cut_off_file_and_writes_atomically(tmp_path, monkeypatch):
    path = tmp_path / "XUSDT.parquet"
    path.write_bytes(b"PAR1 cut off")
    start = pd.Timestamp("2026-01-01", tz="UTC")

    def chunk(session, symbol, s):
        return [[t, "1", "1", "1", "1", "1", t + 59_999, "1", 1, "1", "1", "0"]
                for t in range(s, s + 1000 * D.MINUTE_MS, D.MINUTE_MS)]

    monkeypatch.setattr(D, "_first_available", lambda session, symbol, s: s)
    monkeypatch.setattr(D, "_fetch_chunk", chunk)
    D.download("XUSDT", start=str(start.date()), end="2026-01-02", out_dir=tmp_path, verbose=False)
    assert len(pd.read_parquet(path)) == 1440
    assert sorted(p.name for p in tmp_path.iterdir()) == ["XUSDT.parquet", "XUSDT.parquet.corrupt"]


def test_download_cli_reports_failures(monkeypatch, capsys):
    def fake(sym, *a, **k):
        if sym == "ETHUSDT":
            raise PermissionError("data/ is not writable")

    monkeypatch.setattr(D, "download", fake)
    assert D.main(["download", "--symbols", "BTCUSDT", "ETHUSDT"]) == 1
    assert "1 of 2 downloads failed: ETHUSDT" in capsys.readouterr().out
    assert D.main(["download", "--symbols", "BTCUSDT"]) == 0


# ------------------------------------------------------------ live data cold start

class FakeSession:
    """Binance klines: startTime -> the next 1000 bars; only endTime -> the 1000 bars before it."""

    def __init__(self, listed_days_ago=10_000):
        self.params = []
        now = pd.Timestamp.now(tz="UTC").floor("min")
        self.listed = int((now - pd.Timedelta(days=listed_days_ago)).timestamp() * 1000)

    def get(self, url, params, timeout):
        self.params.append(params)
        end = params["endTime"]
        if "startTime" in params:
            start = max(params["startTime"], self.listed)
            times = range(start, min(end + 1, start + 1000 * D.MINUTE_MS), D.MINUTE_MS)
        else:
            start = max(end - 999 * D.MINUTE_MS, self.listed)
            times = range(start, end + 1, D.MINUTE_MS) if end >= self.listed else range(0)
        rows = [[t, "1.0", "1.1", "0.9", "1.05", "10", t + 59_999, "10.5", 5, "4", "4.2", "0"] for t in times]

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return rows

        return R()


def test_coin_without_history_is_fresh_at_once_then_fills_in(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "KLINE_DIR", tmp_path)
    session = FakeSession()
    store = live.LiveStore(session=session, days=12)
    store.sync("NEWCOINUSDT")
    first = session.params[0]["startTime"]
    days_back = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(first, unit="ms", tz="UTC")).total_seconds() / 86400
    assert days_back == pytest.approx(live.COLD_START_DAYS, abs=0.01)     # not 400 days crawled forward
    m1 = store.frame("NEWCOINUSDT")
    assert m1["close"].dtype == np.float64
    assert np.isfinite(np.log(m1["close"]).ffill().iloc[-1])          # crashed with object dtype before
    assert live.data_health(m1)["ok"]
    # The leftover request budget already fetched the older bars, up to the store's window.
    assert "NEWCOINUSDT" not in store.backfill
    span = store.frame("NEWCOINUSDT").index[-1] - store.frame("NEWCOINUSDT").index[0]
    assert span.days == 11 and not store.frame("NEWCOINUSDT")["gap"].any()


def test_backfill_spreads_over_syncs_and_stops_at_the_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "KLINE_DIR", tmp_path)
    store = live.LiveStore(session=FakeSession(listed_days_ago=30))
    store.sync("NEWCOINUSDT", max_requests=15)            # 12 forward + 3 older pages
    assert "NEWCOINUSDT" in store.backfill
    for _ in range(10):
        store.sync("NEWCOINUSDT", max_requests=15)
    assert "NEWCOINUSDT" not in store.backfill            # Binance returned nothing before the listing
    first = store.frame("NEWCOINUSDT").index[0]
    assert (pd.Timestamp.now(tz="UTC") - first).days == 29


def test_coin_with_no_recent_bars_gives_a_readable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "KLINE_DIR", tmp_path)
    store = live.LiveStore(session=FakeSession(listed_days_ago=-1))   # nothing traded yet
    store.sync("HALTEDUSDT")
    with pytest.raises(ValueError, match="no closed 1-minute bars"):
        store.frame("HALTEDUSDT")


def test_event_rule_says_when_history_is_too_short():
    spec = E.EventSpecialist({"kind": "event", "name": "capitulation", "W": 60, "k": 4.0, "h": 60,
                              "cost": "spot_taker", "symbols": ["SOLUSDT"],
                              "evidence": {"status": "NOT_VALIDATED", "summary": "t"}})
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min"), periods=8 * 1440, freq="1min")
    m1 = pd.DataFrame({"close": 1.0}, index=idx)
    sig = spec.evaluate("SOLUSDT", m1, idx[-1])
    assert sig.action == "NO TRADE"
    assert "days of 1-minute history" in sig.reasons[0]


def test_backfill_pass_shares_one_budget_and_comes_before_the_forward_sync(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "KLINE_DIR", tmp_path)
    session = FakeSession()
    store = live.LiveStore(session=session)
    for coin in ("AAAUSDT", "BBBUSDT"):
        store.sync(coin, backfill=False)                 # forward only: fresh, still owed older history
    assert store.backfill == {"AAAUSDT", "BBBUSDT"}
    n = len(session.params)
    store.backfill_pass(budget=5)
    assert len(session.params) - n == 5                  # one budget for all such coins per run

    calls = []

    class Recorder:
        backfill = set()

        def backfill_pass(self):
            calls.append("backfill")

        def sync(self, symbol, backfill=True):
            calls.append(("sync", symbol, backfill))

        def frame(self, symbol):
            return store.frame("AAAUSDT")

    E.Engine(["AAAUSDT", "BBBUSDT"], [], store=Recorder(), use_book=False).run()
    assert calls == ["backfill", ("sync", "AAAUSDT", False), ("sync", "BBBUSDT", False)]


def test_dry_run_shows_the_rerun_cascade(layout):
    run, _ = fake_runner()
    B.build(B.steps(), [].append, runner=run)
    assert B.to_do(B.steps()) == [False] * 6
    assert B.to_do(B.steps(retrain=True)) == [False] * 5 + [True]
    (X.RESULTS_DIR / "daily_panel_h1.json").unlink()
    assert B.to_do(B.steps()) == [False, False, False, True, True, True]


def test_refresh_updates_fear_and_greed_and_clears_caches(layout, monkeypatch):
    monkeypatch.setattr(X, "FEATURE_DIR", layout / "features")
    X.FEATURE_DIR.mkdir()
    fetched = []
    monkeypatch.setattr(D, "fetch_fng", lambda: fetched.append(1))
    assert B.refresh_history([].append, runner=lambda args, out: 0) == 0
    assert fetched and not X.FEATURE_DIR.exists()

    def offline():
        raise OSError("no network")

    monkeypatch.setattr(D, "fetch_fng", offline)
    log = []
    assert B.refresh_history(log.append, runner=lambda args, out: 0) == 1
    assert any("Fear & Greed" in line for line in log)


def test_build_always_ends_with_a_finished_line(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "_log_path", lambda: tmp_path / "logs" / "build_engine.log")
    monkeypatch.setattr(B, "preflight", lambda out, need_download: None)
    monkeypatch.setattr(B, "steps", lambda retrain=False: [])
    monkeypatch.setattr(B, "summarize", lambda out: None)
    assert B.main([]) == 0
    assert (tmp_path / "logs" / "build_engine.log").read_text().splitlines()[-1].startswith(
        "=== build_engine finished: OK")
    monkeypatch.setattr(B, "build", lambda plan, out: 1)
    assert B.main([]) == 1
    assert "finished: FAILED" in (tmp_path / "logs" / "build_engine.log").read_text().splitlines()[-1]


def test_failed_engine_swap_puts_the_previous_engine_back(tmp_path, monkeypatch):
    from quant import train_engine as T

    target, staging = tmp_path / "engine", tmp_path / "engine.new"
    for d, text in ((target, "old engine"), (staging, "new engine")):
        d.mkdir()
        (d / "trend_evidence.json").write_text(text)
    real, calls = T._replace, []

    def flaky(src, dst, tries=10):
        calls.append(src.name)
        if len(calls) == 2:
            raise PermissionError("held open by antivirus")
        real(src, dst, tries)

    monkeypatch.setattr(T, "_replace", flaky)
    with pytest.raises(PermissionError):
        T.publish(staging, target)
    assert (target / "trend_evidence.json").read_text() == "old engine"     # rolled back
    # A swap killed between its two renames: the next run restores the previous engine first.
    target.rename(tmp_path / "engine.old")
    monkeypatch.setattr(T, "_replace", real)
    T.recover(target)
    assert (target / "trend_evidence.json").read_text() == "old engine"


# ------------------------------------------------------------ dashboard reload

def write_trend(engine_dir, summary):
    engine_dir.mkdir(parents=True, exist_ok=True)
    ev = E.Evidence(status="RISK_OVERLAY", summary=summary, universe=["BTCUSDT"])
    (engine_dir / "trend_evidence.json").write_text(json.dumps(asdict(ev)))


def test_dashboard_picks_up_a_newly_built_engine(layout, monkeypatch):
    from app import server

    monkeypatch.setattr(E.Engine, "run", lambda self, now=None, sync=True: [])
    monkeypatch.setattr(server, "_engine_state", {"engine": None, "at": 0.0, "payload": None, "files": None})
    client = TestClient(server.app)

    resp = client.get("/api/engine")
    assert resp.status_code == 503 and "quant.build_engine" in resp.json()["detail"]

    write_trend(E.ENGINE_DIR, "first build")
    body = client.get("/api/engine").json()
    assert [s["evidence"]["summary"] for s in body["specialists"]] == ["first build"]
    store = server._engine_state["engine"].store

    write_trend(E.ENGINE_DIR, "retrained with a longer holdout")     # within the 5-minute cache window
    body = client.get("/api/engine").json()
    assert [s["evidence"]["summary"] for s in body["specialists"]] == ["retrained with a longer holdout"]
    assert server._engine_state["engine"].store is not store          # re-seeds from fresh history files


def test_blank_engine_symbols_means_the_default_coins(layout, monkeypatch):
    from app import server

    monkeypatch.setenv("ENGINE_SYMBOLS", "")
    monkeypatch.setattr(server, "_engine_state", {"engine": None, "at": 0.0, "payload": None, "files": None})
    write_trend(E.ENGINE_DIR, "x")
    eng = server._get_engine(server._engine_files())
    assert eng.symbols == D.UNIVERSE


def test_train_engine_swaps_in_a_finished_build_only(tmp_path, monkeypatch):
    from quant import train_engine as T

    engine_dir = tmp_path / "models" / "engine"
    monkeypatch.setattr(T, "ENGINE_DIR", engine_dir)
    monkeypatch.setattr(T, "trend_evidence", lambda: E.Evidence(status="RISK_OVERLAY", summary="trend"))
    monkeypatch.setattr(T, "CANDIDATES", [{"name": "model_a"}])
    monkeypatch.setattr(T, "EVENT_CANDIDATES", [])
    monkeypatch.setattr(T, "INFORMATIONAL", [{"name": "view_b", "research_summary": "r"}])
    monkeypatch.setattr(T, "evaluate_candidate", lambda c: {"artifact": {"name": c["name"], "v": 1},
                                                            "evidence": {"status": "NOT_VALIDATED", "summary": "s"}})
    monkeypatch.setattr(T, "informational_model", lambda c, summary: {"name": c["name"]})
    T.main()
    assert sorted(p.name for p in engine_dir.iterdir()) == ["model_a.joblib", "trend_evidence.json", "view_b.joblib"]
    assert sorted(p.name for p in engine_dir.parent.iterdir()) == ["engine"]       # no staging leftovers
    before = {p.name: p.read_bytes() for p in engine_dir.iterdir()}

    def crash(c):
        raise MemoryError("killed halfway")

    monkeypatch.setattr(T, "evaluate_candidate", crash)
    with pytest.raises(MemoryError):
        T.main()                                                     # trend evidence was already rewritten...
    assert {p.name: p.read_bytes() for p in engine_dir.iterdir()} == before   # ...but only in staging
