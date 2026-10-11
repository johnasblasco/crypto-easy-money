"""Autopilot: pretend accounts that copy the signals, and the daily trend guide."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cryptopredict import autopilot as A
from cryptopredict import trend_guide as TG

NOW = pd.Timestamp("2026-10-11 06:00", tz="UTC")
CFG = {"start_cash": 1000.0, "trade_share": 0.10, "report_hour": 1, "trade_alerts": True,
       "coins": ["BTCUSDT", "ETHUSDT"]}


def daily(trend: float, days: int = 300, end: pd.Timestamp = NOW.floor("D")):
    """Completed daily bars indexed by close time, drifting up (trend > 0) or down."""
    rng = np.random.default_rng(1)
    idx = pd.date_range(end=end, periods=days, freq="1D", tz="UTC")
    close = 100 * np.exp(np.cumsum(trend + rng.normal(0, 0.01, days)))
    return pd.DataFrame({"close": close}, index=idx)


def row(symbol="SOLUSDT", conf=0.62, thr=0.58, direction="UP", signal_time=NOW - pd.Timedelta("20min"),
        hold="4 hours", interval="1h", horizon=4, verdict="NO EDGE", price=150.0):
    return {"symbol": symbol, "interval": interval, "horizon": horizon, "verdict": verdict, "status": "NOT VALIDATED",
            "direction": direction, "confidence": conf, "threshold": thr, "price": price, "hold": hold,
            "signal_time": signal_time.isoformat(), "exit_time": (signal_time + pd.Timedelta(hold)).isoformat()}


# ------------------------------------------------------------ trend guide

def test_trend_guide_holds_rising_coins_and_drops_falling_ones():
    up = TG.coin_guide("BTCUSDT", daily(0.004)["close"])
    down = TG.coin_guide("ETHUSDT", daily(-0.004)["close"])
    assert up["trend"] == "UP" and 0 < up["exposure"] <= 1
    assert down["trend"] == "DOWN" and down["exposure"] == 0
    short = TG.coin_guide("BTCUSDT", daily(0.004, days=30)["close"])
    assert short["exposure"] is None and "history" in short["reason"]


def test_guide_refuses_untested_coins_and_survives_errors():
    def fetch(sym):
        if sym == "ETHUSDT":
            raise OSError("offline")
        return daily(0.004)

    g = TG.guide(["BTCUSDT", "ETHUSDT", "FAKECOINUSDT"], fetch, NOW)
    assert [c["trend"] for c in g] == ["UP", "ERROR", "UNTESTED"]


def test_trend_change_messages():
    old = [{"symbol": "BTCUSDT", "exposure": 0.0}, {"symbol": "ETHUSDT", "exposure": 0.4},
           {"symbol": "SOLUSDT", "exposure": 0.2}]
    new = [{"symbol": "BTCUSDT", "exposure": 0.45}, {"symbol": "ETHUSDT", "exposure": 0.0},
           {"symbol": "SOLUSDT", "exposure": 0.5}]
    msgs = TG.changes(old, new)
    assert "BTC trend turned UP: keep 45%" in msgs[0]
    assert "ETH trend turned DOWN" in msgs[1]
    assert "SOL: keep 50%" in msgs[2] and "was 20%" in msgs[2]
    assert TG.changes(None, new) == []                     # first guide: nothing to compare


# ------------------------------------------------------------ copying the model calls

def models_state():
    return A.new_state(NOW, CFG)


def test_copies_a_confident_up_call_once_and_sells_after_the_hold():
    st = models_state()
    costs = {"SOLUSDT_1h_h4": 0.001}
    ev = A.step_models(st, [row()], {"SOLUSDT": 150.0}, costs, NOW, 0.10)
    acct = st["accounts"]["models"]
    assert [e["kind"] for e in ev] == ["buy"]
    pos = acct["positions"][0]
    assert pos["invested"] == pytest.approx(100) and acct["cash"] == pytest.approx(900)
    assert pos["qty"] == pytest.approx(100 * 0.999 / 150)
    # Same signal on the next update: not bought twice.
    assert A.step_models(st, [row()], {"SOLUSDT": 151.0}, costs, NOW + pd.Timedelta("15min"), 0.10) == []
    # After the hold time: sold at the latest price, fees on both sides.
    later = NOW + pd.Timedelta("4h")
    ev = A.step_models(st, [], {"SOLUSDT": 160.0}, costs, later, 0.10)
    t = ev[0]["trade"]
    assert ev[0]["kind"] == "sell" and acct["positions"] == []
    assert t["proceeds"] == pytest.approx(pos["qty"] * 160 * 0.999)
    assert t["pnl"] == pytest.approx(t["proceeds"] - 100)
    assert acct["cash"] == pytest.approx(900 + t["proceeds"])


@pytest.mark.parametrize("bad", [
    row(conf=0.55),                                            # below its own threshold
    row(direction="DOWN"),                                     # spot account: long only
    row(signal_time=NOW - pd.Timedelta("3h")),                 # stale: older than one candle
    row(symbol="PEPEUSDT"),                                    # not a Binance-trained model
])
def test_ignores_calls_it_should_not_copy(bad):
    st = models_state()
    assert A.step_models(st, [bad], {"SOLUSDT": 150.0, "PEPEUSDT": 1e-5}, {"SOLUSDT_1h_h4": 0.001}, NOW, 0.1) == []


def test_never_spends_more_than_it_has():
    st = models_state()
    rows = [row(symbol=f"C{i}USDT") for i in range(12)]
    costs = {f"C{i}USDT_1h_h4": 0.001 for i in range(12)}
    prices = {f"C{i}USDT": 1.0 for i in range(12)}
    ev = A.step_models(st, rows, prices, costs, NOW, 0.10)
    assert sum(e["kind"] == "buy" for e in ev) == 10 and st["accounts"]["models"]["cash"] == pytest.approx(0)
    assert sum(e["kind"] == "skip" for e in ev) == 2


# ------------------------------------------------------------ trend and hold accounts

def test_trend_account_holds_exposure_share_and_rebalances_once_a_day():
    st = models_state()
    st["guide"] = [{"symbol": "BTCUSDT", "exposure": 0.5}, {"symbol": "ETHUSDT", "exposure": 0.0}]
    st["guide_day"] = NOW.floor("D").isoformat()
    prices = {"BTCUSDT": 100.0, "ETHUSDT": 10.0}
    ev = A.step_trend(st, prices, NOW)
    acct = st["accounts"]["trend"]
    assert [e["symbol"] for e in ev] == ["BTCUSDT"]               # 1000 x 0.5 / 2 coins = 250 in BTC
    assert acct["holdings"]["BTCUSDT"] * 100 == pytest.approx(250 * (1 - A.TREND_FEE))
    assert A.step_trend(st, prices, NOW + pd.Timedelta("1h")) == []      # same day: no new trades
    st["guide"] = [{"symbol": "BTCUSDT", "exposure": 0.0}, {"symbol": "ETHUSDT", "exposure": 0.0}]
    st["guide_day"] = (NOW.floor("D") + pd.Timedelta("1D")).isoformat()
    ev = A.step_trend(st, prices, NOW + pd.Timedelta("1D"))
    assert ev[0]["side"] == "sell" and acct["holdings"] == {}


def test_hold_account_buys_once():
    st = models_state()
    A.step_hold(st, {"BTCUSDT": 100.0, "ETHUSDT": 10.0})
    h = st["accounts"]["hold"]["holdings"]
    assert h["BTCUSDT"] * 100 == pytest.approx(500 * (1 - A.TREND_FEE))
    A.step_hold(st, {"BTCUSDT": 50.0, "ETHUSDT": 5.0})
    assert st["accounts"]["hold"]["holdings"] == h


# ------------------------------------------------------------ a full update

def run(tmp_path, now, rows=(), sent=None, prices=None):
    return A.run_once(now=now, path=tmp_path / "state.json", scan_fn=lambda: list(rows),
                      daily_fn=lambda s: daily(0.004 if s == "BTCUSDT" else -0.004, end=now.floor("D")),
                      prices_fn=lambda syms: prices or {"BTCUSDT": 100.0, "ETHUSDT": 10.0, "SOLUSDT": 150.0},
                      costs_fn=lambda: {"SOLUSDT_1h_h4": 0.001},
                      notify=(sent.append if sent is not None else (lambda t: None)), cfg=CFG)


def test_run_once_updates_everything_and_reports_daily(tmp_path):
    sent = []
    st = run(tmp_path, NOW, rows=[row()], sent=sent)
    assert len(st["accounts"]["models"]["positions"]) == 1
    assert st["accounts"]["hold"]["holdings"] and st["accounts"]["trend"]["holdings"].keys() == {"BTCUSDT"}
    assert any(m.startswith("🟢 Pretend BUY SOLUSDT") for m in sent)
    assert sum("Autopilot report" in m for m in sent) == 1
    sent.clear()
    st = run(tmp_path, NOW + pd.Timedelta("15min"), rows=[row()], sent=sent)
    assert len(st["accounts"]["models"]["positions"]) == 1 and sent == []   # nothing new, no second report
    s = A.status(tmp_path / "state.json", now=NOW + pd.Timedelta("20min"))
    assert s["running"] and len(s["accounts"]) == 3 and s["guide_day"] == NOW.floor("D").isoformat()
    assert not A.status(tmp_path / "state.json", now=NOW + pd.Timedelta("2h"))["running"]


def test_alert_failures_never_leak_the_token(tmp_path):
    import requests

    def broken(text):
        raise requests.ConnectionError("https://api.telegram.org/botSECRET123/sendMessage failed")

    st = A.run_once(now=NOW, path=tmp_path / "s.json", scan_fn=lambda: [], daily_fn=lambda s: daily(0.004),
                    prices_fn=lambda syms: {"BTCUSDT": 100.0, "ETHUSDT": 10.0}, costs_fn=dict, notify=broken, cfg=CFG)
    assert st["log"] and not any("SECRET123" in line for line in st["log"])


def test_go_live_checks_start_unmet(tmp_path):
    st = run(tmp_path, NOW)
    checks = A.go_live_checks(st, NOW)
    assert len(checks) == 4 and not any(c["ok"] for c in checks)


def test_autopilot_api(tmp_path, monkeypatch):
    from app import server

    monkeypatch.setattr(A, "STATE_PATH", tmp_path / "state.json")
    client = TestClient(server.app)
    assert client.get("/api/autopilot").json() == {"running": False, "started": None}
    run(tmp_path, pd.Timestamp.now(tz="UTC").floor("min"), rows=[])
    body = client.get("/api/autopilot").json()
    assert body["running"] and [a["key"] for a in body["accounts"]] == ["models", "trend", "hold"]
    assert body["guide"][0]["symbol"] == "BTCUSDT"


# ------------------------------------------------------------ catching up after a pause

class CycleModel:
    """predict_proba cycling through fixed P(UP) values, one per row."""

    def __init__(self, probs):
        self.probs = probs

    def predict_proba(self, X):
        p = np.resize(np.asarray(self.probs, float), len(X))
        return np.column_stack([1 - p, p])


def candles(end, n=500, interval="1h"):
    idx = pd.date_range(end=end, periods=n, freq=interval, tz="UTC")       # open times
    return pd.DataFrame({"timestamp": idx, "open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0 + np.arange(n) * 0.1, "volume": 10.0})


def test_replay_finds_the_calls_the_live_scan_would_have_seen():
    until = pd.Timestamp("2026-10-11 06:00", tz="UTC")
    since = until - pd.Timedelta("6h")
    df = candles(until - pd.Timedelta("1h"))              # last candle closes exactly at `until`
    cfg = {"symbol": "SOLUSDT", "interval": "1h", "horizon": 2, "verdict": "NO EDGE",
           "bundle": {"features": ["close"], "threshold": 0.6, "model": CycleModel([0.7, 0.4, 0.55])}}
    calls, closes, failed = A.replay_calls([cfg], since, until, {"SOLUSDT_1h_h2": 0.001}, load=lambda s, i, n: df)
    assert failed == []
    times = [c["signal_time"] for c in calls]
    assert times and all(since < pd.Timestamp(t) <= until for t in times)
    assert all(c["confidence"] >= 0.6 for c in calls)               # 0.55 is UP but below its threshold
    first = calls[0]
    assert pd.Timestamp(first["exit_time"]) - pd.Timestamp(first["signal_time"]) == pd.Timedelta("2h")
    s = closes[("SOLUSDT", "1h")]
    assert s.index[-1] == until and first["price"] == s[pd.Timestamp(first["signal_time"])]
    assert A.replay_calls([cfg], since, until, {}, load=lambda s, i, n: df) == ([], {}, [])   # not a Binance model

    def offline(s, i, n):
        raise OSError("no network")

    assert A.replay_calls([cfg], since, until, {"SOLUSDT_1h_h2": 0.001}, load=offline) == ([], {}, ["SOLUSDT_1h_h2"])


def test_catch_up_sells_at_the_exit_time_price_and_copies_missed_calls():
    st = A.new_state(NOW - pd.Timedelta("1D"), CFG)
    acct = st["accounts"]["models"]
    acct["cash"] = 900.0
    acct["positions"] = [{"config": "ETHUSDT_1h_h4", "symbol": "ETHUSDT", "interval": "1h", "hold": "4 hours",
                          "opened": (NOW - pd.Timedelta("10h")).isoformat(), "entry_price": 10.0, "qty": 10.0,
                          "invested": 100.0, "fee": 0.0, "exit_time": (NOW - pd.Timedelta("6h")).isoformat(),
                          "confidence": 0.6, "verdict": "NO EDGE"}]
    idx = pd.date_range(NOW - pd.Timedelta("12h"), NOW, freq="1h")
    closes = {("ETHUSDT", "1h"): pd.Series(np.arange(len(idx), dtype=float) + 5.0, index=idx),
              ("SOLUSDT", "1h"): pd.Series(np.full(len(idx), 20.0), index=idx)}
    t1, t2, t3 = NOW - pd.Timedelta("5h"), NOW - pd.Timedelta("4h"), NOW - pd.Timedelta("1h")

    def call(t, hold="2h"):
        return {"config": "SOLUSDT_1h_h2", "symbol": "SOLUSDT", "interval": "1h", "hold": "2 hours",
                "signal_time": t.isoformat(), "exit_time": (t + pd.Timedelta(hold)).isoformat(), "price": 20.0,
                "confidence": 0.7, "verdict": "NO EDGE", "fee": 0.0}

    ev = A.catch_up_models(st, [call(t1), call(t2), call(t3)], closes, NOW, 0.10)
    kinds = [(e["kind"], (e.get("trade") or e.get("position"))["symbol"]) for e in ev]
    # ETH sold at the close at its exit time (6 h ago -> 11.0), not at today's price;
    # SOL bought at t1, the t2 call skipped (still holding), sold at t1+2h, bought again at t3 (still open).
    assert kinds == [("sell", "ETHUSDT"), ("buy", "SOLUSDT"), ("sell", "SOLUSDT"), ("buy", "SOLUSDT")]
    assert ev[0]["trade"]["exit_price"] == 11.0 and ev[0]["trade"]["replayed"]
    assert len(acct["positions"]) == 1 and acct["positions"][0]["opened"] == t3.isoformat()
    # The live scan seeing the t3 candle afterwards does not buy it twice.
    live = row(symbol="SOLUSDT", interval="1h", horizon=2, hold="2 hours", signal_time=t3)
    assert A.step_models(st, [live], {"SOLUSDT": 20.0}, {"SOLUSDT_1h_h2": 0.0}, NOW, 0.10) == []


def test_catch_up_rebalances_the_trend_account_for_each_missed_day():
    st = A.new_state(NOW - pd.Timedelta("5D"), CFG)
    st["accounts"]["trend"]["rebalanced_for"] = (NOW.floor("D") - pd.Timedelta("3D")).isoformat()
    data = {"BTCUSDT": daily(0.004), "ETHUSDT": daily(-0.004)}
    n, done = A.catch_up_trend(st, lambda s: data[s], NOW.floor("D"))
    assert n == 2 and done
    missed = {(NOW.floor("D") - pd.Timedelta(f"{k}D")).isoformat() for k in (1, 2)}
    assert st["accounts"]["trend"]["rebalanced_for"] == (NOW.floor("D") - pd.Timedelta("1D")).isoformat()
    trades = st["accounts"]["trend"]["trades"]
    assert {tr["time"] for tr in trades} <= missed and trades[0]["time"] == min(missed)
    btc = [tr for tr in trades if tr["symbol"] == "BTCUSDT"][0]          # bought at that day's close
    assert btc["price"] == pytest.approx(float(data["BTCUSDT"]["close"][pd.Timestamp(btc["time"])]))


def test_run_once_after_a_pause_replays_and_reports_once(tmp_path):
    first = NOW - pd.Timedelta("10h")
    run(tmp_path, first)
    sent = []
    t = NOW - pd.Timedelta("3h")
    missed = [{"config": "SOLUSDT_1h_h4", "symbol": "SOLUSDT", "interval": "1h", "hold": "4 hours",
               "signal_time": t.isoformat(), "exit_time": (t + pd.Timedelta("4h")).isoformat(), "price": 150.0,
               "confidence": 0.7, "verdict": "NO EDGE", "fee": 0.001}]
    st = A.run_once(now=NOW, path=tmp_path / "state.json", scan_fn=lambda: [],
                    daily_fn=lambda s: daily(0.004 if s == "BTCUSDT" else -0.004, end=NOW.floor("D")),
                    prices_fn=lambda syms: {"BTCUSDT": 100.0, "ETHUSDT": 10.0, "SOLUSDT": 150.0},
                    costs_fn=lambda: {"SOLUSDT_1h_h4": 0.001}, notify=sent.append, cfg=CFG,
                    replay_fn=lambda since, until, costs: (missed, {}, []))
    pos = st["accounts"]["models"]["positions"]
    assert len(pos) == 1 and pos[0]["replayed"] and pos[0]["opened"] == t.isoformat()
    assert sum("was off for 10 hours" in m for m in sent) == 1
    assert not any(m.startswith("🟢 Pretend BUY") for m in sent)    # replayed trades: summed up, not spammed


# ------------------------------------------------------------ review findings (regressions)

def market_prices(market):
    """Like Binance: returns only the symbols asked for (and only those it has)."""
    return lambda syms: {s: market[s] for s in syms if s in market}


def run_with(tmp_path, now, market, rows=(), sent=None, replay=None, daily_fn=None):
    return A.run_once(now=now, path=tmp_path / "state.json", scan_fn=lambda: list(rows),
                      daily_fn=daily_fn or (lambda s: daily(0.004 if s == "BTCUSDT" else -0.004, end=now.floor("D"))),
                      prices_fn=market_prices(market), costs_fn=lambda: {"PEPEUSDT_1h_h4": 0.0},
                      notify=(sent.append if sent is not None else (lambda t: None)), cfg=CFG,
                      replay_fn=replay or (lambda since, until, costs: ([], {}, [])))


def test_a_model_coin_outside_the_guide_is_never_bought_at_an_old_price(tmp_path):
    market = {"BTCUSDT": 100.0, "ETHUSDT": 10.0, "PEPEUSDT": 1000.0}
    t0 = NOW
    run_with(tmp_path, t0, market, rows=[row(symbol="PEPEUSDT", signal_time=t0 - pd.Timedelta("5min"), price=1000.0)])
    for k in range(1, 20):                                     # every 15 minutes: sold after 4 h at 1000
        run_with(tmp_path, t0 + pd.Timedelta(minutes=15 * k), market)
    market["PEPEUSDT"] = 800.0                                 # the coin falls while nothing is held
    t1 = t0 + pd.Timedelta(minutes=15 * 20)
    st = run_with(tmp_path, t1, market, rows=[row(symbol="PEPEUSDT", signal_time=t1 - pd.Timedelta("5min"), price=800.0)])
    pos = st["accounts"]["models"]["positions"]
    assert len(pos) == 1 and pos[0]["entry_price"] == 800.0     # not the 1000 left over from the last trade


def test_offline_restart_after_a_pause_trades_nothing_and_retries_the_catch_up(tmp_path):
    market = {"BTCUSDT": 100.0, "ETHUSDT": 10.0, "PEPEUSDT": 150.0}
    run_with(tmp_path, NOW, market, rows=[row(symbol="PEPEUSDT", signal_time=NOW - pd.Timedelta("5min"))])
    synced = A.load_state(tmp_path / "state.json")["synced_until"]
    later = NOW + pd.Timedelta("3D")                           # the PC was off; it starts before the Wi-Fi
    sent = []

    def offline_replay(since, until, costs):
        return [], {}, ["PEPEUSDT_1h_h4"]

    st = run_with(tmp_path, later, {}, sent=sent, replay=offline_replay,
                  daily_fn=lambda s: (_ for _ in ()).throw(OSError("offline")))
    assert len(st["accounts"]["models"]["positions"]) == 1     # not sold at the pre-pause price
    assert st["synced_until"] == synced and not any("caught up" in m for m in sent)
    # Network back 15 minutes later: the catch-up runs, selling at the close at the exit time.
    exit_time = pd.Timestamp(st["accounts"]["models"]["positions"][0]["exit_time"])
    closes = {("PEPEUSDT", "1h"): pd.Series([120.0], index=[exit_time])}
    st = run_with(tmp_path, later + pd.Timedelta("15min"), {**market, "PEPEUSDT": 90.0}, sent=sent,
                  replay=lambda since, until, costs: ([], closes, []))
    trade = st["accounts"]["models"]["trades"][-1]
    assert trade["exit_price"] == 120.0 and trade["replayed"]
    assert sum("caught up" in m for m in sent) == 1 and st["synced_until"] == (later + pd.Timedelta("15min")).isoformat()


def test_trend_catch_up_stops_at_a_day_with_missing_data():
    st = A.new_state(NOW - pd.Timedelta("5D"), CFG)
    acct = st["accounts"]["trend"]
    acct["rebalanced_for"] = (NOW.floor("D") - pd.Timedelta("3D")).isoformat()
    acct["holdings"] = {"BTCUSDT": 2.0}

    def flaky(sym):
        if sym == "ETHUSDT":
            raise OSError("timeout")
        return daily(0.004)

    n, done = A.catch_up_trend(st, flaky, NOW.floor("D"))
    assert (n, done) == (0, False)
    assert acct["holdings"] == {"BTCUSDT": 2.0} and acct["rebalanced_for"] == (NOW.floor("D") - pd.Timedelta("3D")).isoformat()


def test_trend_and_hold_accounts_wait_for_every_price():
    st = A.new_state(NOW, CFG)
    A.step_hold(st, {"BTCUSDT": 100.0})                         # ETH has no price yet
    assert st["accounts"]["hold"]["holdings"] == {} and st["accounts"]["hold"]["cash"] == 1000
    st["guide"] = [{"symbol": "BTCUSDT", "exposure": 0.5, "trend": "UP"}, {"symbol": "ETHUSDT", "exposure": 0.5, "trend": "UP"}]
    st["guide_day"] = NOW.floor("D").isoformat()
    assert A.step_trend(st, {"BTCUSDT": 100.0}, NOW) is None
    assert st["accounts"]["trend"]["rebalanced_for"] is None   # still due


def test_go_live_needs_real_profit_not_just_losing_less_than_holding():
    st = A.new_state(NOW - pd.Timedelta("120D"), CFG)
    models = st["accounts"]["models"]
    models["stats"] = {"count": 40, "validated": 40, "pnl": -50.0}
    models["cash"] = 950.0                                      # lost 5%
    st["accounts"]["hold"]["cash"] = 700.0                      # buy & hold lost 30%
    money = A.go_live_checks(st, NOW)[2]
    assert not money["ok"] and "-50.00 USDT" in money["detail"]
    models["stats"]["pnl"], models["cash"] = 80.0, 1080.0
    assert A.go_live_checks(st, NOW)[2]["ok"]


def test_a_damaged_save_falls_back_to_the_previous_one(tmp_path):
    p = tmp_path / "state.json"
    first = A.new_state(NOW, CFG)
    A.save_state(first, p)
    second = {**first, "last_run": NOW.isoformat()}
    A.save_state(second, p)                                     # the first save is now the backup
    p.write_text('{"cut off by a power fail')
    assert A.load_state(p)["last_run"] is None                  # dashboard read: backup, file untouched
    assert p.exists()
    assert A.load_state(p, repair=True)["last_run"] is None     # autopilot: backup, damaged file moved aside
    assert not p.exists() and list(tmp_path.glob("state.json.corrupt-*"))


def test_settings_tolerate_typing_variations(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_TRADE_PCT", "10%")
    monkeypatch.setenv("AUTOPILOT_START_CASH", "1,500")
    monkeypatch.setenv("AUTOPILOT_REPORT_HOUR", "ten")
    monkeypatch.setenv("AUTOPILOT_TRADE_ALERTS", "true")
    cfg = A.settings()
    assert cfg["trade_share"] == 0.10 and cfg["start_cash"] == 1500 and cfg["trade_alerts"]
    assert cfg["report_hour"] == 1 and len(cfg["notes"]) == 1 and "AUTOPILOT_REPORT_HOUR" in cfg["notes"][0]


def test_trade_history_is_capped_but_totals_count_everything(monkeypatch):
    monkeypatch.setattr(A, "MAX_HISTORY", 5)
    acct = A.new_state(NOW, CFG)["accounts"]["models"]
    for i in range(12):
        A.record_trade(acct, {"pnl": 1.0, "verdict": "NO EDGE", "i": i})
    assert len(acct["trades"]) == 5 and acct["trades"][-1]["i"] == 11
    assert acct["stats"] == {"count": 12, "validated": 0, "pnl": 12.0}


def test_a_guide_with_missing_data_keeps_the_last_complete_one_and_its_alerts(tmp_path):
    sent = []
    day1 = NOW
    run_with(tmp_path, day1, {"BTCUSDT": 100.0, "ETHUSDT": 10.0}, sent=sent,
             daily_fn=lambda s: daily(-0.004, end=day1.floor("D")))          # both DOWN on day 1
    day2 = day1 + pd.Timedelta("1D")

    def eth_fails(sym):
        if sym == "ETHUSDT":
            raise OSError("timeout")
        return daily(0.004, end=day2.floor("D"))

    st = run_with(tmp_path, day2, {"BTCUSDT": 100.0, "ETHUSDT": 10.0}, sent=sent, daily_fn=eth_fails)
    assert st["guide_day"] == day1.floor("D").isoformat() and all(c["trend"] != "ERROR" for c in st["guide"])
    sent.clear()
    run_with(tmp_path, day2 + pd.Timedelta("15min"), {"BTCUSDT": 100.0, "ETHUSDT": 10.0}, sent=sent,
             daily_fn=lambda s: daily(0.004, end=day2.floor("D")))
    assert any("BTC trend turned UP" in m for m in sent)            # not lost to the failed attempt
