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
