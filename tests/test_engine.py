"""Engine safety properties: NO TRADE unless every gate passes."""
import numpy as np
import pandas as pd
import pytest

from quant.backtest import Calibrator
from quant.engine import (Engine, EventSpecialist, Evidence, ModelSpecialist, TrendSpecialist, consensus,
                          regime_snapshot)
from quant.live import data_health

from test_quant_harness import synthetic_m1


class FixedModel:
    """Predicts a fixed probability regardless of input."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([1 - np.full(len(X), self.p), np.full(len(X), self.p)])


class FakeStore:
    def __init__(self, frames):
        self.frames = frames

    def sync(self, symbol):
        return 0

    def frame(self, symbol):
        return self.frames[symbol]


def artifact(status="VALIDATED", p=0.9, margin=0.0, families=("cal",), decision="ev"):
    cal = Calibrator("none").fit(np.array([0.4, 0.6]), np.array([0, 1]))
    return {
        "name": "test_model", "horizon_min": 60, "families": list(families),
        "columns": ["cal__hour_sin", "cal__hour_cos", "cal__weekend", "cal__dow_sin", "cal__dow_cos"],
        "model": FixedModel(p), "calibrator": cal, "margin": margin, "cost": "perp_taker", "decision": decision,
        "evidence": {"status": status, "summary": "test"}, "trade_sigma": 0.01,
    }


@pytest.fixture(scope="module")
def m1():
    df = synthetic_m1(days=60)
    # Make the series end "now" so freshness checks pass.
    # End on an hour boundary so the 1h decision time equals the last bar (inside the entry window).
    shift = pd.Timestamp.now(tz="UTC").floor("h") - df.index[-1]
    df.index = df.index + shift
    return df


def now_of(m1):
    return m1.index[-1] + pd.Timedelta(seconds=20)


def test_validated_specialist_trades_when_edge_clears_cost(m1):
    spec = ModelSpecialist(artifact(p=0.9))
    sig = spec.evaluate("BTCUSDT", m1, now_of(m1))
    assert sig.action == "LONG"
    assert sig.expected_edge_bps > sig.cost_bps


def test_unvalidated_specialist_never_trades(m1):
    spec = ModelSpecialist(artifact(status="NOT_VALIDATED", p=0.99))
    sig = spec.evaluate("BTCUSDT", m1, now_of(m1))
    assert sig.action == "NO TRADE"
    assert any("did not pass" in r for r in sig.reasons)


def test_small_edge_is_no_trade(m1):
    spec = ModelSpecialist(artifact(p=0.501))
    assert spec.evaluate("BTCUSDT", m1, now_of(m1)).action == "NO TRADE"


def test_prob_rule_uses_probability_margin_not_ev(m1):
    # Under the "prob" rule the margin is a distance from 0.5, not a return hurdle.
    assert ModelSpecialist(artifact(p=0.56, margin=0.05, decision="prob")).evaluate(
        "BTCUSDT", m1, now_of(m1)).action == "LONG"
    assert ModelSpecialist(artifact(p=0.53, margin=0.05, decision="prob")).evaluate(
        "BTCUSDT", m1, now_of(m1)).action == "NO TRADE"
    assert ModelSpecialist(artifact(p=0.40, margin=0.05, decision="prob")).evaluate(
        "BTCUSDT", m1, now_of(m1)).action == "SHORT"


def test_no_margin_from_validation_is_no_trade(m1):
    spec = ModelSpecialist(artifact(p=0.95, margin=None))
    assert spec.evaluate("BTCUSDT", m1, now_of(m1)).action == "NO TRADE"


def test_disabled_health_blocks_trading(m1):
    spec = ModelSpecialist(artifact(p=0.9))
    for _ in range(200):
        spec.health.update(-0.05)
    assert not spec.health.active
    sig = spec.evaluate("BTCUSDT", m1, now_of(m1))
    assert sig.action == "NO TRADE" and any("health" in r for r in sig.reasons)


def test_spot_cost_model_never_shorts(m1):
    a = artifact(p=0.05)
    a["cost"] = "spot_taker"
    sig = ModelSpecialist(a).evaluate("BTCUSDT", m1, now_of(m1))
    assert sig.action == "NO TRADE"


def test_stale_data_forces_no_trade(m1):
    stale_now = m1.index[-1] + pd.Timedelta(minutes=30)
    h = data_health(m1, stale_now)
    assert not h["ok"]
    eng = Engine(["BTCUSDT"], [ModelSpecialist(artifact(p=0.9))], store=FakeStore({"BTCUSDT": m1}), use_book=False)
    out = eng.run(now=stale_now, sync=False)
    assert out[0]["action"] == "NO TRADE"
    assert any("stale" in r for r in out[0]["reasons"])


def test_engine_end_to_end_fresh_data(m1):
    eng = Engine(["BTCUSDT"], [ModelSpecialist(artifact(p=0.9))], store=FakeStore({"BTCUSDT": m1}), use_book=False)
    out = eng.run(now=now_of(m1), sync=False)
    assert out[0]["action"] == "LONG"
    assert out[0]["data_health"]["ok"]
    assert "vol_regime" in out[0]["regime"]


def test_book_veto_blocks_actionable_signal(m1, monkeypatch):
    import quant.engine as E

    monkeypatch.setattr(E, "book_check", lambda sym: {"ok": False, "problems": ["spread 40bp > 5bp"]})
    eng = Engine(["BTCUSDT"], [ModelSpecialist(artifact(p=0.9))], store=FakeStore({"BTCUSDT": m1}), use_book=True)
    out = eng.run(now=now_of(m1), sync=False)
    assert out[0]["action"] == "NO TRADE" and "spread" in out[0]["reasons"][0]


def test_trend_specialist_outputs_exposure(m1):
    ev = Evidence(status="RISK_OVERLAY", summary="test")
    sig = TrendSpecialist(ev, fetch_daily=False).evaluate("BTCUSDT", m1, now_of(m1))
    # 60 days of synthetic data is not enough for the 90-day Donchian -> NO TRADE with a reason
    assert sig.action == "NO TRADE" and sig.reasons


def test_regime_snapshot_keys(m1):
    r = regime_snapshot(m1)
    assert {"vol_regime", "trend_7d", "vol_1d_pct"} <= set(r)


def _sig(spec, action, view, status="VALIDATED", ok=True, edge=10.0, exposure=None):
    return {"symbol": "BTCUSDT", "specialist": spec, "action": action, "view": view, "horizon": "4h",
            "evidence": {"status": status}, "expected_edge_bps": edge, "cost_bps": 12.0, "exposure": exposure,
            "data_health": {"ok": ok, "problems": [] if ok else ["data is 600s old"]}, "regime": {}}


def test_consensus_requires_validated_trade():
    c = consensus([_sig("model", "NO TRADE", "UP", status="NOT_VALIDATED"),
                   _sig("daily_trend", "LONG", "UP", status="RISK_OVERLAY", exposure=0.6)])[0]
    assert c["verdict"] == "NO TRADE"          # an overlay's LONG is allocation, not a trade
    assert c["agreement"] == "agree" and c["exposure"] == 0.6


def test_consensus_conflicting_validated_specialists_is_no_trade():
    c = consensus([_sig("a", "LONG", "UP"), _sig("b", "SHORT", "DOWN")])[0]
    assert c["verdict"] == "NO TRADE" and "conflict" in c["why"] and c["agreement"] == "conflict"


def test_consensus_insufficient_data_is_no_trade():
    c = consensus([_sig("a", "LONG", "UP", ok=False)])[0]
    assert c["verdict"] == "NO TRADE" and c["why"].startswith("insufficient information")


def test_consensus_passes_through_validated_trade():
    c = consensus([_sig("a", "LONG", "UP"), _sig("daily_trend", "FLAT", "DOWN", status="RISK_OVERLAY", exposure=0.0)])[0]
    assert c["verdict"] == "LONG" and c["agreement"] == "conflict"


def capitulation_m1():
    """Fresh synthetic frame whose last closed bar completes a forced, one-sided sell-off."""
    df = synthetic_m1(days=45, seed=3)
    df.index = df.index + (pd.Timestamp.now(tz="UTC").floor("min") - df.index[-1])
    last60 = df.index[-60:]
    df.loc[last60, "taker_buy_base"] = df.loc[last60, "volume"] * 0.05
    df.loc[last60, "taker_buy_quote"] = df.loc[last60, "taker_buy_base"] * df.loc[last60, "close"]
    prev = df["close"].iloc[-2]
    new = prev * 0.94
    df.iloc[-1, df.columns.get_loc("open")] = prev
    df.iloc[-1, df.columns.get_loc("high")] = prev
    df.iloc[-1, df.columns.get_loc("close")] = new
    df.iloc[-1, df.columns.get_loc("low")] = new * 0.999
    df.iloc[-1, df.columns.get_loc("quote_volume")] = df["volume"].iloc[-1] * new
    df.iloc[-1, df.columns.get_loc("taker_buy_quote")] = df["taker_buy_base"].iloc[-1] * new
    return df


def event_artifact(status="VALIDATED"):
    return {"kind": "event", "name": "capitulation", "W": 60, "k": 4.0, "h": 60, "cost": "spot_taker",
            "symbols": ["SOLUSDT"], "evidence": {"status": status, "summary": "test", "ev_bps": 20.0},
            "trade_sigma": 0.02}


def test_event_specialist_trades_fresh_validated_event():
    m = capitulation_m1()
    sig = EventSpecialist(event_artifact()).evaluate("SOLUSDT", m, m.index[-1] + pd.Timedelta(seconds=20))
    assert sig.action == "LONG" and sig.view == "UP"


def test_event_specialist_unvalidated_is_informational():
    m = capitulation_m1()
    sig = EventSpecialist(event_artifact("NOT_VALIDATED")).evaluate("SOLUSDT", m, m.index[-1] + pd.Timedelta(seconds=20))
    assert sig.action == "NO TRADE" and sig.view == "UP"
    assert any("informational" in r for r in sig.reasons)


def test_event_specialist_never_trades_stale_event():
    m = capitulation_m1()
    sig = EventSpecialist(event_artifact()).evaluate("SOLUSDT", m, m.index[-1] + pd.Timedelta(minutes=5))
    assert sig.action == "NO TRADE" and any("stale" in r for r in sig.reasons)


def test_event_specialist_outside_universe_and_quiet_market():
    m = capitulation_m1()
    spec = EventSpecialist(event_artifact())
    assert spec.evaluate("BTCUSDT", m, m.index[-1]).action == "NO TRADE"
    quiet = m.iloc[:-1]
    sig = spec.evaluate("SOLUSDT", quiet, quiet.index[-1] + pd.Timedelta(seconds=20))
    assert sig.action == "NO TRADE" and sig.view is None


def test_model_signal_outside_entry_window_is_no_trade(m1):
    spec = ModelSpecialist(artifact(p=0.9))
    sig = spec.evaluate("BTCUSDT", m1, m1.index[-1] + pd.Timedelta(minutes=30))
    assert sig.action == "NO TRADE" and sig.view == "UP"
    assert any("entry window has passed" in r for r in sig.reasons)


def test_model_specialist_refuses_symbols_outside_its_universe(m1):
    a = artifact(p=0.9)
    a["symbols"] = ["ETHUSDT"]
    assert ModelSpecialist(a).evaluate("BTCUSDT", m1, now_of(m1)).action == "NO TRADE"
    eng = Engine(["BTCUSDT"], [ModelSpecialist(a)], store=FakeStore({"BTCUSDT": m1}), use_book=False)
    assert eng.run(now=now_of(m1), sync=False) == []


def test_model_waits_for_the_decision_bar(m1):
    """A bar lagging the decision time must not turn the previous decision into a fresh one."""
    spec = ModelSpecialist(artifact(p=0.9))
    lagging = m1.iloc[:-1]                       # last bar is 1 minute before the hour
    sig = spec.evaluate("BTCUSDT", lagging, m1.index[-1] + pd.Timedelta(seconds=30))
    assert sig.action == "NO TRADE"
    assert pd.Timestamp(sig.as_of) == m1.index[-1] - pd.Timedelta(hours=1)


# ---------------------------------------------------------------- paper ledger -> health monitor

def test_paper_trade_resolves_like_research_labels(tmp_path, m1):
    from quant.costs import PERP
    from quant.labels import fill_prices
    from quant.paper import PaperLedger

    spec = ModelSpecialist(artifact(p=0.9))
    led = PaperLedger(tmp_path / "t.jsonl", tmp_path / "h.json")
    t0 = m1.index[-200]
    sig = {"symbol": "BTCUSDT", "specialist": spec.name, "action": "LONG", "as_of": t0.isoformat()}
    assert led.record(sig, 60, "perp_taker") and not led.record(sig, 60, "perp_taker")   # recorded once
    assert led.resolve({"BTCUSDT": m1}, [spec]) == 1
    lf = fill_prices(m1)
    entry, exit_ = t0 + pd.Timedelta(minutes=1), t0 + pd.Timedelta(minutes=61)
    expected = lf[exit_] - lf[entry] - PERP.round_trip("BTCUSDT") - PERP.holding(60, 1)
    assert led.trades[0]["net"] == pytest.approx(expected)
    assert spec.health.n == 1


def test_health_state_persists_and_resets_on_revalidation(tmp_path):
    from quant.paper import PaperLedger

    spec = ModelSpecialist(artifact(p=0.9))
    for _ in range(200):
        spec.health.update(-0.05)
    assert not spec.health.active
    led = PaperLedger(tmp_path / "t.jsonl", tmp_path / "h.json")
    led.save_health([spec])
    again = ModelSpecialist(artifact(p=0.9))
    PaperLedger(tmp_path / "t.jsonl", tmp_path / "h.json").load_health([again])
    assert not again.health.active                      # a restart does not re-enable it
    a = artifact(p=0.9)
    a["evidence"]["summary"] = "fresh validation"
    fresh = ModelSpecialist(a)
    PaperLedger(tmp_path / "t.jsonl", tmp_path / "h.json").load_health([fresh])
    assert fresh.health.active                          # new evidence -> fresh monitor


def test_engine_records_validated_trades_only(tmp_path, m1):
    from quant.paper import PaperLedger

    led = PaperLedger(tmp_path / "t.jsonl", tmp_path / "h.json")
    specs = [ModelSpecialist(artifact(p=0.9)), ModelSpecialist({**artifact(status="NOT_VALIDATED", p=0.9), "name": "info"})]
    eng = Engine(["BTCUSDT"], specs, store=FakeStore({"BTCUSDT": m1}), use_book=False, paper=led)
    eng.run(now=now_of(m1), sync=False)
    eng.run(now=now_of(m1), sync=False)
    assert [t["specialist"] for t in led.trades] == ["test_model"]


def test_unvalidated_trend_rule_never_suggests_exposure_as_action(m1):
    ev = Evidence(status="NOT_VALIDATED", summary="x")
    sig = TrendSpecialist(ev, fetch_daily=False).evaluate("BTCUSDT", synthetic_m1(days=260, seed=2), now_of(m1))
    assert sig.action in ("NO TRADE", "FLAT")


def test_trend_rule_is_not_applied_to_untested_coins(m1):
    ev = Evidence(status="RISK_OVERLAY", summary="x", universe=["BTCUSDT"])
    spec = TrendSpecialist(ev, fetch_daily=False)
    assert spec.evaluate("PEPEUSDT", m1, now_of(m1)).action == "NO TRADE"
    eng = Engine(["PEPEUSDT"], [spec], store=FakeStore({"PEPEUSDT": m1}), use_book=False)
    assert eng.run(now=now_of(m1), sync=False) == []


def test_book_veto_limits_depend_on_the_specialist():
    from quant.engine import book_veto

    book = {"ok": False, "problems": ["spread 24.6bp > 5bp"], "spread_bps": 24.6, "buy_impact_bps": 13.0,
            "sell_impact_bps": 12.0}
    assert book_veto(book, ModelSpecialist(artifact(p=0.9))).startswith("spread 24.6bp")   # intraday: veto
    assert book_veto(book, TrendSpecialist(Evidence(status="RISK_OVERLAY", summary="x"), fetch_daily=False)) is None
    assert book_veto({"ok": False, "problems": ["empty order book"]}, ModelSpecialist(artifact(p=0.9))) == "empty order book"
