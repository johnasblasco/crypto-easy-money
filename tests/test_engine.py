"""Engine safety properties: NO TRADE unless every gate passes."""
import numpy as np
import pandas as pd
import pytest

from quant.backtest import Calibrator
from quant.engine import Engine, Evidence, ModelSpecialist, TrendSpecialist, regime_snapshot
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


def artifact(status="VALIDATED", p=0.9, margin=0.0, families=("cal",)):
    cal = Calibrator("none").fit(np.array([0.4, 0.6]), np.array([0, 1]))
    return {
        "name": "test_model", "horizon_min": 60, "families": list(families),
        "columns": ["cal__hour_sin", "cal__hour_cos", "cal__weekend", "cal__dow_sin", "cal__dow_cos"],
        "model": FixedModel(p), "calibrator": cal, "margin": margin, "cost": "perp_taker", "decision": "ev",
        "evidence": {"status": status, "summary": "test"}, "trade_sigma": 0.01,
    }


@pytest.fixture(scope="module")
def m1():
    df = synthetic_m1(days=60)
    # Make the series end "now" so freshness checks pass.
    shift = pd.Timestamp.now(tz="UTC").floor("min") - df.index[-1]
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
    sig = TrendSpecialist(ev).evaluate("BTCUSDT", m1, now_of(m1))
    # 60 days of synthetic data is not enough for the 90-day Donchian -> NO TRADE with a reason
    assert sig.action == "NO TRADE" and sig.reasons


def test_regime_snapshot_keys(m1):
    r = regime_snapshot(m1)
    assert {"vol_regime", "trend_7d", "vol_1d_pct"} <= set(r)
