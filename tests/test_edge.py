import numpy as np
import pytest

from cryptopredict.experiments import holm, run_grid
from cryptopredict.train import THRESHOLDS, backtest, choose_threshold, edge_verdict, p_value, threshold_analysis


def test_backtest_charges_fees_on_position_changes():
    pred = np.array([1, 1, 0, 1])
    ret = np.array([0.01, 0.01, 0.05, -0.02])
    out = backtest(pred, ret, fee=0.001)
    # in, hold, out, in -> 3 position changes
    assert out["trades"] == 3
    expected = (1.01 - 0.001) * 1.01 * (1 - 0.001) * (1 - 0.02 - 0.001) - 1
    assert out["strategy_return"] == pytest.approx(expected)
    assert out["buy_and_hold_return"] == pytest.approx(np.prod(1 + ret) - 1)


def test_backtest_horizon_uses_non_overlapping_holds():
    pred = np.ones(6, dtype=int)
    ret = np.arange(6) / 100
    out = backtest(pred, ret, fee=0.0, horizon=3)
    # decisions at rows 0 and 3 only
    assert out["buy_and_hold_return"] == pytest.approx((1 + 0.00) * (1 + 0.03) - 1)


def test_p_value_detects_real_skill_but_not_chance():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    good = np.where(rng.random(2000) < 0.6, y, 1 - y)  # 60% accurate
    coin = rng.integers(0, 2, 2000)
    assert p_value(good, y, 0.5) < 0.001
    assert p_value(coin, y, 0.5) > 0.05


def test_threshold_analysis_trades_less_as_threshold_rises():
    rng = np.random.default_rng(1)
    prob = rng.random(500)
    y = rng.integers(0, 2, 500)
    rows = threshold_analysis(prob, y, rng.normal(0, 0.01, 500), fee=0.001)
    assert [r["threshold"] for r in rows] == list(THRESHOLDS)
    assert rows[0]["coverage"] == 1.0
    calls = [r["calls"] for r in rows]
    assert calls == sorted(calls, reverse=True)


def test_choose_threshold_ignores_thresholds_with_too_few_trades():
    rows = [
        {"threshold": 0.5, "calls": 500, "strategy_return": -0.10},
        {"threshold": 0.55, "calls": 100, "strategy_return": 0.02},
        {"threshold": 0.6, "calls": 5, "strategy_return": 0.50},  # too few to trust
    ]
    assert choose_threshold(rows) == 0.55


@pytest.mark.parametrize(
    "p, strat, hold, verdict",
    [
        (0.01, 0.10, 0.05, "POSSIBLE EDGE"),
        (0.01, -0.02, 0.05, "NO PROFIT"),
        (0.40, 0.10, 0.05, "UNPROVEN"),
        (0.40, -0.10, 0.05, "NO EDGE"),
    ],
)
def test_edge_verdict(p, strat, hold, verdict):
    out = edge_verdict({"p_vs_baseline": p}, {"strategy_return": strat, "buy_and_hold_return": hold})
    assert out["verdict"] == verdict


def test_holm_adjustment():
    adjusted = holm([0.01, 0.04, 0.03])
    # sorted: 0.01*3=0.03, 0.03*2=0.06, max(0.06, 0.04*1)=0.06
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_run_grid_on_synthetic(tmp_path):
    summary = run_grid(["AAAUSDT", "BBBUSDT"], ["1h"], [1], source="synthetic", limit=1200,
                       out_dir=tmp_path, verbose=False, jobs=2)
    assert len(summary) == 2
    assert set(summary["symbol"]) == {"AAAUSDT", "BBBUSDT"}
    assert (summary["p_adjusted"] >= summary["p_vs_baseline"] - 1e-12).all()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "AAAUSDT_1h_h1" / "model.joblib").exists()
