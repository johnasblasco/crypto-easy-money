"""Leakage and correctness tests for the research harness (quant/)."""
import numpy as np
import pandas as pd
import pytest

from quant import features as F
from quant import metrics as M
from quant.backtest import Dataset, choose_threshold, positions_from_prob, simulate, walk_forward_predict
from quant.costs import PERP, SPOT, ZERO
from quant.data import resample
from quant.labels import decision_times, forward_returns, triple_barrier
from quant.models import BaseRate, logistic
from quant.validation import check_no_leakage, walk_forward


def synthetic_m1(days: int = 45, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2024-01-01 00:01", periods=n, freq="1min", tz="UTC")
    r = rng.normal(0, 0.0008, n)
    close = 30000 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[30000], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0003, n)))
    vol = rng.lognormal(0, 0.5, n)
    tb = vol * rng.uniform(0.2, 0.8, n)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
        "quote_volume": vol * close, "trades": rng.integers(10, 100, n),
        "taker_buy_base": tb, "taker_buy_quote": tb * close,
    }, index=idx)
    df.index.name = "ts"
    df["gap"] = False
    return df


@pytest.fixture(scope="module")
def m1():
    return synthetic_m1()


def test_features_are_point_in_time(m1):
    """Changing anything after T must not change any feature at or before T."""
    times = decision_times(m1, "60min")
    cut = times[len(times) // 2 + 7]
    base = F.build(m1, times, "AAAUSDT", families=["mom", "vol", "act", "flow", "shape", "range", "regime", "cal", "ta", "trend"])
    altered = m1.copy()
    after = altered.index > cut
    rng = np.random.default_rng(1)
    for c in ["open", "high", "low", "close"]:
        altered.loc[after, c] *= 1.3
    altered.loc[after, "high"] *= 1.05
    altered.loc[after, "volume"] *= rng.uniform(0.1, 10, after.sum())
    altered.loc[after, "quote_volume"] *= 7
    altered.loc[after, "taker_buy_quote"] *= 0.1
    altered.loc[after, "trades"] *= 3
    pert = F.build(altered, times, "AAAUSDT", families=["mom", "vol", "act", "flow", "shape", "range", "regime", "cal", "ta", "trend"])
    before = base.index <= cut
    pd.testing.assert_frame_equal(base[before], pert[before])
    # and something after the cut did change (the test is not vacuous)
    assert not base[~before].equals(pert[~before])


def test_cross_asset_features_are_point_in_time(m1):
    times = decision_times(m1, "60min")
    other = synthetic_m1(seed=5)
    cut = times[len(times) // 2]
    others = {"BTCUSDT": np.log(other["close"]).ffill(), "AAAUSDT": np.log(m1["close"]).ffill()}
    a = F.build(m1, times, "AAAUSDT", others=others, families=["xa"])
    o2 = other.copy()
    o2.loc[o2.index > cut, "close"] *= 2
    others2 = {"BTCUSDT": np.log(o2["close"]).ffill(), "AAAUSDT": np.log(m1["close"]).ffill()}
    b = F.build(m1, times, "AAAUSDT", others=others2, families=["xa"])
    pd.testing.assert_frame_equal(a[a.index <= cut], b[b.index <= cut])


def test_resample_labels_bars_by_close_time(m1):
    h = resample(m1, "1h")
    # The 01:00 bar contains the 1m bars closing 00:01..01:00 (60 bars)
    t = pd.Timestamp("2024-01-01 01:00", tz="UTC")
    window = m1[(m1.index > t - pd.Timedelta("1h")) & (m1.index <= t)]
    assert len(window) == 60
    assert h.loc[t, "close"] == window["close"].iloc[-1]
    assert h.loc[t, "open"] == window["open"].iloc[0]
    assert h.loc[t, "volume"] == pytest.approx(window["volume"].sum())


def test_resample_drops_unfinished_last_bar(m1):
    partial = m1.iloc[:-17]  # last 1m close is not on an hour boundary
    h = resample(partial, "1h")
    assert h.index[-1] <= partial.index[-1]


def test_forward_returns_use_delayed_entry(m1):
    times = decision_times(m1, "60min")[:50]
    lab = forward_returns(m1, times, 60, latency_min=1)
    t = times[10]
    entry = np.log(m1.loc[t + pd.Timedelta("1min"), "close"])
    exit_ = np.log(m1.loc[t + pd.Timedelta("61min"), "close"])
    assert lab.loc[t, "fwd_ret"] == pytest.approx(exit_ - entry)
    assert lab.loc[t, "exit_ts"] == t + pd.Timedelta("61min")


def test_triple_barrier_hits_the_right_barrier():
    idx = pd.date_range("2024-01-01 00:01", periods=200, freq="1min", tz="UTC")
    close = np.full(200, 100.0)
    close[5:] = 101.5   # +1.5% shortly after entry -> take-profit
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)
    times = pd.DatetimeIndex([idx[1]])
    tb = triple_barrier(df, times, 60, up=np.array([0.01]), down=np.array([0.01]))
    assert tb["label"].iloc[0] == 1
    close2 = np.full(200, 100.0)
    close2[5:] = 98.5
    df2 = pd.DataFrame({"open": close2, "high": close2, "low": close2, "close": close2}, index=idx)
    assert triple_barrier(df2, times, 60, up=np.array([0.01]), down=np.array([0.01]))["label"].iloc[0] == -1
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)
    assert triple_barrier(flat, times, 60, up=np.array([0.01]), down=np.array([0.01]))["label"].iloc[0] == 0


def test_walk_forward_purges_overlapping_labels():
    times = pd.date_range("2022-01-01", periods=24 * 800, freq="1h", tz="UTC")
    label_end = times + pd.Timedelta("4h")   # 4h labels on an hourly grid overlap
    folds = walk_forward(times, label_end, test_size="60D", min_train="365D", embargo="2h")
    assert len(folds) >= 5
    check_no_leakage(folds, times, label_end)
    for f in folds:
        assert label_end[f.train].max() < f.test_start
        assert times[f.train].max() < f.test_start - pd.Timedelta("2h")
        assert times[f.test].min() >= f.test_start
        # validation strictly after fit
        assert times[f.fit].max() < times[f.valid].min()


def test_simulate_charges_costs_on_changes():
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    pos = pd.Series([1, 1, 0, -1, -1, 1], index=idx)
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.0, 0.02], index=idx)
    sim = simulate(pos, r, "BTCUSDT", PERP, 60)
    side = PERP.one_side("BTCUSDT")
    # turnover: enter 1, hold 0, exit 1, enter short 1, hold 0, flip 2, final close 1
    expected_turnover = 1 + 0 + 1 + 1 + 0 + 2 + 1
    hold = PERP.holding(60) * 5   # 5 periods with a position
    assert sim["cost"].sum() == pytest.approx(expected_turnover * side + hold)
    assert sim["gross"].sum() == pytest.approx(0.01 - 0.02 + 0.01 + 0.0 + 0.02)
    spot = simulate(pos, r, "BTCUSDT", SPOT, 60)
    assert (spot["pos"] >= 0).all()


def test_simulate_treats_time_gaps_as_flat():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 05:00"], tz="UTC")
    pos = pd.Series([1, 1, 1], index=idx)
    r = pd.Series([0.0, 0.0, 0.0], index=idx)
    sim = simulate(pos, r, "BTCUSDT", ZERO.__class__("t", fee_bps=10, impact_bps={}, other_impact_bps=0), 60)
    # enter, hold, exit before gap, re-enter after gap, final exit = 4 sides
    assert sim["cost"].sum() == pytest.approx(4 * 0.001)


def test_threshold_abstains_without_edge():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.4, 0.6, 3000)
    r = rng.normal(0, 0.005, 3000)       # returns unrelated to p
    margin, _ = choose_threshold(p, r, "BTCUSDT", PERP, 60)
    assert margin is None


def test_threshold_trades_with_real_edge():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, 5000)
    r = (p - 0.5) * 0.05 + rng.normal(0, 0.003, 5000)   # strong, cost-beating edge
    margin, _ = choose_threshold(p, r, "BTCUSDT", PERP, 60)
    assert margin is not None


def test_walk_forward_predict_has_no_skill_on_noise():
    rng = np.random.default_rng(3)
    times = pd.date_range("2021-01-01", periods=24 * 700, freq="1h", tz="UTC")
    X = pd.DataFrame(rng.normal(size=(len(times), 5)), index=times, columns=[f"f{i}" for i in range(5)])
    r = pd.Series(rng.normal(0, 0.005, len(times)), index=times)
    ds = Dataset("BTCUSDT", 60, X, r, pd.Series(times + pd.Timedelta("61min"), index=times))
    folds = walk_forward(ds.times, pd.DatetimeIndex(ds.label_end), test_size="90D", min_train="365D")
    pred = walk_forward_predict(ds, folds, logistic(), cost=PERP)
    assert abs(M.auc(pred["y"], pred["p"]) - 0.5) < 0.03
    # Decision layer should almost always abstain on pure noise.
    assert (pred["pos"] != 0).mean() < 0.05


def test_metrics_sanity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    p = np.clip(y * 0.2 + 0.4 + rng.normal(0, 0.05, 5000), 0.01, 0.99)
    assert M.auc(y, p) > 0.9
    assert M.ece(y, np.full(5000, y.mean())) < 0.02
    noise = rng.normal(0, 1, 1000)
    p_val, lo, hi = M.mean_pvalue(noise + 0.0)
    assert p_val > 0.01
    p_val2, lo2, hi2 = M.mean_pvalue(noise + 0.5)
    assert p_val2 < 0.01 and lo2 > 0
    assert M.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    # PBO of pure-noise strategies should be around 0.5, not near 0
    perf = rng.normal(0, 1, (400, 20))
    assert 0.2 < M.pbo_cscv(perf, n_blocks=8) < 0.8
    # DSR penalises many trials
    r = rng.normal(0.05, 1, 500)
    assert M.deflated_sharpe(r, 1000, 0.01) < M.deflated_sharpe(r, 2, 0.01)


def test_positions_from_prob():
    p = np.array([0.4, 0.5, 0.52, 0.6])
    assert positions_from_prob(p, 0.05, True).tolist() == [-1, 0, 0, 1]
    assert positions_from_prob(p, 0.05, False).tolist() == [0, 0, 0, 1]


def test_base_rate_model():
    m = BaseRate().fit(np.zeros((10, 1)), np.array([1] * 7 + [0] * 3))
    assert m.predict_proba(np.zeros((2, 1)))[:, 1].tolist() == [0.7, 0.7]


def test_panel_dataset_keeps_symbols_separate():
    from quant.backtest import evaluate, simulate_panel

    rng = np.random.default_rng(7)
    times = pd.date_range("2021-01-01", periods=24 * 500, freq="1h", tz="UTC")
    parts = []
    for sym in ["AAAUSDT", "BBBUSDT"]:
        X = pd.DataFrame(rng.normal(size=(len(times), 3)), index=times, columns=["a", "b", "c"])
        r = pd.Series(rng.normal(0, 0.004, len(times)), index=times)
        reg = pd.DataFrame({"vol": rng.choice(["low", "high"], len(times))}, index=times)
        parts.append(Dataset(sym, 60, X, r, pd.Series(times + pd.Timedelta("61min"), index=times), reg))
    panel = Dataset.pool(parts)
    assert len(panel.X) == 2 * len(times)
    assert panel.X.index.is_monotonic_increasing
    assert set(panel.sym) == {"AAAUSDT", "BBBUSDT"}
    folds = walk_forward(panel.times, pd.DatetimeIndex(panel.label_end), test_size="60D", min_train="200D")
    check_no_leakage(folds, panel.times, pd.DatetimeIndex(panel.label_end))
    pred = walk_forward_predict(panel, folds, logistic(), cost=PERP)
    # force trades to exercise the simulator
    pred["pos"] = 1
    sim = simulate_panel(pred, PERP, 60)
    assert (sim.index == pred.index).all()
    for sym in ["AAAUSDT", "BBBUSDT"]:
        g = pred[pred["sym"] == sym]
        alone = simulate(g["pos"], g["fwd_ret"], sym, PERP, 60)
        assert sim[sim["sym"] == sym]["net"].sum() == pytest.approx(alone["net"].sum())
    rep, _ = evaluate(panel, pred, PERP, n_boot=50)
    assert rep["trading"]["periods"] == pred.index.nunique()
    assert {"vol=low", "vol=high"} <= set(rep["by_regime"])


def test_skill_gate_and_ev_decisions():
    from quant.backtest import choose_ev_margin, expected_edge, has_skill, positions_from_ev

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    assert not has_skill(np.full(2000, 0.5) + rng.normal(0, 0.05, 2000), y, 0.5)
    assert has_skill(np.clip(0.5 + (y - 0.5) * 0.3, 0.01, 0.99), y, 0.5)
    # Same probability edge is worth more when volatility is higher.
    assert expected_edge(0.55, 0.02) > expected_edge(0.55, 0.005) > 0
    pos = positions_from_ev([0.55, 0.55, 0.45], [0.002, 0.05, 0.05], np.full(3, 0.002), 0.0, True)
    assert pos.tolist() == [0, 1, -1]
    # Noise -> abstain; real cost-beating edge -> trade.
    p = rng.uniform(0.4, 0.6, 4000)
    sig = np.full(4000, 0.01)
    assert choose_ev_margin(p, sig, rng.normal(0, 0.01, 4000), "BTCUSDT", PERP, 240)[0] is None
    r = (2 * p - 1) * 0.01 * 0.8 + rng.normal(0, 0.003, 4000)
    assert choose_ev_margin(p, sig, r, "BTCUSDT", PERP, 240)[0] is not None


def test_monitor_quiet_when_healthy_and_trips_when_edge_disappears():
    from quant.monitor import SignalHealth, simulate_monitoring

    rng = np.random.default_rng(0)
    mu0, sd = 0.002, 0.01
    trips = [simulate_monitoring(rng.normal(mu0, sd, 300), mu0, sd)["tripped_at"] for _ in range(50)]
    false_alarm_rate = np.mean([t is not None for t in trips])
    assert false_alarm_rate < 0.15
    broken = np.concatenate([rng.normal(mu0, sd, 100), rng.normal(-0.003, sd, 300)])
    res = simulate_monitoring(broken, mu0, sd)
    assert res["tripped_at"] is not None and 100 < res["tripped_at"] < 260
    mon = SignalHealth("x", mu0, sd)
    for x in rng.normal(-0.01, sd, 100):
        mon.update(x)
    assert not mon.active
    mon.revalidate(mu0, sd)
    assert mon.active


def test_btc_keeps_cross_asset_columns(m1):
    """Regression: pooled panels used to drop every BTC row because BTC lacked xa__btc_* columns."""
    times = decision_times(m1, "60min")
    other = synthetic_m1(seed=9)
    closes = {"BTCUSDT": np.log(m1["close"]).ffill(), "ETHUSDT": np.log(other["close"]).ffill()}
    xb = F.build(m1, times, "BTCUSDT", others=closes, families=["xa"])
    xe = F.build(other, times, "ETHUSDT", others=closes, families=["xa"])
    assert set(xb.columns) == set(xe.columns)
    assert any(c.startswith("xa__btc_") for c in xb.columns)


def test_skill_gate_rejects_noise_models():
    """Regression: calibrating and judging skill on the same slice made the gate always pass."""
    rng = np.random.default_rng(11)
    times = pd.date_range("2021-01-01", periods=24 * 700, freq="1h", tz="UTC")
    X = pd.DataFrame(rng.normal(size=(len(times), 8)), index=times, columns=[f"f{i}" for i in range(8)])
    r = pd.Series(rng.normal(0, 0.005, len(times)), index=times)
    ds = Dataset("BTCUSDT", 60, X, r, pd.Series(times + pd.Timedelta("61min"), index=times))
    folds = walk_forward(ds.times, pd.DatetimeIndex(ds.label_end), test_size="90D", min_train="365D")
    from quant.models import lightgbm
    pred = walk_forward_predict(ds, folds, lightgbm(n_estimators=100, min_child_samples=50), cost=PERP)
    assert pred.groupby("fold")["skilled"].first().mean() < 0.5


def test_clustered_se_wider_for_duplicated_timestamps():
    from quant.backtest import _ev_and_se

    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 200)
    net = np.repeat(base, 5)               # 5 identical rows per timestamp (perfectly correlated coins)
    groups = np.repeat(np.arange(200), 5)
    _, se_naive = _ev_and_se(net, None)
    _, se_cluster = _ev_and_se(net, groups)
    assert se_cluster > 1.8 * se_naive


def test_trend_daily_bars_keep_partial_outage_days():
    from quant.studies import trend as T

    idx = pd.date_range("2024-01-01 00:01", periods=3 * 1440, freq="1min", tz="UTC")
    m1 = synthetic_m1(days=3)
    m1.loc[m1.index[1440:1440 + 400], ["open", "high", "low", "close"]] = np.nan   # 400-minute outage on day 2
    m1.loc[m1.index[1440:1440 + 400], "gap"] = True
    d = resample(m1, "1D")
    full = d.reindex(pd.date_range(d.index[0], d.index[-1], freq="1D", tz="UTC"))
    full.loc[full["gap_frac"] >= 0.999, "close"] = np.nan
    outage_day = full[(full["gap_frac"] > 0.2) & (full["gap_frac"] < 0.4)]
    assert len(outage_day) == 1 and outage_day["close"].notna().all()
