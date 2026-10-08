"""Trade planner: first-touch replay of entry / take-profit / stop-loss levels."""
import numpy as np
import pandas as pd
import pytest

from cryptopredict import planner


def candles(rows):
    """rows of (open, high, low, close)."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)
    df.insert(0, "timestamp", pd.date_range("2026-01-01", periods=len(df), freq="h", tz="UTC"))
    return df


def first(df, reward, risk, side="long", max_bars=3, cost=0.0):
    return planner.replay(df, reward, risk, side, max_bars, cost).iloc[0]


def test_side_follows_the_levels():
    assert planner.side_of(100, 95, 110) == "long"
    assert planner.side_of(100, 105, 90) == "short"
    for bad in [(100, 105, 110), (100, 95, 90), (100, 100, 110), (-1, 95, 110)]:
        with pytest.raises(ValueError):
            planner.side_of(*bad)


def test_target_first():
    df = candles([(100, 100, 100, 100), (100, 101, 99.5, 101), (101, 102.5, 100.5, 102), (102, 103, 101, 102), (102, 102, 102, 102)])
    t = first(df, 0.02, 0.01, cost=0.001)
    assert t.outcome == "target" and t.bars == 2
    assert t.net == pytest.approx(0.02 - 0.002)
    assert t.hold_net == pytest.approx(0.02 - 0.002)          # close of the exit candle is 102


def test_stop_first_and_gap_fills_at_the_open():
    df = candles([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (97, 97.5, 96, 97), (97, 97, 97, 97), (97, 97, 97, 97)])
    t = first(df, 0.02, 0.01)
    assert t.outcome == "stop" and t.bars == 2
    assert t.net == pytest.approx(-0.03)                       # opened at 97, below the 99 stop


def test_same_candle_tie_counts_as_a_loss():
    df = candles([(100, 100, 100, 100), (100, 105, 95, 100), (100, 100, 100, 100), (100, 100, 100, 100)])
    t = first(df, 0.02, 0.01)
    assert t.outcome == "stop" and t.net == pytest.approx(-0.01)


def test_timeout_closes_at_the_last_candle():
    df = candles([(100, 100, 100, 100), (100, 100.5, 99.5, 100.2), (100, 100.5, 99.5, 100.4), (100, 100.5, 99.5, 100.5)])
    t = first(df, 0.02, 0.01)
    assert t.outcome == "timeout" and t.bars == 3 and t.net == pytest.approx(0.005)


def test_short_is_mirrored():
    df = candles([(100, 100, 100, 100), (100, 100.5, 98.5, 99), (99, 99, 97.5, 98), (98, 98, 98, 98), (98, 98, 98, 98)])
    t = first(df, 0.02, 0.01, side="short")
    assert t.outcome == "target" and t.bars == 2 and t.net == pytest.approx(0.02)
    up = candles([(100, 100, 100, 100), (100, 101.5, 99.8, 101), (101, 101, 101, 101), (101, 101, 101, 101)])
    assert first(up, 0.02, 0.01, side="short").outcome == "stop"


def test_only_starts_with_a_full_window():
    df = candles([(100, 100, 100, 100)] * 10)
    assert len(planner.replay(df, 0.02, 0.01, "long", 4, 0.0)) == 6


def test_random_walk_hits_the_closer_level_proportionally_more_often():
    """No drift, no fees: P(target first) ~ risk / (reward + risk), and the average is ~0."""
    rng = np.random.default_rng(0)
    ticks = np.exp(np.cumsum(rng.normal(0, 0.001, 30_000 * 10))).reshape(-1, 10)
    df = candles(np.column_stack([ticks[:, 0], ticks.max(1), ticks.min(1), ticks[:, -1]]))
    reward, risk = np.log(1.02), -np.log(0.99)                # symmetric in log space
    stats = planner.summarize(planner.replay(df, np.expm1(reward), 0.01, "long", 300, 0.0), 0.02, 0.01, 0.0)
    assert stats["timeout_rate"] < 0.01
    assert stats["win_rate"] == pytest.approx(risk / (reward + risk), abs=0.04)
    assert stats["avg_net_low"] < 0 < stats["avg_net_high"]
    assert planner.verdict(stats)[0] == "NO PROVEN EDGE"


def test_fees_make_a_random_walk_lose():
    rng = np.random.default_rng(1)
    ticks = np.exp(np.cumsum(rng.normal(0, 0.001, 20_000 * 10))).reshape(-1, 10)
    df = candles(np.column_stack([ticks[:, 0], ticks.max(1), ticks.min(1), ticks[:, -1]]))
    stats = planner.summarize(planner.replay(df, 0.006, 0.003, "long", 200, 0.002), 0.006, 0.003, 0.002)
    assert planner.verdict(stats)[0] == "LOSES ON AVERAGE"


def test_breakeven_win_rate():
    assert planner.breakeven_win_rate(0.02, 0.01, 0.0) == pytest.approx(1 / 3)
    assert planner.breakeven_win_rate(0.02, 0.01, 0.001) == pytest.approx(0.012 / 0.03)
    assert planner.breakeven_win_rate(0.001, 0.01, 0.01) == 1.0


def test_position_size_risks_a_fixed_share():
    s = planner.position_size(100, 98, 104, account=1000, risk_share=0.01, cost=0.0)
    assert s["quantity"] == pytest.approx(5) and s["notional"] == pytest.approx(500)
    assert s["loss_at_stop"] == pytest.approx(10) and s["gain_at_target"] == pytest.approx(20)
    assert "capped_quantity" not in s
    tight = planner.position_size(100, 99.9, 100.2, account=1000, risk_share=0.01, cost=0.0)
    assert tight["leverage_needed"] == pytest.approx(10)
    assert tight["capped_quantity"] == pytest.approx(10) and tight["capped_loss_at_stop"] == pytest.approx(1)
    with_fees = planner.position_size(100, 98, 104, account=1000, risk_share=0.01, cost=0.001)
    assert with_fees["loss_at_stop"] == pytest.approx(10) and with_fees["quantity"] < 5


def test_suggest_is_a_long_or_short_template():
    df = candles([(100, 101, 99, 100)] * 30)                   # true range 2 every candle
    s = planner.suggest(df)
    assert s == {"entry": 100, "stop": pytest.approx(97), "target": pytest.approx(106)}
    short = planner.suggest(df, "short")
    assert planner.side_of(short["entry"], short["stop"], short["target"]) == "short"
