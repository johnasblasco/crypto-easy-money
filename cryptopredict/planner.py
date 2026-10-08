"""Trade planner: check an entry / take-profit / stop-loss plan against the coin's own history.

Drawing levels on a chart does not create an edge. In a market with no
predictable direction, a target twice as far away as the stop is hit first
about one time in three, which exactly cancels the bigger payoff, and fees
push the average below zero. What a planner *can* do honestly:

* show the plan on the chart and its reward/risk;
* say which win rate it needs to break even after fees;
* replay a trade with the same percentage distances from every past candle
  of this coin and timeframe, and report how often the target came first,
  how often the stop came first, and what that averaged after fees;
* compare with entering at the same moments and simply selling after the
  same number of candles (no levels), so you see what the levels added;
* size the position so that hitting the stop loses a fixed share of the account.

Replay rules (conservative where the candles can't tell):

* entry at a candle's close; the next ``max_bars`` candles decide the outcome;
* a long hits the target when a candle's high reaches it and the stop when
  a candle's low reaches it (mirrored for a short);
* if both happen inside the same candle, the order is unknown and the trade
  counts as a loss;
* a candle that opens beyond the stop fills at its open (a gap), not at the stop;
* if neither level is hit, the trade closes at the last candle's close (timeout);
* every trade pays ``cost`` (fee + half-tick spread) on the way in and out.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MAX_BARS_LIMIT = 500
ATR_PERIOD = 14
SUGGEST_STOP_ATR = 1.5   # stop 1.5 x ATR away
SUGGEST_REWARD_RISK = 2  # target twice as far as the stop


def side_of(entry: float, stop: float, target: float) -> str:
    """'long' if target > entry > stop, 'short' if target < entry < stop; otherwise a ValueError."""
    for name, v in (("entry", entry), ("stop", stop), ("target", target)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise ValueError(f"{name} must be a positive price")
    if stop < entry < target:
        return "long"
    if target < entry < stop:
        return "short"
    raise ValueError("For a long, the stop must be below the entry and the target above it; "
                     "for a short, the other way round.")


def replay(df: pd.DataFrame, reward: float, risk: float, side: str, max_bars: int, cost: float) -> pd.DataFrame:
    """Replay the plan from every candle that has ``max_bars`` candles after it.

    ``reward`` and ``risk`` are fractions of the entry price (0.044 = 4.4%).
    Returns one row per start: outcome ('target', 'stop' or 'timeout'), bars
    held, net return of the plan, and net return of holding the same number
    of candles without levels.
    """
    if not 1 <= max_bars <= MAX_BARS_LIMIT:
        raise ValueError(f"max_bars must be between 1 and {MAX_BARS_LIMIT}")
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    starts = len(c) - max_bars
    if starts < 1:
        return pd.DataFrame(columns=["timestamp", "outcome", "bars", "net", "hold_net"])

    win = np.lib.stride_tricks.sliding_window_view
    entry = c[:starts, None]
    # Candles after each entry: rows are starts, columns are the 1..max_bars candles that follow.
    fo, fh, fl, fc = (win(x[1:], max_bars)[:starts] for x in (o, h, l, c))
    if side == "long":
        tp, sl = entry * (1 + reward), entry * (1 - risk)
        hit_tp, hit_sl = fh >= tp, fl <= sl
        sl_fill = np.minimum(sl, fo)           # a gap down fills at the open
    else:
        tp, sl = entry * (1 - reward), entry * (1 + risk)
        hit_tp, hit_sl = fl <= tp, fh >= sl
        sl_fill = np.maximum(sl, fo)

    big = max_bars + 1
    first_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), big)
    first_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), big)
    rows = np.arange(starts)
    stopped = (first_sl <= first_tp) & (first_sl < big)     # a same-candle tie counts as a loss
    target = (first_tp < first_sl)
    exit_col = np.where(stopped, first_sl, np.where(target, first_tp, max_bars - 1))
    exit_px = np.where(stopped, sl_fill[rows, exit_col],
                       np.where(target, tp[:, 0], fc[rows, max_bars - 1]))
    sign = 1.0 if side == "long" else -1.0
    gross = sign * (exit_px / entry[:, 0] - 1)
    hold_gross = sign * (fc[rows, exit_col] / entry[:, 0] - 1)
    return pd.DataFrame({
        "timestamp": df["timestamp"].iloc[:starts].to_numpy(),
        "outcome": np.where(stopped, "stop", np.where(target, "target", "timeout")),
        "bars": exit_col + 1,
        "net": gross - 2 * cost,
        "hold_net": hold_gross - 2 * cost,
    })


def breakeven_win_rate(reward: float, risk: float, cost: float) -> float:
    """Share of trades that must hit the target (the rest hitting the stop) to break even after fees."""
    return min(1.0, (risk + 2 * cost) / (reward + risk))


def summarize(trades: pd.DataFrame, reward: float, risk: float, cost: float) -> dict:
    n = len(trades)
    out = {
        "starts": n,
        "breakeven_win_rate": breakeven_win_rate(reward, risk, cost),
        # With no predictable direction, the closer level is hit first proportionally more often.
        "random_walk_win_rate": risk / (reward + risk),
    }
    if n == 0:
        return out
    counts = trades["outcome"].value_counts()
    avg_bars = float(trades["bars"].mean())
    # Trades start on every candle, so neighbours overlap and share most of their path. Count
    # roughly one independent trade per holding period when judging how sure the numbers are.
    n_eff = max(1.0, n / avg_bars)
    net, hold = trades["net"], trades["hold_net"]
    se = float(net.std(ddof=1) / math.sqrt(n_eff)) if n > 1 else float("nan")
    diff = net - hold
    se_diff = float(diff.std(ddof=1) / math.sqrt(n_eff)) if n > 1 else float("nan")
    out.update({
        "period": [trades["timestamp"].iloc[0].isoformat(), trades["timestamp"].iloc[-1].isoformat()],
        "independent_trades": round(n_eff, 1),
        "win_rate": float(counts.get("target", 0) / n),
        "loss_rate": float(counts.get("stop", 0) / n),
        "timeout_rate": float(counts.get("timeout", 0) / n),
        "avg_bars": avg_bars,
        "avg_net": float(net.mean()),
        "avg_net_low": float(net.mean() - 2 * se),
        "avg_net_high": float(net.mean() + 2 * se),
        "median_net": float(net.median()),
        "share_profitable": float((net > 0).mean()),
        "hold_avg_net": float(hold.mean()),
        "levels_vs_hold": float(diff.mean()),
        "levels_vs_hold_low": float(diff.mean() - 2 * se_diff),
        "levels_vs_hold_high": float(diff.mean() + 2 * se_diff),
    })
    return out


def verdict(stats: dict) -> tuple[str, str]:
    if stats.get("starts", 0) == 0 or stats.get("independent_trades", 0) < 10:
        return "NOT ENOUGH HISTORY", "Too few past candles to replay this plan; shorten the hold or load more history."
    lo, hi = stats["avg_net_low"], stats["avg_net_high"]
    if hi < 0:
        return "LOSES ON AVERAGE", ("Replayed from every past candle, this plan lost money on average after fees, "
                                    "and luck does not explain it.")
    if lo > 0:
        text = "This plan made money on average after fees in this coin's recent history. "
        if stats["levels_vs_hold_low"] <= 0:
            text += ("But buying at the same moments and selling after the same time without levels did about as well: "
                     "the profit came from the trend in this period, not from the levels. ")
        return "HELD UP IN THE PAST", text + ("That is the past of one coin; if you tried several levels and kept "
                                              "the best, it is likely luck. Not a forecast.")
    return "NO PROVEN EDGE", ("The average after fees is too close to zero to tell from luck. "
                              "The levels set your risk, they don't predict the direction.")


def position_size(entry: float, stop: float, target: float, account: float, risk_share: float, cost: float) -> dict:
    """How much to buy so that hitting the stop (fees included) loses ``risk_share`` of ``account``."""
    loss_per_unit = abs(entry - stop) + cost * (entry + stop)
    gain_per_unit = abs(target - entry) - cost * (entry + target)
    risk_amount = account * risk_share
    qty = risk_amount / loss_per_unit
    notional = qty * entry
    out = {
        "account": account,
        "risk_share": risk_share,
        "risk_amount": risk_amount,
        "quantity": qty,
        "notional": notional,
        "loss_at_stop": qty * loss_per_unit,
        "gain_at_target": qty * gain_per_unit,
        "leverage_needed": notional / account,
    }
    if notional > account:
        capped = account / entry
        out.update({"capped_quantity": capped, "capped_loss_at_stop": capped * loss_per_unit})
    return out


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Average true range of the last ``period`` candles."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
                   axis=1).max(axis=1)
    return float(tr.iloc[-period:].mean())


def suggest(df: pd.DataFrame, side: str = "long") -> dict:
    """Entry at the last close, stop 1.5 ATR away, target at 2x the stop distance.

    A common volatility-scaled starting point: the stop sits outside one
    candle's normal noise. It is a template, not a forecast.
    """
    entry = float(df["close"].iloc[-1])
    dist = min(SUGGEST_STOP_ATR * atr(df), 0.5 * entry)
    sign = 1 if side == "long" else -1
    return {"entry": entry, "stop": entry - sign * dist, "target": entry + sign * SUGGEST_REWARD_RISK * dist}


def plan(df: pd.DataFrame, entry: float, stop: float, target: float, max_bars: int, cost: float,
         account: float | None = None, risk_share: float | None = None) -> dict:
    side = side_of(entry, stop, target)
    reward, risk = abs(target - entry) / entry, abs(entry - stop) / entry
    trades = replay(df, reward, risk, side, max_bars, cost)
    stats = summarize(trades, reward, risk, cost)
    label, text = verdict(stats)
    return {
        "plan": {"side": side, "entry": entry, "stop": stop, "target": target, "reward_pct": reward,
                 "risk_pct": risk, "reward_risk": reward / risk, "max_bars": max_bars},
        "cost_per_side": cost,
        "history": stats,
        "verdict": label,
        "verdict_text": text,
        "size": position_size(entry, stop, target, account, risk_share, cost) if account and risk_share else None,
    }
