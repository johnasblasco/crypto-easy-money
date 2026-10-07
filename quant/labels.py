"""Prediction targets, built so they can never leak into features.

Timing convention (see quant.data): bars are indexed by CLOSE time. A decision
made at time ``t`` may use every bar with close <= t. The trade is filled one
``latency`` later at the close of the 1m bar ending at ``t + latency``, and
exits at the close of the 1m bar ending at ``t + latency + horizon``.

Filling at a later bar than the one that generated the signal does two things:
it models a realistic reaction delay, and it removes the bid-ask-bounce
artifact where the signal bar's own closing tick (at the bid or the ask)
mechanically "predicts" a reversal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decision_times(m1: pd.DataFrame, step: str, start=None, end=None) -> pd.DatetimeIndex:
    """Regular decision grid (bar close times) inside the data range."""
    idx = m1.index
    first = idx[0].ceil(step)
    times = pd.date_range(first, idx[-1], freq=step, tz="UTC")
    if start is not None:
        times = times[times >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        times = times[times < pd.Timestamp(end, tz="UTC")]
    return times


def fill_prices(m1: pd.DataFrame, price: str = "vwap") -> pd.Series:
    """Log fill price per 1m bar: the bar's VWAP (quote volume / volume), or its close.

    VWAP of the bar after the signal is a more realistic taker fill than its
    last trade and removes most bid-ask bounce from labels. Falls back to the
    close when the bar had no volume.
    """
    close = m1["close"]
    if price == "close":
        return np.log(close)
    vwap = m1["quote_volume"] / m1["volume"].where(m1["volume"] > 0)
    vwap = vwap.where((vwap >= m1["low"] * 0.999) & (vwap <= m1["high"] * 1.001))
    return np.log(vwap.fillna(close))


def forward_returns(m1: pd.DataFrame, times: pd.DatetimeIndex, horizon_min: int,
                    latency_min: int = 1, price: str = "vwap") -> pd.DataFrame:
    """Forward log return from the delayed entry to the exit, per decision time.

    Entry is filled during the 1m bar that closes at ``t + latency`` (at its
    VWAP by default), exit during the bar closing ``horizon`` later.
    Columns: ``entry_ts``, ``exit_ts`` (label end, used for purging), ``fwd_ret``.
    A label touching a missing price (exchange outage) is NaN.
    """
    lc = fill_prices(m1, price)
    entry_ts = times + pd.Timedelta(minutes=latency_min)
    exit_ts = entry_ts + pd.Timedelta(minutes=horizon_min)
    entry = lc.reindex(entry_ts).to_numpy()
    exit_ = lc.reindex(exit_ts).to_numpy()
    return pd.DataFrame(
        {"entry_ts": entry_ts, "exit_ts": exit_ts, "fwd_ret": exit_ - entry},
        index=times,
    )


def triple_barrier(m1: pd.DataFrame, times: pd.DatetimeIndex, horizon_min: int, up: np.ndarray,
                   down: np.ndarray, latency_min: int = 1, chunk: int = 4096) -> pd.DataFrame:
    """First-touch triple-barrier outcome for each decision time.

    ``up``/``down`` are per-sample barrier distances in log-return units
    (positive numbers), typically k * ex-ante volatility. Barriers are checked on
    1m highs/lows after entry; if both are touched in the same minute the
    outcome is treated as the STOP (conservative). Returns ``label`` in
    {+1 (take-profit first), -1 (stop first), 0 (time barrier)}, the realised
    log return ``tb_ret`` and ``exit_ts``.
    """
    lc = np.log(m1["close"]).to_numpy()
    lh = np.log(m1["high"]).to_numpy()
    ll = np.log(m1["low"]).to_numpy()
    pos = m1.index.get_indexer(times + pd.Timedelta(minutes=latency_min))
    n = len(times)
    label = np.zeros(n, dtype=np.int8)
    tb_ret = np.full(n, np.nan)
    exit_off = np.full(n, horizon_min, dtype=np.int64)
    steps = np.arange(1, horizon_min + 1)
    for s in range(0, n, chunk):
        p = pos[s:s + chunk]
        ok = (p >= 0) & (p + horizon_min < len(lc))
        rows = np.where(ok)[0]
        if not len(rows):
            continue
        base = p[rows]
        win = base[:, None] + steps[None, :]
        entry = lc[base]
        hi = lh[win] - entry[:, None]
        lo = ll[win] - entry[:, None]
        u = up[s:s + chunk][rows][:, None]
        d = down[s:s + chunk][rows][:, None]
        hit_up = hi >= u
        hit_dn = lo <= -d
        first_up = np.where(hit_up.any(1), hit_up.argmax(1), horizon_min)
        first_dn = np.where(hit_dn.any(1), hit_dn.argmax(1), horizon_min)
        lab = np.where((first_dn <= first_up) & (first_dn < horizon_min), -1,
                       np.where(first_up < horizon_min, 1, 0))
        ret = np.where(lab == 1, u[:, 0], np.where(lab == -1, -d[:, 0], lc[base + horizon_min] - entry))
        off = np.minimum(first_up, first_dn) + 1
        off = np.where(lab == 0, horizon_min, off)
        idx = s + rows
        label[idx] = lab
        tb_ret[idx] = ret
        exit_off[idx] = off
        bad = ~np.isfinite(ret)
        tb_ret[idx[bad]] = np.nan
    entry_ts = times + pd.Timedelta(minutes=latency_min)
    return pd.DataFrame({
        "label": label,
        "tb_ret": tb_ret,
        "exit_ts": entry_ts + pd.to_timedelta(exit_off, unit="min"),
    }, index=times)
