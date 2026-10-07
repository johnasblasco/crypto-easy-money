"""Pre-registered event hypotheses (definitions fixed before looking at results).

Sources: research sweep (Kitron & Wengrowicz 2026 flow-driven reversal; cascade/
capitulation and idiosyncratic-shock hypotheses; Osler stop cascades for sweeps;
Quantpedia MAX breakout). Every cell (hypothesis x parameter x horizon) is
reported; p-values are Holm- and BH-adjusted across ALL cells.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from quant import metrics as M
from quant.costs import COST_MODELS
from quant.data import KLINE_DIR, UNIVERSE
from quant.events import Bars, outcomes, placebo, select, summarize
from quant.experiment import HOLDOUT_START, m1_frame, save

HORIZONS = (15, 60, 240)
COSTS = ("spot_taker", "perp_taker", "perp_maker")


def research_frame(sym: str) -> pd.DataFrame:
    m1 = m1_frame(sym)
    return m1[m1.index < pd.Timestamp(HOLDOUT_START, tz="UTC") - pd.Timedelta(days=2)]


# ------------------------------------------------------------- definitions

def h_reversal_15m_control(b: Bars) -> np.ndarray:
    """Positive control: bet against the last 15m return, at quarter-hour marks."""
    r = b.ret(15)
    sig = -np.sign(r)
    sig[(b.index.minute % 15 != 0)] = 0
    sig[b.gaps(15) > 0] = 0
    return sig


def h_flow_driven_reversal(b: Bars, W: int, k: float) -> np.ndarray:
    r = b.ret(W)
    ti = b.imbalance(W)
    z = r / (b.sigma * np.sqrt(W) + 1e-12)
    pct = b.trailing_pctile(np.abs(ti))
    ev = (np.sign(ti) == np.sign(r)) & (np.abs(z) >= k) & (pct >= 0.8) & (b.gaps(W) == 0)
    return np.where(ev, -np.sign(r), 0)


def h_cascade_reversal(b: Bars, W: int, k: float = 4.0) -> np.ndarray:
    r = b.ret(W)
    ti = b.imbalance(W)
    z = r / (b.sigma * np.sqrt(W) + 1e-12)
    intensity = b.same_hour_ratio(b.trades, W)
    down = (z <= -k) & (ti <= -0.3) & (intensity >= 3)
    up = (z >= k) & (ti >= 0.3) & (intensity >= 3)
    ok = b.gaps(W) == 0
    return np.where(down & ok, 1, np.where(up & ok, -1, 0))


def h_sweep_reclaim(b: Bars, M_: int = 240) -> np.ndarray:
    """Trade through the prior M-minute high/low (level >= 15 min old), close back inside, wick-heavy, high volume."""
    hi = pd.Series(b.high).shift(15).rolling(M_ - 15, min_periods=M_ - 20).max().to_numpy()
    lo = pd.Series(b.low).shift(15).rolling(M_ - 15, min_periods=M_ - 20).min().to_numpy()
    rng = b.high - b.low
    upper_wick = (b.high - np.maximum(b.lc, b.lc - b.r)) / (rng + 1e-12)
    lower_wick = (np.minimum(b.lc, b.lc - b.r) - b.low) / (rng + 1e-12)
    vol_ratio = b.same_hour_ratio(b.q, 1)
    sweep_up = (b.high > hi) & (b.lc < hi) & (upper_wick >= 0.5) & (vol_ratio >= 2)
    sweep_dn = (b.low < lo) & (b.lc > lo) & (lower_wick >= 0.5) & (vol_ratio >= 2)
    return np.where(sweep_up, -1, np.where(sweep_dn, 1, 0))


def h_breakout_continuation(b: Bars, M_: int = 1440) -> np.ndarray:
    """Close above the prior 24h high with buy-side flow -> continuation (mirror for lows)."""
    hi = pd.Series(b.high).shift(1).rolling(M_, min_periods=M_ - 30).max().to_numpy()
    lo = pd.Series(b.low).shift(1).rolling(M_, min_periods=M_ - 30).min().to_numpy()
    ti = b.imbalance(15)
    up = (b.lc > hi) & (ti > 0)
    dn = (b.lc < lo) & (ti < 0)
    return np.where(up, 1, np.where(dn, -1, 0))


def h_weekend_reversal(b: Bars) -> np.ndarray:
    """At Monday 00:00 UTC, bet against a large Fri 20:00 -> Mon 00:00 move (|z| >= 1.5 vs prior 26 weekends)."""
    idx = b.index
    sig = np.zeros(b.n)
    mon = np.flatnonzero((idx.dayofweek == 0) & (idx.hour == 0) & (idx.minute == 0))
    W = 52 * 60
    rets = []
    for p in mon:
        if p - W < 0:
            rets.append(np.nan)
            continue
        rets.append(b.lc[p] - b.lc[p - W])
    rets = np.array(rets)
    for i, p in enumerate(mon):
        past = rets[max(0, i - 26):i]
        past = past[np.isfinite(past)]
        if len(past) < 20 or not np.isfinite(rets[i]):
            continue
        z = rets[i] / (past.std() + 1e-12)
        if abs(z) >= 1.5:
            sig[p] = -np.sign(rets[i])
    return sig


def h_idio_vs_market(b: Bars, market: dict, W: int = 60, k: float = 4.0):
    """Own 1h move >= k sigma. Idiosyncratic (low breadth) -> reversal; market-wide (high breadth) -> continuation."""
    r = b.ret(W)
    z = r / (b.sigma * np.sqrt(W) + 1e-12)
    m = market["mkt"]
    breadth = market["breadth_up"]
    big = (np.abs(z) >= k) & (b.gaps(W) == 0)
    same_share = np.where(r > 0, breadth, 1 - breadth)
    idio = big & (same_share < 0.5) & (np.abs(r - m) / (b.sigma * np.sqrt(W) + 1e-12) >= 3)
    mkt_wide = big & (same_share >= 0.8)
    return np.where(idio, -np.sign(r), 0), np.where(mkt_wide, np.sign(r), 0)


_CLOSE_CACHE: dict = {}


def log_close(sym: str) -> pd.Series:
    """Light-weight 1m log close (close-time index), cached without the full frame."""
    if sym not in _CLOSE_CACHE:
        df = pd.read_parquet(KLINE_DIR / f"{sym}.parquet", columns=["open_time", "close"])
        idx = pd.to_datetime(df["open_time"] + 60_000, unit="ms", utc=True)
        _CLOSE_CACHE[sym] = pd.Series(np.log(df["close"].to_numpy()), index=idx)
    return _CLOSE_CACHE[sym]


def market_arrays(symbols, ref_index, W: int = 60) -> dict:
    rets = []
    for s in symbols:
        lc = log_close(s).reindex(ref_index).ffill()
        rets.append((lc - lc.shift(W)).to_numpy())
    R = np.vstack(rets)
    with np.errstate(invalid="ignore"):
        return {"mkt": np.nanmean(R, axis=0), "breadth_up": np.nanmean(R > 0, axis=0),
                "count": np.sum(np.isfinite(R), axis=0)}


# ------------------------------------------------------------------ runner

def cells_for(b: Bars, sym: str, market):
    yield "control_reversal_15m", {}, h_reversal_15m_control(b)
    for W in (15, 60):
        for k in (3.0, 4.0):
            yield "flow_driven_reversal", {"W": W, "k": k}, h_flow_driven_reversal(b, W, k)
    for W in (15, 30):
        yield "cascade_reversal", {"W": W, "k": 4.0}, h_cascade_reversal(b, W)
    yield "sweep_reclaim", {"M": 240}, h_sweep_reclaim(b)
    yield "breakout_continuation", {"M": 1440}, h_breakout_continuation(b)
    yield "weekend_reversal", {}, h_weekend_reversal(b)
    if market is not None:
        idio, mk = h_idio_vs_market(b, market)
        yield "idio_shock_reversal", {"W": 60, "k": 4.0}, idio
        yield "market_shock_continuation", {"W": 60, "k": 4.0}, mk


def main(symbols=None):
    symbols = symbols or [s for s in UNIVERSE if (KLINE_DIR / f"{s}.parquet").exists()]
    t0 = time.time()
    pooled: dict = {}
    for sym in symbols:
        m1 = research_frame(sym)
        b = Bars(m1)
        others = [s for s in symbols if s != sym]
        market = market_arrays(others, m1.index) if others else None
        for name, params, sig in cells_for(b, sym, market):
            for h in HORIZONS if name != "weekend_reversal" else (1200,):
                idx = select(sig, cooldown=h)
                if len(idx) == 0:
                    continue
                ev = outcomes(b, idx, np.sign(sig[idx]).astype(int), h)
                pl = placebo(b, ev, h, n_per_event=3)
                key = (name, json.dumps(params, sort_keys=True), h)
                group = "majors" if sym in ("BTCUSDT", "ETHUSDT") else "alts"
                pooled.setdefault(key, {"majors": [], "alts": [], "placebo": []})
                pooled[key][group].append(ev.assign(sym=sym))
                pooled[key]["placebo"].append(pl)
        import quant.experiment as E
        E._M1_CACHE.pop(sym, None)
        print(f"{sym} done ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for (name, params, h), parts in pooled.items():
        for group in ("majors", "alts"):
            if not parts[group]:
                continue
            ev = pd.concat(parts[group], ignore_index=True)
            pl = pd.concat(parts["placebo"], ignore_index=True)
            for cname in COSTS:
                cost = COST_MODELS[cname]
                rt = np.array([cost.round_trip(s) for s in ev["sym"]]) + cost.holding(h)
                if not cost.allow_short:
                    keep = ev["dir"] > 0
                    e2, rt2 = ev[keep], rt[keep.to_numpy()]
                    pl_c = pl[pl["dir"] > 0]
                else:
                    e2, rt2, pl_c = ev, rt, pl
                if len(e2) == 0:
                    continue
                s = summarize(e2.assign(ret=e2["ret"] - rt2 + rt2.mean()), pl_c, float(rt2.mean()))
                rows.append({"hypothesis": name, "params": params, "h": h, "group": group, "cost": cname, **s})
    df = pd.DataFrame(rows)
    # Multiple-testing adjustment across every cell that has data.
    ok = df["p_net"].notna()
    df.loc[ok, "p_holm"] = M.holm(df.loc[ok, "p_net"].to_numpy())
    df.loc[ok, "p_bh"] = M.benjamini_hochberg(df.loc[ok, "p_net"].to_numpy())
    save("event_hypotheses", df.to_dict(orient="records"))
    cols = ["hypothesis", "params", "h", "group", "cost", "n", "days", "gross_bps", "placebo_gross_bps",
            "excess_vs_placebo_bps", "net_bps", "p_net", "p_holm", "positive_years"]
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 30):
        print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
