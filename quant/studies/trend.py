"""H: daily time-series trend (Donchian + past-return ensemble), long/flat, vol-sized.

Pre-registered from the literature (Zarattini-Pagani-Barbon 2025; Liu-Tsyvinski
2021; Quantpedia MAX revisit) before looking at our data:

* Daily bars close at 00:00 UTC. Signal uses bars closed by then; the trade
  fills at the VWAP of the next 1m bar (00:00-00:01) and is held to the next
  day's equivalent fill.
* Donchian L in {10,20,30,60,90}: long when close >= max(close, prior L days),
  exit when close <= min(close, prior L//2 days). Past-return L in {7,14,28,56}:
  long when ln(P_t / P_{t-L}) > 0. Ensemble S = mean of the 9 components.
* Weight w = S * min(1, TARGET_VOL / sigma_t), sigma = EWMA(halflife 20d) of
  daily log returns, annualised (365). Spot: no leverage, no shorts.

Controls (the trend signal has to beat these, not just buy & hold):
1. buy & hold (w = 1)
2. vol-targeted buy & hold (w = min(1, TARGET_VOL / sigma)) -> isolates timing from sizing
3. random-timing null: the S series circularly shifted by random offsets
   (same time in market, same turnover profile, no timing information)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import metrics as M
from quant.costs import COST_MODELS
from quant.data import resample
from quant.experiment import HOLDOUT_START, m1_frame, save
from quant.labels import fill_prices

TARGET_VOL = 0.5
DONCHIAN = (10, 20, 30, 60, 90)
PASTRET = (7, 14, 28, 56)


def daily_bars(symbol: str, m1: pd.DataFrame | None = None) -> pd.DataFrame:
    m1 = m1_frame(symbol) if m1 is None else m1
    d = resample(m1, "1D")
    full = pd.date_range(d.index[0], d.index[-1], freq="1D", tz="UTC")
    d = d.reindex(full)
    # A day with an exchange outage still has a valid last price; only a fully missing day is NaN.
    d.loc[d["gap_frac"] >= 0.999, ["open", "high", "low", "close"]] = np.nan
    # Fill price for a decision at day close t: VWAP of the 1m bar closing at t + 1min.
    lp = fill_prices(m1, "vwap")
    d["fill"] = lp.reindex(d.index + pd.Timedelta("1min")).to_numpy()
    return d


def donchian_state(close: pd.Series, L: int) -> pd.Series:
    hi = close.shift(1).rolling(L).max()
    lo = close.shift(1).rolling(max(L // 2, 2)).min()
    state = np.zeros(len(close))
    on = 0
    c, h, l_ = close.to_numpy(), hi.to_numpy(), lo.to_numpy()
    for i in range(len(c)):
        if np.isnan(h[i]) or np.isnan(l_[i]) or np.isnan(c[i]):
            # Not enough history (or no price): the component is unknown, not "flat".
            on = 0
            state[i] = np.nan
            continue
        elif not on and c[i] >= h[i]:
            on = 1
        elif on and c[i] <= l_[i]:
            on = 0
        state[i] = on
    return pd.Series(state, index=close.index)


def ensemble(close: pd.Series) -> pd.Series:
    comps = [donchian_state(close, L) for L in DONCHIAN]
    comps += [(np.log(close / close.shift(L)) > 0).astype(float).where(close.shift(L).notna()) for L in PASTRET]
    return pd.concat(comps, axis=1).mean(axis=1, skipna=False)


def ewma_vol(close: pd.Series, halflife: int = 20) -> pd.Series:
    r = np.log(close).diff()
    return np.sqrt((r ** 2).ewm(halflife=halflife, min_periods=halflife).mean() * 365)


def backtest_weights(w: pd.Series, d: pd.DataFrame, symbol: str, cost) -> pd.Series:
    """Daily net log-ish returns of holding weight w_t from fill t to fill t+1."""
    fwd = d["fill"].shift(-1) - d["fill"]           # log return fill->fill
    w = w.clip(0, 1).fillna(0)
    gross = w * (np.expm1(fwd))                       # simple return on the risky fraction
    turnover = (w - w.shift(1).fillna(0)).abs()
    side = cost.one_side(symbol)
    if isinstance(side, pd.Series):          # a per-day cost (e.g. tick-size dependent)
        side = side.reindex(w.index)
    net_simple = gross - turnover * side
    return np.log1p(net_simple).where(fwd.notna())


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    ann = 365
    return {
        "days": int(len(r)),
        "cagr": float(np.expm1(r.mean() * ann)),
        "vol": float(r.std() * np.sqrt(ann)),
        "sharpe": float(M.sharpe(r.to_numpy()) * np.sqrt(ann)),
        "sortino": float(M.sortino(r.to_numpy()) * np.sqrt(ann)),
        "max_dd": M.max_drawdown(r.to_numpy()),
    }


def alpha_vs(r: pd.Series, bench: pd.Series) -> dict:
    """OLS r = a + b*bench with Newey-West t-stat on alpha (daily)."""
    df = pd.concat([r, bench], axis=1).dropna()
    beta, t, _ = M.hac_ols(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy().reshape(-1, 1), lags=10)
    return {"alpha_ann": float(beta[0] * 365), "beta": float(beta[1]), "alpha_t": float(t[0])}


def run_symbol(symbol: str, cost_name: str = "spot_taker", end: str = HOLDOUT_START, n_null: int = 500, seed: int = 0):
    d = daily_bars(symbol)
    d = d[d.index < pd.Timestamp(end, tz="UTC")]
    cost = COST_MODELS[cost_name] if isinstance(cost_name, str) else cost_name
    S = ensemble(d["close"])
    sig = ewma_vol(d["close"])
    size = (TARGET_VOL / sig).clip(upper=1.0)
    start = S.dropna().index[0]
    d, S, size = d[d.index >= start], S[S.index >= start], size[size.index >= start]

    r_trend = backtest_weights(S * size, d, symbol, cost)
    r_bh = backtest_weights(pd.Series(1.0, index=d.index), d, symbol, cost)
    r_vt = backtest_weights(size, d, symbol, cost)

    out = {
        "symbol": symbol, "cost": getattr(cost, "name", str(cost_name)),
        "period": [str(d.index[0].date()), str(d.index[-1].date())],
        "time_in_market": float((S > 0).mean()), "avg_weight": float((S * size).mean()),
        "trend": stats(r_trend), "buy_hold": stats(r_bh), "voltarget_hold": stats(r_vt),
        "alpha_vs_bh": alpha_vs(r_trend, r_bh), "alpha_vs_voltarget": alpha_vs(r_trend, r_vt),
    }
    # Random-timing null: circularly shift the signal (keeps its persistence and time in market).
    rng = np.random.default_rng(seed)
    s_vals = S.to_numpy()
    null_sharpes = []
    for _ in range(n_null):
        k = rng.integers(60, len(s_vals) - 60)
        Sn = pd.Series(np.roll(s_vals, k), index=S.index)
        null_sharpes.append(stats(backtest_weights(Sn * size, d, symbol, cost))["sharpe"])
    null_sharpes = np.array(null_sharpes)
    out["null_sharpe_mean"] = float(null_sharpes.mean())
    out["null_sharpe_p95"] = float(np.percentile(null_sharpes, 95))
    out["p_vs_random_timing"] = float(((null_sharpes >= out["trend"]["sharpe"]).sum() + 1) / (len(null_sharpes) + 1))
    # Stability by year
    by_year = {}
    for y in sorted(set(d.index.year)):
        m = d.index.year == y
        by_year[int(y)] = {"trend": stats(r_trend[m])["sharpe"], "voltarget": stats(r_vt[m])["sharpe"],
                           "bh": stats(r_bh[m])["sharpe"],
                           "trend_ret": float(r_trend[m].sum()), "vt_ret": float(r_vt[m].sum())}
    out["by_year"] = by_year
    # Robustness: each component alone (no cherry-picking -- report all)
    comp = {}
    for L in DONCHIAN:
        comp[f"donchian_{L}"] = stats(backtest_weights(donchian_state(d["close"], L) * size, d, symbol, cost))["sharpe"]
    for L in PASTRET:
        s_l = (np.log(d["close"] / d["close"].shift(L)) > 0).astype(float)
        comp[f"pastret_{L}"] = stats(backtest_weights(s_l * size, d, symbol, cost))["sharpe"]
    out["components_sharpe"] = comp
    return out, {"trend": r_trend, "bh": r_bh, "voltarget": r_vt}


def main():
    from quant.data import UNIVERSE, KLINE_DIR

    results = []
    for sym in UNIVERSE:
        if not (KLINE_DIR / f"{sym}.parquet").exists():
            continue
        res, _ = run_symbol(sym)
        results.append(res)
        t, b, v = res["trend"], res["buy_hold"], res["voltarget_hold"]
        print(f"{sym:9s} trend SR {t['sharpe']:.2f} DD {t['max_dd']:.2f} CAGR {t['cagr']:.2%} | "
              f"B&H SR {b['sharpe']:.2f} DD {b['max_dd']:.2f} | VT SR {v['sharpe']:.2f} DD {v['max_dd']:.2f} | "
              f"alpha vs VT {res['alpha_vs_voltarget']['alpha_ann']:.2%} t={res['alpha_vs_voltarget']['alpha_t']:.2f} | "
              f"random-timing p={res['p_vs_random_timing']:.3f} (null mean {res['null_sharpe_mean']:.2f})", flush=True)
    save("trend_daily", results)


if __name__ == "__main__":
    main()
