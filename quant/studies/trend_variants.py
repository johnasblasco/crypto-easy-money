"""Pre-registered refinements of the daily trend rule (from the research sweep).

V1 efficiency gate: only take trend exposure when the 28-day efficiency ratio
   |ln(P_t/P_{t-28})| / sum|daily log returns| is above its trailing 1-year median.
V2 crowding risk gate: halve exposure when the coin is 'crowded' -- trailing 14d
   return in its top decile (vs trailing year) AND 7d taker imbalance > 0 AND
   7d quote volume > 1.5x its 90d average (a spot proxy for leveraged trend-chasing,
   predicted to raise downside tail risk).
V3 jump-variance exclusion (weekly cross-section): drop coins in the top tercile of
   prior-week realised variance from an equal-weight long basket.

Each variant is compared with the base rule on the same portfolio, same costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import metrics as M
from quant.costs import COST_MODELS
from quant.data import KLINE_DIR, UNIVERSE
from quant.experiment import HOLDOUT_START, m1_frame, save
from quant.studies import trend as T
from quant.studies.trend_robust import daily_bars_offset


def daily_flow(symbol: str) -> pd.DataFrame:
    from quant.data import resample

    d = resample(m1_frame(symbol), "1D")
    return pd.DataFrame({"qv": d["quote_volume"], "imb": (2 * d["taker_buy_quote"] - d["quote_volume"]) / d["quote_volume"]})


def weights(d: pd.DataFrame, flow: pd.DataFrame, variant: str) -> pd.Series:
    close = d["close"]
    S = T.ensemble(close)
    size = (T.TARGET_VOL / T.ewma_vol(close)).clip(upper=1.0)
    w = S * size
    if variant == "er_gate":
        r = np.log(close).diff()
        er = np.log(close / close.shift(28)).abs() / r.abs().rolling(28).sum()
        med = er.shift(1).rolling(365, min_periods=180).median()   # trailing, excludes today
        w = w.where(er > med, 0.0)
    elif variant == "crowding_gate":
        f = flow.reindex(d.index)
        r14 = np.log(close / close.shift(14))
        top = r14 > r14.shift(1).rolling(365, min_periods=180).quantile(0.9)
        imb7 = (f["imb"] * f["qv"]).rolling(7).sum() / f["qv"].rolling(7).sum()
        surge = f["qv"].rolling(7).mean() > 1.5 * f["qv"].shift(7).rolling(90, min_periods=60).mean()
        crowded = top & (imb7 > 0) & surge
        w = w.where(~crowded, w * 0.5)
    valid = S.notna() & size.notna()
    return w.where(valid)


def portfolio(symbols, variant, cost_name="spot_taker", end=HOLDOUT_START):
    cost = COST_MODELS[cost_name]
    rets = {}
    for s in symbols:
        d = daily_bars_offset(s)
        d = d[d.index < pd.Timestamp(end, tz="UTC")]
        w = weights(d, daily_flow(s), variant)
        r = T.backtest_weights(w, d, s, cost).where(w.notna())
        first = d["close"].first_valid_index()
        r[r.index < first + pd.Timedelta(days=365)] = np.nan
        rets[s] = r
    df = pd.DataFrame(rets)
    return np.log(np.expm1(df).mean(axis=1, skipna=True) + 1).dropna()


def jump_variance_exclusion(symbols, end=HOLDOUT_START):
    """Weekly equal-weight long basket with vs without the top-tercile prior-week RV coins."""
    from quant.studies.xsection import weekly_panel

    closes, fills = weekly_panel(symbols)
    spot = COST_MODELS["spot_taker"]
    rv = np.log(closes).diff().pow(2).rolling(7).sum()
    first_seen = closes.apply(lambda c: c.first_valid_index())
    out = []
    prev = {"all": pd.Series(dtype=float), "ex": pd.Series(dtype=float)}
    for t in closes.index[closes.index.dayofweek == 0]:
        nxt = t + pd.Timedelta(days=7)
        if t >= pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=8) or nxt not in fills.index:
            continue
        tm1 = t - pd.Timedelta(days=1)
        elig = [s for s in closes.columns if pd.notna(first_seen[s]) and t - first_seen[s] >= pd.Timedelta(days=60)]
        v = rv.loc[tm1, elig].dropna()
        fwd = (fills.loc[nxt, v.index] - fills.loc[t, v.index]).dropna()
        v = v[fwd.index]
        if len(v) < 6:
            continue
        keep = v[v <= v.quantile(2 / 3)].index
        r = {}
        for k, names in (("all", v.index), ("ex", keep)):
            wk = pd.Series(1 / len(names), index=names)
            gross = float((wk * np.expm1(fwd[names])).sum())
            turn = float(wk.sub(prev[k], fill_value=0).abs().sum())
            r[k] = np.log1p(gross - turn * np.mean([spot.one_side(s) for s in names]))
            prev[k] = wk
        out.append({"t": t, **r})
    df = pd.DataFrame(out).set_index("t")
    diff = (df["ex"] - df["all"]).to_numpy()
    p, lo, hi = M.mean_pvalue(diff, n_boot=2000)
    return {"weeks": len(df), "all_sharpe": M.sharpe(df["all"]) * np.sqrt(52), "ex_sharpe": M.sharpe(df["ex"]) * np.sqrt(52),
            "ex_minus_all_ann": float(diff.mean() * 52), "p": p,
            "by_year": {int(y): round(float(g.mean() * 52), 3) for y, g in (df["ex"] - df["all"]).groupby(df.index.year)}}


def main():
    syms = [s for s in UNIVERSE if (KLINE_DIR / f"{s}.parquet").exists()]
    res = {}
    base = portfolio(syms, "base")
    res["base"] = T.stats(base)
    print("base          ", res["base"], flush=True)
    for v in ("er_gate", "crowding_gate"):
        r = portfolio(syms, v)
        st = T.stats(r)
        df = pd.concat([r, base], axis=1).dropna()
        d = (df.iloc[:, 0] - df.iloc[:, 1]).to_numpy()
        se = np.sqrt(M.newey_west_var(d, 10) / len(d))
        res[v] = {**st, "diff_vs_base_ann": float(d.mean() * 365), "t_vs_base": float(d.mean() / se) if se > 0 else float("nan")}
        print(f"{v:14s}", res[v], flush=True)
    res["jump_variance_exclusion"] = jump_variance_exclusion(syms)
    print("jump var excl ", res["jump_variance_exclusion"], flush=True)
    save("trend_variants", res)


if __name__ == "__main__":
    main()
