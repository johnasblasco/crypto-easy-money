"""H: cross-sectional momentum among liquid coins (weekly rebalance).

Pre-registered (Liu-Tsyvinski-Wu; Dobrynskaya; Fieberg et al. 2024): rank the
eligible coins each Monday 00:00 UTC by past L-day return (L in 7, 14, 28),
skipping the most recent day to avoid short-term reversal/bounce. Hold one week.

  long_top   long the top tercile, equal weight (spot)       vs  equal-weight all (spot)
  long_short top tercile minus bottom tercile (perp)         vs  zero

Point-in-time eligibility: >= 60 days of history at the rebalance. Our coin list
is today's survivors, so results are an UPPER BOUND (dead coins are missing,
which flatters 'buy losers' far more than 'buy winners', but still biases).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import metrics as M
from quant.costs import COST_MODELS
from quant.data import KLINE_DIR, UNIVERSE
from quant.experiment import HOLDOUT_START, save
from quant.studies.trend import daily_bars


def weekly_panel(symbols):
    fills, closes = {}, {}
    for s in symbols:
        d = daily_bars(s)
        fills[s] = d["fill"]
        closes[s] = d["close"]
    return pd.DataFrame(closes), pd.DataFrame(fills)


def run(L: int, closes: pd.DataFrame, fills: pd.DataFrame, end=HOLDOUT_START):
    spot, perp = COST_MODELS["spot_taker"], COST_MODELS["perp_taker"]
    mondays = closes.index[(closes.index.dayofweek == 0)]
    mondays = mondays[mondays < pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=8)]
    first_seen = closes.apply(lambda c: c.first_valid_index())
    rows = []
    prev_w = {"top": pd.Series(dtype=float), "ew": pd.Series(dtype=float), "ls": pd.Series(dtype=float)}
    for t in mondays:
        nxt = t + pd.Timedelta(days=7)
        if nxt not in fills.index:
            continue
        tm1 = t - pd.Timedelta(days=1)
        if tm1 not in closes.index:
            continue
        elig = [s for s in closes.columns if pd.notna(first_seen[s]) and t - first_seen[s] >= pd.Timedelta(days=60)]
        elig = [s for s in elig if np.isfinite(closes.at[tm1, s])]
        if len(elig) < 6:
            continue
        past = np.log(closes.loc[t - pd.Timedelta(days=1), elig] / closes.loc[t - pd.Timedelta(days=1 + L), elig])
        past = past.dropna()
        fwd = (fills.loc[nxt, past.index] - fills.loc[t, past.index]).dropna()
        past = past[fwd.index]
        if len(past) < 6:
            continue
        n3 = len(past) // 3
        ranked = past.sort_values()
        top, bot = ranked.index[-n3:], ranked.index[:n3]
        w = {
            "top": pd.Series(1 / n3, index=top),
            "ew": pd.Series(1 / len(past), index=past.index),
            "ls": pd.concat([pd.Series(1 / n3, index=top), pd.Series(-1 / n3, index=bot)]),
        }
        r = {}
        for k, wk in w.items():
            gross = float((wk * np.expm1(fwd.reindex(wk.index))).sum())
            turnover = float(wk.sub(prev_w[k], fill_value=0).abs().sum())
            cost = spot if k != "ls" else perp
            side = np.mean([cost.one_side(s) for s in wk.index])
            hold = cost.holding(7 * 1440) * float(wk.abs().sum()) if k == "ls" else 0.0
            r[k] = np.log1p(gross - turnover * side - hold)
            prev_w[k] = wk
        rows.append({"t": t, **r, "n": len(past)})
    df = pd.DataFrame(rows).set_index("t")
    ann = 52
    out = {"L": L, "weeks": len(df)}
    for k in ("top", "ew", "ls"):
        x = df[k].to_numpy()
        p, lo, hi = M.mean_pvalue(x, n_boot=2000)
        out[k] = {"ann_ret": float(np.expm1(x.mean() * ann)), "sharpe": float(M.sharpe(x) * np.sqrt(ann)),
                  "max_dd": M.max_drawdown(x), "p_mean_gt0": p}
    diff = (df["top"] - df["ew"]).to_numpy()
    p, lo, hi = M.mean_pvalue(diff, n_boot=2000)
    out["top_minus_ew"] = {"ann": float(diff.mean() * ann), "p": p, "ci90_ann": (lo * ann, hi * ann)}
    out["by_year_top_minus_ew"] = {int(y): round(float(g.mean() * ann), 3) for y, g in (df["top"] - df["ew"]).groupby(df.index.year)}
    out["by_year_ls"] = {int(y): round(float(g.mean() * ann), 3) for y, g in df["ls"].groupby(df.index.year)}
    return out


def main():
    syms = [s for s in UNIVERSE if (KLINE_DIR / f"{s}.parquet").exists()]
    closes, fills = weekly_panel(syms)
    res = []
    for L in (7, 14, 28):
        r = run(L, closes, fills)
        res.append(r)
        print(f"L={L:2d}d weeks={r['weeks']} | top SR {r['top']['sharpe']:.2f} ann {r['top']['ann_ret']:.1%} | "
              f"EW SR {r['ew']['sharpe']:.2f} ann {r['ew']['ann_ret']:.1%} | top-EW {r['top_minus_ew']['ann']:.1%}/yr "
              f"p={r['top_minus_ew']['p']:.3f} | L/S SR {r['ls']['sharpe']:.2f} ann {r['ls']['ann_ret']:.1%} "
              f"p={r['ls']['p_mean_gt0']:.3f}", flush=True)
        print("   by year top-EW:", r["by_year_top_minus_ew"], " L/S:", r["by_year_ls"], flush=True)
    save("xsection_momentum", res)


if __name__ == "__main__":
    main()
