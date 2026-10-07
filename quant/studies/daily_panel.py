"""Daily horizon, pooled across coins: does ML on all available information beat
the simple trend rule after costs?

Decision at 00:00 UTC, fill at the VWAP of 00:00-00:01, hold 1 day (or H days
on an H-day grid). One model is trained on the pooled panel (all coins), with
walk-forward 90-day test folds and an inner validation slice for calibration
and the NO-TRADE threshold. Every candidate is simulated with the same
simulator on the same test rows:

  bh        always long (spot) -- beta benchmark
  trend     long when the 9-component trend ensemble S >= 0.5 (rule, not fitted)
  logit     walk-forward logistic regression on all families
  lgbm      walk-forward LightGBM on all families
  trend_meta LightGBM trained only on days the trend rule is long (meta-labeling:
            learns WHEN to follow the trend; flat otherwise)

Comparisons on the equal-weight portfolio's daily net returns use a paired
Newey-West test, so "ML beats trend" needs evidence, not a bigger number.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quant import ledger
from quant import metrics as M
from quant import models
from quant.backtest import Dataset, portfolio_returns, simulate_panel, trading_stats, walk_forward_predict
from quant.costs import COST_MODELS
from quant.data import KLINE_DIR, UNIVERSE
from quant.experiment import make_dataset, save
from quant.studies import trend as T
from quant.validation import walk_forward

FAMS = ["mom", "vol", "act", "flow", "range", "regime", "cal", "xa", "trend", "fng", "ta"]


def trend_score(ds: Dataset, symbol: str) -> pd.Series:
    d = T.daily_bars(symbol)
    S = T.ensemble(d["close"])
    return S.reindex(ds.times)


def paired_test(a: pd.Series, b: pd.Series) -> dict:
    df = pd.concat([a, b], axis=1).dropna()
    d = (df.iloc[:, 0] - df.iloc[:, 1]).to_numpy()
    if len(d) < 30:
        return {"mean_diff_bps": float("nan"), "p": float("nan")}
    v = M.newey_west_var(d, lags=10)
    z = d.mean() / np.sqrt(v / len(d)) if v > 0 else float("nan")
    from scipy import stats
    return {"mean_diff_bps": float(d.mean() * 1e4), "t": float(z), "p": float(1 - stats.norm.cdf(z))}


def main(horizon_days: int = 1, cost_names=("spot_taker", "perp_taker")):
    h = 1440 * horizon_days
    syms = [s for s in UNIVERSE if (KLINE_DIR / f"{s}.parquet").exists()]
    t0 = time.time()
    parts = []
    for s in syms:
        ds = make_dataset(s, h, families=FAMS)
        X = ds.X.assign(trend__ensemble=trend_score(ds, s).to_numpy())
        # Drop each coin's first 60 days (thin, freshly listed markets).
        keep = (ds.times >= ds.times[0] + pd.Timedelta(days=60)) & X["trend__ensemble"].notna().to_numpy()
        parts.append(Dataset(s, h, X[keep], ds.fwd_ret[keep], ds.label_end[keep], ds.regimes[keep]))
        print(f"{s}: {keep.sum()} days ({time.time() - t0:.0f}s)", flush=True)
    panel = Dataset.pool(parts)
    print("panel", panel.X.shape, flush=True)
    folds = walk_forward(panel.times, pd.DatetimeIndex(panel.label_end), test_size="90D", min_train="365D")

    # Meta-labeling panel: only rows where the trend rule is long; the model learns when to follow it.
    on = panel.X["trend__ensemble"].to_numpy() >= 0.5
    meta_panel = Dataset("PANEL", h, panel.X[on], panel.fwd_ret[on], panel.label_end[on],
                         panel.regimes[on] if not panel.regimes.empty else pd.DataFrame(), sym=panel.sym[on])
    meta_folds = walk_forward(meta_panel.times, pd.DatetimeIndex(meta_panel.label_end), test_size="90D", min_train="365D")
    results = {"horizon_days": horizon_days, "symbols": syms, "n_rows": int(len(panel.X))}
    port = {}
    for cname in cost_names:
        cost = COST_MODELS[cname]
        preds = {}
        preds["logit"] = walk_forward_predict(panel, folds, models.logistic(), cost=cost)
        preds["lgbm"] = walk_forward_predict(panel, folds, models.lightgbm(n_estimators=200, min_child_samples=200), cost=cost)
        base = preds["logit"].copy()
        # Rule-based comparators on exactly the same test rows.
        key = pd.MultiIndex.from_arrays([panel.times, panel.sym])
        S = pd.Series(panel.X["trend__ensemble"].to_numpy(), index=key)
        s_vals = S.reindex(pd.MultiIndex.from_arrays([base.index, base["sym"]])).to_numpy()
        preds["trend"] = base.assign(pos=np.where(s_vals >= 0.5, 1, 0))
        preds["bh"] = base.assign(pos=1)
        # Meta-labeling: trained only on trend-long rows; long only if the trend is long AND the model agrees.
        meta = walk_forward_predict(meta_panel, meta_folds, models.lightgbm(n_estimators=200, min_child_samples=100),
                                    cost=COST_MODELS["spot_taker"] if not cost.allow_short else cost)
        mkey = pd.MultiIndex.from_arrays([meta.index, meta["sym"]])
        mpos = pd.Series(np.clip(meta["pos"].to_numpy(), 0, 1), index=mkey)
        preds["trend_meta"] = base.assign(pos=mpos.reindex(pd.MultiIndex.from_arrays([base.index, base["sym"]])).fillna(0).astype(int).to_numpy())
        for name, pred in preds.items():
            sim = simulate_panel(pred, cost, h)
            pr = portfolio_returns(sim)
            port[(cname, name)] = pr
            st = trading_stats(sim, h)
            cl = {"auc": M.auc(pred["y"], pred["p"]) if name in ("logit", "lgbm") else float("nan")}
            results[f"{cname}:{name}"] = {**st, **cl}
            ledger.record({"study": "daily_panel", "h": h, "model": name, "cost": cname, "families": FAMS},
                          {**st, **cl}, pr)
            print(f"{cname:11s} {name:9s} SR {st['sharpe_ann']:.2f} DD {st['max_drawdown']:.2f} "
                  f"ann {st['ann_return']:.1%} cov {st['coverage']:.2f} ev {st['ev_bps']:.1f}bps "
                  f"p {st['ev_pvalue']} auc {cl['auc']:.4f}", flush=True)
        for a in ("logit", "lgbm", "trend_meta"):
            results[f"{cname}:{a}_vs_trend"] = paired_test(port[(cname, a)], port[(cname, "trend")])
            results[f"{cname}:{a}_vs_bh"] = paired_test(port[(cname, a)], port[(cname, "bh")])
            print(f"  {a} vs trend: {results[f'{cname}:{a}_vs_trend']}", flush=True)
        by_year = {}
        for name in preds:
            r = port[(cname, name)]
            by_year[name] = {int(y): round(float(M.sharpe(g.to_numpy()) * np.sqrt(365 / horizon_days)), 2)
                             for y, g in r.groupby(r.index.year)}
        results[f"{cname}:by_year_sharpe"] = by_year
        print("  by-year Sharpe:", by_year, flush=True)
    save(f"daily_panel_h{horizon_days}", results)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
