"""Adversarial dissection of the daily-panel LightGBM result (3-day horizon).

The headline (perp costs: 340 position-rows, 69% hit, +316bp/trade, p=0.007)
is suspicious on its face: overall AUC is only 0.517, it trades in two of five
years, and its portfolio does not beat buy & hold (paired p=0.25). Questions:

1. How many independent decisions is it? (distinct decision dates; coins on the
   same date are one correlated bet)
2. Is it market beta? Direction mix, and the excess over the equal-weight
   market return on the same dates in the same direction.
3. Is it one episode? Contribution of the best dates/months; leave-one-month-out.
4. Is it the selection? Same rows with random direction-preserving date shuffles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import models
from quant.backtest import Dataset, simulate_panel, walk_forward_predict
from quant.costs import COST_MODELS
from quant.data import KLINE_DIR, UNIVERSE
from quant.experiment import make_dataset, save
from quant.studies.daily_panel import FAMS, trend_score
from quant.validation import walk_forward


def build_panel(horizon_days: int) -> Dataset:
    h = 1440 * horizon_days
    parts = []
    for s in [s for s in UNIVERSE if (KLINE_DIR / f"{s}.parquet").exists()]:
        ds = make_dataset(s, h, families=FAMS)
        X = ds.X.assign(trend__ensemble=trend_score(ds, s).to_numpy())
        keep = (ds.times >= ds.times[0] + pd.Timedelta(days=60)) & X["trend__ensemble"].notna().to_numpy()
        parts.append(Dataset(s, h, X[keep], ds.fwd_ret[keep], ds.label_end[keep], ds.regimes[keep]))
    return Dataset.pool(parts)


def main(horizon_days: int = 3, cost_name: str = "perp_taker", n_null: int = 2000, seed: int = 0):
    panel = build_panel(horizon_days)
    h = panel.horizon_min
    cost = COST_MODELS[cost_name]
    folds = walk_forward(panel.times, pd.DatetimeIndex(panel.label_end), test_size="90D", min_train="365D")
    pred = walk_forward_predict(panel, folds, models.lightgbm(n_estimators=200, min_child_samples=200), cost=cost)
    sim = simulate_panel(pred, cost, h)
    act = pred[pred["pos"] != 0]
    out = {"cost": cost_name, "horizon_days": horizon_days, "rows": int(len(pred)), "active_rows": int(len(act))}

    # 1. independent decisions
    dates = act.index.unique()
    out["distinct_dates"] = int(len(dates))
    out["coins_per_active_date"] = float(len(act) / max(len(dates), 1))
    out["long_rows"], out["short_rows"] = int((act["pos"] > 0).sum()), int((act["pos"] < 0).sum())
    out["active_folds"] = sorted(int(f) for f in act["fold"].unique())
    out["margin_by_fold"] = {int(k): (None if pd.isna(v) else float(v)) for k, v in pred.groupby("fold")["margin"].first().items()}
    out["skilled_by_fold"] = {int(k): bool(v) for k, v in pred.groupby("fold")["skilled"].first().items()}

    # 2. beta: equal-weight market forward return on the same date (all coins in the panel that date)
    mkt = pred.groupby(level=0)["fwd_ret"].mean()
    a = act.assign(mkt=mkt.reindex(act.index).to_numpy())
    a = a.assign(signed=a["pos"] * a["fwd_ret"], signed_mkt=a["pos"] * a["mkt"])
    per_date = a.groupby(level=0).agg(signed=("signed", "mean"), signed_mkt=("signed_mkt", "mean"), n=("pos", "size"),
                                      direction=("pos", "mean"))
    out["gross_bps_per_row"] = float(a["signed"].mean() * 1e4)
    out["market_same_direction_bps"] = float(a["signed_mkt"].mean() * 1e4)
    out["excess_over_market_bps"] = float((a["signed"] - a["signed_mkt"]).mean() * 1e4)
    # date-level t-stats (dates are the independent units)
    for col in ("signed", "signed_mkt"):
        x = per_date[col].to_numpy()
        out[f"date_level_{col}_t"] = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")
    ex = (per_date["signed"] - per_date["signed_mkt"]).to_numpy()
    out["date_level_excess_t"] = float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))) if len(ex) > 2 else float("nan")

    # 3. concentration
    per_date = per_date.sort_values("signed", ascending=False)
    tot = per_date["signed"].sum()
    out["share_of_gross_from_top3_dates"] = float(per_date["signed"].head(3).sum() / tot) if tot else float("nan")
    out["share_of_gross_from_top5_dates"] = float(per_date["signed"].head(5).sum() / tot) if tot else float("nan")
    months = a.groupby(a.index.to_period("M"))["signed"].agg(["sum", "size"])
    out["by_month"] = {str(k): {"rows": int(v["size"]), "gross_bps_sum": round(float(v["sum"] * 1e4), 1)} for k, v in months.iterrows()}
    rt = cost.round_trip("BTCUSDT")
    lomo = {}
    for m in months.index:
        rest = a[a.index.to_period("M") != m]
        lomo[str(m)] = round(float((rest["signed"].mean() - rt) * 1e4), 1) if len(rest) else None
    out["leave_one_month_out_net_bps_approx"] = lomo
    out["dates"] = [{"date": str(d.date()), "n": int(r["n"]), "dir": round(float(r["direction"]), 2),
                     "gross_bps": round(float(r["signed"] * 1e4), 1), "mkt_same_dir_bps": round(float(r["signed_mkt"] * 1e4), 1)}
                    for d, r in per_date.sort_index().iterrows()]

    # 4. date-shuffle null: same number of active dates, same per-date direction and coin count,
    #    dates drawn at random from the same test folds -> is the TIMING informative?
    rng = np.random.default_rng(seed)
    test_dates = pred.index.unique()
    by_date = {d: g for d, g in pred.groupby(level=0)}
    obs = per_date["signed"].mean()
    null = np.empty(n_null)
    dirs = per_date["direction"].to_numpy()
    for i in range(n_null):
        pick = rng.choice(test_dates, size=len(per_date), replace=False)
        vals = []
        for d, dr in zip(pick, dirs):
            g = by_date[d]
            vals.append(np.sign(dr if dr != 0 else 1) * g["fwd_ret"].mean())
        null[i] = np.mean(vals)
    out["timing_null_mean_bps"] = float(null.mean() * 1e4)
    out["timing_null_p"] = float(((null >= obs).sum() + 1) / (n_null + 1))
    print({k: v for k, v in out.items() if k not in ("dates", "by_month", "leave_one_month_out_net_bps_approx")}, flush=True)
    save(f"dissect_panel_h{horizon_days}_{cost_name}", out)
    return out


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3, sys.argv[2] if len(sys.argv) > 2 else "perp_taker")
