"""Can the statistically real 1h-4h directional signal be converted into net profit?

Levers tested (each reported, nothing selected silently):
* decision rule: probability margin vs expected-edge-vs-cost (uses ex-ante vol)
* pooling BTC+ETH (more data, shared structure) vs single-coin models
* cost model: perp taker (long/short) and spot taker (long/flat)
All with the skill gate on, walk-forward 90-day folds, research period only.
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
from quant.experiment import make_dataset, save
from quant.validation import walk_forward

FAMS = ["mom", "vol", "act", "flow", "shape", "range", "regime", "cal", "ta", "xa"]


def run(ds, label, h, rows):
    folds = walk_forward(ds.times, pd.DatetimeIndex(ds.label_end), test_size="90D", min_train="365D")
    for cname in ("perp_taker", "spot_taker"):
        cost = COST_MODELS[cname]
        for decision in ("prob", "ev"):
            t0 = time.time()
            pred = walk_forward_predict(ds, folds, models.lightgbm(), cost=cost, decision=decision)
            sim = simulate_panel(pred, cost, h)
            st = trading_stats(sim, h)
            skilled = float(pred.groupby("fold")["skilled"].first().mean())
            traded = int(pred.groupby("fold")["margin"].first().notna().sum())
            row = {"set": label, "h": h, "cost": cname, "decision": decision, "auc": M.auc(pred["y"], pred["p"]),
                   "skilled_folds": round(skilled, 2), "trading_folds": traded, "folds": int(pred["fold"].nunique()),
                   **{k: st[k] for k in ("active_rows", "coverage", "hit_rate", "ev_bps", "ev_pvalue",
                                          "sharpe_ann", "max_drawdown", "profit_factor")}}
            by_year = {}
            for y, g in sim.groupby(sim.index.year):
                a = g["net"][g["pos"] != 0]
                by_year[int(y)] = (int(len(a)), round(float(a.mean() * 1e4), 1) if len(a) else None)
            row["by_year"] = by_year
            rows.append(row)
            ledger.record({"study": "intraday_conversion", "set": label, "h": h, "cost": cname,
                           "decision": decision, "model": "lgbm", "families": FAMS}, row, portfolio_returns(sim))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                  f"({time.time() - t0:.0f}s)", flush=True)


def main():
    rows = []
    for h in (240, 60):
        parts = [make_dataset(s, h, families=FAMS) for s in ("BTCUSDT", "ETHUSDT")]
        run(Dataset.pool(parts), "BTC+ETH pooled", h, rows)
        for p in parts:
            run(p, p.symbol, h, rows)
    save("intraday_conversion", rows)


if __name__ == "__main__":
    main()
