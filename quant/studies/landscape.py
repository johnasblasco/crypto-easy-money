"""Horizon x model x cost landscape on the research period (pre-holdout).

Question: at which horizons is direction predictable at all (AUC, log-loss vs
base rate), and does any of it survive costs? Uses all single-asset feature
families (no cross-asset yet), fixed hyperparameters, walk-forward 90-day folds.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from quant import ledger, models
from quant.backtest import portfolio_returns, simulate_panel
from quant.experiment import make_dataset, run, save
from quant.metrics import diebold_mariano, log_loss

FAMS = ["mom", "vol", "act", "flow", "shape", "range", "regime", "cal", "ta"]
COSTS = ("perp_maker", "perp_taker", "spot_taker")


def main(symbols=("BTCUSDT", "ETHUSDT"), horizons=(15, 60, 240, 1440)):
    out = []
    for sym in symbols:
        for h in horizons:
            ds = make_dataset(sym, h, families=FAMS)
            for mname, fac in [("logit", models.logistic()), ("lgbm", models.lightgbm())]:
                t0 = time.time()
                reps, preds = run(ds, fac, cost_names=COSTS)
                base = float(preds[COSTS[0]]["y"].mean())
                for c in COSTS:
                    r = reps[c]
                    pred = preds[c]
                    ll_model = -(pred["y"] * np.log(pred["p"]) + (1 - pred["y"]) * np.log(1 - pred["p"]))
                    # base-rate loss uses the walk-forward training base rate proxy (overall mean, generous to base)
                    ll_base = -(pred["y"] * np.log(base) + (1 - pred["y"]) * np.log(1 - base))
                    d, p_dm = diebold_mariano(ll_base.to_numpy(), ll_model.to_numpy())
                    cfg = {"study": "landscape", "symbol": sym, "horizon": h, "model": mname, "cost": c,
                           "families": FAMS}
                    sim = simulate_panel(pred, __import__("quant.costs", fromlist=["COST_MODELS"]).COST_MODELS[c], h)
                    ledger.record(cfg, r, portfolio_returns(sim))
                    row = {"symbol": sym, "h": h, "model": mname, "cost": c,
                           "auc": r["classification"]["auc"], "ll_gain": d, "dm_p": p_dm,
                           "ece": r["classification"]["ece"], "slope": r["classification"]["cal_slope"],
                           **{k: r["trading"][k] for k in ("active_rows", "coverage", "ev_bps", "ev_pvalue",
                                                            "sharpe_ann", "max_drawdown", "profit_factor")},
                           "abstain": f"{r['abstain_folds']}/{r['folds']}",
                           "by_year_ev": {y: round(v["ev_bps"], 1) if v["ev_bps"] == v["ev_bps"] else None
                                          for y, v in r["by_year"].items()}}
                    out.append(row)
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
                print(f"  ({sym} h={h} {mname}: {time.time() - t0:.0f}s)", flush=True)
    save("landscape", out)


if __name__ == "__main__":
    main()
