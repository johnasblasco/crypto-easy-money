"""Which information carries the (statistical) directional signal?

For each horizon/symbol: walk-forward logistic models on (a) each feature
family alone and (b) all families minus one. Out-of-sample log loss is compared
with the base rate and with the full model via Diebold-Mariano (HAC) tests.
Per-year AUC shows whether a family's contribution is stable or comes from a
single period (e.g. Binance's 2022-07..2023-03 zero-fee BTC campaign).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from quant import metrics as M
from quant import models
from quant.backtest import walk_forward_predict
from quant.experiment import make_dataset, save
from quant.validation import walk_forward

FAMS = ["mom", "vol", "act", "flow", "shape", "range", "regime", "cal", "ta"]


def oos(ds, factory):
    folds = walk_forward(ds.times, pd.DatetimeIndex(ds.label_end), test_size="90D", min_train="365D")
    return walk_forward_predict(ds, folds, factory, decide=False)


def per_sample_ll(pred, base):
    p = np.clip(pred["p"].to_numpy(), 1e-6, 1 - 1e-6)
    y = pred["y"].to_numpy()
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)), -(y * np.log(base) + (1 - y) * np.log(1 - base))


def by_year_auc(pred):
    return {int(y): round(M.auc(g["y"], g["p"]), 4) for y, g in pred.groupby(pred.index.year)}


def main(symbols=("BTCUSDT", "ETHUSDT"), horizons=(60, 240)):
    rows = []
    for sym in symbols:
        for h in horizons:
            full_ds = make_dataset(sym, h, families=FAMS)
            t0 = time.time()
            full = oos(full_ds, models.logistic())
            base = float(full["y"].mean())
            ll_full, ll_base = per_sample_ll(full, base)
            rows.append({"symbol": sym, "h": h, "set": "ALL", "auc": M.auc(full["y"], full["p"]),
                         "ll_gain_vs_base": float((ll_base - ll_full).mean()),
                         "p_vs_base": M.diebold_mariano(ll_base, ll_full)[1], "by_year_auc": by_year_auc(full)})
            print(rows[-1], f"({time.time() - t0:.0f}s)", flush=True)
            for fam in FAMS:
                for mode in ("only", "drop"):
                    fams = [fam] if mode == "only" else [f for f in FAMS if f != fam]
                    ds = make_dataset(sym, h, families=fams)
                    pred = oos(ds, models.logistic())
                    pred = pred.reindex(full.index).dropna(subset=["p"])
                    ll, llb = per_sample_ll(pred, base)
                    ll_f = ll_full[full.index.isin(pred.index)]
                    row = {"symbol": sym, "h": h, "set": f"{mode}:{fam}", "auc": M.auc(pred["y"], pred["p"]),
                           "ll_gain_vs_base": float((llb - ll).mean()),
                           "p_vs_base": M.diebold_mariano(llb, ll)[1]}
                    if mode == "drop":
                        # does removing the family hurt? (full better than drop)
                        row["ll_loss_from_dropping"] = float((ll - ll_f).mean())
                        row["p_family_adds"] = M.diebold_mariano(ll, ll_f)[1]
                    else:
                        row["by_year_auc"] = by_year_auc(pred)
                    rows.append(row)
                    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    save("ablation", rows)


if __name__ == "__main__":
    main()
