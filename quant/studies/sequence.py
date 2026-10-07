"""Do models that learn from RAW recent sequences find information the hand-built
features miss? (Escalation test before considering GRU/TCN/Transformers.)

Target: BTC (and ETH) 1h direction, hourly decisions, research period.
Raw sequence inputs at decision t (all from bars closed by t):
  * last 24 five-minute bars: return / trailing sigma, taker imbalance, log volume vs 30d mean
  * last 24 hourly bars: the same three channels
Models (walk-forward, same folds, same calibration):
  A  LightGBM on hand features (reference)
  B  MLP (64, 32) on raw sequences
  C  LightGBM on raw sequences
  D  LightGBM on hand features + raw sequences
Comparison: out-of-sample log loss, Diebold-Mariano vs A.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant import metrics as M
from quant import models
from quant.backtest import Dataset, walk_forward_predict
from quant.data import resample
from quant.experiment import make_dataset, m1_frame, save
from quant.validation import walk_forward

FAMS = ["mom", "vol", "act", "flow", "shape", "range", "regime", "cal", "ta"]


def raw_sequences(symbol: str, times: pd.DatetimeIndex) -> pd.DataFrame:
    m1 = m1_frame(symbol)
    sig = np.sqrt((np.log(m1["close"]).ffill().diff() ** 2).ewm(halflife=1440, min_periods=1440).mean())
    cols = {}
    for rule, n, mins in (("5min", 24, 5), ("1h", 24, 60)):
        b = resample(m1, rule)
        r = np.log(b["close"]).diff()
        s = sig.reindex(b.index) * np.sqrt(mins)
        z = r / s
        imb = (2 * b["taker_buy_quote"] - b["quote_volume"]) / b["quote_volume"].replace(0, np.nan)
        lv = np.log(b["quote_volume"] + 1)
        lvz = lv - lv.rolling(int(30 * 1440 / mins), min_periods=int(10 * 1440 / mins)).mean()
        frame = pd.DataFrame({"z": z, "imb": imb, "lv": lvz})
        pos = frame.index.searchsorted(times, side="right") - 1   # last bar closed by t
        arr = frame.to_numpy()
        for lag in range(n):
            p = pos - lag
            vals = np.where((p >= 0)[:, None], arr[np.maximum(p, 0)], np.nan)
            for j, c in enumerate(frame.columns):
                cols[f"seq_{rule}_{c}_{lag}"] = vals[:, j]
    return pd.DataFrame(cols, index=times)


def mlp():
    return lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                                 early_stopping=True, validation_fraction=0.15,
                                                                 max_iter=200, random_state=0))


def ll(pred):
    p = np.clip(pred["p"].to_numpy(), 1e-6, 1 - 1e-6)
    y = pred["y"].to_numpy()
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main(symbols=("BTCUSDT", "ETHUSDT"), h=60):
    out = []
    for sym in symbols:
        ds = make_dataset(sym, h, families=FAMS)
        seq = raw_sequences(sym, ds.times).clip(-20, 20)
        variants = {
            "A_hand_lgbm": (ds.X, models.lightgbm()),
            "B_raw_mlp": (seq, mlp()),
            "C_raw_lgbm": (seq, models.lightgbm()),
            "D_hand+raw_lgbm": (pd.concat([ds.X, seq], axis=1), models.lightgbm()),
        }
        preds = {}
        for name, (X, fac) in variants.items():
            t0 = time.time()
            d = Dataset(sym, h, X.fillna(0.0), ds.fwd_ret, ds.label_end)
            folds = walk_forward(d.times, pd.DatetimeIndex(d.label_end), test_size="90D", min_train="365D")
            preds[name] = walk_forward_predict(d, folds, fac, decide=False)
            print(f"{sym} {name}: auc {M.auc(preds[name]['y'], preds[name]['p']):.4f} ({time.time() - t0:.0f}s)", flush=True)
        ref = preds["A_hand_lgbm"]
        for name, pr in preds.items():
            common = ref.index.intersection(pr.index)
            dm, p = M.diebold_mariano(ll(ref.loc[common]), ll(pr.loc[common]))
            row = {"symbol": sym, "variant": name, "auc": M.auc(pr["y"], pr["p"]), "logloss": float(ll(pr).mean()),
                   "ll_gain_vs_hand": dm, "p_better_than_hand": p,
                   "by_year_auc": {int(y): round(M.auc(g["y"], g["p"]), 4) for y, g in pr.groupby(pr.index.year)}}
            out.append(row)
            print(row, flush=True)
    save("sequence_models", out)


if __name__ == "__main__":
    main()
