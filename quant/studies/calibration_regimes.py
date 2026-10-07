"""Calibration, regime specialisation and evidence agreement (BTC+ETH pooled, 4h and 1h).

Q1 Is calibrated confidence honest? Reliability (ECE, slope) overall and within
   ex-ante volatility regimes and trend regimes, out-of-sample.
Q2 Regime-conditional calibration: separate Platt maps per volatility regime
   (fitted on the validation slice) vs one global map -> test log loss / ECE.
Q3 Regime specialisation: one model per volatility regime vs one global model
   (with regime features) -> test log loss, DM test.
Q4 Agreement: when the intraday model's direction agrees with the daily trend
   ensemble, is it more accurate / higher EV than when they conflict?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import metrics as M
from quant import models
from quant.backtest import Calibrator, Dataset, walk_forward_predict
from quant.experiment import make_dataset, save
from quant.studies import trend as T
from quant.validation import walk_forward

FAMS = ["mom", "vol", "act", "flow", "shape", "range", "regime", "cal", "ta"]


def ll_vec(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def regime_calibrated(ds, folds, factory):
    """Global Platt vs per-vol-regime Platt, same fitted model per fold."""
    X, y = ds.X.to_numpy(float), ds.y.to_numpy()
    reg = ds.regimes["vol"].astype(str).to_numpy()
    rows = []
    for k, f in enumerate(folds):
        if len(f.fit) < 100 or len(f.valid) < 30 or not len(f.test):
            continue
        m = factory().fit(X[f.fit], y[f.fit])
        pv, pt = m.predict_proba(X[f.valid])[:, 1], m.predict_proba(X[f.test])[:, 1]
        g = Calibrator("platt").fit(pv, y[f.valid])
        p_global = g.transform(pt)
        p_reg = p_global.copy()
        for rv in np.unique(reg[f.valid]):
            vm, tm = reg[f.valid] == rv, reg[f.test] == rv
            if vm.sum() >= 200 and tm.any():
                p_reg[tm] = Calibrator("platt").fit(pv[vm], y[f.valid][vm]).transform(pt[tm])
        rows.append(pd.DataFrame({"y": y[f.test], "p_global": p_global, "p_regime": p_reg, "reg": reg[f.test]},
                                 index=ds.times[f.test]))
    return pd.concat(rows)


def specialised(ds, folds, factory):
    """One model per volatility regime (fit on that regime's rows only)."""
    X, y = ds.X.to_numpy(float), ds.y.to_numpy()
    reg = ds.regimes["vol"].astype(str).to_numpy()
    rows = []
    for f in folds:
        if len(f.fit) < 100 or len(f.valid) < 30 or not len(f.test):
            continue
        p = np.full(len(f.test), np.nan)
        for rv in np.unique(reg[f.test]):
            fm, vm, tm = reg[f.fit] == rv, reg[f.valid] == rv, reg[f.test] == rv
            if fm.sum() < 300 or vm.sum() < 100:
                continue
            m = factory().fit(X[f.fit][fm], y[f.fit][fm])
            cal = Calibrator("platt").fit(m.predict_proba(X[f.valid][vm])[:, 1], y[f.valid][vm])
            p[tm] = cal.transform(m.predict_proba(X[f.test][tm])[:, 1])
        rows.append(pd.DataFrame({"y": y[f.test], "p": p, "reg": reg[f.test]}, index=ds.times[f.test]))
    return pd.concat(rows)


def main(horizons=(240, 60)):
    out = {}
    for h in horizons:
        parts = [make_dataset(s, h, families=FAMS) for s in ("BTCUSDT", "ETHUSDT")]
        ds = Dataset.pool(parts)
        folds = walk_forward(ds.times, pd.DatetimeIndex(ds.label_end), test_size="90D", min_train="365D")
        fac = models.lightgbm()
        base = walk_forward_predict(ds, folds, fac, decide=False)
        res = {"overall": {"auc": M.auc(base["y"], base["p"]), "ece": M.ece(base["y"], base["p"]),
                           "slope": M.calibration_slope(base["y"], base["p"])[0],
                           "reliability": M.reliability_table(base["y"], base["p"], 10).round(4).to_dict("records")}}
        # Q1: calibration by regime
        for col in ("regime_vol", "regime_trend7d"):
            res[f"by_{col}"] = {str(v): {"n": int(len(g)), "auc": round(M.auc(g["y"], g["p"]), 4),
                                        "ece": round(M.ece(g["y"], g["p"]), 4),
                                        "slope": round(M.calibration_slope(g["y"], g["p"])[0], 3),
                                        "mean_p": round(float(g["p"].mean()), 4), "up_rate": round(float(g["y"].mean()), 4)}
                                for v, g in base.groupby(col)}
        # Q2: regime-conditional calibration
        rc = regime_calibrated(ds, folds, fac)
        lg, lr = ll_vec(rc["y"].to_numpy(), rc["p_global"].to_numpy()), ll_vec(rc["y"].to_numpy(), rc["p_regime"].to_numpy())
        d, p = M.diebold_mariano(lg, lr)
        res["regime_calibration"] = {"logloss_global": float(lg.mean()), "logloss_regime": float(lr.mean()),
                                     "gain": d, "p_regime_better": p,
                                     "ece_global": M.ece(rc["y"], rc["p_global"]), "ece_regime": M.ece(rc["y"], rc["p_regime"])}
        # Q3: specialised models
        sp = specialised(ds, folds, fac).dropna(subset=["p"])
        common = base.index.intersection(sp.index)
        b2 = base[~base.index.duplicated()].reindex(sp.index)
        ok = b2["p"].notna().to_numpy()
        lb_, ls_ = ll_vec(sp["y"].to_numpy()[ok], b2["p"].to_numpy()[ok]), ll_vec(sp["y"].to_numpy()[ok], sp["p"].to_numpy()[ok])
        d, p = M.diebold_mariano(lb_, ls_)
        res["specialised_models"] = {"logloss_global": float(lb_.mean()), "logloss_specialised": float(ls_.mean()),
                                     "gain": d, "p_specialised_better": p, "auc_specialised": M.auc(sp["y"], sp["p"])}
        # Q4: agreement with the daily trend ensemble
        agree_rows = []
        for sym in ("BTCUSDT", "ETHUSDT"):
            S = T.ensemble(T.daily_bars(sym)["close"])
            g = base[base["sym"] == sym]
            s_at = S.reindex(g.index.floor("D")).to_numpy()   # trend state from the last completed day
            agree_rows.append(g.assign(trend_long=s_at >= 0.5))
        ag = pd.concat(agree_rows).dropna(subset=["trend_long"])
        conf = (ag["p"] - 0.5).abs()
        strong = conf >= conf.quantile(0.8)
        model_up = ag["p"] >= 0.5
        agree = model_up == ag["trend_long"].astype(bool)
        res["agreement_top_quintile_confidence"] = {}
        for name, m in (("agree", strong & agree), ("conflict", strong & ~agree)):
            g = ag[m]
            hit = ((g["p"] >= 0.5).astype(int) == g["y"]).mean()
            signed = np.where(g["p"] >= 0.5, 1, -1) * g["fwd_ret"]
            res["agreement_top_quintile_confidence"][name] = {"n": int(len(g)), "hit_rate": round(float(hit), 4),
                                                              "gross_bps": round(float(signed.mean() * 1e4), 2)}
        out[f"h{h}"] = res
        print(f"h={h}", {k: v for k, v in res.items() if k != "overall"}, flush=True)
        print("  overall:", {k: v for k, v in res["overall"].items() if k != "reliability"}, flush=True)
    save("calibration_regimes", out)


if __name__ == "__main__":
    main()
