"""POST-HOC, after the one-time holdout verdicts (changes no verdict, selects nothing).

Both frozen model candidates abstained on the whole holdout because their skill
gate failed on the latest validation slice. Was abstaining right? For each, fit
exactly as frozen (research data only), then on the holdout report:

* AUC and log loss vs the recent base rate (did directional information persist?)
* what FORCED trading would have earned: skill gate off, zero margin (every
  prediction whose expected edge or direction clears the bare rule is traded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import metrics as M
from quant import models
from quant.backtest import (positions_from_ev, positions_from_prob, round_trips, simulate_panel, trading_stats)
from quant.costs import COST_MODELS
from quant.experiment import save
from quant.train_engine import CANDIDATES, HOLDOUT, _fit_freeze, candidate_dataset


def main():
    out = {}
    for c in CANDIDATES:
        cost = COST_MODELS[c["cost"]]
        ds = candidate_dataset(c)
        factory = getattr(models, c["model"])(**c.get("model_kwargs", {}))
        art = _fit_freeze(ds, HOLDOUT, factory, cost, c["decision"])
        rows = np.flatnonzero(ds.times >= HOLDOUT)
        X = ds.X.to_numpy(float)[rows]
        p = art["calibrator"].transform(art["model"].predict_proba(X)[:, 1])
        y = ds.y.to_numpy()[rows]
        base = float(np.mean(ds.y.to_numpy()[(ds.times < HOLDOUT)][-len(rows):]))
        res = {"rows": int(len(rows)), "auc": M.auc(y, p), "logloss_model": M.log_loss(y, p),
               "logloss_recent_base_rate": M.log_loss(y, np.full(len(y), base)), "skill_gate_passed": art["skilled"]}
        for label, margin in (("forced_margin_0", 0.0),):
            if c["decision"] == "ev":
                pos = positions_from_ev(p, ds.sigma.to_numpy(float)[rows], round_trips(ds.sym[rows], cost, ds.horizon_min),
                                        margin, cost.allow_short)
            else:
                pos = positions_from_prob(p, margin, cost.allow_short)
            pred = pd.DataFrame({"p": p, "y": y, "fwd_ret": ds.fwd_ret.to_numpy()[rows], "pos": pos,
                                 "sym": ds.sym[rows], "fold": 0}, index=ds.times[rows])
            st = trading_stats(simulate_panel(pred, cost, ds.horizon_min), ds.horizon_min)
            res[label] = {k: st[k] for k in ("trades", "active_rows", "hit_rate", "ev_bps", "ev_pvalue", "total_logret",
                                             "sharpe_ann", "max_drawdown")}
        out[c["name"]] = res
        print(c["name"], res, flush=True)
    save("holdout_posthoc", out)


if __name__ == "__main__":
    main()
