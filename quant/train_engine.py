"""Freeze candidates, evaluate each ONCE on the holdout, write engine artifacts.

Follows docs/PREREGISTRATION.md mechanically:

1. Configuration frozen from research-period results (CANDIDATES below).
2. Fit on the research period (inner last-20% slice for calibration and the
   NO-TRADE margin), predict the holdout [HOLDOUT_START, now) once.
3. Status from the pre-registered pass criteria.
4. Refit on all data with the same procedure for live use; the artifact keeps
   the HOLDOUT evidence (not the refit's in-sample numbers).

Usage:
    python -m quant.train_engine            # trend evidence + all model candidates
"""
from __future__ import annotations

import json
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd

from . import ledger
from . import metrics as M
from . import models
from .backtest import (Calibrator, Dataset, choose_ev_margin, choose_threshold, has_skill, portfolio_returns,
                       positions_from_ev, positions_from_prob, round_trips, simulate_panel, split_validation,
                       trading_stats)
from .costs import COST_MODELS
from .data import KLINE_DIR, UNIVERSE
from .engine import ENGINE_DIR, Evidence
from .events import day_bootstrap
from .experiment import HOLDOUT_START, make_dataset

HOLDOUT = pd.Timestamp(HOLDOUT_START, tz="UTC")

# Frozen from research-period evidence (see docs/RESEARCH_REPORT.md). Filled in
# after the research studies, before any holdout evaluation.
CANDIDATES: list[dict] = []


# ------------------------------------------------------------------ trend

def trend_evidence() -> Evidence:
    from .studies import trend as T
    from .studies.trend_robust import daily_bars_offset, trend_returns

    syms = [s for s in UNIVERSE if (KLINE_DIR / f"{s}.parquet").exists()]
    cost = COST_MODELS["spot_taker"]
    rets = {"trend": {}, "voltarget": {}}
    for s in syms:
        d = daily_bars_offset(s)
        for mode in rets:
            r = trend_returns(d, s, cost, mode=mode)
            first = d["close"].first_valid_index()
            r[r.index < first + pd.Timedelta(days=365)] = np.nan
            rets[mode][s] = r
    port = {m: np.log(np.expm1(pd.DataFrame(v)).mean(axis=1, skipna=True) + 1).dropna() for m, v in rets.items()}
    hold = {m: r[r.index >= HOLDOUT] for m, r in port.items()}
    res = {m: r[r.index < HOLDOUT] for m, r in port.items()}
    hs, hv = T.stats(hold["trend"]), T.stats(hold["voltarget"])
    rs, rv = T.stats(res["trend"]), T.stats(res["voltarget"])
    a_res = T.alpha_vs(res["trend"], res["voltarget"])
    a_hold = T.alpha_vs(hold["trend"], hold["voltarget"])
    overlay_ok = (hs["max_dd"] >= 0.8 * hv["max_dd"]) and (hs["sharpe"] >= hv["sharpe"] - 0.2)   # dd are negative
    alpha_ok = overlay_ok and a_hold["alpha_ann"] > 0 and a_res["alpha_t"] >= 2
    status = "VALIDATED" if alpha_ok else "RISK_OVERLAY" if overlay_ok else "NOT_VALIDATED"
    summary = (
        f"Daily trend ensemble, long/flat, vol-sized. Holdout ({hold['trend'].index[0].date()}->{hold['trend'].index[-1].date()}): "
        f"Sharpe {hs['sharpe']:.2f} vs vol-targeted hold {hv['sharpe']:.2f}; max drawdown {hs['max_dd']:.0%} vs {hv['max_dd']:.0%}. "
        f"Research period: Sharpe {rs['sharpe']:.2f} vs {rv['sharpe']:.2f}, drawdown {rs['max_dd']:.0%} vs {rv['max_dd']:.0%}, "
        f"timing alpha {a_res['alpha_ann']:+.1%}/yr (t={a_res['alpha_t']:.2f}, not significant)."
    )
    ev = Evidence(status=status, summary=summary, period=f"holdout {HOLDOUT_START} onward", sharpe=hs["sharpe"],
                  max_drawdown=hs["max_dd"], benchmark="volatility-targeted buy & hold (spot costs)",
                  notes=[f"holdout alpha vs vol-target {a_hold['alpha_ann']:+.1%}/yr (t={a_hold['alpha_t']:.2f})",
                         f"criteria: overlay={overlay_ok}, alpha={alpha_ok} (pre-registered)"])
    return ev


# ------------------------------------------------------------------ models

def _fit_freeze(ds: Dataset, end: pd.Timestamp, factory, cost, decision: str, calibration: str = "platt"):
    """Fit on data with labels known before ``end``: last 20% (by time) is the validation slice."""
    known = pd.DatetimeIndex(ds.label_end) < end
    times = ds.times[known]
    v_start = times[0] + (times[-1] - times[0]) * 0.8
    pos = np.flatnonzero(known)
    valid = pos[times >= v_start]
    fit = pos[(times < v_start) & (pd.DatetimeIndex(ds.label_end)[pos] < v_start)]
    X, y, r = ds.X.to_numpy(float), ds.y.to_numpy(), ds.fwd_ret.to_numpy(float)
    model = factory().fit(X[fit], y[fit])
    cal_rows, sel_rows = split_validation(ds.times, valid)
    cal = Calibrator(calibration).fit(model.predict_proba(X[cal_rows])[:, 1], y[cal_rows])
    p_sel = cal.transform(model.predict_proba(X[sel_rows])[:, 1])
    skilled = has_skill(p_sel, y[sel_rows], float(np.mean(y[cal_rows])))   # same recent base rate as the backtest
    groups = ds.times[sel_rows].asi8
    margin = None
    if skilled:
        if decision == "ev":
            margin, _ = choose_ev_margin(p_sel, ds.sigma.to_numpy(float)[sel_rows], r[sel_rows], ds.sym[sel_rows], cost,
                                         ds.horizon_min, groups=groups)
        else:
            margin, _ = choose_threshold(p_sel, r[sel_rows], ds.sym[sel_rows], cost, ds.horizon_min, groups=groups)
    return {"model": model, "calibrator": cal, "margin": margin, "skilled": skilled,
            "fit_rows": int(len(fit)), "valid_rows": int(len(valid)), "valid_period": [str(times[times >= v_start][0]), str(times[-1])]}


def _predict_positions(ds: Dataset, art: dict, rows: np.ndarray, cost, decision: str) -> pd.DataFrame:
    X = ds.X.to_numpy(float)[rows]
    p = art["calibrator"].transform(art["model"].predict_proba(X)[:, 1])
    if art["margin"] is None:
        pos = np.zeros(len(rows), dtype=int)
    elif decision == "ev":
        pos = positions_from_ev(p, ds.sigma.to_numpy(float)[rows], round_trips(ds.sym[rows], cost, ds.horizon_min),
                                art["margin"], cost.allow_short)
    else:
        pos = positions_from_prob(p, art["margin"], cost.allow_short)
    return pd.DataFrame({"p": p, "y": ds.y.to_numpy()[rows], "fwd_ret": ds.fwd_ret.to_numpy()[rows],
                         "pos": pos, "sym": ds.sym[rows], "fold": 0}, index=ds.times[rows])


def evaluate_candidate(c: dict) -> dict:
    cost = COST_MODELS[c["cost"]]
    parts = [make_dataset(s, c["horizon_min"], families=c["families"], holdout=True) for s in c["symbols"]]
    ds = Dataset.pool(parts) if len(parts) > 1 else parts[0]
    factory = getattr(models, c["model"])(**c.get("model_kwargs", {}))
    art = _fit_freeze(ds, HOLDOUT, factory, cost, c["decision"])
    rows = np.flatnonzero(ds.times >= HOLDOUT)
    pred = _predict_positions(ds, art, rows, cost, c["decision"])
    sim = simulate_panel(pred, cost, ds.horizon_min)
    st = trading_stats(sim, ds.horizon_min)
    act = sim[sim["pos"] != 0]
    ev_day = pd.DataFrame({"ts": act.index, "net": act["net"].to_numpy()})
    mean, p_day, lo = day_bootstrap(ev_day, "net") if len(ev_day) >= 5 else (float("nan"),) * 3
    # Deflated Sharpe over every research-period trial of this study in the ledger.
    trials = [t for t in ledger.trials() if t["config"].get("study") == c["study"] and t["config"].get("h") == c["horizon_min"]]
    dsr = float("nan")
    if c.get("ledger_id"):
        try:
            dsr = ledger.selection_stats({"study": c["study"], "h": c["horizon_min"]}, c["ledger_id"])["deflated_sharpe"]
        except KeyError:
            pass
    passed = (len(act) >= 30 and np.isfinite(mean) and mean > 0 and p_day < 0.05 and np.isfinite(dsr) and dsr > 0.5)
    status = "VALIDATED" if passed else "NOT_VALIDATED"
    summary = (f"{c['description']} Holdout: {len(act)} trades, mean net {mean * 1e4 if np.isfinite(mean) else float('nan'):+.1f}bp "
               f"(day-clustered p={p_day:.3f}), Sharpe {st['sharpe_ann']:.2f}; research-period DSR {dsr:.2f} "
               f"over {len(trials)} trials. Verdict: {status} (pre-registered criteria).")
    evidence = Evidence(status=status, summary=summary, period=f"holdout {HOLDOUT_START} onward", trades=int(len(act)),
                        ev_bps=float(mean * 1e4) if np.isfinite(mean) else None,
                        ev_p5_bps=float(lo * 1e4) if np.isfinite(lo) else None,
                        sharpe=st["sharpe_ann"], max_drawdown=st["max_drawdown"], benchmark="zero (flat) after costs",
                        notes=[f"validation slice {art['valid_period']}", f"skill gate passed: {art['skilled']}",
                               f"margin chosen on validation: {art['margin']}"])
    # Live refit on everything (same procedure), but keep the HOLDOUT evidence.
    live = _fit_freeze(ds, pd.Timestamp.now(tz="UTC"), factory, cost, c["decision"])
    trade_sigma = float(np.std(act["net"])) if len(act) > 2 else 0.01
    artifact = {"name": c["name"], "horizon_min": c["horizon_min"], "families": c["families"],
                "columns": list(ds.X.columns), "model": live["model"], "calibrator": live["calibrator"],
                "margin": live["margin"], "cost": c["cost"], "decision": c["decision"],
                "evidence": asdict(evidence), "trade_sigma": trade_sigma, "symbols": c["symbols"]}
    return {"artifact": artifact, "holdout_stats": st, "evidence": asdict(evidence)}


def informational_model(c: dict, research_summary: str) -> dict:
    """A model that did NOT qualify for the holdout: fitted for display only, always NO TRADE."""
    cost = COST_MODELS[c["cost"]]
    parts = [make_dataset(s, c["horizon_min"], families=c["families"], holdout=True) for s in c["symbols"]]
    ds = Dataset.pool(parts) if len(parts) > 1 else parts[0]
    factory = getattr(models, c["model"])(**c.get("model_kwargs", {}))
    live = _fit_freeze(ds, pd.Timestamp.now(tz="UTC"), factory, cost, c["decision"])
    evidence = Evidence(status="NOT_VALIDATED", summary=research_summary, period="research 2020-2025-09",
                        benchmark="zero (flat) after costs",
                        notes=["did not meet the research-period criterion to be tested on the holdout",
                               "shown for information only: the engine never trades it"])
    return {"name": c["name"], "horizon_min": c["horizon_min"], "families": c["families"],
            "columns": list(ds.X.columns), "model": live["model"], "calibrator": live["calibrator"],
            "margin": None, "cost": c["cost"], "decision": c["decision"], "evidence": asdict(evidence),
            "trade_sigma": 0.01, "symbols": c["symbols"]}


# Models shown for information only (research verdict: statistically skilful, not cost-surviving).
INFORMATIONAL: list[dict] = []


def main():
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    ev = trend_evidence()
    (ENGINE_DIR / "trend_evidence.json").write_text(json.dumps(asdict(ev), indent=2, default=str))
    print("trend:", ev.status, "|", ev.summary, flush=True)
    for c in CANDIDATES:
        res = evaluate_candidate(c)
        joblib.dump(res["artifact"], ENGINE_DIR / f"{c['name']}.joblib")
        print(c["name"], res["evidence"]["status"], "|", res["evidence"]["summary"], flush=True)
    for c in INFORMATIONAL:
        art = informational_model(c, c["research_summary"])
        joblib.dump(art, ENGINE_DIR / f"{c['name']}.joblib")
        print(c["name"], "NOT_VALIDATED (informational)", flush=True)


if __name__ == "__main__":
    main()
