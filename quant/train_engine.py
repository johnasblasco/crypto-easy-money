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
                       trades, trading_stats)
from .costs import COST_MODELS
from .data import KLINE_DIR, UNIVERSE
from .engine import ENGINE_DIR, Evidence
from .events import day_bootstrap
from .experiment import HOLDOUT_START, make_dataset

HOLDOUT = pd.Timestamp(HOLDOUT_START, tz="UTC")

# Frozen from research-period evidence (see docs/RESEARCH_REPORT.md). Filled in
# after the research studies, before any holdout evaluation.
# Selection rule (docs/PREREGISTRATION.md): research net EV per trade > 0 after the cost
# model it will use, beats its simplest baseline, and at most ONE variant per idea, chosen by
# its research-period result. Every variant is in the trial ledger / results JSON.
PANEL_SYMBOLS = [s for s in UNIVERSE if (KLINE_DIR / f"{s}.parquet").exists()]
ALTS = [s for s in PANEL_SYMBOLS if s not in ("BTCUSDT", "ETHUSDT")]

CANDIDATES: list[dict] = [
    {
        # Idea: ML on the pooled daily panel. Best of 20 ledger trials (daily_panel h1/h3 x
        # logit/lgbm/trend/meta/bh x spot/perp): h3 LightGBM, perp costs, +316bp/trade,
        # day-clustered p=0.007, beats B&H on point estimate (paired p=0.25). Dissection
        # (dissect_panel_h3_perp_taker.json): 42 decision dates, mostly market timing (excess over
        # the same-direction market move 12bp, t=1.29); timing beats random dates p=0.023.
        "name": "daily_panel_3d", "study": "daily_panel", "ledger_id": "b37dbd69f8a2", "horizon_min": 4320,
        "families": ["mom", "vol", "act", "flow", "range", "regime", "cal", "xa", "trend", "fng", "ta", "ens"],
        "symbols": PANEL_SYMBOLS, "model": "lightgbm",
        "model_kwargs": {"n_estimators": 200, "min_child_samples": 200},
        "cost": "perp_taker", "decision": "prob", "min_history_days": 60, "require_cols": ["ens__trend_ensemble"],
        "description": "LightGBM on the pooled 16-coin daily panel, 3-day hold, long/short perps; it trades only "
                       "when calibrated confidence clears the margin chosen on validation (research: mostly "
                       "market-timing calls on a few dates).",
    },
    {
        # Idea: intraday direction model (order flow, bar shape, momentum, ...). Variant chosen among the
        # 24 intraday_conversion trials: positive research net EV, >= 100 trades, lowest p-value.
        # ETH 4h, perp costs, EV-vs-cost rule: 130 trades, +20.6bp/trade, p=0.052; DSR 0.50, PBO 0.56.
        "name": "eth_4h_flow_model", "study": "intraday_conversion", "ledger_id": "c9758efaeebb", "horizon_min": 240,
        "families": ["mom", "vol", "act", "flow", "shape", "range", "regime", "cal", "ta", "xa"],
        "symbols": ["ETHUSDT"], "model": "lightgbm", "cost": "perp_taker", "decision": "ev",
        "description": "LightGBM on 1-minute order-flow, bar-shape and momentum features for ETH, 4-hour hold, "
                       "long/short perps; it trades only when expected edge clears round-trip cost plus a "
                       "validated margin.",
    },
]

EVENT_CANDIDATES: list[dict] = [
    {
        # Idea: flow-driven capitulation reversal. Best of 48 side-aware cells (execution_realism.json),
        # found among 204 event cells (event_hypotheses.json) -> 252 trials for the deflated Sharpe.
        "name": "alt_capitulation_reversal_1h", "W": 60, "k": 4.0, "h": 60, "cost": "spot_taker",
        "symbols": ALTS, "n_trials": 252, "research_net_bps": 33.2,
        "description": "Buy an altcoin after a forced, one-sided sell-off (60-minute drop of at least 4 ex-ante "
                       "sigmas with top-quintile taker selling), hold 1 hour; spot, aggressor-side fills.",
    },
]


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


def _subset(ds: Dataset, keep: np.ndarray) -> Dataset:
    return Dataset(ds.symbol, ds.horizon_min, ds.X[keep], ds.fwd_ret[keep], ds.label_end[keep],
                   ds.regimes[keep] if not ds.regimes.empty else ds.regimes,
                   sym=ds.sym[keep], sigma=None if ds.sigma is None else ds.sigma[keep])


def candidate_dataset(c: dict) -> Dataset:
    """The candidate's pooled dataset (research + holdout), with the research study's row filters."""
    parts = []
    for s in c["symbols"]:
        ds = make_dataset(s, c["horizon_min"], families=c["families"], holdout=True)
        keep = np.ones(len(ds.X), dtype=bool)
        if c.get("min_history_days"):
            keep &= ds.times >= ds.times[0] + pd.Timedelta(days=c["min_history_days"])
        for col in c.get("require_cols", []):
            keep &= ds.X[col].notna().to_numpy()
        parts.append(_subset(ds, keep))
    return Dataset.pool(parts) if len(parts) > 1 else parts[0]


def evaluate_candidate(c: dict) -> dict:
    cost = COST_MODELS[c["cost"]]
    ds = candidate_dataset(c)
    factory = getattr(models, c["model"])(**c.get("model_kwargs", {}))
    art = _fit_freeze(ds, HOLDOUT, factory, cost, c["decision"])
    rows = np.flatnonzero(ds.times >= HOLDOUT)
    pred = _predict_positions(ds, art, rows, cost, c["decision"])
    sim = simulate_panel(pred, cost, ds.horizon_min)
    st = trading_stats(sim, ds.horizon_min)
    # Trade-level P&L (each trade includes its exit cost), clustered by entry day.
    act = trades(sim)
    mean, p_day, lo = day_bootstrap(act, "net") if len(act) >= 5 else (float("nan"),) * 3
    # Deflated Sharpe over every research-period trial of this study in the ledger (pre-registered).
    trials = [t for t in ledger.trials() if t["config"].get("study") == c["study"]]
    dsr = float("nan")
    if c.get("ledger_id"):
        try:
            dsr = ledger.selection_stats({"study": c["study"]}, c["ledger_id"])["deflated_sharpe"]
        except KeyError:
            pass
    passed = (len(act) >= 30 and np.isfinite(mean) and mean > 0 and p_day < 0.05 and np.isfinite(dsr) and dsr > 0.5)
    status = "VALIDATED" if passed else "NOT_VALIDATED"
    if len(act):
        holdout_txt = (f"Holdout: {len(act)} trades, mean net {mean * 1e4 if np.isfinite(mean) else float('nan'):+.1f}bp "
                       f"(day-clustered p={p_day:.3f}), Sharpe {st['sharpe_ann']:.2f}")
    else:
        holdout_txt = ("Holdout: 0 trades - its skill gate failed on the latest validation slice, so it abstained "
                       "throughout" if not art["skilled"] else "Holdout: 0 trades - no prediction cleared the margin")
    summary = (f"{c['description']} {holdout_txt}; research-period DSR {dsr:.2f} over {len(trials)} trials. "
               f"Verdict: {status} (pre-registered criteria).")
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
                "evidence": asdict(evidence), "trade_sigma": trade_sigma, "symbols": c["symbols"],
                "xa_universe": PANEL_SYMBOLS}
    return {"artifact": artifact, "holdout_stats": st, "evidence": asdict(evidence)}


def informational_model(c: dict, research_summary: str) -> dict:
    """A model that did NOT qualify for the holdout: fitted for display only, always NO TRADE."""
    cost = COST_MODELS[c["cost"]]
    ds = candidate_dataset(c)
    factory = getattr(models, c["model"])(**c.get("model_kwargs", {}))
    live = _fit_freeze(ds, pd.Timestamp.now(tz="UTC"), factory, cost, c["decision"])
    evidence = Evidence(status="NOT_VALIDATED", summary=research_summary, period="research 2020-2025-09",
                        benchmark="zero (flat) after costs",
                        notes=["did not meet the research-period criterion to be tested on the holdout",
                               "shown for information only: the engine never trades it"])
    return {"name": c["name"], "horizon_min": c["horizon_min"], "families": c["families"],
            "columns": list(ds.X.columns), "model": live["model"], "calibrator": live["calibrator"],
            "margin": None, "cost": c["cost"], "decision": c["decision"], "evidence": asdict(evidence),
            "trade_sigma": 0.01, "symbols": c["symbols"], "xa_universe": PANEL_SYMBOLS}


# ------------------------------------------------------------------ events

def event_research_dsr(c: dict) -> tuple[float, int]:
    """Deflated Sharpe of the chosen event cell over every event cell tried in research."""
    from .experiment import RESULTS_DIR

    cells = json.loads((RESULTS_DIR / "execution_realism.json").read_text())["flow_reversal_side_fills"]
    srs = np.array([r["sr_day"] for r in cells if np.isfinite(r.get("sr_day", np.nan))])
    best = next(r for r in cells if (r["W"], r["k"], r["h"], r["group"], r["cost"]) ==
                (c["W"], c["k"], c["h"], "alts", c["cost"]))
    sr0 = M.expected_max_sharpe(c["n_trials"], float(srs.var(ddof=1)))
    return M.probabilistic_sharpe(best["sr_day"], best["n_day"], best["skew_day"], best["kurt_day"], sr0), c["n_trials"]


def evaluate_event_candidate(c: dict) -> dict:
    """Frozen event rule on the holdout, aggressor-side fills, exactly as in research."""
    from .events import Bars, select
    from .experiment import m1_frame
    from .studies.event_hypotheses import h_flow_driven_reversal
    from .studies.execution_realism import OWN_IMPACT, side_fills, side_outcomes

    cost = COST_MODELS[c["cost"]]
    parts = []
    for sym in c["symbols"]:
        m1 = m1_frame(sym)
        b = Bars(m1)
        buy, sell = side_fills(m1)
        sig = h_flow_driven_reversal(b, c["W"], c["k"])
        idx = select(sig, cooldown=c["h"])
        idx = idx[b.index[idx] >= HOLDOUT]
        ev = side_outcomes(b, buy, sell, idx, np.sign(sig[idx]).astype(int), c["h"])
        if not cost.allow_short:
            ev = ev[ev["dir"] > 0]
        parts.append(ev.assign(sym=sym))
    ev = pd.concat(parts, ignore_index=True)
    fund = np.where(ev["dir"] > 0, cost.holding(c["h"], 1), 0.0)
    ev["net"] = ev["ret_side"] - 2 * (cost.fee_bps / 1e4 + OWN_IMPACT) - fund
    mean, p_day, lo = day_bootstrap(ev, "net") if len(ev) >= 5 else (float("nan"),) * 3
    dsr, n_trials = event_research_dsr(c)
    passed = len(ev) >= 30 and np.isfinite(mean) and mean > 0 and p_day < 0.05 and dsr > 0.5
    status = "VALIDATED" if passed else "NOT_VALIDATED"
    by_month = {str(k): {"n": int(len(g)), "net_bps": round(float(g["net"].mean() * 1e4), 1)}
                for k, g in ev.groupby(ev["ts"].dt.to_period("M"))}
    span = f"{ev['ts'].min():%Y-%m-%d} -> {ev['ts'].max():%Y-%m-%d}" if len(ev) else "no events"
    summary = (f"{c['description']} Research 2020-2025-09: {c['research_net_bps']:+.1f}bp/event net "
               f"(aggressor-side fills), concentrated in 2020-21. Holdout ({span}): {len(ev)} events, "
               f"mean net {mean * 1e4 if np.isfinite(mean) else float('nan'):+.1f}bp (day-clustered p={p_day:.3f}); "
               f"research deflated Sharpe {dsr:.2f} over {n_trials} event cells. Verdict: {status} (pre-registered criteria).")
    evidence = Evidence(status=status, summary=summary, period=f"holdout {HOLDOUT_START} onward", trades=int(len(ev)),
                        ev_bps=float(mean * 1e4) if np.isfinite(mean) else None,
                        ev_p5_bps=float(lo * 1e4) if np.isfinite(lo) else None,
                        benchmark="zero (flat) after fees and aggressor-side fills",
                        notes=[f"hit rate {float((ev['net'] > 0).mean()):.1%}" if len(ev) else "no holdout events",
                               f"deflated Sharpe {dsr:.3f} (N={n_trials})", f"by month: {by_month}"])
    artifact = {"kind": "event", "name": c["name"], "W": c["W"], "k": c["k"], "h": c["h"], "cost": c["cost"],
                "symbols": c["symbols"], "evidence": asdict(evidence),
                "trade_sigma": float(ev["net"].std()) if len(ev) > 2 else 0.01}
    return {"artifact": artifact, "evidence": asdict(evidence), "by_month": by_month}


# Models shown for information only (research verdict: statistically skilful, not cost-surviving).
INFORMATIONAL: list[dict] = [
    {
        "name": "btc_4h_flow_view", "horizon_min": 240, "symbols": ["BTCUSDT"], "model": "lightgbm",
        "families": ["mom", "vol", "act", "flow", "shape", "range", "regime", "cal", "ta", "xa"],
        "cost": "perp_taker", "decision": "ev",
        "research_summary": "Same model family as eth_4h_flow_model, fitted for BTC. Research walk-forward "
                            "(2021-2025-09): AUC 0.562, skill over the base rate in about half the folds, but "
                            "after perp taker costs +6.7bp/trade with p=0.28 (EV rule). Not a holdout candidate "
                            "(one variant per idea). Shown as a calibrated directional view; it never trades.",
    },
]


def main():
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    ev = trend_evidence()
    (ENGINE_DIR / "trend_evidence.json").write_text(json.dumps(asdict(ev), indent=2, default=str))
    print("trend:", ev.status, "|", ev.summary, flush=True)
    for c in CANDIDATES:
        res = evaluate_candidate(c)
        joblib.dump(res["artifact"], ENGINE_DIR / f"{c['name']}.joblib")
        print(c["name"], res["evidence"]["status"], "|", res["evidence"]["summary"], flush=True)
    for c in EVENT_CANDIDATES:
        res = evaluate_event_candidate(c)
        joblib.dump(res["artifact"], ENGINE_DIR / f"{c['name']}.joblib")
        print(c["name"], res["evidence"]["status"], "|", res["evidence"]["summary"], flush=True)
    for c in INFORMATIONAL:
        art = informational_model(c, c["research_summary"])
        joblib.dump(art, ENGINE_DIR / f"{c['name']}.joblib")
        print(c["name"], "NOT_VALIDATED (informational)", flush=True)


if __name__ == "__main__":
    main()
