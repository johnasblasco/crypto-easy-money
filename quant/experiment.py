"""Glue: build a dataset for (symbol, horizon), run a model through walk-forward, report.

Feature matrices are cached per (symbol, decision step) under data/features/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .backtest import Dataset, evaluate, walk_forward_predict
from .costs import COST_MODELS
from .data import ROOT, UNIVERSE, load
from .labels import decision_times, forward_returns
from .validation import check_no_leakage, walk_forward

FEATURE_DIR = ROOT / "data" / "features"
RESULTS_DIR = ROOT / "data" / "results"

# Untouched final holdout. Research runs end here; the final engine is
# evaluated on [HOLDOUT_START, now) exactly once. Do not move this date after
# looking at results.
HOLDOUT_START = "2025-10-01"

_M1_CACHE: dict = {}


def m1_frame(symbol: str) -> pd.DataFrame:
    if symbol not in _M1_CACHE:
        _M1_CACHE[symbol] = load(symbol)
    return _M1_CACHE[symbol]


_CLOSE_CACHE: dict = {}


def log_close(symbol: str) -> pd.Series:
    """1m log close on a complete close-time grid (forward-filled), without loading the full frame."""
    if symbol not in _CLOSE_CACHE:
        from .data import KLINE_DIR, MINUTE_MS

        df = pd.read_parquet(KLINE_DIR / f"{symbol}.parquet", columns=["open_time", "close"])
        idx = pd.DatetimeIndex(pd.to_datetime(df["open_time"] + MINUTE_MS, unit="ms", utc=True))
        s = pd.Series(np.log(df["close"].to_numpy()), index=idx)
        full = pd.date_range(idx[0], idx[-1], freq="1min", tz="UTC")
        _CLOSE_CACHE[symbol] = s.reindex(full).ffill()
    return _CLOSE_CACHE[symbol]


def log_closes(symbols, ref_index=None) -> dict:
    out = {}
    for s in symbols:
        try:
            out[s] = log_close(s)
        except FileNotFoundError:
            continue
    return out


def _family_path(symbol: str, step_min: int, family: str, others: list[str]) -> Path:
    tag = family
    if family == "xa":  # cross-asset features depend on which symbols exist
        import hashlib
        tag += "_" + hashlib.sha1(",".join(sorted(others)).encode()).hexdigest()[:8]
    return FEATURE_DIR / f"{symbol}_{step_min}m_{tag}.parquet"


def feature_matrix(symbol: str, step_min: int, families=None, universe=None, refresh: bool = False) -> pd.DataFrame:
    """Feature matrix for the requested families, cached one parquet per family."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    families = list(families or F.FAMILIES)
    universe = [s for s in (universe or UNIVERSE) if (ROOT / "data" / "klines_1m" / f"{s}.parquet").exists()]
    m1 = None
    parts = []
    for fam in families:
        path = _family_path(symbol, step_min, fam, universe)
        if path.exists() and not refresh:
            parts.append(pd.read_parquet(path))
            continue
        if m1 is None:
            m1 = m1_frame(symbol)
            times = decision_times(m1, f"{step_min}min")
        others = log_closes(universe) if fam == "xa" else None
        X = F.build(m1, times, symbol, others=others, families=[fam])
        X.to_parquet(path)
        parts.append(X)
    return pd.concat(parts, axis=1)


def regimes_for(X: pd.DataFrame) -> pd.DataFrame:
    """Ex-ante regime labels (from past-only features) for stratified reporting."""
    reg = pd.DataFrame(index=X.index)
    if "regime__vol_pctile_1y" in X:
        reg["vol"] = pd.cut(X["regime__vol_pctile_1y"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["low", "mid", "high"])
    if "mom__ret_10080" in X:
        reg["trend7d"] = np.where(X["mom__ret_10080"] > 0, "up", "down")
    return reg


def make_dataset(symbol: str, horizon_min: int, families=None, latency_min: int = 1,
                 start: str | None = None, holdout: bool = False, price: str = "vwap") -> Dataset:
    """Research dataset (ends before HOLDOUT_START), or the full one if ``holdout``."""
    families = list(families or F.FAMILIES)
    X_all = feature_matrix(symbol, horizon_min, sorted(set(families) | {"regime", "mom"}))
    X = X_all[[c for c in X_all.columns if c.split("__")[0] in families]]
    if start:
        X = X[X.index >= pd.Timestamp(start, tz="UTC")]
    if not holdout:
        # Drop samples whose label would end inside the holdout as well.
        cutoff = pd.Timestamp(HOLDOUT_START, tz="UTC") - pd.Timedelta(minutes=horizon_min + latency_min)
        X = X[X.index < cutoff]
    m1 = m1_frame(symbol)
    lab = forward_returns(m1, X.index, horizon_min, latency_min, price=price)
    return Dataset(symbol=symbol, horizon_min=horizon_min, X=X, fwd_ret=lab["fwd_ret"],
                   label_end=lab["exit_ts"], regimes=regimes_for(X_all.reindex(X.index)),
                   sigma=ex_ante_sigma(m1, X.index, horizon_min))


def ex_ante_sigma(m1: pd.DataFrame, times: pd.DatetimeIndex, horizon_min: int, halflife_min: int = 1440) -> pd.Series:
    """Std of the next ``horizon`` log return predicted from an EWMA of past 1m squared returns.

    Uses 1m bars closed by each decision time only (EWMA is causal).
    """
    r = np.log(m1["close"]).ffill().diff()
    v = (r ** 2).ewm(halflife=halflife_min, min_periods=halflife_min).mean()
    return np.sqrt(v.reindex(times) * horizon_min)


def run(ds: Dataset, model_factory, cost_names=("perp_taker", "spot_taker"), test_size="90D",
        min_train="365D", calibration="platt", train_window=None, min_trades: int = 30):
    folds = walk_forward(ds.times, pd.DatetimeIndex(ds.label_end), test_size=test_size,
                         min_train=min_train, train_window=train_window)
    check_no_leakage(folds, ds.times, pd.DatetimeIndex(ds.label_end))
    reports = {}
    preds = {}
    for cname in cost_names:
        cost = COST_MODELS[cname]
        pred = walk_forward_predict(ds, folds, model_factory, calibration=calibration, cost=cost,
                                    min_trades=min_trades)
        rep, sim = evaluate(ds, pred, cost)
        reports[cname] = rep
        preds[cname] = pred.assign(net=sim["net"])
    return reports, preds


def save(name: str, obj) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o)))
    return path
