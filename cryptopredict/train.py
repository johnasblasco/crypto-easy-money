"""Train and compare UP/DOWN classifiers, then save the best one.

Usage:
    python -m cryptopredict.train --source binance --symbol BTCUSDT --interval 1h
    python -m cryptopredict.train --source synthetic        # offline demo

Methodology (see README):
  * Chronological train/test split -- never shuffle time series.
  * A gap of ``horizon`` rows between train and test (and between CV folds)
    so no training target peeks into the test period.
  * Walk-forward cross-validation on the training set picks the best model;
    the held-out test set is only used for the final report.
  * Models are compared against naive baselines; beating them is the bar.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .data import load_data
from .features import FEATURE_COLUMNS, build_dataset
from .models import get_models

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_FEE = 0.001  # 0.1% per side, a typical spot taker fee


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:  # only one class present
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def backtest(pred: np.ndarray, future_return: np.ndarray, fee: float, horizon: int = 1) -> dict:
    """Long-only toy strategy: hold for one interval when the model says UP.

    Only meaningful for horizon=1, where consecutive positions don't overlap.
    Ignores slippage and funding; it is a sanity check, not a trading system.
    """
    if horizon != 1:
        return {}
    position = pred.astype(float)
    trades = np.abs(np.diff(np.concatenate([[0.0], position])))
    strat = position * future_return - trades * fee
    return {
        "strategy_return": float(np.prod(1 + strat) - 1),
        "buy_and_hold_return": float(np.prod(1 + future_return) - 1),
        "trades": int(trades.sum()),
        "time_in_market": float(position.mean()),
    }


def walk_forward_cv(model, X, y, n_splits: int, gap: int) -> dict:
    scores = {"accuracy": [], "roc_auc": []}
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        m = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = m.predict_proba(X.iloc[val_idx])[:, 1]
        scores["accuracy"].append(accuracy_score(y.iloc[val_idx], prob >= 0.5))
        scores["roc_auc"].append(roc_auc_score(y.iloc[val_idx], prob))
    return {
        "cv_accuracy": float(np.mean(scores["accuracy"])),
        "cv_accuracy_std": float(np.std(scores["accuracy"])),
        "cv_roc_auc": float(np.mean(scores["roc_auc"])),
    }


def run(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    source: str,
    horizon: int = 1,
    test_size: float = 0.2,
    cv_splits: int = 5,
    fee: float = DEFAULT_FEE,
    out_dir: Path = MODELS_DIR,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    X, y, frame = build_dataset(df, horizon)
    if len(X) < 300:
        raise ValueError(f"Only {len(X)} usable rows; load more candles (at least ~500).")

    split = int(len(X) * (1 - test_size))
    train_end = split - horizon  # gap: last train targets must not reach into the test set
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    test_frame = frame.iloc[split:]
    future_ret = test_frame["future_return"].to_numpy()

    log = print if verbose else (lambda *a, **k: None)
    log(f"Dataset: {len(X)} rows  |  train {len(X_train)}  |  test {len(X_test)}")
    log(f"Train period: {frame['timestamp'].iloc[0]} -> {frame['timestamp'].iloc[train_end - 1]}")
    log(f"Test period:  {test_frame['timestamp'].iloc[0]} -> {test_frame['timestamp'].iloc[-1]}")
    log(f"UP share: train {y_train.mean():.1%}  |  test {y_test.mean():.1%}\n")

    results = []

    # Baselines a model has to beat.
    majority = int(y_train.mean() >= 0.5)
    base_pred = np.full(len(y_test), majority)
    results.append({
        "model": f"Baseline: always {'UP' if majority else 'DOWN'}",
        "baseline": True,
        **classification_metrics(y_test, base_pred, np.full(len(y_test), y_train.mean())),
        **backtest(base_pred, future_ret, fee, horizon),
    })
    persist_pred = (X_test["ret_1"] > 0).astype(int).to_numpy()
    results.append({
        "model": "Baseline: repeat last move",
        "baseline": True,
        **classification_metrics(y_test, persist_pred, persist_pred),
        **backtest(persist_pred, future_ret, fee, horizon),
    })

    fitted, test_probs = {}, {}
    for name, model in get_models(random_state).items():
        log(f"Training {name} ...")
        cv = walk_forward_cv(model, X_train, y_train, cv_splits, gap=horizon)
        m = clone(model).fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        fitted[name], test_probs[name] = m, prob
        results.append({
            "model": name,
            "baseline": False,
            **cv,
            **classification_metrics(y_test, pred, prob),
            **backtest(pred, future_ret, fee, horizon),
        })

    candidates = [r for r in results if not r["baseline"]]
    best = max(candidates, key=lambda r: r["cv_accuracy"])
    best_name = best["model"]
    log(f"\nBest model by walk-forward CV accuracy: {best_name}")

    importance = permutation_importance(
        fitted[best_name], X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=random_state
    )
    feature_importance = sorted(
        zip(FEATURE_COLUMNS, importance.importances_mean.round(5).tolist()),
        key=lambda kv: -kv[1],
    )

    # Refit on every labelled row so the live model has seen the latest market.
    final_model = clone(get_models(random_state)[best_name]).fit(X, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = {
        "symbol": symbol,
        "interval": interval,
        "source": source,
        "horizon": horizon,
        "trained_at": trained_at,
        "fee": fee,
        "rows": {"total": len(X), "train": len(X_train), "test": len(X_test)},
        "test_period": [str(test_frame["timestamp"].iloc[0]), str(test_frame["timestamp"].iloc[-1])],
        "test_up_share": float(y_test.mean()),
        "best_model": best_name,
        "results": results,
        "feature_importance": feature_importance,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    preds = test_frame[["timestamp", "close", "future_return", "target"]].copy()
    preds["prob_up"] = test_probs[best_name]
    preds["prediction"] = (preds["prob_up"] >= 0.5).astype(int)
    preds.to_csv(out_dir / "test_predictions.csv", index=False)

    joblib.dump(
        {
            "model": final_model,
            "model_name": best_name,
            "features": FEATURE_COLUMNS,
            "symbol": symbol,
            "interval": interval,
            "source": source,
            "horizon": horizon,
            "trained_at": trained_at,
            # Last candle whose outcome the final model was fitted on; anything
            # after this is out-of-sample for the dashboard.
            "data_end": frame["timestamp"].iloc[-1],
        },
        out_dir / "model.joblib",
    )

    report = format_report(metrics)
    (out_dir / "report.md").write_text(report)
    log("\n" + report)
    log(f"Saved model, metrics and predictions to {out_dir}/")
    return metrics


def _pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}%"


def format_report(metrics: dict) -> str:
    lines = [
        f"# {metrics['symbol']} {metrics['interval']} — next {metrics['horizon']} candle(s) direction",
        "",
        f"Source: {metrics['source']} · Test period: {metrics['test_period'][0]} → "
        f"{metrics['test_period'][1]} · Test rows: {metrics['rows']['test']} · "
        f"UP share in test: {_pct(metrics['test_up_share'])}",
        "",
        "| Model | CV acc | Accuracy | Precision | Recall | F1 | ROC AUC | Strategy | Buy & hold |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in metrics["results"]:
        name = f"**{r['model']}**" if r["model"] == metrics["best_model"] else r["model"]
        lines.append(
            f"| {name} | {_pct(r.get('cv_accuracy'))} | {_pct(r['accuracy'])} | "
            f"{_pct(r['precision'])} | {_pct(r['recall'])} | {_pct(r['f1'])} | "
            f"{r['roc_auc']:.3f} | {_pct(r.get('strategy_return'))} | {_pct(r.get('buy_and_hold_return'))} |"
        )
    lines += [
        "",
        f"Strategy = long-only, hold one candle when the model predicts UP, "
        f"{metrics['fee'] * 100:.2f}% fee per trade, no slippage.",
        "",
        "Top features (permutation importance, drop in test ROC AUC):",
        "",
    ]
    for feat, score in metrics["feature_importance"][:8]:
        lines.append(f"- `{feat}`: {score:+.4f}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["binance", "csv", "synthetic"], default="binance")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h", help="1m 5m 15m 30m 1h 4h 1d")
    p.add_argument("--limit", type=int, default=5000, help="number of candles to load")
    p.add_argument("--csv", help="CSV path for --source csv (default: data/<SYMBOL>_<interval>.csv)")
    p.add_argument("--horizon", type=int, default=1, help="predict the direction this many candles ahead")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--fee", type=float, default=DEFAULT_FEE)
    p.add_argument("--out", type=Path, default=MODELS_DIR)
    args = p.parse_args(argv)

    df = load_data(args.source, args.symbol, args.interval, args.limit, args.csv)
    run(
        df,
        symbol=args.symbol.upper(),
        interval=args.interval,
        source=args.source,
        horizon=args.horizon,
        test_size=args.test_size,
        fee=args.fee,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
