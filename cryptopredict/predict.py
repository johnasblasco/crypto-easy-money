"""Predict the direction of the next candle with a trained model.

Usage:
    python -m cryptopredict.predict                 # uses models/model.joblib
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd

from .data import load_data
from .features import add_features
from .train import MODELS_DIR


def load_bundle(path: Path = MODELS_DIR / "model.joblib") -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python -m cryptopredict.train` first.")
    return joblib.load(path)


def _clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def describe_indicators(row: pd.Series) -> list[dict]:
    """Human-readable signal for each headline indicator on the latest candle."""
    rsi_v = row["rsi_14"]
    if rsi_v >= 70:
        rsi_sig = "Overbought"
    elif rsi_v <= 30:
        rsi_sig = "Oversold"
    else:
        rsi_sig = "Bullish" if rsi_v >= 50 else "Bearish"

    pct_b = row["bb_pct_b"]
    if pct_b > 1:
        bb_sig = "Above upper band"
    elif pct_b < 0:
        bb_sig = "Below lower band"
    else:
        bb_sig = "Upper half" if pct_b >= 0.5 else "Lower half"

    vol = row["volume_change"]
    return [
        {"name": "RSI (14)", "value": _clean(rsi_v), "format": "number", "signal": rsi_sig},
        {
            "name": "MACD histogram",
            "value": _clean(row["macd_hist"]),
            "format": "number",
            "signal": "Bullish" if row["macd_hist"] > 0 else "Bearish",
        },
        {
            "name": "EMA 12 / 26 trend",
            "value": _clean(row["ema12_vs_ema26"]),
            "format": "percent",
            "signal": "Bullish" if row["ema_12"] > row["ema_26"] else "Bearish",
        },
        {"name": "Bollinger %B", "value": _clean(pct_b), "format": "number", "signal": bb_sig},
        {
            "name": "Volume change",
            "value": _clean(vol),
            "format": "percent",
            "signal": "Increasing" if vol > 0 else "Decreasing",
        },
        {
            "name": "Price vs SMA 20",
            "value": _clean(row["close_vs_sma20"]),
            "format": "percent",
            "signal": "Above" if row["close_vs_sma20"] > 0 else "Below",
        },
    ]


def predict_latest(df: pd.DataFrame, bundle: dict) -> dict:
    """Predict the candle after the last row of ``df``."""
    feats = add_features(df)
    latest = feats.dropna(subset=bundle["features"]).iloc[-1]
    prob_up = float(bundle["model"].predict_proba(latest[bundle["features"]].to_frame().T.astype(float))[0, 1])
    direction = "UP" if prob_up >= 0.5 else "DOWN"
    prev_close = feats["close"].iloc[-2]
    return {
        "symbol": bundle["symbol"],
        "interval": bundle["interval"],
        "horizon": bundle["horizon"],
        "model": bundle["model_name"],
        "trained_at": bundle["trained_at"],
        "as_of": latest["timestamp"].isoformat(),
        "price": float(latest["close"]),
        "change": float(latest["close"] / prev_close - 1),
        "direction": direction,
        "prob_up": prob_up,
        "confidence": prob_up if direction == "UP" else 1 - prob_up,
        "indicators": describe_indicators(latest),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=MODELS_DIR / "model.joblib")
    p.add_argument("--source", choices=["binance", "csv", "synthetic"], help="default: same as training")
    p.add_argument("--csv")
    args = p.parse_args(argv)

    bundle = load_bundle(args.model)
    df = load_data(args.source or bundle["source"], bundle["symbol"], bundle["interval"], 300, args.csv, save=False)
    print(json.dumps(predict_latest(df, bundle), indent=2))


if __name__ == "__main__":
    main()
