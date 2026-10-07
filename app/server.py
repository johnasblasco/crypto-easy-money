"""Web dashboard: candlestick chart, indicators and the model's next-candle call.

Run:
    uvicorn app.server:app --reload
then open http://127.0.0.1:8000

Environment variables:
    MODEL_PATH   trained bundle (default models/model.joblib)
    DATA_SOURCE  binance | csv | synthetic (default: the source used in training)
    CANDLES      candles shown on the chart (default 300)
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cryptopredict.data import load_data
from cryptopredict.features import add_features
from cryptopredict.predict import load_bundle, predict_latest
from cryptopredict.train import MODELS_DIR

STATIC_DIR = Path(__file__).resolve().parent / "static"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", MODELS_DIR / "model.joblib"))
CANDLES = int(os.environ.get("CANDLES", 300))
# Indicators need ~50 candles of warm-up before the first chart candle.
WARMUP = 60
CACHE_SECONDS = 30

app = FastAPI(title="Crypto Direction Predictor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_cache: dict = {"at": 0.0, "payload": None, "model_mtime": None}


def _num(v):
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def _unix(ts: pd.Timestamp) -> int:
    return int(ts.timestamp())


def _out_of_sample_predictions(feats: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Predictions the model made without having seen the outcome.

    Combines the held-out test-set predictions saved at training time with
    fresh predictions for candles that arrived after training. In-sample
    candles are never shown, because their hit rate would be flattering.
    """
    frames = []
    test_path = MODEL_PATH.parent / "test_predictions.csv"
    if test_path.exists():
        test = pd.read_csv(test_path, parse_dates=["timestamp"])
        frames.append(test[["timestamp", "prob_up"]])

    data_end = bundle.get("data_end")
    if data_end is not None:
        newer = feats[feats["timestamp"] > data_end].dropna(subset=bundle["features"])
        if len(newer):
            prob = bundle["model"].predict_proba(newer[bundle["features"]].astype(float))[:, 1]
            frames.append(pd.DataFrame({"timestamp": newer["timestamp"].to_numpy(), "prob_up": prob}))

    if not frames:
        return pd.DataFrame(columns=["timestamp", "prob_up", "prediction", "actual"])
    preds = pd.concat(frames).drop_duplicates("timestamp", keep="last")
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)

    horizon = bundle["horizon"]
    outcome = feats[["timestamp", "close"]].copy()
    outcome["actual"] = (outcome["close"].shift(-horizon) > outcome["close"]).astype(float)
    outcome.loc[outcome["close"].shift(-horizon).isna(), "actual"] = float("nan")
    preds = preds.merge(outcome[["timestamp", "actual"]], on="timestamp", how="inner")
    preds["prediction"] = (preds["prob_up"] >= 0.5).astype(int)
    return preds.sort_values("timestamp")


def build_dashboard() -> dict:
    try:
        bundle = load_bundle(MODEL_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    source = os.environ.get("DATA_SOURCE", bundle["source"])
    try:
        df = load_data(source, bundle["symbol"], bundle["interval"], CANDLES + WARMUP, save=False)
    except Exception as exc:  # network errors, missing CSV, ...
        raise HTTPException(status_code=502, detail=f"Could not load market data: {exc}") from exc

    prediction = predict_latest(df, bundle)
    feats = add_features(df).iloc[-CANDLES:]
    preds = _out_of_sample_predictions(feats, bundle)

    candles = [
        {
            "time": _unix(r.timestamp),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "ema12": _num(r.ema_12),
            "ema26": _num(r.ema_26),
            "rsi": _num(r.rsi_14),
        }
        for r in feats.itertuples()
    ]
    history = [
        {
            "time": _unix(r.timestamp),
            "prob_up": float(r.prob_up),
            "prediction": int(r.prediction),
            "actual": None if math.isnan(r.actual) else int(r.actual),
        }
        for r in preds.itertuples()
    ]
    scored = [h for h in history if h["actual"] is not None]
    hits = sum(h["prediction"] == h["actual"] for h in scored)

    metrics_path = MODEL_PATH.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    return {
        "source": source,
        "prediction": prediction,
        "candles": candles,
        "history": history,
        "history_accuracy": hits / len(scored) if scored else None,
        "history_count": len(scored),
        "metrics": metrics,
    }


@app.get("/api/dashboard")
def dashboard():
    mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    fresh = time.time() - _cache["at"] < CACHE_SECONDS and _cache["model_mtime"] == mtime
    if not fresh or _cache["payload"] is None:
        _cache.update(payload=build_dashboard(), at=time.time(), model_mtime=mtime)
    return _cache["payload"]


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
