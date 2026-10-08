"""Web dashboard: candlestick chart, indicators and the model's next-candle call.

Run:
    uvicorn app.server:app --reload
then open http://127.0.0.1:8000

Environment variables:
    MODEL_PATH       trained bundle (default models/model.joblib)
    EXPERIMENTS_DIR  models from `cryptopredict.experiments` (default models/experiments),
                     used by the market scanner and the per-token charts
    DATA_SOURCE      binance | csv | synthetic (default: the source used in training)
    CANDLES          candles shown on the chart (default 300)
    PLAN_HISTORY     candles the trade planner replays a plan over (default 2000)
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cryptopredict import planner
from cryptopredict.data import load_data, tick_size
from cryptopredict.features import add_features
from cryptopredict.predict import load_bundle, predict_latest
from cryptopredict.scanner import scan
from cryptopredict.train import DEFAULT_FEE, MODELS_DIR

STATIC_DIR = Path(__file__).resolve().parent / "static"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", MODELS_DIR / "model.joblib"))
EXPERIMENTS_DIR = Path(os.environ.get("EXPERIMENTS_DIR", MODELS_DIR / "experiments"))
CANDLES = int(os.environ.get("CANDLES", 300))
# Indicators need ~50 candles of warm-up before the first chart candle.
WARMUP = 60
PLAN_HISTORY = int(os.environ.get("PLAN_HISTORY", 2000))
CACHE_SECONDS = 30
SCAN_CACHE_SECONDS = 120
CONFIG_NAME = re.compile(r"^[A-Z0-9]{2,20}_[0-9]{1,2}[mhd]_h[0-9]{1,3}$")

app = FastAPI(title="Crypto Direction Predictor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# (kind, model path) -> {"at": time, "mtime": model mtime, "payload": ...}
_cache: dict = {}


def _num(v):
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def _json_safe(obj):
    """Replace NaN/inf (invalid in JSON) with None, recursively."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _unix(ts: pd.Timestamp) -> int:
    return int(ts.timestamp())


def _out_of_sample_predictions(feats: pd.DataFrame, bundle: dict, model_dir: Path) -> pd.DataFrame:
    """Predictions the model made without having seen the outcome.

    Combines the held-out test-set predictions saved at training time with
    fresh predictions for candles that arrived after training. In-sample
    candles are never shown, because their hit rate would be flattering.
    """
    frames = []
    test_path = model_dir / "test_predictions.csv"
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


def build_dashboard(model_path: Path) -> dict:
    try:
        bundle = load_bundle(model_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    source = os.environ.get("DATA_SOURCE", bundle["source"])
    try:
        df = load_data(source, bundle["symbol"], bundle["interval"], CANDLES + WARMUP, save=False)
    except Exception as exc:  # network errors, missing CSV, ...
        raise HTTPException(status_code=502, detail=f"Could not load market data: {exc}") from exc

    prediction = predict_latest(df, bundle)
    feats = add_features(df).iloc[-CANDLES:]
    preds = _out_of_sample_predictions(feats, bundle, model_path.parent)

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

    metrics_path = model_path.parent / "metrics.json"
    metrics = _json_safe(json.loads(metrics_path.read_text())) if metrics_path.exists() else None

    return {
        "source": source,
        "prediction": prediction,
        "candles": candles,
        "history": history,
        "history_accuracy": hits / len(scored) if scored else None,
        "history_count": len(scored),
        "metrics": metrics,
    }


def _cached(key, path: Path, ttl: float, build):
    mtime = path.stat().st_mtime if path.exists() else None
    entry = _cache.get(key)
    if entry is None or time.time() - entry["at"] >= ttl or entry["mtime"] != mtime:
        entry = {"payload": build(), "at": time.time(), "mtime": mtime}
        _cache[key] = entry
    return entry["payload"]


def _model_path(config: str | None) -> Path:
    if config is None:
        return MODEL_PATH
    if not CONFIG_NAME.match(config):
        raise HTTPException(status_code=400, detail="Invalid config name")
    model_path = EXPERIMENTS_DIR / config / "model.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"No trained model for {config}")
    return model_path


@app.get("/api/dashboard")
def dashboard(config: str | None = None):
    """Chart data for the main model, or for one experiments model (e.g. ETHUSDT_4h_h1)."""
    model_path = _model_path(config)
    return _cached(("dashboard", str(model_path)), model_path, CACHE_SECONDS, lambda: build_dashboard(model_path))


def _plan_market(model_path: Path) -> dict:
    """Symbol, recent candles and per-side trading cost for the chart's coin."""
    try:
        bundle = load_bundle(model_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    source = os.environ.get("DATA_SOURCE", bundle["source"])
    try:
        df = load_data(source, bundle["symbol"], bundle["interval"], PLAN_HISTORY, save=False)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not load market data: {exc}") from exc
    metrics_path = model_path.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    cost = metrics.get("cost_per_trade")
    if cost is None:  # models trained before the half-tick spread was charged
        cost = metrics.get("fee", DEFAULT_FEE)
        tick = tick_size(bundle["symbol"]) if source == "binance" else None
        if tick:
            cost += 0.5 * tick / float(df["close"].median())
    return {"symbol": bundle["symbol"], "interval": bundle["interval"], "source": source, "df": df, "cost": cost}


@app.get("/api/plan")
def trade_plan(config: str | None = None, entry: float | None = None, stop: float | None = None,
               target: float | None = None, side: str = "long", max_bars: int = 48,
               account: float | None = None, risk_pct: float = 1.0):
    """Replay an entry / stop-loss / take-profit plan over the coin's recent history.

    Without levels, suggests a volatility-scaled template (stop 1.5 ATR away, target at 2R).
    """
    model_path = _model_path(config)
    levels = (entry, stop, target)
    if any(v is None for v in levels) and not all(v is None for v in levels):
        raise HTTPException(status_code=400, detail="Give entry, stop and target together, or none to get a suggestion.")
    if side not in ("long", "short"):
        raise HTTPException(status_code=400, detail="side must be long or short")
    if not 1 <= max_bars <= planner.MAX_BARS_LIMIT:
        raise HTTPException(status_code=400, detail=f"max_bars must be between 1 and {planner.MAX_BARS_LIMIT}")
    if account is not None and not (math.isfinite(account) and account > 0):
        raise HTTPException(status_code=400, detail="account must be a positive amount")
    if not (math.isfinite(risk_pct) and 0 < risk_pct <= 100):
        raise HTTPException(status_code=400, detail="risk_pct must be between 0 and 100")

    market = _cached(("plan", str(model_path)), model_path, CACHE_SECONDS, lambda: _plan_market(model_path))
    df = market["df"]
    suggested = entry is None
    if suggested:
        levels = planner.suggest(df, side)
        entry, stop, target = levels["entry"], levels["stop"], levels["target"]
    try:
        result = planner.plan(df, entry, stop, target, max_bars, market["cost"], account, risk_pct / 100)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["plan"]["suggested"] = suggested
    return _json_safe({
        "symbol": market["symbol"],
        "interval": market["interval"],
        "source": market["source"],
        "price": float(df["close"].iloc[-1]),
        "as_of": df["timestamp"].iloc[-1].isoformat(),
        **result,
    })


@app.get("/api/scanner")
def scanner():
    """Current signal for every token/timeframe trained by `cryptopredict.experiments`."""
    summary = EXPERIMENTS_DIR / "summary.csv"

    def build():
        rows = scan(EXPERIMENTS_DIR, os.environ.get("DATA_SOURCE"))
        for r in rows:
            r["config"] = f"{r['symbol']}_{r['interval']}_h{r['horizon']}"
        return {"rows": _json_safe(rows), "scanned_at": pd.Timestamp.now(tz="UTC").isoformat()}

    return _cached(("scanner", str(summary)), summary, SCAN_CACHE_SECONDS, build)


# ------------------------------------------------------------ signal engine

ENGINE_TTL_SECONDS = int(os.environ.get("ENGINE_TTL", 300))
_engine_state: dict = {"engine": None, "at": 0.0, "payload": None}


def _get_engine():
    from quant.data import UNIVERSE
    from quant.engine import Engine, load_specialists

    if _engine_state["engine"] is None:
        specs = load_specialists()
        if not specs:
            raise HTTPException(status_code=503, detail="Signal engine not trained yet: run `python -m quant.train_engine`.")
        symbols = os.environ.get("ENGINE_SYMBOLS", ",".join(UNIVERSE)).split(",")
        from quant.paper import PaperLedger

        _engine_state["engine"] = Engine(symbols, specs, use_book=os.environ.get("ENGINE_BOOK", "1") == "1",
                                         paper=PaperLedger())
    return _engine_state["engine"]


@app.get("/api/engine")
def engine_signals():
    """Evidence-gated signals for every symbol and specialist (NO TRADE unless justified)."""
    if _engine_state["payload"] is not None and time.time() - _engine_state["at"] < ENGINE_TTL_SECONDS:
        return _engine_state["payload"]
    eng = _get_engine()
    signals = eng.run()
    specialists = [{"name": sp.name, "horizon": sp.horizon, "evidence": _json_safe(json.loads(json.dumps(
        sp.evidence.__dict__, default=str))), "health": sp.health.snapshot()} for sp in eng.specialists]
    from quant.engine import consensus

    payload = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "signals": _json_safe(signals),
               "consensus": _json_safe(consensus(signals)), "specialists": specialists,
               "paper": eng.paper.summary() if eng.paper is not None else None}
    _engine_state.update(payload=payload, at=time.time())
    return payload


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
