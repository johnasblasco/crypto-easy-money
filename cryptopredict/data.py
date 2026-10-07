"""Market data loading: Binance public klines, local CSV, or synthetic candles.

Every loader returns a DataFrame with the columns in ``COLUMNS``, sorted by
time, with ``timestamp`` as a timezone-aware UTC datetime.
"""
from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Public market-data hosts, tried in order. data-api.binance.vision serves
# market data only and works in most regions; api.binance.us is the fallback
# for the US where api.binance.com is geo-blocked.
BINANCE_HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api.binance.us",
]
BINANCE_MAX_LIMIT = 1000

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

SYNTHETIC_LENGTH = 10_000

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _klines_to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame([r[:6] for r in rows], columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in COLUMNS[1:]:
        df[col] = df[col].astype(float)
    return df


def tick_size(symbol: str, timeout: float = 10.0) -> float | None:
    """Binance's price step for ``symbol`` (None if unavailable).

    The bid-ask spread can never be smaller than one tick. For sub-cent meme
    coins (PEPE, SHIB, BONK) one tick is 0.1-0.3% of the price, so it matters.
    """
    for host in BINANCE_HOSTS:
        try:
            r = requests.get(f"{host}/api/v3/exchangeInfo", params={"symbol": symbol}, timeout=timeout)
            r.raise_for_status()
            filters = r.json()["symbols"][0]["filters"]
            return float(next(f["tickSize"] for f in filters if f["filterType"] == "PRICE_FILTER"))
        except Exception:
            continue
    return None


def fetch_binance(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 5000,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """Download the most recent ``limit`` closed candles from Binance.

    Binance returns at most 1000 candles per request, so this pages backwards
    with ``endTime`` until ``limit`` candles are collected.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval {interval!r}; use one of {list(INTERVAL_MS)}")

    last_error: Exception | None = None
    for host in BINANCE_HOSTS:
        try:
            return _fetch_binance_from(host, symbol, interval, limit, timeout)
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not download {symbol} {interval} from Binance: {last_error}")


def _fetch_binance_from(host, symbol, interval, limit, timeout) -> pd.DataFrame:
    rows: list = []
    end_time = None
    while len(rows) < limit:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(BINANCE_MAX_LIMIT, limit - len(rows)),
        }
        if end_time is not None:
            params["endTime"] = end_time
        resp = requests.get(f"{host}/api/v3/klines", params=params, timeout=timeout)
        resp.raise_for_status()
        batch = resp.json()
        if not isinstance(batch, list):
            raise ValueError(f"Unexpected response from {host}: {batch}")
        if not batch:
            break
        rows = batch + rows
        end_time = batch[0][0] - 1
        if len(batch) < params["limit"]:
            break
        time.sleep(0.1)  # be polite to the public API

    df = _klines_to_frame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    # The newest candle is still forming; its close is not final yet.
    now = pd.Timestamp.now(tz="UTC")
    still_open = df["timestamp"] + pd.Timedelta(milliseconds=INTERVAL_MS[interval]) > now
    return df[~still_open].reset_index(drop=True)


def generate_synthetic(
    n: int = 5000,
    interval: str = "1h",
    start_price: float = 60_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic-looking OHLCV candles for offline testing.

    Returns follow a regime-switching random walk with volatility clustering
    and a weak short-term momentum term. This exists so the pipeline can be
    developed and tested without network access; model results on synthetic
    data say nothing about real markets.

    The series is always generated at full length and then trimmed, so a
    shorter request returns exactly the tail of a longer one (the dashboard
    relies on this to line up with the training data).
    """
    requested, n = n, max(n, SYNTHETIC_LENGTH)
    rng = np.random.default_rng(seed)
    step = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    end = pd.Timestamp("2026-01-01", tz="UTC")
    timestamps = end - step * np.arange(n)[::-1]

    # GARCH(1,1) variance with a long-run hourly volatility of about 0.6%.
    long_run_var = 0.006**2
    alpha, beta = 0.08, 0.90
    omega = long_run_var * (1 - alpha - beta)

    returns = np.empty(n)
    var = long_run_var
    drift = 0.0
    prev = 0.0
    for i in range(n):
        if rng.random() < 0.01:  # occasional regime change
            drift = rng.normal(0, 0.0006)
        var = omega + alpha * prev**2 + beta * var
        prev = drift + 0.08 * prev + rng.normal(0, np.sqrt(var))
        returns[i] = prev

    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, 0.5, size=(2, n))) * np.maximum(np.abs(returns), 0.002)
    high = np.maximum(open_, close) * (1 + wick[0])
    low = np.minimum(open_, close) * (1 - wick[1])
    volume = rng.lognormal(mean=6.5, sigma=0.4, size=n) * (1 + 60 * np.abs(returns))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    ).iloc[-requested:].reset_index(drop=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load candles from a CSV with at least the columns in ``COLUMNS``.

    ``timestamp`` may be an ISO date string or Unix epoch milliseconds.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[COLUMNS].copy()
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def cache_path(symbol: str, interval: str) -> Path:
    return DATA_DIR / f"{symbol.upper()}_{interval}.csv"


def load_data(
    source: str = "binance",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 5000,
    csv_path: str | Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Load candles from ``source`` ("binance", "csv" or "synthetic")."""
    if source == "binance":
        df = fetch_binance(symbol, interval, limit)
        if save:
            DATA_DIR.mkdir(exist_ok=True)
            df.to_csv(cache_path(symbol, interval), index=False)
        return df
    if source == "csv":
        return load_csv(csv_path or cache_path(symbol, interval))
    if source == "synthetic":
        # A different (but repeatable) series per symbol.
        return generate_synthetic(n=limit, interval=interval, seed=zlib.crc32(symbol.upper().encode()))
    raise ValueError(f"Unknown data source {source!r}")
