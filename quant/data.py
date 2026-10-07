"""1-minute Binance spot klines: download, cache, load, quality-check, resample.

Bars are stored with Binance's *open* time. A bar with open time ``t`` covers
``[t, t + 1m)`` and is only known once it has closed, at ``t + 1m``. All
downstream code indexes bars by **close time** (``ts``) so a feature computed
"at time ts" can only use bars whose close is <= ts. Using open times as the
index is a classic one-bar look-ahead bug.

Usage:
    python -m quant.data download --symbols BTCUSDT ETHUSDT --start 2020-01-01
    python -m quant.data check --symbols BTCUSDT
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
KLINE_DIR = ROOT / "data" / "klines_1m"
API = "https://data-api.binance.vision/api/v3/klines"
MINUTE_MS = 60_000
CHUNK = 1000  # Binance max candles per request

# Liquid spot pairs with multi-year history. Chosen today, so this list carries
# survivorship bias for anything except BTC/ETH; see the research report.
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT", "LINKUSDT",
    "LTCUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT", "ATOMUSDT", "BCHUSDT", "ETCUSDT", "XLMUSDT",
]

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
PRICE_COLS = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]


def _session() -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter)
    return s


def _fetch_chunk(session, symbol: str, start_ms: int, retries: int = 6) -> list:
    params = {"symbol": symbol, "interval": "1m", "limit": CHUNK, "startTime": start_ms,
              "endTime": start_ms + CHUNK * MINUTE_MS - 1}
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=30)
            if r.status_code in (418, 429):  # rate limited: back off hard
                time.sleep(int(r.headers.get("Retry-After", 30)))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{symbol}: failed to fetch chunk starting {start_ms}")


def _first_available(session, symbol: str, start_ms: int) -> int:
    """Open time of the first 1m bar at or after start_ms (handles later listings)."""
    r = session.get(API, params={"symbol": symbol, "interval": "1m", "limit": 1, "startTime": start_ms}, timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise ValueError(f"{symbol}: no data after {start_ms}")
    return rows[0][0]


def download(symbol: str, start: str = "2020-01-01", end: str | None = None, threads: int = 12,
             out_dir: Path = KLINE_DIR, verbose: bool = True) -> Path:
    """Download all closed 1m bars in [start, end) and write ``<out_dir>/<symbol>.parquet``.

    Resumes from an existing file: only bars after its last open time are fetched.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}.parquet"
    session = _session()
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else now_ms
    end_ms = min(end_ms, now_ms // MINUTE_MS * MINUTE_MS)  # exclude the still-forming bar

    existing = None
    if path.exists():
        existing = pd.read_parquet(path)
        start_ms = max(start_ms, int(existing["open_time"].iloc[-1]) + MINUTE_MS)
    else:
        start_ms = _first_available(session, symbol, start_ms)
    if start_ms >= end_ms:
        return path

    starts = list(range(start_ms, end_ms, CHUNK * MINUTE_MS))
    t0 = time.time()
    rows: list = []
    with cf.ThreadPoolExecutor(threads) as ex:
        for i, chunk in enumerate(ex.map(lambda s: _fetch_chunk(session, symbol, s), starts)):
            rows.extend(chunk)
            if verbose and i and i % 500 == 0:
                print(f"  {symbol}: {i}/{len(starts)} chunks ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows, columns=COLUMNS).drop(columns=["ignore", "close_time"])
    df = df[df["open_time"] < end_ms]
    for c in PRICE_COLS:
        df[c] = df[c].astype("float64")
    for c in VOLUME_COLS:
        df[c] = df[c].astype("float64")
    df["trades"] = df["trades"].astype("int64")
    df["open_time"] = df["open_time"].astype("int64")
    if existing is not None:
        df = pd.concat([existing, df])
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df.to_parquet(path, index=False)
    if verbose:
        print(f"{symbol}: {len(df):,} bars -> {path} ({time.time() - t0:.0f}s)", flush=True)
    return path


def load(symbol: str, start: str | None = None, end: str | None = None, data_dir: Path = KLINE_DIR) -> pd.DataFrame:
    """Load 1m bars indexed by bar CLOSE time (UTC), on a complete 1-minute grid.

    Missing minutes (exchange outages, no trades) are inserted as rows with NaN
    prices and zero volume and flagged in ``gap``; nothing is forward-filled,
    so a feature window spanning an outage is visibly contaminated.
    """
    df = pd.read_parquet(data_dir / f"{symbol}.parquet")
    df["ts"] = pd.to_datetime(df["open_time"] + MINUTE_MS, unit="ms", utc=True)
    df = df.drop(columns=["open_time"]).set_index("ts")
    full = pd.date_range(df.index[0], df.index[-1], freq="1min", tz="UTC")
    df = df.reindex(full)
    df.index.name = "ts"
    df["gap"] = df["close"].isna()
    df[VOLUME_COLS + ["trades"]] = df[VOLUME_COLS + ["trades"]].fillna(0)
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate close-time-indexed 1m bars into ``rule`` bars, labelled by close time.

    A 1h bar labelled 10:00 contains the 1m bars that closed in (09:00, 10:00].
    ``gap_frac`` is the share of missing minutes inside the bar.
    """
    g = df.resample(rule, label="right", closed="right")
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "quote_volume": g["quote_volume"].sum(),
        "trades": g["trades"].sum(),
        "taker_buy_base": g["taker_buy_base"].sum(),
        "taker_buy_quote": g["taker_buy_quote"].sum(),
        "gap_frac": g["gap"].mean(),
    })
    # Drop a trailing bar that hasn't closed yet (its label is after the last 1m close).
    return out[out.index <= df.index[-1]]


def quality_report(df: pd.DataFrame) -> dict:
    """Basic integrity checks on a 1m frame from ``load``."""
    valid = df.dropna(subset=["close"])
    bad_ohlc = ((valid["high"] < valid[["open", "close"]].max(axis=1)) |
                (valid["low"] > valid[["open", "close"]].min(axis=1))).sum()
    ret = np.log(valid["close"]).diff().abs()
    gaps = df["gap"].astype(int)
    runs = gaps.groupby((gaps != gaps.shift()).cumsum()).sum()
    return {
        "bars": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "missing_minutes": int(df["gap"].sum()),
        "missing_pct": float(df["gap"].mean() * 100),
        "longest_gap_minutes": int(runs.max()) if len(runs) else 0,
        "zero_volume_bars": int(((valid["volume"] == 0)).sum()),
        "bad_ohlc_bars": int(bad_ohlc),
        "taker_buy_gt_volume": int((valid["taker_buy_base"] > valid["volume"] * (1 + 1e-9)).sum()),
        "max_abs_1m_logret": float(ret.max()),
        "bars_over_10pct_1m": int((ret > 0.10).sum()),
    }


FNG_PATH = ROOT / "data" / "fear_greed.csv"


def fetch_fng(path: Path = FNG_PATH) -> pd.DataFrame:
    """Download the full Crypto Fear & Greed history (alternative.me) to CSV."""
    r = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=30)
    r.raise_for_status()
    rows = r.json()["data"]
    df = pd.DataFrame({"date": pd.to_datetime([int(x["timestamp"]) for x in rows], unit="s", utc=True),
                       "fng": [int(x["value"]) for x in rows]}).sort_values("date")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_fng(path: Path = FNG_PATH) -> pd.Series:
    """Fear & Greed indexed by the time it is safely AVAILABLE (label date + 1 day).

    The value labelled day D is published around 00:00 UTC on D; we only let
    decisions use it from D+1 00:00 to rule out any publication-time leakage.
    """
    if not path.exists():
        fetch_fng(path)
    df = pd.read_csv(path, parse_dates=["date"])
    s = pd.Series(df["fng"].to_numpy(float), index=pd.DatetimeIndex(df["date"]).tz_convert("UTC") + pd.Timedelta("1D"))
    return s[~s.index.duplicated()].sort_index()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--symbols", nargs="+", default=UNIVERSE)
    d.add_argument("--start", default="2020-01-01")
    d.add_argument("--end")
    d.add_argument("--threads", type=int, default=12)
    c = sub.add_parser("check")
    c.add_argument("--symbols", nargs="+", default=UNIVERSE)
    args = p.parse_args(argv)

    if args.cmd == "download":
        for sym in args.symbols:
            try:
                download(sym, args.start, args.end, args.threads)
            except Exception as exc:  # keep going with the rest of the universe
                print(f"!! {sym}: {exc}", flush=True)
    else:
        rows = []
        for sym in args.symbols:
            path = KLINE_DIR / f"{sym}.parquet"
            if path.exists():
                rows.append({"symbol": sym, **quality_report(load(sym))})
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
