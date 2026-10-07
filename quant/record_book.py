"""Forward-record live order-book state so book hypotheses can be tested later.

Historical order books are not available (and book signals decay within
seconds), so the only honest way to test "does book imbalance predict
anything at the horizons we can trade?" is to record it now and evaluate
later with the same walk-forward harness. Each snapshot stores, per symbol:

  ts, snapshot latency, mid, spread (bp), depth within 5/10/25/50 bp on each
  side (quote volume), banded imbalance (bid - ask)/(bid + ask), and the cost
  in bp to fill 1k / 10k USDT by walking the book.

Usage (best on your own machine; websockets would be faster but REST is enough
for minute-scale research):
    python -m quant.record_book --symbols BTCUSDT ETHUSDT SOLUSDT --every 10
Files: data/book/<SYMBOL>/<YYYY-MM-DD>.parquet (one per UTC day).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import requests

from .data import ROOT

BOOK_DIR = ROOT / "data" / "book"
DEPTH_URL = "https://data-api.binance.vision/api/v3/depth"
BANDS_BP = (5, 10, 25, 50)


def snapshot(symbol: str, session: requests.Session) -> dict:
    t0 = time.time()
    r = session.get(DEPTH_URL, params={"symbol": symbol, "limit": 1000}, timeout=10)
    r.raise_for_status()
    latency = time.time() - t0
    book = r.json()
    bids = np.array(book["bids"], float)
    asks = np.array(book["asks"], float)
    mid = (bids[0, 0] + asks[0, 0]) / 2
    row = {"ts": pd.Timestamp.now(tz="UTC"), "latency_s": latency, "mid": mid,
           "spread_bp": (asks[0, 0] - bids[0, 0]) / mid * 1e4, "last_update_id": book.get("lastUpdateId")}
    for bp in BANDS_BP:
        b = bids[bids[:, 0] >= mid * (1 - bp / 1e4)]
        a = asks[asks[:, 0] <= mid * (1 + bp / 1e4)]
        bq, aq = float((b[:, 0] * b[:, 1]).sum()), float((a[:, 0] * a[:, 1]).sum())
        row[f"bid_depth_{bp}bp"], row[f"ask_depth_{bp}bp"] = bq, aq
        row[f"imbalance_{bp}bp"] = (bq - aq) / (bq + aq) if bq + aq > 0 else np.nan
    for notional in (1_000, 10_000):
        for side, levels, sign in (("buy", asks, 1), ("sell", bids, -1)):
            need, cost, qty = notional, 0.0, 0.0
            for px, q in levels:
                take = min(need, px * q)
                cost, qty, need = cost + take, qty + take / px, need - take
                if need <= 0:
                    break
            row[f"{side}_cost_{notional}_bp"] = sign * (cost / qty - mid) / mid * 1e4 if need <= 0 else np.nan
    return row


def run(symbols, every: float, minutes: float | None = None):
    session = requests.Session()
    buf: dict = {s: [] for s in symbols}
    t_end = time.time() + minutes * 60 if minutes else None
    last_flush = time.time()
    while t_end is None or time.time() < t_end:
        t0 = time.time()
        for s in symbols:
            try:
                buf[s].append(snapshot(s, session))
            except requests.RequestException as exc:
                print(f"{s}: {exc}", flush=True)
        if time.time() - last_flush > 300 or (t_end and time.time() >= t_end):
            flush(buf)
            last_flush = time.time()
        time.sleep(max(0.0, every - (time.time() - t0)))
    flush(buf)


def flush(buf: dict) -> None:
    for s, rows in buf.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for day, g in df.groupby(df["ts"].dt.strftime("%Y-%m-%d")):
            path = BOOK_DIR / s / f"{day}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                g = pd.concat([pd.read_parquet(path), g])
            g.to_parquet(path, index=False)
        rows.clear()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p.add_argument("--every", type=float, default=10.0, help="seconds between snapshot rounds")
    p.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")
    args = p.parse_args(argv)
    run(args.symbols, args.every, args.minutes)


if __name__ == "__main__":
    main()
