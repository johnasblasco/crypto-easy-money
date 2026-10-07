"""Live market data for the engine: local 1m store + incremental sync + freshness checks.

The engine computes exactly the same point-in-time features as the research
harness, from a frame in the same format as ``quant.data.load`` (close-time
index, complete 1-minute grid, ``gap`` flag). Live bars come from the same
Binance spot endpoint as the history, only closed bars are ever used, and every
evaluation reports how stale its data is.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from .data import API, COLUMNS, KLINE_DIR, MINUTE_MS, PRICE_COLS, VOLUME_COLS

MAX_STALENESS = pd.Timedelta(minutes=3)     # newest closed bar older than this -> NO TRADE
MAX_RECENT_GAP_FRAC = 0.02                   # >2% missing minutes in the last day -> NO TRADE


def _to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame([r[:11] for r in rows], columns=COLUMNS[:11]).drop(columns=["close_time"])
    for c in PRICE_COLS + VOLUME_COLS:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype("int64")
    df["open_time"] = df["open_time"].astype("int64")
    return df


def _gridded(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["open_time"] + MINUTE_MS, unit="ms", utc=True)
    df = df.drop(columns=["open_time"]).drop_duplicates("ts").set_index("ts").sort_index()
    full = pd.date_range(df.index[0], df.index[-1], freq="1min", tz="UTC")
    df = df.reindex(full)
    df.index.name = "ts"
    df["gap"] = df["close"].isna()
    df[VOLUME_COLS + ["trades"]] = df[VOLUME_COLS + ["trades"]].fillna(0)
    return df


@dataclass
class LiveStore:
    """Rolling window of 1m bars per symbol, seeded from the local parquet history."""

    days: int = 400
    session: requests.Session = field(default_factory=requests.Session)
    raw: dict = field(default_factory=dict)       # symbol -> raw frame (open_time int64 + values)
    last_sync: dict = field(default_factory=dict)

    def seed(self, symbol: str) -> None:
        path = KLINE_DIR / f"{symbol}.parquet"
        cutoff = int((time.time() - self.days * 86400) * 1000)
        if path.exists():
            df = pd.read_parquet(path, filters=[("open_time", ">=", cutoff)])
        else:
            df = pd.DataFrame(columns=COLUMNS[:11]).drop(columns=["close_time"])
        self.raw[symbol] = df

    def sync(self, symbol: str, max_requests: int = 60) -> int:
        """Fetch closed bars after the last stored one. Returns the number of new bars."""
        if symbol not in self.raw:
            self.seed(symbol)
        df = self.raw[symbol]
        now_ms = int(time.time() * 1000)
        last_closed_open = now_ms // MINUTE_MS * MINUTE_MS - MINUTE_MS   # open time of the newest CLOSED bar
        start = int(df["open_time"].iloc[-1]) + MINUTE_MS if len(df) else now_ms - self.days * 86_400_000
        new = []
        for _ in range(max_requests):
            if start > last_closed_open:
                break
            r = self.session.get(API, params={"symbol": symbol, "interval": "1m", "limit": 1000,
                                              "startTime": start, "endTime": last_closed_open}, timeout=15)
            r.raise_for_status()
            rows = r.json()
            if not rows:
                break
            new.extend(rows)
            start = rows[-1][0] + MINUTE_MS
        if new:
            add = _to_frame(new)
            add = add[add["open_time"] <= last_closed_open]
            df = pd.concat([df, add]).drop_duplicates("open_time").sort_values("open_time")
            cutoff = now_ms - self.days * 86_400_000
            self.raw[symbol] = df[df["open_time"] >= cutoff].reset_index(drop=True)
        self.last_sync[symbol] = pd.Timestamp.now(tz="UTC")
        return len(new)

    def frame(self, symbol: str) -> pd.DataFrame:
        return _gridded(self.raw[symbol])


def data_health(m1: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
    """Freshness and completeness of a live 1m frame; ``ok`` False means NO TRADE."""
    now = now or pd.Timestamp.now(tz="UTC")
    last = m1["close"].last_valid_index()
    staleness = now - last if last is not None else pd.Timedelta.max
    recent = m1[m1.index > now - pd.Timedelta(days=1)]
    gap_frac = float(recent["gap"].mean()) if len(recent) else 1.0
    r = np.log(recent["close"].dropna()).diff().abs()
    jump = float(r.max()) if len(r) else 0.0
    problems = []
    if staleness > MAX_STALENESS:
        problems.append(f"stale data: newest closed bar is {staleness} old")
    if gap_frac > MAX_RECENT_GAP_FRAC:
        problems.append(f"{gap_frac:.1%} of the last day's minutes are missing")
    if jump > 0.15:
        problems.append(f"suspicious 1m move of {jump:.1%} in the last day")
    return {"ok": not problems, "problems": problems, "last_bar": str(last),
            "staleness_s": float(staleness.total_seconds()) if last is not None else None,
            "missing_last_day": gap_frac}


def book_check(symbol: str, session: requests.Session | None = None, notional_usdt: float = 1000.0,
               max_spread_bps: float = 5.0, max_impact_bps: float = 10.0) -> dict:
    """Live order-book liquidity veto (spread and slippage to fill ``notional`` by walking the book).

    Book data predicts direction only over seconds (research sweep), so it is
    used here purely as a cost/liquidity sanity check, never as a signal.
    """
    s = session or requests.Session()
    t0 = time.time()
    r = s.get("https://data-api.binance.vision/api/v3/depth", params={"symbol": symbol, "limit": 100}, timeout=10)
    r.raise_for_status()
    latency = time.time() - t0
    book = r.json()
    bids = np.array(book["bids"], float)
    asks = np.array(book["asks"], float)
    if not len(bids) or not len(asks):
        return {"ok": False, "problems": ["empty order book"]}
    mid = (bids[0, 0] + asks[0, 0]) / 2
    spread_bps = (asks[0, 0] - bids[0, 0]) / mid * 1e4

    def impact(levels, sign):
        need, cost, qty = notional_usdt, 0.0, 0.0
        for px, q in levels:
            take = min(need, px * q)
            cost += take
            qty += take / px
            need -= take
            if need <= 0:
                break
        if need > 0:
            return float("inf")
        vwap = cost / qty
        return sign * (vwap - mid) / mid * 1e4

    buy_impact, sell_impact = impact(asks, 1), impact(bids, -1)
    problems = []
    if spread_bps > max_spread_bps:
        problems.append(f"spread {spread_bps:.1f}bp > {max_spread_bps}bp")
    if max(buy_impact, sell_impact) > max_impact_bps:
        problems.append(f"filling {notional_usdt:.0f} USDT would cost {max(buy_impact, sell_impact):.1f}bp")
    return {"ok": not problems, "problems": problems, "spread_bps": round(spread_bps, 3),
            "buy_impact_bps": round(buy_impact, 3), "sell_impact_bps": round(sell_impact, 3),
            "snapshot_latency_s": round(latency, 3)}
