"""Does the daily trend overlay generalise to meme coins? An out-of-sample test across assets.

The trend rule (quant/studies/trend.py) was designed and pre-registered on 16
large coins. None of the meme coins below were used, so applying the frozen
rule to them, unchanged, is an out-of-sample test on new assets.

Costs: spot taker fee plus the larger of 3 bp and half a price tick per side,
with today's Binance tick size relative to each day's price (sub-cent memes
like PEPE pay 10-15 bp per side for the tick alone).

Survivorship: only meme coins still trading on Binance in October 2026 have
data. Delisted and dead memes are missing, so buy & hold here is far better
than what a meme buyer really got. The overlay comparison is less affected
(both arms hold the same coins), but still only describes survivors.

Reports, per coin and for an equal-weight meme portfolio (a coin enters after
365 days of history as pre-registered, and after the rule's own warm-up as a
sensitivity): trend vs volatility-targeted hold vs buy & hold, by period, and
the pre-registered overlay criterion (max drawdown <= 0.8 x the vol-targeted
hold's, Sharpe >= its Sharpe - 0.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from quant.data import KLINE_DIR
from quant.experiment import HOLDOUT_START, save
from quant.studies import trend as T
from quant.studies import trend_robust as TR

MEMES = ["DOGEUSDT", "SHIBUSDT", "PEOPLEUSDT", "PEPEUSDT", "FLOKIUSDT", "MEMEUSDT", "BONKUSDT", "1000SATSUSDT",
         "WIFUSDT", "BOMEUSDT", "TURBOUSDT", "DOGSUSDT", "NEIROUSDT", "PNUTUSDT", "ACTUSDT", "PENGUUSDT",
         "TRUMPUSDT", "1MBABYDOGEUSDT", "TSTUSDT", "MUBARAKUSDT", "BROCCOLI714USDT", "BANANAS31USDT"]
FEE_BPS, FLOOR_BPS = 10.0, 3.0
FAR_FUTURE = "2100-01-01"


def tick_sizes(symbols) -> dict:
    info = requests.get("https://data-api.binance.vision/api/v3/exchangeInfo", timeout=60).json()
    out = {}
    for s in info["symbols"]:
        if s["symbol"] in symbols:
            out[s["symbol"]] = float(next(f["tickSize"] for f in s["filters"] if f["filterType"] == "PRICE_FILTER"))
    return out


class TickCost:
    """Per-side cost = fee + max(floor, half a tick relative to that day's price)."""

    name = "spot_taker+half_tick"

    def __init__(self, ticks: dict):
        self.ticks = ticks
        self._cache: dict = {}

    def one_side(self, symbol: str):
        if symbol not in self._cache:
            close = TR.daily_bars_offset(symbol)["close"]
            half_tick = (0.5 * self.ticks.get(symbol, 0.0) / close).fillna(0.0)
            self._cache[symbol] = FEE_BPS / 1e4 + np.maximum(FLOOR_BPS / 1e4, half_tick)
        return self._cache[symbol]

    def round_trip(self, symbol: str):
        return 2 * self.one_side(symbol)


def window_stats(port: dict, start=None, end=None) -> dict:
    sl = {m: r[(r.index >= pd.Timestamp(start, tz="UTC")) if start else slice(None)] for m, r in port.items()}
    sl = {m: r[r.index < pd.Timestamp(end, tz="UTC")] if end else r for m, r in sl.items()}
    out = {m: T.stats(r) for m, r in sl.items()}
    if min(len(r.dropna()) for r in sl.values()) > 60:
        out["alpha_vs_voltarget"] = T.alpha_vs(sl["trend"], sl["voltarget"])
    tr, vt = out["trend"], out["voltarget"]
    out["overlay_criterion"] = {"drawdown_ok": bool(tr["max_dd"] >= 0.8 * vt["max_dd"]),
                                "sharpe_ok": bool(tr["sharpe"] >= vt["sharpe"] - 0.2)}
    out["overlay_criterion"]["passed"] = all(out["overlay_criterion"].values())
    return out


def main():
    syms = [s for s in MEMES if (KLINE_DIR / f"{s}.parquet").exists()]
    ticks = tick_sizes(syms)
    cost = TickCost(ticks)
    TR.prepare(syms)
    per_coin = []
    for s in syms:
        d = TR.daily_bars_offset(s)
        days = int(d["close"].notna().sum())
        row = {"symbol": s, "first_day": str(d["close"].first_valid_index().date()), "days": days,
               "tick_bps_now": round(float(ticks.get(s, 0) / d["close"].dropna().iloc[-1] * 1e4), 2)}
        if days < 250:              # warm-up (90 days) plus enough history to say anything
            row["skipped"] = "under 250 days of history"
            per_coin.append(row)
            continue
        res, _ = T.run_symbol(s, cost_name=cost, end=FAR_FUTURE, n_null=300)
        row.update({k: res[k] for k in ("trend", "buy_hold", "voltarget_hold", "alpha_vs_voltarget",
                                         "p_vs_random_timing", "time_in_market")})
        per_coin.append(row)
        t, b, v = res["trend"], res["buy_hold"], res["voltarget_hold"]
        print(f"{s:16s} {row['first_day']} trend SR {t['sharpe']:.2f} DD {t['max_dd']:.0%} | B&H SR {b['sharpe']:.2f} "
              f"DD {b['max_dd']:.0%} CAGR {b['cagr']:.0%} | VT SR {v['sharpe']:.2f} DD {v['max_dd']:.0%} | "
              f"random-timing p={res['p_vs_random_timing']:.3f}", flush=True)
    out = {"symbols": syms, "tick_sizes": ticks, "per_coin": per_coin, "portfolio": {}}
    for label, min_hist in (("min_history_365d", 365), ("after_warmup_only", 0)):
        port = TR.portfolio(syms, cost_name=cost, end=FAR_FUTURE, min_history_days=min_hist)
        out["portfolio"][label] = {
            "full": window_stats(port),
            "before_holdout": window_stats(port, end=HOLDOUT_START),
            "holdout_period": window_stats(port, start=HOLDOUT_START),
            # Coins in the portfolio by the end of each year (history >= max(min_hist, 90-day warm-up)).
            "coins_by_year_end": {
                y: sum(TR.daily_bars_offset(s)["close"].first_valid_index() + pd.Timedelta(days=max(min_hist, 90))
                       <= pd.Timestamp(f"{y}-12-31", tz="UTC") for s in syms)
                for y in range(2020, 2027)},
        }
        f = out["portfolio"][label]["full"]
        print(f"portfolio {label}: trend SR {f['trend']['sharpe']:.2f} DD {f['trend']['max_dd']:.0%} | "
              f"VT SR {f['voltarget']['sharpe']:.2f} DD {f['voltarget']['max_dd']:.0%} | "
              f"B&H SR {f['bh']['sharpe']:.2f} DD {f['bh']['max_dd']:.0%} | criterion {f['overlay_criterion']}", flush=True)
    save("meme_coins", out)
    return out


if __name__ == "__main__":
    main()
