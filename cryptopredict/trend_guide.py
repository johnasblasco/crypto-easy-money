"""Trend guide: how much of each coin to hold, from the one rule that held up on unseen data.

The rule is the frozen daily trend ensemble from the research (quant/studies/trend.py):
nine trend components (five Donchian breakouts, four past-return signs) on daily closes
at 00:00 UTC, and a size cap that shrinks the position when the coin is very volatile.
Exposure = share of long components x min(1, 50% / annualised volatility).

What it is and isn't (docs/RESEARCH_REPORT.md): on the year it had never seen
(Oct 2025 - Oct 2026) it cut the worst drop of the 16 main coins from -54% to -22% versus
a volatility-sized hold, and on meme coins from -94% to -28% versus buy & hold. It did not
make money in that falling year. It is a risk tool: it tells you when to hold less, not what
will go up.

Reading it: "keep 45%" means of the money you set aside for that coin, keep 45% in the coin
and 55% in USDT. 0% means the trend is down: hold none.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data import UNIVERSE
from quant.studies.meme_coins import MEMES
from quant.studies.trend import TARGET_VOL, ensemble, ewma_vol

# Coins the rule was tested on: the 16 main coins and the meme coins of the meme study.
TESTED = set(UNIVERSE) | set(MEMES)
CHANGE_ALERT = 0.25          # also alert when the suggested share moves this much while staying in


def coin_guide(symbol: str, close: pd.Series) -> dict:
    """Today's suggestion for one coin from its completed daily closes (index = close time)."""
    close = close.dropna()
    row = {"symbol": symbol, "price": float(close.iloc[-1]) if len(close) else None,
           "as_of": close.index[-1].isoformat() if len(close) else None}
    s, v = ensemble(close).iloc[-1], ewma_vol(close).iloc[-1]
    if not np.isfinite(s) or not np.isfinite(v):
        return {**row, "exposure": None, "trend": "UNKNOWN", "reason": "not enough daily history yet (needs ~90 days)"}
    cap = min(1.0, TARGET_VOL / v)
    exposure = round(float(s * cap), 2)
    return {**row, "exposure": exposure, "trend": "UP" if exposure > 0 else "DOWN",
            "components_long": int(round(s * 9)), "volatility": round(float(v), 3),
            "reason": f"{int(round(s * 9))} of 9 trend signals point up; volatility {v:.0%} a year "
                      f"caps the size at {cap:.0%}"}


def guide(symbols: list[str], daily, now: pd.Timestamp) -> list[dict]:
    """``daily(symbol)`` returns completed daily bars (a ``close`` column indexed by close time)."""
    out = []
    for sym in symbols:
        if sym not in TESTED:
            out.append({"symbol": sym, "exposure": None, "trend": "UNTESTED",
                        "reason": "the trend rule was not tested on this coin"})
            continue
        try:
            close = daily(sym)["close"]
            out.append(coin_guide(sym, close[close.index <= now]))
        except Exception as exc:          # one coin failing must not stop the others
            out.append({"symbol": sym, "exposure": None, "trend": "ERROR", "reason": f"no data: {exc}"})
    return out


def coin_name(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def changes(old: list[dict] | None, new: list[dict]) -> list[str]:
    """Plain-language alerts for coins whose suggestion changed since the last guide."""
    if not old:
        return []
    before = {c["symbol"]: c.get("exposure") for c in old}
    msgs = []
    for c in new:
        was, now = before.get(c["symbol"]), c.get("exposure")
        if was is None or now is None:
            continue
        name = coin_name(c["symbol"])
        if was <= 0 < now:
            msgs.append(f"📈 {name} trend turned UP: keep {now:.0%} of your {name} money in {name}, the rest in USDT.")
        elif now <= 0 < was:
            msgs.append(f"📉 {name} trend turned DOWN: move your {name} money to USDT (keep 0%).")
        elif was > 0 and now > 0 and abs(now - was) >= CHANGE_ALERT:
            msgs.append(f"↕ {name}: keep {now:.0%} in {name} now (was {was:.0%}).")
    return msgs
