"""Execution realism: aggressor-side fills and state-dependent spreads (research period).

Every other study fills at the 1m bar's VWAP. A market buy actually pays roughly
the price aggressive buyers paid in that minute, and a market sell receives the
price aggressive sellers received. Binance klines carry the taker-buy base and
quote volume, so both are exact per minute:

    buy fill  = taker_buy_quote / taker_buy_base
    sell fill = (quote_volume - taker_buy_quote) / (volume - taker_buy_base)

Their gap is an effective-spread proxy (it also contains intra-minute drift, so
it is conservative). This matters most for reversal signals: they buy when
sellers dominate, i.e. when the VWAP sits near the bid, so a VWAP fill flatters
them by about half the effective spread at exactly the worst moments.

Q1 Is the spread constant? Effective spread at random times vs at the entry
   minute of flow-driven events, per coin.
Q2 Does the best event cell (flow-driven reversal) survive side-aware fills?
   Long: enter at the buy fill, exit at the sell fill; short: the reverse. On top
   of that only exchange fees + 1bp own impact per side (the spread is already in
   the fills). All four pre-registered parameter cells are reported, not just
   the best one.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quant import metrics as M
from quant.costs import COST_MODELS
from quant.data import KLINE_DIR, UNIVERSE
from quant.events import Bars, day_bootstrap, select
from quant.experiment import save
from quant.studies.event_hypotheses import h_flow_driven_reversal, research_frame

HORIZONS = (15, 60, 240)
OWN_IMPACT = 1e-4


def side_fills(m1: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Log price paid by aggressive buyers and received by aggressive sellers, per minute.

    A minute without takers on a side falls back to the bar's high (buy) or low
    (sell) -- the conservative bound for that side.
    """
    v, q = m1["volume"].to_numpy(float), m1["quote_volume"].to_numpy(float)
    bb, bq = m1["taker_buy_base"].to_numpy(float), m1["taker_buy_quote"].to_numpy(float)
    hi, lo = m1["high"].ffill().to_numpy(float), m1["low"].ffill().to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        buy = np.where(bb > 0, bq / bb, np.nan)
        sell = np.where(v - bb > 0, (q - bq) / (v - bb), np.nan)
    # Guard against rounding junk outside the bar's range.
    buy = np.where((buy >= lo * 0.999) & (buy <= hi * 1.001), buy, hi)
    sell = np.where((sell >= lo * 0.999) & (sell <= hi * 1.001), sell, lo)
    return np.log(buy), np.log(sell)


def side_outcomes(b: Bars, buy: np.ndarray, sell: np.ndarray, idx: np.ndarray, d: np.ndarray, h: int) -> pd.DataFrame:
    ent, ex = idx + 1, idx + 1 + h
    ok = ex < b.n
    idx, d, ent, ex = idx[ok], d[ok], ent[ok], ex[ok]
    gap_in = np.array([b.gap[i + 1:j + 1].any() for i, j in zip(idx, ex)], dtype=bool)
    long_r = sell[ex] - buy[ent]
    short_r = sell[ent] - buy[ex]
    side = np.where(d > 0, long_r, short_r)
    vwap = (b.fill[ex] - b.fill[ent]) * d
    out = pd.DataFrame({"ts": b.index[idx], "dir": d, "ret_side": side, "ret_vwap": vwap,
                        "spread_entry": buy[ent] - sell[ent]})
    return out[~gap_in & np.isfinite(out["ret_side"]) & np.isfinite(out["ret_vwap"])].reset_index(drop=True)


def summarize(ev: pd.DataFrame, fee_side: float, funding: float, allow_short: bool) -> dict:
    if not allow_short:
        ev = ev[ev["dir"] > 0]
    if len(ev) < 30:
        return {"n": int(len(ev))}
    fund = np.where(ev["dir"] > 0, funding, 0.0)
    net_side = ev["ret_side"] - 2 * (fee_side + OWN_IMPACT) - fund
    m, p, lo = day_bootstrap(ev.assign(net=net_side), "net")
    # Day-level series (events on the same day are one correlated bet) for the deflated Sharpe.
    day = net_side.groupby(ev["ts"].dt.floor("D")).mean()
    from scipy import stats as _st

    day_stats = {"sr_day": float(day.mean() / day.std(ddof=1)) if len(day) > 2 else float("nan"),
                 "n_day": int(len(day)), "skew_day": float(_st.skew(day)),
                 "kurt_day": float(_st.kurtosis(day, fisher=False))}
    by_year = {int(y): {"n": int(len(g)), "net_bps": round(float(net_side[g.index].mean() * 1e4), 1)}
               for y, g in ev.groupby(ev["ts"].dt.year)}
    return {"n": int(len(ev)), "days": int(ev["ts"].dt.floor("D").nunique()),
            "gross_vwap_bps": float(ev["ret_vwap"].mean() * 1e4), "gross_side_bps": float(ev["ret_side"].mean() * 1e4),
            "fill_drag_bps": float((ev["ret_vwap"] - ev["ret_side"]).mean() * 1e4),
            "net_side_bps": float(m * 1e4), "p_net": p, "net_p5_bps": float(lo * 1e4),
            "median_entry_spread_bps": float(ev["spread_entry"].median() * 1e4), **day_stats, "by_year": by_year,
            "positive_years": f"{sum(1 for v in by_year.values() if v['n'] >= 20 and v['net_bps'] > 0)}/"
                              f"{sum(1 for v in by_year.values() if v['n'] >= 20)}"}


def main(symbols=None):
    symbols = symbols or [s for s in UNIVERSE if (KLINE_DIR / f"{s}.parquet").exists()]
    t0 = time.time()
    spreads, events = {}, {}
    rng = np.random.default_rng(0)
    for sym in symbols:
        m1 = research_frame(sym)
        b = Bars(m1)
        buy, sell = side_fills(m1)
        spr = buy - sell
        ok = np.flatnonzero(np.isfinite(spr) & ~b.gap & (m1["volume"].to_numpy() > 0))
        samp = rng.choice(ok, size=min(200_000, len(ok)), replace=False)
        spreads[sym] = {"median_bps_random": float(np.median(spr[samp]) * 1e4),
                        "p90_bps_random": float(np.percentile(spr[samp], 90) * 1e4)}
        group = "majors" if sym in ("BTCUSDT", "ETHUSDT") else "alts"
        for W in (15, 60):
            for k in (3.0, 4.0):
                sig = h_flow_driven_reversal(b, W, k)
                for h in HORIZONS:
                    idx = select(sig, cooldown=h)
                    if not len(idx):
                        continue
                    ev = side_outcomes(b, buy, sell, idx, np.sign(sig[idx]).astype(int), h)
                    events.setdefault((W, k, h, group), []).append(ev.assign(sym=sym))
                    if (W, k, h) == (60, 4.0, 60):
                        spreads[sym]["median_bps_at_events"] = float(ev["spread_entry"].median() * 1e4) if len(ev) else None
        import quant.experiment as E
        E._M1_CACHE.pop(sym, None)
        print(f"{sym} spread {spreads[sym]} ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for (W, k, h, group), parts in events.items():
        ev = pd.concat(parts, ignore_index=True)
        for cname in ("spot_taker", "perp_taker"):
            cost = COST_MODELS[cname]
            s = summarize(ev, cost.fee_bps / 1e4, cost.holding(h), cost.allow_short)
            rows.append({"W": W, "k": k, "h": h, "group": group, "cost": cname, **s})
    df = pd.DataFrame(rows)
    ok = df["p_net"].notna() if "p_net" in df else pd.Series(False, index=df.index)
    df.loc[ok, "p_holm"] = M.holm(df.loc[ok, "p_net"].to_numpy())
    save("execution_realism", {"spreads": spreads, "flow_reversal_side_fills": df.to_dict(orient="records")})
    cols = ["W", "k", "h", "group", "cost", "n", "gross_vwap_bps", "gross_side_bps", "fill_drag_bps", "net_side_bps",
            "p_net", "p_holm", "positive_years"]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(df[[c for c in cols if c in df]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
