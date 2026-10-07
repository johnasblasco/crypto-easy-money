"""Two pre-registered regression tests of new information sources (research period).

Q1 Quarter-hour opening imbalance (Kim & Hansen 2026, perps -> spot replication):
   OIq(H) = sum over boundary minutes (minute-of-hour in 0,15,30,45) within (t-H, t] of
            (2*taker_buy_quote - quote_volume) / sum of their quote_volume
   TIall(H) = the same over ALL minutes (control), r(H) = past return (control).
   Target: forward log return over h in {4h, 8h, 12h} from the next-minute VWAP.
   Regression y = a + b1*OIq + b2*TIall + b3*r/sigma, hourly samples; HAC (Newey-West, lags = h/1h)
   standard errors. Prediction: b1 > 0 (continuation), significant beyond TIall.

Q2 Stablecoin premium (capital-inflow proxy): USDC priced in USDT.
   prem = ln(USDCUSDT close), its 24h change, and 24h taker imbalance on USDCUSDT.
   Prediction (weak prior): USDT at a premium to USDC (prem < 0) or USDT being bought -> fresh
   buying capital -> higher BTC returns over the next 1-3 days. Daily samples at 00:00 UTC, HAC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import metrics as M
from quant.data import KLINE_DIR
from quant.experiment import HOLDOUT_START, m1_frame, save
from quant.labels import fill_prices


def hac_ols(y, X, lags):
    """OLS with Newey-West standard errors. Returns (beta, t-stats)."""
    X = np.column_stack([np.ones(len(X)), X])
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * u[L:, None]).T @ (X[:-L] * u[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return beta, beta / np.sqrt(np.diag(V)), int(n)


def quarter_hour(symbol: str):
    m1 = m1_frame(symbol)
    m1 = m1[m1.index < pd.Timestamp(HOLDOUT_START, tz="UTC") - pd.Timedelta(days=1)]
    q = m1["quote_volume"].to_numpy(float)
    s = 2 * m1["taker_buy_quote"].to_numpy(float) - q
    # First minute of each quarter-hour = the bar OPENING at :00/:15/:30/:45, which our
    # close-time index labels :01/:16/:31/:46.
    boundary = (m1.index.minute % 15 == 1)
    cq_b = np.concatenate([[0], np.cumsum(np.where(boundary, q, 0))])
    cs_b = np.concatenate([[0], np.cumsum(np.where(boundary, s, 0))])
    cq, cs = np.concatenate([[0], np.cumsum(q)]), np.concatenate([[0], np.cumsum(s)])
    lc = np.log(m1["close"]).ffill().to_numpy()
    lf = fill_prices(m1).ffill().to_numpy()
    sig = np.sqrt(pd.Series(np.diff(lc, prepend=lc[0]) ** 2).ewm(halflife=1440, min_periods=1440).mean().to_numpy())
    times = np.arange(1440 * 30, len(m1) - 13 * 60, 60)
    times = times[m1.index[times].minute == 0]
    res = {}
    for H in (240, 720):
        e = times + 1
        b = e - H
        oiq = (cs_b[e] - cs_b[b]) / (cq_b[e] - cq_b[b] + 1e-9)
        tiall = (cs[e] - cs[b]) / (cq[e] - cq[b] + 1e-9)
        rz = (lc[times] - lc[times - H]) / (sig[times] * np.sqrt(H) + 1e-12)
        for h in (240, 480, 720):
            y = (lf[times + 1 + h] - lf[times + 1]) / (sig[times] * np.sqrt(h) + 1e-12)   # vol-normalised
            beta, t, n = hac_ols(y, np.column_stack([oiq, tiall, rz]), lags=h // 60 + 1)
            res[f"H{H // 60}h_h{h // 60}h"] = {"b_OIq": round(beta[1], 4), "t_OIq": round(t[1], 2),
                                              "b_TIall": round(beta[2], 4), "t_TIall": round(t[2], 2),
                                              "b_r": round(beta[3], 4), "t_r": round(t[3], 2), "n": n}
    return res


def stablecoin_premium():
    if not (KLINE_DIR / "USDCUSDT.parquet").exists():
        return {"skipped": "USDCUSDT not downloaded"}
    u = m1_frame("USDCUSDT")
    b = m1_frame("BTCUSDT")
    end = pd.Timestamp(HOLDOUT_START, tz="UTC") - pd.Timedelta(days=4)
    days = pd.date_range(max(u.index[0], b.index[0]).ceil("D") + pd.Timedelta(days=3), end, freq="1D", tz="UTC")
    lu = np.log(u["close"]).ffill()
    prem = lu.reindex(days).to_numpy()
    dprem = prem - lu.reindex(days - pd.Timedelta("1D")).to_numpy()
    uq = u["quote_volume"].rolling(1440, min_periods=1200).sum()
    us = (2 * u["taker_buy_quote"] - u["quote_volume"]).rolling(1440, min_periods=1200).sum()
    uimb = (us / uq).reindex(days).to_numpy()
    lb = fill_prices(b).ffill()
    out = {}
    for hd in (1, 3):
        y = lb.reindex(days + pd.Timedelta(minutes=1 + 1440 * hd)).to_numpy() - lb.reindex(days + pd.Timedelta(minutes=1)).to_numpy()
        beta, t, n = hac_ols(y, np.column_stack([prem * 1e4, dprem * 1e4, uimb]), lags=hd + 2)
        out[f"btc_{hd}d"] = {"b_prem_bps": float(beta[1]), "t_prem": round(t[1], 2), "b_dprem_bps": float(beta[2]),
                            "t_dprem": round(t[2], 2), "b_usdc_imb": float(beta[3]), "t_usdc_imb": round(t[3], 2), "n": n}
    return out


def main():
    out = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        out[f"quarter_hour_{sym}"] = quarter_hour(sym)
        print(sym, out[f"quarter_hour_{sym}"], flush=True)
    out["stablecoin_premium"] = stablecoin_premium()
    print("stablecoin premium", out["stablecoin_premium"], flush=True)
    save("flow_regressions", out)


if __name__ == "__main__":
    main()
