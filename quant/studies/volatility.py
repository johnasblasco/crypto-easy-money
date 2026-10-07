"""Volatility is forecastable; how well, and does it predict TRADEABILITY?

(1) Forecast next-h realised variance (h = 1h, 4h, 24h) with:
      EWMA (halflife 1d of 1m squared returns)            -- current engine default
      HAR  (log RV over 1h, 1d, 1w windows, + hour-of-week seasonal factor + trade-count term),
           fitted by OLS on a trailing 365-day window, refit every 30 days (walk-forward)
    Loss: QLIKE and MSE on log RV; Diebold-Mariano test of HAR vs EWMA.
(2) Tradeability: AUC of predicting |r_h| > round-trip cost from sigma_hat / cost alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import metrics as M
from quant.costs import COST_MODELS
from quant.experiment import HOLDOUT_START, m1_frame, save
from quant.labels import forward_returns


def rv_series(m1: pd.DataFrame) -> pd.Series:
    r = np.log(m1["close"]).ffill().diff().fillna(0.0)
    return r ** 2


def study(symbol: str, h: int):
    m1 = m1_frame(symbol)
    m1 = m1[m1.index < pd.Timestamp(HOLDOUT_START, tz="UTC")]
    r2 = rv_series(m1)
    cs = np.concatenate([[0.0], np.cumsum(r2.to_numpy())])
    times = pd.date_range(m1.index[0].ceil(f"{h}min") + pd.Timedelta(days=30), m1.index[-1] - pd.Timedelta(minutes=h + 2),
                          freq=f"{h}min", tz="UTC")
    pos = m1.index.get_indexer(times)

    def past(L):
        out = (cs[pos + 1] - cs[np.maximum(pos + 1 - L, 0)]) / L * h
        out[pos + 1 - L < 0] = np.nan
        return out

    fut = cs[pos + 1 + h] - cs[pos + 1]                       # realised variance over the next h minutes
    trades = m1["trades"].to_numpy(float)
    ct = np.concatenate([[0.0], np.cumsum(trades)])
    act = np.log((ct[pos + 1] - ct[np.maximum(pos + 1 - 60, 0)] + 1) / ((ct[pos + 1] - ct[np.maximum(pos + 1 - 10080, 0)]) / 168 + 1))
    ewma = (r2.ewm(halflife=1440, min_periods=1440).mean().reindex(times).to_numpy()) * h
    df = pd.DataFrame({"fut": fut, "ewma": ewma, "rv1h": past(60), "rv1d": past(1440), "rv1w": past(10080),
                       "act": act}, index=times)
    # hour-of-week seasonal factor from the trailing 8 weeks (past only)
    how = times.dayofweek * 24 + times.hour
    df["how"] = how
    df["lfut"] = np.log(df["fut"] + 1e-12)
    for c in ("rv1h", "rv1d", "rv1w", "ewma"):
        df[f"l{c}"] = np.log(df[c] + 1e-12)
    lf = df["lfut"].to_numpy()
    # vectorised: mean of past realised log RV for the same hour-of-week within 8 weeks, minus overall past mean
    s = pd.Series(lf, index=df.index)
    by_how = s.groupby(df["how"].to_numpy())
    seas = by_how.transform(lambda x: x.shift(1).rolling(8, min_periods=4).mean()).to_numpy()
    overall = s.shift(1).rolling(int(8 * 7 * 1440 / h), min_periods=50).mean().to_numpy()
    df["seas"] = seas - overall
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # Walk-forward HAR: refit every 30 days on the trailing 365 days whose targets are already known.
    feats = ["lrv1h", "lrv1d", "lrv1w", "seas", "act"]
    pred = pd.Series(np.nan, index=df.index)
    starts = pd.date_range(df.index[0] + pd.Timedelta(days=365), df.index[-1], freq="30D", tz="UTC")
    for t0 in starts:
        t1 = t0 + pd.Timedelta(days=30)
        tr = df[(df.index >= t0 - pd.Timedelta(days=365)) & (df.index < t0 - pd.Timedelta(minutes=h))]
        te = df[(df.index >= t0) & (df.index < t1)]
        if len(tr) < 200 or not len(te):
            continue
        X = np.column_stack([np.ones(len(tr)), tr[feats].to_numpy()])
        beta = np.linalg.lstsq(X, tr["lfut"].to_numpy(), rcond=None)[0]
        resid_var = np.var(tr["lfut"].to_numpy() - X @ beta)
        Xt = np.column_stack([np.ones(len(te)), te[feats].to_numpy()])
        pred[te.index] = np.exp(Xt @ beta + resid_var / 2)       # log-normal mean correction
    ev = df.assign(har=pred).dropna(subset=["har"])
    qlike = lambda f, y: y / f - np.log(y / f) - 1  # noqa: E731
    y = ev["fut"].to_numpy() + 1e-12
    q_e, q_h = qlike(ev["ewma"].to_numpy(), y), qlike(ev["har"].to_numpy(), y)
    d, p = M.diebold_mariano(q_e, q_h)
    # Tradeability: does sigma_hat / cost rank |r| > cost?
    lab = forward_returns(m1, ev.index, h, 1)
    absr = lab["fwd_ret"].abs().to_numpy()
    rt = COST_MODELS["perp_taker"].round_trip(symbol)
    ok = np.isfinite(absr)
    trade_auc = M.auc((absr[ok] > rt).astype(int), np.sqrt(ev["har"].to_numpy()[ok]))
    corr = np.corrcoef(np.log(ev["har"]), np.log(y))[0, 1]
    return {"symbol": symbol, "h": h, "n": int(len(ev)), "qlike_ewma": float(q_e.mean()), "qlike_har": float(q_h.mean()),
            "qlike_gain": d, "p_har_better": p, "corr_log_har_vs_realised": float(corr),
            "share_moves_above_cost": float((absr[ok] > rt).mean()), "tradeability_auc": trade_auc}


def main():
    out = []
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        for h in (60, 240, 1440):
            r = study(sym, h)
            out.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    save("volatility_forecast", out)


if __name__ == "__main__":
    main()
