"""Adversarial checks on the stablecoin-premium result (research period only).

flow_regressions found: ln(USDC/USDT) at 00:00 UTC predicts the next day's BTC
return with t = -3.4 (USDT at a premium to USDC -> higher BTC returns). The
premium is persistent and has one huge episode (the March 2023 USDC de-peg),
so a single event could manufacture the t-stat. Checks:

1. exclude the de-peg window (2023-03-09 .. 2023-03-31)
2. winsorise the premium at its 1st/99th percentiles
3. coefficient by calendar year (sign stability)
4. the premium's own level vs its change (is it a slow regime or news?)
5. a tradeable version: long BTC when prem < its trailing 90-day median, flat otherwise
   (spot costs), vs the always-long benchmark on the same days
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import metrics as M
from quant.costs import SPOT
from quant.experiment import HOLDOUT_START, m1_frame, save
from quant.labels import fill_prices

DEPEG = (pd.Timestamp("2023-03-09", tz="UTC"), pd.Timestamp("2023-03-31", tz="UTC"))


def frame() -> pd.DataFrame:
    u, b = m1_frame("USDCUSDT"), m1_frame("BTCUSDT")
    end = pd.Timestamp(HOLDOUT_START, tz="UTC") - pd.Timedelta(days=4)
    days = pd.date_range(max(u.index[0], b.index[0]).ceil("D") + pd.Timedelta(days=3), end, freq="1D", tz="UTC")
    lu = np.log(u["close"]).ffill()
    lb = fill_prices(b).ffill()
    # USDCUSDT was halted for months (2022-09 -> 2023-03): a forward-filled price there is
    # not a premium. Keep only days whose previous 24h actually traded.
    traded = (~u["gap"]).rolling(1440, min_periods=1).mean().reindex(days).to_numpy() > 0.5
    days = days[traded]
    df = pd.DataFrame(index=days)
    df["prem_bps"] = lu.reindex(days).to_numpy() * 1e4
    df["dprem_bps"] = df["prem_bps"] - lu.reindex(days - pd.Timedelta("1D")).to_numpy() * 1e4
    df["y1"] = lb.reindex(days + pd.Timedelta(minutes=1441)).to_numpy() - lb.reindex(days + pd.Timedelta(minutes=1)).to_numpy()
    return df.dropna()


def reg(df: pd.DataFrame, cols=("prem_bps", "dprem_bps")) -> dict:
    beta, t, n = M.hac_ols(df["y1"].to_numpy(), df[list(cols)].to_numpy(), lags=4)
    return {"b_prem_bps": float(beta[1]), "t_prem": float(t[1]), "n": n,
            **({"t_dprem": float(t[2])} if len(cols) > 1 else {})}


def main():
    df = frame()
    out = {"all": reg(df)}
    ex = df[(df.index < DEPEG[0]) | (df.index > DEPEG[1])]
    out["ex_depeg"] = reg(ex)
    lo, hi = df["prem_bps"].quantile([0.01, 0.99])
    out["winsorised"] = reg(df.assign(prem_bps=df["prem_bps"].clip(lo, hi)))
    out["ex_depeg_winsorised"] = reg(ex.assign(prem_bps=ex["prem_bps"].clip(lo, hi)))
    out["by_year"] = {int(y): reg(g, ("prem_bps",)) for y, g in df.groupby(df.index.year) if len(g) > 100}
    # Premium distribution: is it a slow regime?
    out["prem_autocorr_1d"] = float(df["prem_bps"].autocorr(1))
    out["prem_quantiles_bps"] = {str(q): float(df["prem_bps"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    # Tradeable version (decided at 00:00 from data <= t; trailing median uses past days only)
    med = df["prem_bps"].shift(1).rolling(90, min_periods=60).median()
    sig = (df["prem_bps"] < med).astype(float).where(med.notna())
    d = df.assign(pos=sig).dropna(subset=["pos"])
    turnover = d["pos"].diff().abs().fillna(d["pos"].iloc[0])
    net = d["pos"] * np.expm1(d["y1"]) - turnover * SPOT.one_side("BTCUSDT")
    bh = np.expm1(d["y1"])
    diff = (net - bh * d["pos"].mean()).to_numpy()   # vs a constant exposure equal to its average
    v = M.newey_west_var(diff, lags=10)
    out["rule"] = {"days": int(len(d)), "time_in_market": float(d["pos"].mean()),
                   "sharpe": float(M.sharpe(np.log1p(net.to_numpy())) * np.sqrt(365)),
                   "bh_sharpe": float(M.sharpe(d["y1"].to_numpy()) * np.sqrt(365)),
                   "excess_vs_same_exposure_ann": float(diff.mean() * 365),
                   "t_excess": float(diff.mean() / np.sqrt(v / len(diff))) if v > 0 else float("nan"),
                   "by_year_net": {int(y): round(float(g.sum()), 3) for y, g in net.groupby(net.index.year)}}
    print(out, flush=True)
    save("stablecoin_robust", out)


if __name__ == "__main__":
    main()
