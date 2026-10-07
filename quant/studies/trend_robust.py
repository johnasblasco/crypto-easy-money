"""Robustness of the daily trend result: portfolio level, parameters, timing, costs.

A real effect should (1) hold for an equal-weight portfolio built point in
time, (2) sit on a plateau of nearby parameters rather than a spike, (3) not
depend on the arbitrary 00:00 UTC day boundary or on executing within a
minute, (4) survive doubled costs. Each variant is reported, not selected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import metrics as M
from quant.costs import CostModel, COST_MODELS
from quant.data import KLINE_DIR, UNIVERSE, resample
from quant.experiment import HOLDOUT_START, m1_frame, save
from quant.labels import fill_prices
from quant.studies import trend as T


_BARS: dict = {}


def prepare(symbols, variants=((0, 1),)) -> None:
    """Precompute daily bars for every (offset_h, delay_min) variant, loading each coin's 1m data once."""
    import quant.experiment as E

    for sym in symbols:
        todo = [v for v in variants if (sym, *v) not in _BARS]
        for off, delay in todo:
            _BARS[(sym, off, delay)] = _daily_bars_offset(sym, off, delay)
        E._M1_CACHE.pop(sym, None)


def daily_bars_offset(symbol: str, offset_h: int = 0, delay_min: int = 1) -> pd.DataFrame:
    key = (symbol, offset_h, delay_min)
    if key not in _BARS:
        _BARS[key] = _daily_bars_offset(symbol, offset_h, delay_min)
    return _BARS[key]


def _daily_bars_offset(symbol: str, offset_h: int = 0, delay_min: int = 1) -> pd.DataFrame:
    """Daily bars whose boundary is offset_h UTC; fill delay_min after the boundary."""
    m1 = m1_frame(symbol)
    rule = "1D"
    shifted = m1.copy()
    shifted.index = shifted.index - pd.Timedelta(hours=offset_h)
    d = resample(shifted, rule)
    d.index = d.index + pd.Timedelta(hours=offset_h)
    full = pd.date_range(d.index[0], d.index[-1], freq="1D", tz="UTC")
    d = d.reindex(full)
    d.loc[d["gap_frac"] >= 0.999, ["open", "high", "low", "close"]] = np.nan
    lp = fill_prices(m1, "vwap")
    d["fill"] = lp.reindex(d.index + pd.Timedelta(minutes=delay_min)).to_numpy()
    return d


def trend_returns(d: pd.DataFrame, symbol: str, cost, donchian=T.DONCHIAN, pastret=T.PASTRET,
                  target_vol=T.TARGET_VOL, mode="trend"):
    close = d["close"]
    comps = [T.donchian_state(close, L) for L in donchian]
    comps += [(np.log(close / close.shift(L)) > 0).astype(float).where(close.shift(L).notna()) for L in pastret]
    S = pd.concat(comps, axis=1).mean(axis=1, skipna=False)
    size = (target_vol / T.ewma_vol(close)).clip(upper=1.0)
    w = {"trend": S * size, "voltarget": size, "bh": pd.Series(1.0, index=d.index)}[mode]
    valid = S.notna() & size.notna()
    return T.backtest_weights(w.where(valid), d, symbol, cost).where(valid)


def portfolio(symbols, cost_name="spot_taker", end=HOLDOUT_START, min_history_days=365, **kw):
    """Equal weight across symbols that have >= min_history_days of data at each date."""
    cost = COST_MODELS[cost_name] if isinstance(cost_name, str) else cost_name
    rets = {}
    for s in symbols:
        d = daily_bars_offset(s, kw.get("offset_h", 0), kw.get("delay_min", 1))
        d = d[d.index < pd.Timestamp(end, tz="UTC")]
        first = d["close"].first_valid_index()
        out = {}
        for mode in ("trend", "voltarget", "bh"):
            r = trend_returns(d, s, cost, kw.get("donchian", T.DONCHIAN), kw.get("pastret", T.PASTRET),
                              kw.get("target_vol", T.TARGET_VOL), mode)
            r[r.index < first + pd.Timedelta(days=min_history_days)] = np.nan
            out[mode] = r
        rets[s] = out
    port = {}
    for mode in ("trend", "voltarget", "bh"):
        df = pd.DataFrame({s: rets[s][mode] for s in rets})
        port[mode] = np.log(np.expm1(df).mean(axis=1, skipna=True) + 1).dropna()
    return port


def summary(port: dict) -> dict:
    out = {m: T.stats(r) for m, r in port.items()}
    out["alpha_vs_voltarget"] = T.alpha_vs(port["trend"], port["voltarget"])
    out["alpha_vs_bh"] = T.alpha_vs(port["trend"], port["bh"])
    return out


def main(only: list[str] | None = None):
    import sys

    only = only or sys.argv[1:] or None
    syms = [s for s in UNIVERSE if (KLINE_DIR / f"{s}.parquet").exists()]
    prepare(syms, ((0, 1), (8, 1), (16, 1), (0, 60), (0, 240), (0, 720)))
    results = {"symbols": syms}

    def wanted(name):
        return only is None or name in only

    def show(name, res):
        t, v, b, a = res["trend"], res["voltarget"], res["bh"], res["alpha_vs_voltarget"]
        print(f"{name:34s} trend SR {t['sharpe']:.2f} DD {t['max_dd']:.2f} CAGR {t['cagr']:.1%} | "
              f"VT SR {v['sharpe']:.2f} DD {v['max_dd']:.2f} | BH SR {b['sharpe']:.2f} DD {b['max_dd']:.2f} | "
              f"alpha vs VT {a['alpha_ann']:.1%} t={a['alpha_t']:.2f}", flush=True)
        results[name] = res

    base = portfolio(syms)
    if wanted("portfolio_all"):
        show("portfolio_all", summary(base))
    if wanted("portfolio_btc_eth"):
        show("portfolio_btc_eth", summary(portfolio(["BTCUSDT", "ETHUSDT"])))
    by_year = {}
    for y in sorted(set(base["trend"].index.year)):
        m = {k: v[v.index.year == y] for k, v in base.items()}
        by_year[int(y)] = {k: round(T.stats(v)["sharpe"], 2) for k, v in m.items()}
    results["portfolio_by_year_sharpe"] = by_year
    print("by year (trend/voltarget/bh Sharpe):", by_year, flush=True)

    # Parameter plateau
    for name, kw in {
        "lookbacks_x0.5": {"donchian": tuple(max(2, L // 2) for L in T.DONCHIAN), "pastret": tuple(max(2, L // 2) for L in T.PASTRET)},
        "lookbacks_x2": {"donchian": tuple(L * 2 for L in T.DONCHIAN), "pastret": tuple(L * 2 for L in T.PASTRET)},
        "donchian_only": {"pastret": ()},
        "pastret_only": {"donchian": ()},
        "target_vol_0.3": {"target_vol": 0.3},
        "target_vol_0.8": {"target_vol": 0.8},
    }.items():
        if wanted(name):
            show(name, summary(portfolio(syms, **kw)))
    # Timing: day boundary and execution delay
    for off in (8, 16):
        if wanted(f"day_boundary_{off:02d}utc"):
            show(f"day_boundary_{off:02d}utc", summary(portfolio(syms, offset_h=off)))
    for delay in (60, 240, 720):
        if wanted(f"exec_delay_{delay}min"):
            show(f"exec_delay_{delay}min", summary(portfolio(syms, delay_min=delay)))
    # Costs
    double = CostModel("spot_2x", fee_bps=20.0, other_impact_bps=6.0, impact_bps={"BTCUSDT": 2.0, "ETHUSDT": 2.0})
    if wanted("cost_2x"):
        show("cost_2x", summary(portfolio(syms, cost_name=double)))
    save("trend_robustness" if only is None else "trend_robustness_part2", results)


if __name__ == "__main__":
    main()
