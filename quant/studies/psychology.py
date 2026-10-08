"""Eight pre-registered trader-psychology hypotheses (docs/PREREGISTRATION_PSYCHOLOGY.md).

Research period only (2020-01-01 to 2025-09-28). One primary cell per
hypothesis; Holm across the eight decides, together with each hypothesis's
extra pass condition. A few cheap secondary cells are reported descriptively;
they cannot rescue a failed primary.

R1 52-week-high anchoring (weekly cross-section)     R5 Deribit monthly expiry rebound
R2 intraday momentum into the NY close               R6 round-number first touch (fade)
R3 round-number break continuation                   R7 turn-of-month
R4 lottery demand / MAX effect (weekly cross-section) R8 kernel chart patterns (H&S, double top/bottom)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from quant import metrics as M
from quant.costs import PERP, SPOT
from quant.data import KLINE_DIR, UNIVERSE, resample
from quant.events import Bars, day_bootstrap, outcomes, placebo, select
from quant.experiment import save
from quant.studies import trend as T
from quant.studies.event_hypotheses import research_frame

LARGE = [s for s in UNIVERSE if (KLINE_DIR / f"{s}.parquet").exists()]
DAILY_END = pd.Timestamp("2025-09-29", tz="UTC")     # last daily fill used (before the holdout)


# ------------------------------------------------------------------ shared statistics

def nw_p(x, lags: int, two_sided: bool = False) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    v = M.newey_west_var(x, lags=lags)
    t = x.mean() / np.sqrt(v / len(x)) if v > 0 else float("nan")
    p = 2 * (1 - stats.norm.cdf(abs(t))) if two_sided else 1 - stats.norm.cdf(t)
    return float(t), float(p)


def week_block_bootstrap_p(s: pd.Series, two_sided: bool = False, n_boot: int = 5000, seed: int = 0) -> float:
    s = s.dropna()
    weeks = s.groupby(s.index.to_period("W-SUN").start_time)
    sums, counts = weeks.sum().to_numpy(), weeks.count().to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sums), (n_boot, len(sums)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = float((means <= 0).mean()), float((means >= 0).mean())
    return min(1.0, 2 * min(lo, hi)) if two_sided else lo


def event_net(ev: pd.DataFrame, cost, h: int) -> pd.Series:
    rt = np.array([cost.round_trip(s) for s in ev["sym"]])
    fund = np.where(ev["dir"] > 0, cost.holding(h, 1), 0.0)
    return ev["ret"] - rt - fund


def delta_bootstrap(a: pd.DataFrame, b: pd.DataFrame, value: str = "ret", n_boot: int = 4000, seed: int = 0):
    """mean(a) - mean(b) with a day-clustered bootstrap that resamples calendar days; one-sided p for > 0."""
    da, db = a.assign(day=a["ts"].dt.floor("D")), b.assign(day=b["ts"].dt.floor("D"))
    days = pd.Index(sorted(set(da["day"]) | set(db["day"])))
    ga = da.groupby("day")[value].agg(["sum", "count"]).reindex(days, fill_value=0)
    gb = db.groupby("day")[value].agg(["sum", "count"]).reindex(days, fill_value=0)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(days), (n_boot, len(days)))
    ma = ga["sum"].to_numpy()[pick].sum(1) / np.maximum(ga["count"].to_numpy()[pick].sum(1), 1)
    mb = gb["sum"].to_numpy()[pick].sum(1) / np.maximum(gb["count"].to_numpy()[pick].sum(1), 1)
    delta = float(a[value].mean() - b[value].mean())
    return delta, float(((ma - mb) <= 0).mean())


def by_year(ev: pd.DataFrame, col: str = "net") -> dict:
    return {int(y): {"n": int(len(g)), "mean_bps": round(float(g[col].mean() * 1e4), 1)}
            for y, g in ev.groupby(ev["ts"].dt.year)}


# ------------------------------------------------------------------ weekly cross-sections (R1, R4)

def daily_frames(symbols):
    out = {}
    for s in symbols:
        d = T.daily_bars(s)
        out[s] = d[d.index <= DAILY_END]
    keys = ("close", "high", "fill", "quote_volume")
    return {k: pd.DataFrame({s: out[s][k] for s in symbols}) for k in keys}


def weekly_book(members: dict, fill_log: pd.DataFrame, cost) -> tuple[pd.Series, pd.Series]:
    """Equal-weight buy-and-hold from each Monday's fill to the next; returns (daily gross, daily cost)."""
    P = np.exp(fill_log).ffill(limit=3)
    mondays = sorted(members)
    gross, costs = {}, {}
    prev_drift: dict = {}
    for k, D in enumerate(mondays):
        names = [s for s in members[D] if np.isfinite(P.at[D, s])]
        if not names:
            continue
        end = mondays[k + 1] if k + 1 < len(mondays) else D + pd.Timedelta(days=7)
        w_new = {s: 1 / len(names) for s in names}
        turn = sum(abs(w_new.get(s, 0) - prev_drift.get(s, 0)) * cost.one_side(s)
                   for s in set(w_new) | set(prev_drift))
        days = P.index[(P.index > D) & (P.index <= end)]
        rel = P.loc[days, names].div(P.loc[D, names]).ffill()
        V = rel.mean(axis=1)
        r = V / V.shift(1).fillna(1.0) - 1
        for i, day in enumerate(days):
            gross[day] = float(r.iloc[i])
            costs[day] = float(turn) if i == 0 else 0.0
        last = rel.iloc[-1] if len(rel) else pd.Series(1.0, index=names)
        prev_drift = (last / len(names) / last.mean()).to_dict() if len(rel) else {}
    return pd.Series(gross).sort_index(), pd.Series(costs).sort_index()


def _eligible_only(nxt: pd.DataFrame, members: dict) -> pd.DataFrame:
    """Next-week returns on each sort date, kept only for that week's eligible coins."""
    y = pd.DataFrame(np.nan, index=sorted(members), columns=nxt.columns)
    for D, names in members.items():
        y.loc[D, names] = nxt.loc[D, names]
    return y


def fama_macbeth(y: pd.DataFrame, X: dict, lags: int = 4) -> dict:
    """Weekly cross-sectional OLS of y on standardised regressors; NW t of the mean slope of the first one."""
    names = list(X)
    slopes = []
    for D in y.index:
        df = pd.DataFrame({"y": y.loc[D], **{k: X[k].loc[D] for k in names}}).dropna()
        if len(df) < len(names) + 3:
            continue
        Z = (df[names] - df[names].mean()) / df[names].std(ddof=0).replace(0, np.nan)
        if Z.isna().any().any():
            continue
        A = np.column_stack([np.ones(len(df)), Z.to_numpy()])
        beta = np.linalg.lstsq(A, df["y"].to_numpy(), rcond=None)[0]
        slopes.append(beta[1])
    s = np.array(slopes)
    t, _ = nw_p(s, lags)
    return {"weeks": int(len(s)), "mean_slope": float(s.mean()), "nw_t": t}


def r1_nearness52():
    F = daily_frames(LARGE)
    close, high, fill, qv = F["close"], F["high"], F["fill"], F["quote_volume"]
    first = close.apply(lambda c: c.first_valid_index())
    hi365 = high.rolling(365, min_periods=1).max()
    n_hi = high.rolling(365, min_periods=1).count()
    n52 = close / hi365
    S = pd.DataFrame({s: T.ensemble(close[s]) for s in LARGE})
    lr = np.log(close).diff()
    ret28 = np.log(close.shift(1) / close.shift(29))
    vol30 = lr.rolling(30).std()
    lqv = np.log(qv.rolling(30).median())
    mondays = pd.date_range("2021-01-04", "2025-09-22", freq="7D", tz="UTC")
    top, ew = {}, {}
    for D in mondays:
        elig = [s for s in LARGE if first[s] is not None and first[s] <= D - pd.Timedelta(days=365)
                and close.loc[D - pd.Timedelta(days=7):D, s].notna().all() and n_hi.at[D, s] >= 330]
        if len(elig) < 9:
            continue
        v = n52.loc[D, elig].sort_index().sort_values(ascending=False, kind="mergesort")
        top[D], ew[D] = list(v.index[:len(elig) // 3]), elig
    g_top, c_top = weekly_book(top, fill, SPOT)
    g_ew, c_ew = weekly_book(ew, fill, SPOT)
    s = (g_top - c_top) - (g_ew - c_ew)
    t_nw, p_nw = nw_p(s, 7)
    p_bs = week_block_bootstrap_p(s)
    nxt = _eligible_only(fill.shift(-7) - fill, ew)
    fm = fama_macbeth(nxt, {"n52": n52.loc[mondays], "ret28": ret28.loc[mondays],
                                          "vol30": vol30.loc[mondays], "lqv": lqv.loc[mondays], "trend": S.loc[mondays]})
    p = max(p_nw, p_bs)
    return {"primary": {"mean_bps_per_day": float(s.mean() * 1e4), "ann": float(s.mean() * 365), "days": int(s.notna().sum()),
                        "nw_t": t_nw, "p_nw": p_nw, "p_block_bootstrap": p_bs, "p": p},
            "extra_condition": {"fama_macbeth": fm, "passed": bool(fm["mean_slope"] > 0 and fm["nw_t"] >= 1.65)},
            "secondary": {"top_net_ann": float((g_top - c_top).mean() * 365), "ew_net_ann": float((g_ew - c_ew).mean() * 365),
                          "by_year_spread_ann": {int(y): round(float(g.mean() * 365), 3) for y, g in s.groupby(s.index.year)}}}


def r4_max_effect():
    F = daily_frames(LARGE)
    close, fill = F["close"], F["fill"]
    first = close.apply(lambda c: c.first_valid_index())
    lr = np.log(close).diff()
    mx = lr.rolling(7).max()
    ret7 = np.log(close / close.shift(7))
    ret28 = np.log(close.shift(1) / close.shift(29))
    vol30 = lr.rolling(30).std()
    mondays = pd.date_range("2020-03-02", "2025-09-22", freq="7D", tz="UTC")
    low, high_, ew = {}, {}, {}
    for D in mondays:
        elig = [s for s in LARGE if first[s] is not None and first[s] <= D - pd.Timedelta(days=60)
                and close.loc[D - pd.Timedelta(days=7):D, s].notna().all() and np.isfinite(mx.at[D, s])]
        if len(elig) < 9:
            continue
        v = mx.loc[D, elig].sort_index().sort_values(kind="mergesort")
        n3 = len(elig) // 3
        low[D], high_[D], ew[D] = list(v.index[:n3]), list(v.index[-n3:]), elig
    # 30-day idiosyncratic volatility at each sort date, as pre-registered: one OLS (with intercept) of the
    # coin's daily log returns over D-29..D on the mean return of that week's eligible coins.
    idio = pd.DataFrame(np.nan, index=mondays, columns=LARGE)
    for D, names in ew.items():
        win = lr.loc[D - pd.Timedelta(days=29):D, names]
        mkt = win.mean(axis=1)
        for s_ in names:
            df = pd.concat([win[s_], mkt], axis=1).dropna()
            if len(df) >= 20:
                A = np.column_stack([np.ones(len(df)), df.iloc[:, 1]])
                resid = df.iloc[:, 0].to_numpy() - A @ np.linalg.lstsq(A, df.iloc[:, 0].to_numpy(), rcond=None)[0]
                idio.at[D, s_] = resid.std(ddof=1)
    g_lo, c_lo = weekly_book(low, fill, PERP)
    g_hi, c_hi = weekly_book(high_, fill, PERP)
    fund = PERP.holding(1440, 1)
    s = (g_lo - c_lo - fund) - (g_hi + c_hi)          # long LOW (pays funding), short HIGH (no funding credit)
    t_nw, p_nw = nw_p(s, 7, two_sided=True)
    p_bs = week_block_bootstrap_p(s, two_sided=True)
    nxt = _eligible_only(fill.shift(-7) - fill, ew)
    fm = fama_macbeth(nxt, {"max": mx.loc[mondays], "ret7": ret7.loc[mondays], "ret28": ret28.loc[mondays],
                                          "vol30": vol30.loc[mondays], "idio30": idio})
    sign_spread = np.sign(s.mean())
    # Spread is LOW - HIGH, so a positive spread means a NEGATIVE MAX slope.
    passed = bool(np.sign(fm["mean_slope"]) == -sign_spread and abs(fm["nw_t"]) >= 1.96)
    return {"primary": {"mean_bps_per_day": float(s.mean() * 1e4), "ann": float(s.mean() * 365), "days": int(s.notna().sum()),
                        "nw_t": t_nw, "p_nw_two_sided": p_nw, "p_block_bootstrap_two_sided": p_bs, "p": max(p_nw, p_bs)},
            "extra_condition": {"fama_macbeth": fm, "passed": passed},
            "secondary": {"by_year_spread_ann": {int(y): round(float(g.mean() * 365), 3) for y, g in s.groupby(s.index.year)}}}


def r7_turn_of_month():
    F = daily_frames(LARGE)
    fill = F["fill"]
    r = np.expm1(fill.shift(-1) - fill)               # return from D's fill to D+1's fill, labelled D
    r_ew = r.mean(axis=1, skipna=True)
    r_ew = r_ew[(r_ew.index >= pd.Timestamp("2020-01-01", tz="UTC")) & (r_ew.index < DAILY_END)].dropna()
    idx = r_ew.index
    month_starts = pd.date_range("2019-12-01", "2025-10-01", freq="MS", tz="UTC")

    def window_dummy(start_offset: int) -> np.ndarray:
        """1 on the 4 days starting `start_offset` days after each month start (-1 = last day of prior month)."""
        days = set()
        for m in month_starts:
            st = m + pd.Timedelta(days=start_offset)
            days.update(st + pd.Timedelta(days=k) for k in range(4))
        return np.isin(idx, pd.DatetimeIndex(sorted(days))).astype(float)

    tom = window_dummy(-1)
    beta, t, n = M.hac_ols(r_ew.to_numpy(), tom.reshape(-1, 1), lags=5)
    p = float(1 - stats.norm.cdf(t[1]))
    placebo_b = {k: float(M.hac_ols(r_ew.to_numpy(), window_dummy(k).reshape(-1, 1), lags=5)[0][1]) for k in range(-1, 27)}
    rank = int(sum(v >= placebo_b[-1] for v in placebo_b.values()))
    return {"primary": {"b_bps_per_day": float(beta[1] * 1e4), "t": float(t[1]), "p": p, "days": n},
            "extra_condition": {"window_bps": float(4 * beta[1] * 1e4), "cost_bps": 25.0,
                                "passed": bool(4 * beta[1] > 0.0025)},
            "secondary": {"tom_rank_among_28_start_days": rank,
                          "by_year_b_bps": {int(y): round(float(M.hac_ols(g.to_numpy(), tom[idx.year == y].reshape(-1, 1), 5)[0][1] * 1e4), 1)
                                            for y, g in r_ew.groupby(idx.year)}}}


# ------------------------------------------------------------------ event studies (R2, R3, R5, R6, R8)

NYSE_CLOSED = set(pd.to_datetime([
    "2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25", "2020-07-03", "2020-09-07", "2020-11-26", "2020-12-25",
    "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31", "2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24",
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20", "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26",
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23",
    "2023-12-25", "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-11-28", "2024-12-25", "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01",
    # 13:00 ET early closes, excluded
    "2020-11-27", "2020-12-24", "2021-11-26", "2022-11-25", "2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03"]).date)


def nyse_closes() -> pd.DatetimeIndex:
    days = pd.bdate_range("2020-01-02", "2025-09-26")
    days = [d for d in days if d.date() not in NYSE_CLOSED]
    return pd.DatetimeIndex([pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta(hours=16)
                             for d in days]).tz_convert("UTC")


def r2_intraday_momentum(h: int = 60):
    closes = nyse_closes()
    evs = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        m1 = research_frame(sym)
        b = Bars(m1)
        g60 = b.gaps(60)
        pos_c = b.index.get_indexer(closes)
        rows = []
        for k in range(1, len(closes)):
            pc, pp = pos_c[k], pos_c[k - 1]
            if pc < 0 or pp < 0:
                continue
            t = pc - h
            if b.gap[t] or b.gap[pp] or g60[t] > 0:
                continue
            r = b.lc[t] - b.lc[pp]
            m = t - pp
            z = r / (b.sigma[t] * np.sqrt(m)) if np.isfinite(b.sigma[t]) and b.sigma[t] > 0 else 0.0
            if abs(z) >= 1.0:
                rows.append((t, int(np.sign(r))))
        if rows:
            idx, d = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
            evs.append(outcomes(b, idx, d, h).assign(sym=sym))
    ev = pd.concat(evs, ignore_index=True)
    ev["net"] = event_net(ev, PERP, h)
    m, p, lo = day_bootstrap(ev, "net")
    return {"primary": {"n": int(len(ev)), "gross_bps": float(ev["ret"].mean() * 1e4), "net_bps": float(m * 1e4), "p": p},
            "extra_condition": {"passed": True},
            "secondary": {"by_year": by_year(ev), "long_only_spot_net_bps":
                          float((ev[ev["dir"] > 0]["ret"] - 2 * SPOT.one_side("BTCUSDT")).mean() * 1e4)}}


def r5_deribit_expiry(h: int = 60):
    months = pd.date_range("2020-01-01", "2025-09-01", freq="MS")
    expiry = [(m + pd.offsets.MonthEnd(0)) - pd.Timedelta(days=((m + pd.offsets.MonthEnd(0)).weekday() - 4) % 7)
              for m in months]
    exp_days = {e.date() for e in expiry}
    exp_months = {(e.year, e.month) for e in expiry}
    evs = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        m1 = research_frame(sym)
        b = Bars(m1)
        g60 = b.gaps(60)
        cands = pd.date_range("2020-01-01 08:00", "2025-09-27 08:00", freq="1D", tz="UTC")
        pos = b.index.get_indexer(cands)
        keep = [(p_, c) for p_, c in zip(pos, cands) if p_ >= 0 and g60[p_] == 0 and not b.gap[p_]]
        idx = np.array([k[0] for k in keep])
        ev = outcomes(b, idx, np.ones(len(idx), int), h).assign(sym=sym)
        evs.append(ev)
    ev = pd.concat(evs, ignore_index=True)
    day = ev["ts"].dt.date
    is_exp = day.isin(exp_days)
    in_month = [(d.year, d.month) in exp_months for d in ev["ts"]]
    null = (~is_exp) & ev["ts"].dt.weekday.isin([0, 1, 2, 3]) & pd.Series(in_month)
    E, N = ev[is_exp], ev[null]
    delta, p = delta_bootstrap(E, N)
    net_e = event_net(E, PERP, h)
    fri = ev[(~is_exp) & (ev["ts"].dt.weekday == 4)]
    q = E[E["ts"].dt.month.isin([3, 6, 9, 12])]
    return {"primary": {"n_expiry_events": int(len(E)), "n_null_events": int(len(N)), "delta_bps": delta * 1e4, "p": p},
            "extra_condition": {"expiry_net_bps": float(net_e.mean() * 1e4), "passed": bool(net_e.mean() > 0)},
            "secondary": {"ordering_gross_bps": {"quarterly": float(q["ret"].mean() * 1e4), "monthly": float(E["ret"].mean() * 1e4),
                                                 "other_fridays": float(fri["ret"].mean() * 1e4),
                                                 "mon_thu": float(N["ret"].mean() * 1e4)}}}


def _grid_arrays(m1: pd.DataFrame):
    close = m1["close"].to_numpy(float)
    day = (m1.index - pd.Timedelta(minutes=1)).floor("D")
    c_ref = m1["close"].reindex(day).to_numpy(float)          # close of the bar closing at D 00:00
    S = 10.0 ** (np.floor(np.log10(c_ref)) - 1)
    # tick filter from the previous day's smallest positive 1m close change
    dc = np.abs(np.diff(close, prepend=np.nan))
    dc[~(dc > 0)] = np.nan
    tick = pd.Series(dc, index=day).groupby(level=0).min()
    tick_prev = tick.shift(1).reindex(day).to_numpy()
    ok = np.isfinite(S) & np.isfinite(tick_prev) & (tick_prev / c_ref <= 5e-4)
    return close, S, ok


def round_signals(m1: pd.DataFrame, b: Bars, kind: str, offset_frac: float) -> np.ndarray:
    close, S, ok = _grid_arrays(m1)
    hi, lo = m1["high"].to_numpy(float), m1["low"].to_numpy(float)
    g = S / 2
    o = offset_frac * S
    prev = np.concatenate([[np.nan], close[:-1]])
    g241 = b.gaps(241)
    sig = np.zeros(len(close))
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "break":
            hmax = pd.Series(hi).rolling(226).max().shift(15).to_numpy()
            lmin = pd.Series(lo).rolling(226).min().shift(15).to_numpy()
            L_up = np.floor((close / 1.001 - o) / g) * g + o
            up = (prev < L_up * 1.001) & (hmax < L_up)
            L_dn = np.ceil((close / 0.999 - o) / g) * g + o
            dn = (prev > L_dn * 0.999) & (lmin > L_dn)
            sig = np.where(up, 1, np.where(dn, -1, 0))
        else:                                              # first touch, fade
            lmin = pd.Series(lo).rolling(240).min().shift(1).to_numpy()
            hmax = pd.Series(hi).rolling(240).max().shift(1).to_numpy()
            L_s = (np.ceil((close - o) / g) - 1) * g + o           # highest level strictly below close
            sup = (lo <= L_s * 1.0005) & (lmin > L_s * 1.0005)
            L_r = (np.floor((close - o) / g) + 1) * g + o          # lowest level strictly above close
            res = (hi >= L_r * 0.9995) & (hmax < L_r * 0.9995)
            sig = np.where(sup & ~res, 1, np.where(res & ~sup, -1, 0))
    sig[~ok | (g241 > 0)] = 0
    return sig


def round_study(kind: str, h: int = 60, symbols=None):
    symbols = symbols or LARGE
    grids = {"round": [0.0], "control": [0.13, 0.37]}
    out = {k: [] for k in grids}
    for sym in symbols:
        m1 = research_frame(sym)
        b = Bars(m1)
        for name, offs in grids.items():
            for off in offs:
                sig = round_signals(m1, b, kind, off)
                idx = select(sig, cooldown=h)
                if len(idx):
                    out[name].append(outcomes(b, idx, np.sign(sig[idx]).astype(int), h).assign(sym=sym))
        import quant.experiment as E
        E._M1_CACHE.pop(sym, None)
    R = pd.concat(out["round"], ignore_index=True)
    C = pd.concat(out["control"], ignore_index=True)
    delta, p = delta_bootstrap(R, C)
    R["net"] = event_net(R, PERP, h)
    return R, C, delta, p


def r3_round_break():
    R, C, delta, p = round_study("break")
    return {"primary": {"n_round": int(len(R)), "n_control": int(len(C)), "round_gross_bps": float(R["ret"].mean() * 1e4),
                        "control_gross_bps": float(C["ret"].mean() * 1e4), "delta_bps": delta * 1e4, "p": p},
            "extra_condition": {"round_net_bps": float(R["net"].mean() * 1e4), "passed": bool(R["net"].mean() > 0)},
            "secondary": {"by_year_round_net": by_year(R),
                          "spot_long_only_net_bps": float((R[R["dir"] > 0]["ret"] - 2 * SPOT.one_side("ETHUSDT")).mean() * 1e4)}}


def r6_round_touch():
    R, C, delta, p = round_study("touch")
    return {"primary": {"n_round": int(len(R)), "n_control": int(len(C)), "round_gross_bps": float(R["ret"].mean() * 1e4),
                        "control_gross_bps": float(C["ret"].mean() * 1e4), "delta_bps": delta * 1e4, "p": p},
            "extra_condition": {"round_net_bps": float(R["net"].mean() * 1e4), "passed": bool(R["net"].mean() > 0)},
            "secondary": {"by_year_round_net": by_year(R)}}


# ------------------------------------------------------------------ R8 kernel chart patterns

BW_GRID = np.round(np.arange(0.5, 10.0001, 0.1), 2)
WIN = 38


def _kernels():
    i = np.arange(WIN)
    D2 = (i[:, None] - i[None, :]) ** 2
    K = np.exp(-0.5 * D2[None] / BW_GRID[:, None, None] ** 2)
    K_loo = K.copy()
    K_loo[:, i, i] = 0
    return D2, K_loo / K_loo.sum(2, keepdims=True)


def smooth_windows(X: np.ndarray, D2, W_loo) -> np.ndarray:
    """Nadaraya-Watson fit per window; bandwidth = 0.3 x leave-one-out CV choice."""
    fits = np.empty_like(X)
    for a in range(0, len(X), 1500):
        x = X[a:a + 1500]
        m_loo = np.einsum("hij,wj->hwi", W_loo, x)
        err = ((m_loo - x[None]) ** 2).sum(2)                  # (h, w)
        bw = 0.3 * BW_GRID[err.argmin(0)]
        for u in np.unique(bw):
            sel = np.flatnonzero(bw == u)
            K = np.exp(-0.5 * D2 / u ** 2)
            K = K / K.sum(1, keepdims=True)
            fits[a + sel] = x[sel] @ K.T
    return fits


def extrema(fit: np.ndarray, closes: np.ndarray) -> list:
    d = np.diff(fit)
    ex = []
    for tau in range(1, WIN - 1):
        kind = 1 if d[tau - 1] > 0 and d[tau] < 0 else -1 if d[tau - 1] < 0 and d[tau] > 0 else 0
        if not kind:
            continue
        lo_, hi_ = tau - 1, min(tau + 2, WIN)
        seg = closes[lo_:hi_]
        j = lo_ + (int(np.argmax(seg)) if kind > 0 else int(np.argmin(seg)))
        if j > WIN - 4:                                        # 3-bar confirmation lag
            continue
        if ex and ex[-1][0] == kind:                           # keep alternation: the more extreme one
            if (kind > 0 and closes[j] > ex[-1][2]) or (kind < 0 and closes[j] < ex[-1][2]):
                ex[-1] = (kind, j, closes[j])
            continue
        ex.append((kind, j, closes[j]))
    return ex


def detect(ex: list) -> dict:
    tol = lambda a, b: abs(a - b) <= 0.015 * (a + b) / 2          # noqa: E731
    out = {"HS": False, "IHS": False, "DTOP": False, "DBOT": False}
    if len(ex) >= 5:
        e = ex[-5:]
        p = [x[2] for x in e]
        if e[0][0] > 0 and p[2] > p[0] and p[2] > p[4] and tol(p[0], p[4]) and tol(p[1], p[3]):
            out["HS"] = True
        if e[0][0] < 0 and p[2] < p[0] and p[2] < p[4] and tol(p[0], p[4]) and tol(p[1], p[3]):
            out["IHS"] = True
    if ex:
        k1, t1, p1 = ex[0]
        later = [x for x in ex[1:] if x[0] == k1 and x[1] - t1 > 22]
        if later:
            a = max(later, key=lambda x: x[2]) if k1 > 0 else min(later, key=lambda x: x[2])
            if tol(p1, a[2]):
                out["DTOP" if k1 > 0 else "DBOT"] = True
    return out


def r8_chart_patterns(h: int = 1440, symbols=None):
    symbols = symbols or LARGE
    D2, W_loo = _kernels()
    evs, pls = [], []
    for sym in symbols:
        m1 = research_frame(sym)
        b = Bars(m1)
        bars = resample(m1, "4h")
        c = bars["close"].where(bars["gap_frac"] <= 5 / 240)
        x = np.log(c.to_numpy(float))
        n = len(x)
        starts = np.arange(0, n - WIN + 1)
        X = np.lib.stride_tricks.sliding_window_view(x, WIN)
        good = np.isfinite(X).all(1)
        fits = np.full_like(X, np.nan)
        fits[good] = smooth_windows(X[good], D2, W_loo)
        last_fire = {k: -10 ** 9 for k in ("HS", "IHS", "DTOP", "DBOT")}
        prev_det = {k: False for k in last_fire}
        rows = []
        for w in starts:
            t = w + WIN - 1
            if not good[w]:
                prev_det = {k: False for k in prev_det}
                continue
            det = detect(extrema(fits[w], np.exp(X[w])))
            fire = {k: det[k] and not prev_det[k] and t - last_fire[k] >= WIN for k in det}
            prev_det = det
            for k, f in fire.items():
                if f:
                    last_fire[k] = t
            bull = fire["IHS"] or fire["DBOT"]
            bear = fire["HS"] or fire["DTOP"]
            if bull != bear:
                rows.append((bars.index[t], 1 if bull else -1))
        if rows:
            pos = b.index.get_indexer(pd.DatetimeIndex([r[0] for r in rows]))
            ok = pos >= 0
            idx = pos[ok]
            d = np.array([r[1] for r in rows])[ok]
            ev = outcomes(b, idx, d, h).assign(sym=sym)
            evs.append(ev)
            pls.append(placebo(b, ev, h, n_per_event=3))
        import quant.experiment as E
        E._M1_CACHE.pop(sym, None)
    ev = pd.concat(evs, ignore_index=True)
    pl = pd.concat(pls, ignore_index=True)
    ev["net"] = event_net(ev, PERP, h)
    m, p, lo = day_bootstrap(ev, "net")
    excess = float(ev["ret"].mean() - pl["ret"].mean())
    return {"primary": {"n": int(len(ev)), "gross_bps": float(ev["ret"].mean() * 1e4), "net_bps": float(m * 1e4), "p": p},
            "extra_condition": {"excess_vs_placebo_bps": excess * 1e4, "passed": bool(excess > 0)},
            "secondary": {"by_year": by_year(ev), "bullish_n": int((ev["dir"] > 0).sum()), "bearish_n": int((ev["dir"] < 0).sum())}}


# ------------------------------------------------------------------ main

def main():
    t0 = time.time()
    res = {}
    for key, fn in (("R1_nearness52", r1_nearness52), ("R2_intraday_momentum", r2_intraday_momentum),
                    ("R3_round_break", r3_round_break), ("R4_max_effect", r4_max_effect),
                    ("R5_deribit_expiry", r5_deribit_expiry), ("R6_round_touch", r6_round_touch),
                    ("R7_turn_of_month", r7_turn_of_month), ("R8_chart_patterns", r8_chart_patterns)):
        res[key] = fn()
        print(f"{key}: {res[key]['primary']} | extra {res[key]['extra_condition']} ({time.time() - t0:.0f}s)", flush=True)
    keys = list(res)
    p = np.array([res[k]["primary"]["p"] for k in keys])
    holm, bh = M.holm(p), M.benjamini_hochberg(p)
    for k, ph, pb in zip(keys, holm, bh):
        res[k]["p_holm"], res[k]["p_bh"] = float(ph), float(pb)
        res[k]["verdict"] = "PASS" if ph < 0.05 and res[k]["extra_condition"]["passed"] else "FAIL"
    save("psychology", res)
    print("\n".join(f"{k:22s} p={res[k]['primary']['p']:.4f} holm={res[k]['p_holm']:.4f} extra={res[k]['extra_condition']['passed']} "
                    f"-> {res[k]['verdict']}" for k in keys), flush=True)
    return res


if __name__ == "__main__":
    main()
