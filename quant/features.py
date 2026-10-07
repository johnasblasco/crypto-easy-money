"""Point-in-time feature families computed from 1m bars.

Every feature for decision time ``t`` is computed from 1m bars whose CLOSE
time is <= t (the index of the frames from ``quant.data.load``). Windows are
evaluated with cumulative sums on the 1-minute grid, sampled at decision
positions, so each value is exactly "the last L minutes up to and including
t". Missing minutes count as zero-return/zero-volume and a window with more
than ``MAX_GAP_FRAC`` missing minutes yields NaN.

Columns are named ``<family>__<name>`` so experiments can ablate whole
families by prefix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GAP_FRAC = 0.05
EPS = 1e-12


class Grid:
    """Cumulative-sum helpers over a 1m frame for fast trailing-window sums."""

    def __init__(self, m1: pd.DataFrame, times: pd.DatetimeIndex):
        self.m1 = m1
        self.times = times
        self.pos = m1.index.get_indexer(times)
        if (self.pos < 0).any():
            raise ValueError("decision times must be on the 1m grid")
        close = m1["close"].to_numpy(float)
        self.logc = np.log(close)
        lc_ff = pd.Series(self.logc).ffill().to_numpy()
        r = np.diff(lc_ff, prepend=np.nan)
        r[~np.isfinite(r)] = 0.0
        self.r1 = r
        gap = m1["gap"].to_numpy(bool)
        self.lc_ff = lc_ff
        self._cs = {}
        self._base = {
            "r": r,
            "r2": r * r,
            "absr": np.abs(r),
            "gap": gap.astype(float),
            "vol": m1["volume"].to_numpy(float),
            "qvol": m1["quote_volume"].to_numpy(float),
            "trades": m1["trades"].to_numpy(float),
            "tbq": m1["taker_buy_quote"].to_numpy(float),
        }
        q = self._base["qvol"]
        self._base["sqv"] = 2 * self._base["tbq"] - q          # signed quote volume (taker buy - taker sell)
        self._base["sqv_r"] = self._base["sqv"] * r            # for flow/return covariance
        self._base["sqv2"] = self._base["sqv"] ** 2
        hl = np.log(m1["high"].to_numpy(float)) - np.log(m1["low"].to_numpy(float))
        hl[~np.isfinite(hl)] = 0.0
        self._base["pk"] = hl * hl / (4 * np.log(2))            # Parkinson variance per minute
        self._base["illiq"] = np.abs(r) / np.maximum(q, 1.0)    # Amihud per minute

    def csum(self, key: str) -> np.ndarray:
        if key not in self._cs:
            self._cs[key] = np.concatenate([[0.0], np.cumsum(self._base[key])])
        return self._cs[key]

    def wsum(self, key: str, L: int) -> np.ndarray:
        """Sum of ``key`` over the L minutes ending at (and including) each decision time."""
        cs = self.csum(key)
        end = self.pos + 1
        start = np.maximum(end - L, 0)
        out = cs[end] - cs[start]
        out[end - L < 0] = np.nan  # not enough history
        return out

    def valid(self, L: int) -> np.ndarray:
        return self.wsum("gap", L) <= MAX_GAP_FRAC * L

    def logret(self, L: int) -> np.ndarray:
        p = self.pos
        prev = p - L
        out = np.full(len(p), np.nan)
        ok = prev >= 0
        out[ok] = self.lc_ff[p[ok]] - self.lc_ff[prev[ok]]
        return out


def _mask(x, ok):
    x = np.asarray(x, float)
    x[~ok] = np.nan
    return x


# ------------------------------------------------------------------ families

def momentum(g: Grid, windows=(5, 15, 60, 240, 720, 1440, 4320, 10080)) -> dict:
    """Trailing log returns, raw and scaled by trailing 1-day volatility."""
    sig = np.sqrt(g.wsum("r2", 1440) / 1440)  # per-minute vol over last day
    out = {}
    for L in windows:
        ret = _mask(g.logret(L), g.valid(L))
        out[f"mom__ret_{L}"] = ret
        out[f"mom__z_{L}"] = ret / (sig * np.sqrt(L) + EPS)
    return out


def volatility(g: Grid) -> dict:
    out = {}
    rv = {L: np.sqrt(g.wsum("r2", L) / L) for L in (60, 240, 1440, 10080, 43200)}
    for L, v in rv.items():
        out[f"vol__rv_{L}"] = _mask(np.log(v + EPS), g.valid(L))
    pk = np.sqrt(g.wsum("pk", 1440) / 1440)
    out["vol__parkinson_1440"] = np.log(pk + EPS)
    out["vol__ratio_60_1440"] = np.log((rv[60] + EPS) / (rv[1440] + EPS))
    out["vol__ratio_1440_43200"] = np.log((rv[1440] + EPS) / (rv[43200] + EPS))
    # Jumpiness: share of variance from the largest moves is approximated by
    # realised variance vs Parkinson (range) variance.
    out["vol__rv_vs_range_1440"] = np.log((rv[1440] + EPS) / (pk + EPS))
    return out


def activity(g: Grid) -> dict:
    """Volume / trade-count / trade-size, relative to their own trailing history."""
    out = {}
    q60, q1d, q30d = g.wsum("qvol", 60), g.wsum("qvol", 1440), g.wsum("qvol", 43200)
    n60, n1d, n30d = g.wsum("trades", 60), g.wsum("trades", 1440), g.wsum("trades", 43200)
    out["act__qvol_60_vs_30d"] = np.log((q60 * 720 + 1) / (q30d + 1))
    out["act__qvol_1d_vs_30d"] = np.log((q1d * 30 + 1) / (q30d + 1))
    out["act__trades_60_vs_30d"] = np.log((n60 * 720 + 1) / (n30d + 1))
    size60 = q60 / np.maximum(n60, 1)
    size30d = q30d / np.maximum(n30d, 1)
    out["act__tradesize_60_vs_30d"] = np.log((size60 + 1) / (size30d + 1))
    # Same hour-of-day last 7 days: volume seasonality adjustment.
    same_hour = np.zeros(len(g.pos))
    for d in range(1, 8):
        p = g.pos - d * 1440
        ok = p - 60 >= 0
        cs = g.csum("qvol")
        same_hour[ok] += cs[p[ok] + 1] - cs[p[ok] + 1 - 60]
    out["act__qvol_60_vs_samehour"] = np.log((q60 * 7 + 1) / (same_hour + 1))
    out["act__illiq_1440"] = np.log(g.wsum("illiq", 1440) / 1440 + EPS)
    return out


def order_flow(g: Grid, windows=(5, 15, 60, 240, 1440)) -> dict:
    """Taker-initiated (aggressive) flow from taker-buy volume.

    imbalance = (taker buy - taker sell) / total quote volume over the window.
    Absorption: heavy one-sided aggression that failed to move price, measured
    as imbalance minus what the window's return would "explain".
    """
    out = {}
    sig = np.sqrt(g.wsum("r2", 1440) / 1440)
    for L in windows:
        s, q = g.wsum("sqv", L), g.wsum("qvol", L)
        imb = s / (q + EPS)
        ok = g.valid(L) & (q > 0)
        out[f"flow__imb_{L}"] = _mask(imb, ok)
        ret_z = g.logret(L) / (sig * np.sqrt(L) + EPS)
        # Absorption: flow pushed one way, price did not follow (or moved against).
        out[f"flow__absorb_{L}"] = _mask(imb * 3 - np.tanh(ret_z), ok)
    # Kyle-lambda-like price impact over the last day: cov(r, signed vol) / var(signed vol).
    L = 1440
    n = L
    sr, ss, sq = g.wsum("sqv_r", L), g.wsum("sqv", L), g.wsum("sqv2", L)
    rr = g.wsum("r", L)
    cov = sr / n - (ss / n) * (rr / n)
    var = sq / n - (ss / n) ** 2
    lam = cov / (var + EPS)
    q1d = g.wsum("qvol", L) / n
    out["flow__impact_1440"] = _mask(np.sign(lam) * np.log1p(np.abs(lam * q1d) * 1e4), g.valid(L))
    # Persistent flow: 4h imbalance vs 1d imbalance.
    out["flow__imb_trend"] = out["flow__imb_240"] - out["flow__imb_1440"]
    return out


def bar_shape(g: Grid, m1: pd.DataFrame, scales=("15min", "1h", "4h")) -> dict:
    """Close location within range, wicks, and liquidity sweeps on completed bars.

    A sweep: the last completed bar traded beyond the extreme of the previous
    N bars but closed back inside it (stop-run / failed breakout).
    """
    out = {}
    for rule in scales:
        b = _bars(m1, rule)
        rng = (b["high"] - b["low"]).replace(0, np.nan)
        clv = (b["close"] - b["low"]) / rng
        upper = (b["high"] - b[["open", "close"]].max(axis=1)) / rng
        lower = (b[["open", "close"]].min(axis=1) - b["low"]) / rng
        N = 20
        prev_hi = b["high"].shift(1).rolling(N, min_periods=N - 2).max()
        prev_lo = b["low"].shift(1).rolling(N, min_periods=N - 2).min()
        sweep_hi = ((b["high"] > prev_hi) & (b["close"] < prev_hi)).astype(float)
        sweep_lo = ((b["low"] < prev_lo) & (b["close"] > prev_lo)).astype(float)
        breakout = ((b["close"] > prev_hi).astype(float) - (b["close"] < prev_lo).astype(float))
        frame = pd.DataFrame({
            f"shape__clv_{rule}": clv,
            f"shape__upwick_{rule}": upper,
            f"shape__lowwick_{rule}": lower,
            f"shape__sweep_{rule}": sweep_lo - sweep_hi,     # +1 = swept lows and reclaimed
            f"shape__breakout_{rule}": breakout,
        })
        aligned = _asof(frame, g.times)
        out.update({c: aligned[c].to_numpy(float) for c in frame.columns})
    return out


def range_position(g: Grid, m1: pd.DataFrame) -> dict:
    """Where price sits in its trailing range; distance from highs/lows."""
    d = _bars(m1, "1h")
    out = {}
    for days in (1, 7, 30):
        n = 24 * days
        hi = d["high"].rolling(n, min_periods=int(n * 0.9)).max()
        lo = d["low"].rolling(n, min_periods=int(n * 0.9)).min()
        pos = (d["close"] - lo) / (hi - lo).replace(0, np.nan)
        out[f"range__pos_{days}d"] = pos
    out["range__dd_from_60d_high"] = np.log(d["close"] / d["high"].rolling(24 * 60, min_periods=24 * 54).max())
    frame = pd.DataFrame(out)
    aligned = _asof(frame, g.times)
    return {c: aligned[c].to_numpy(float) for c in frame.columns}


def regime(g: Grid) -> dict:
    """Trend efficiency and volatility percentile (ex-ante regime descriptors)."""
    out = {}
    for L in (240, 1440, 10080):
        net = np.abs(g.logret(L))
        path = g.wsum("absr", L)
        out[f"regime__efficiency_{L}"] = _mask(net / (path + EPS), g.valid(L))
    # Variance ratio: variance of 60m returns vs 60 x variance of 1m returns, last 7d.
    # Approximated from 1d realised variance vs squared hourly-return proxy.
    rv1 = g.wsum("r2", 10080)
    hr = np.zeros(len(g.pos))
    for k in range(168):
        p = g.pos - k * 60
        ok = p - 60 >= 0
        hr[ok] += (g.lc_ff[p[ok]] - g.lc_ff[p[ok] - 60]) ** 2
    out["regime__varratio_60_7d"] = np.log((hr + EPS) / (rv1 + EPS))
    v = pd.Series(np.log(np.sqrt(g.wsum("r2", 1440)) + EPS), index=g.times)
    # Percentile of today's vol within the trailing year of decision times (past only).
    out["regime__vol_pctile_1y"] = _rolling_pctile(v, "365D")
    return out


def calendar(g: Grid) -> dict:
    t = g.times
    hour = t.hour + t.minute / 60
    return {
        "cal__hour_sin": np.sin(2 * np.pi * hour / 24),
        "cal__hour_cos": np.cos(2 * np.pi * hour / 24),
        "cal__weekend": (t.dayofweek >= 5).astype(float),
        "cal__dow_sin": np.sin(2 * np.pi * t.dayofweek / 7),
        "cal__dow_cos": np.cos(2 * np.pi * t.dayofweek / 7),
    }


def cross_asset(g: Grid, symbol: str, others: dict[str, pd.Series], windows=(15, 60, 240, 1440)) -> dict:
    """BTC lead, market breadth/dispersion, residual (beta-hedged) returns.

    ``others`` maps symbol -> 1m log close (forward-filled) on the same grid.
    """
    out = {}
    if not others:
        return out
    t = g.times
    rets = {}
    for L in windows:
        cols = {}
        for s, lc in others.items():
            arr = lc.to_numpy()
            p = lc.index.get_indexer(t)
            v = np.full(len(t), np.nan)
            ok = (p - L >= 0) & (p >= 0)
            v[ok] = arr[p[ok]] - arr[p[ok] - L]
            cols[s] = v
        frame = pd.DataFrame(cols, index=t)
        rets[L] = frame
        own = g.logret(L)
        out[f"xa__breadth_{L}"] = (frame > 0).mean(axis=1).to_numpy()
        out[f"xa__mkt_{L}"] = frame.mean(axis=1).to_numpy()
        out[f"xa__dispersion_{L}"] = frame.std(axis=1).to_numpy()
        out[f"xa__rel_{L}"] = own - frame.mean(axis=1).to_numpy()
        if symbol != "BTCUSDT" and "BTCUSDT" in frame:
            out[f"xa__btc_{L}"] = frame["BTCUSDT"].to_numpy()
    # Residual vs market with a trailing 30-day beta from hourly returns (past only).
    return out


# ------------------------------------------------------------------ helpers

def _bars(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    from .data import resample

    return resample(m1, rule)


def _asof(frame: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Last row of ``frame`` (indexed by bar close time) at or before each time."""
    frame = frame.sort_index()
    pos = frame.index.searchsorted(times, side="right") - 1
    out = frame.iloc[np.maximum(pos, 0)].copy()
    out.iloc[pos < 0] = np.nan
    out.index = times
    return out


def _rolling_pctile(v: pd.Series, window: str) -> np.ndarray:
    """Percentile of each value within the trailing ``window`` of DAILY samples before its day.

    The reference distribution for day D is the series sampled at each day's
    first decision time over the ``window`` days strictly before D, so it uses
    past data only and costs O(days * log window) instead of O(n * window).
    """
    vals = v.to_numpy(float)
    days = v.index.floor("D")
    day_codes, uniq = pd.factorize(days)
    first = ~pd.Index(day_codes).duplicated()
    dvals = np.full(len(uniq), np.nan)
    dvals[day_codes[first]] = vals[first]
    n_days = int(pd.Timedelta(window) / pd.Timedelta("1D"))
    out = np.full(len(v), np.nan)
    order = np.argsort(day_codes, kind="stable")
    bounds = np.searchsorted(day_codes[order], np.arange(len(uniq) + 1))
    for k in range(len(uniq)):
        ref = dvals[max(0, k - n_days):k]
        ref = np.sort(ref[np.isfinite(ref)])
        if len(ref) < 30:
            continue
        rows = order[bounds[k]:bounds[k + 1]]
        x = vals[rows]
        pct = np.searchsorted(ref, x, side="left") / len(ref)
        pct[~np.isfinite(x)] = np.nan
        out[rows] = pct
    return out


def long_trend(g: Grid, m1: pd.DataFrame) -> dict:
    """Slow trend state from daily bars closed by t (the documented cost-surviving family)."""
    d = _bars(m1, "1D")["close"]
    out = {}
    for L in (14, 28, 56, 90, 180):
        out[f"trend__ret_{L}d"] = np.log(d / d.shift(L))
    for L in (20, 55):
        hi = d.shift(1).rolling(L, min_periods=L - 2).max()
        lo = d.shift(1).rolling(L, min_periods=L - 2).min()
        out[f"trend__donchian_pos_{L}d"] = (d - lo) / (hi - lo).replace(0, np.nan)
    out["trend__dist_ma200"] = np.log(d / d.rolling(200, min_periods=180).mean())
    out["trend__dist_ma50"] = np.log(d / d.rolling(50, min_periods=45).mean())
    # Daily-vol-scaled 28d momentum (risk-adjusted trend strength)
    dv = np.log(d).diff().rolling(60, min_periods=50).std()
    out["trend__z_28d"] = np.log(d / d.shift(28)) / (dv * np.sqrt(28))
    frame = pd.DataFrame(out)
    aligned = _asof(frame, g.times)
    return {c: aligned[c].to_numpy(float) for c in frame.columns}


def fear_greed(g: Grid) -> dict:
    from .data import load_fng

    try:
        s = load_fng()
    except Exception:
        return {}
    frame = pd.DataFrame({"fng__value": s, "fng__chg_7d": s - s.shift(7)})
    aligned = _asof(frame, g.times)
    # stale if the last available value is more than 3 days old
    last = frame.index[np.maximum(frame.index.searchsorted(g.times, side="right") - 1, 0)]
    stale = (g.times - last) > pd.Timedelta("3D")
    out = {c: aligned[c].to_numpy(float) for c in frame.columns}
    for c in out:
        out[c][stale] = np.nan
    return out


FAMILIES = ["mom", "vol", "act", "flow", "shape", "range", "regime", "cal", "xa", "ta", "trend", "fng"]


def classic_indicators(m1: pd.DataFrame, times: pd.DatetimeIndex, rule: str = "1h") -> dict:
    """RSI / MACD / Bollinger on ``rule`` bars: the baseline family to beat."""
    b = _bars(m1, rule)
    c = b["close"]
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    frame = pd.DataFrame({
        f"ta__rsi_{rule}": rsi,
        f"ta__macdhist_{rule}": hist / c,
        f"ta__bbpctb_{rule}": (c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan),
        f"ta__ema_gap_{rule}": ema12 / ema26 - 1,
    })
    aligned = _asof(frame, times)
    return {k: aligned[k].to_numpy(float) for k in frame.columns}


def build(m1: pd.DataFrame, times: pd.DatetimeIndex, symbol: str, others: dict | None = None,
          families=None) -> pd.DataFrame:
    """All requested families for ``symbol`` at ``times``."""
    families = families or FAMILIES
    g = Grid(m1, times)
    cols: dict = {}
    if "mom" in families:
        cols.update(momentum(g))
    if "vol" in families:
        cols.update(volatility(g))
    if "act" in families:
        cols.update(activity(g))
    if "flow" in families:
        cols.update(order_flow(g))
    if "shape" in families:
        cols.update(bar_shape(g, m1))
    if "range" in families:
        cols.update(range_position(g, m1))
    if "regime" in families:
        cols.update(regime(g))
    if "cal" in families:
        cols.update(calendar(g))
    if "xa" in families and others:
        cols.update(cross_asset(g, symbol, {k: v for k, v in others.items() if k != symbol}))
    if "ta" in families:
        cols.update(classic_indicators(m1, times))
    if "trend" in families:
        cols.update(long_trend(g, m1))
    if "fng" in families:
        cols.update(fear_greed(g))
    X = pd.DataFrame(cols, index=times)
    return X.replace([np.inf, -np.inf], np.nan)
