"""Event-study machinery for sparse, conditional hypotheses on 1m bars.

An event fires at the close of 1m bar ``t`` using only bars <= t, with a
direction d in {+1, -1}. The trade enters at the VWAP of bar t+1 and exits at
the VWAP of bar t+1+h. Per symbol, events closer than ``cooldown`` to the
previous kept event are dropped (no overlapping trades).

Inference: events cluster in time (a crash fires on every coin at once), so
the unit of independent evidence is the calendar day: we bootstrap over days.
Every event set is compared with a placebo of random non-event times matched
on hour-of-day and volatility tercile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import fill_prices

EPS = 1e-12


class Bars:
    """Arrays over a symbol's full 1m grid for vectorised event definitions."""

    def __init__(self, m1: pd.DataFrame):
        self.index = m1.index
        self.n = len(m1)
        c = m1["close"].ffill().to_numpy(float)
        self.lc = np.log(c)
        self.r = np.diff(self.lc, prepend=np.nan)
        self.r[~np.isfinite(self.r)] = 0.0
        self.gap = m1["gap"].to_numpy(bool)
        self.q = m1["quote_volume"].to_numpy(float)
        self.tbq = m1["taker_buy_quote"].to_numpy(float)
        self.trades = m1["trades"].to_numpy(float)
        self.high = np.log(m1["high"].ffill().to_numpy(float))
        self.low = np.log(m1["low"].ffill().to_numpy(float))
        self.fill = fill_prices(m1, "vwap").ffill().to_numpy(float)
        self.hour = m1.index.hour.to_numpy()
        # Ex-ante per-minute volatility: EWMA of squared 1m returns, halflife 1 day, using bars <= t.
        self.sigma = np.sqrt(pd.Series(self.r ** 2).ewm(halflife=1440, min_periods=1440).mean().to_numpy())

    def wsum(self, x: np.ndarray, W: int) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(x)])
        out = np.full(self.n, np.nan)
        out[W - 1:] = cs[W:] - cs[:-W]
        return out

    def ret(self, W: int) -> np.ndarray:
        out = np.full(self.n, np.nan)
        out[W:] = self.lc[W:] - self.lc[:-W]
        return out

    def imbalance(self, W: int) -> np.ndarray:
        s = self.wsum(2 * self.tbq - self.q, W)
        q = self.wsum(self.q, W)
        return np.where(q > 0, s / (q + EPS), np.nan)

    def gaps(self, W: int) -> np.ndarray:
        return self.wsum(self.gap.astype(float), W)

    def same_hour_ratio(self, x: np.ndarray, W: int, days: int = 7) -> np.ndarray:
        """Window sum of x vs the mean of the same clock window over the previous ``days`` days."""
        cur = self.wsum(x, W)
        ref = np.zeros(self.n)
        cnt = np.zeros(self.n)
        for d in range(1, days + 1):
            lag = d * 1440
            past = np.full(self.n, np.nan)
            past[lag:] = cur[:-lag]
            ok = np.isfinite(past)
            ref[ok] += past[ok]
            cnt[ok] += 1
        ref = np.where(cnt >= days - 1, ref / np.maximum(cnt, 1), np.nan)
        return cur / (ref + EPS)

    def trailing_pctile(self, x: np.ndarray, days: int = 30, step: int = 60) -> np.ndarray:
        """Percentile of x[t] among x sampled every ``step`` minutes over the previous ``days`` days."""
        out = np.full(self.n, np.nan)
        samp_idx = np.arange(0, self.n, step)
        samp = x[samp_idx]
        win = days * 1440 // step
        # day-chunked: reference set refreshed once per day
        for start in range(0, self.n, 1440):
            k = start // step
            ref = samp[max(0, k - win):k]
            ref = np.sort(ref[np.isfinite(ref)])
            if len(ref) < win // 2:
                continue
            seg = x[start:start + 1440]
            p = np.searchsorted(ref, seg) / len(ref)
            p[~np.isfinite(seg)] = np.nan
            out[start:start + 1440] = p
        return out


def select(signal: np.ndarray, cooldown: int) -> np.ndarray:
    """Positions of nonzero signals, keeping only those >= cooldown after the previous kept one."""
    idx = np.flatnonzero(np.nan_to_num(signal) != 0)
    keep = []
    last = -10 ** 12
    for i in idx:
        if i - last >= cooldown:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=np.int64)


def outcomes(b: Bars, idx: np.ndarray, direction: np.ndarray, h: int, latency: int = 1) -> pd.DataFrame:
    """Directional forward log return for each event (entry VWAP t+latency, exit +h)."""
    ent = idx + latency
    ex = ent + h
    ok = ex < b.n
    idx, direction, ent, ex = idx[ok], direction[ok], ent[ok], ex[ok]
    bad = b.wsum(b.gap.astype(float), h + latency)  # gaps inside the holding window
    gap_in = np.array([b.gap[i + 1:j + 1].any() for i, j in zip(idx, ex)]) if len(idx) < 50000 else bad[ex] > 0
    raw = b.fill[ex] - b.fill[ent]
    out = pd.DataFrame({
        "ts": b.index[idx],
        "dir": direction,
        "ret": raw * direction,
        "sigma_h": b.sigma[idx] * np.sqrt(h),
        "hour": b.hour[idx],
    })
    return out[~gap_in & np.isfinite(out["ret"])].reset_index(drop=True)


def placebo(b: Bars, ev: pd.DataFrame, h: int, n_per_event: int = 5, seed: int = 0,
            direction: str = "same") -> pd.DataFrame:
    """Random non-event times matched on hour-of-day and volatility tercile, same direction mix."""
    rng = np.random.default_rng(seed)
    sig = b.sigma
    valid = np.flatnonzero(np.isfinite(sig) & ~b.gap)
    valid = valid[(valid > 2880) & (valid < b.n - h - 2)]
    terc = np.full(b.n, -1)
    q1, q2 = np.nanpercentile(sig[valid], [33.3, 66.7])
    terc[valid] = np.digitize(sig[valid], [q1, q2])
    key_arr = b.hour[valid] * 3 + terc[valid]
    order = np.argsort(key_arr, kind="stable")
    sorted_keys = key_arr[order]
    rows = []
    ev_pos = b.index.get_indexer(ev["ts"])
    for pos, d in zip(ev_pos, ev["dir"].to_numpy()):
        k = b.hour[pos] * 3 + (terc[pos] if terc[pos] >= 0 else 1)
        lo, hi = np.searchsorted(sorted_keys, [k, k + 1])
        if hi <= lo:
            continue
        pick = valid[order[rng.integers(lo, hi, n_per_event)]]
        rows.append(pd.DataFrame({"idx": pick, "dir": d}))
    if not rows:
        return pd.DataFrame(columns=["ts", "dir", "ret", "sigma_h", "hour"])
    pl = pd.concat(rows)
    return outcomes(b, pl["idx"].to_numpy(), pl["dir"].to_numpy(), h)


def day_bootstrap(ev: pd.DataFrame, value: str = "net", n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of ``value`` with a day-clustered bootstrap: (mean, p(mean<=0), 5th pct)."""
    if len(ev) < 5:
        return float("nan"), float("nan"), float("nan")
    days = ev["ts"].dt.floor("D")
    g = ev.groupby(days)[value].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), (n_boot, len(g)))
    means = s[pick].sum(1) / c[pick].sum(1)
    m = ev[value].mean()
    centered = means - means.mean()
    p = float(np.mean(centered >= m))
    return float(m), max(p, 1 / n_boot), float(np.percentile(means, 5))


def summarize(ev: pd.DataFrame, pl: pd.DataFrame, rt_cost: float) -> dict:
    if len(ev) == 0:
        return {"n": 0}
    ev = ev.assign(net=ev["ret"] - rt_cost)
    m, p, lo = day_bootstrap(ev, "net")
    m_gross, p_gross, _ = day_bootstrap(ev, "ret")
    out = {
        "n": int(len(ev)),
        "days": int(ev["ts"].dt.floor("D").nunique()),
        "gross_bps": float(ev["ret"].mean() * 1e4),
        "net_bps": float(m * 1e4),
        "p_net": p,
        "p_gross": p_gross,
        "net_p5_bps": float(lo * 1e4),
        "hit_rate_net": float((ev["net"] > 0).mean()),
        "gross_in_sigma": float((ev["ret"] / ev["sigma_h"]).mean()),
        "placebo_gross_bps": float(pl["ret"].mean() * 1e4) if len(pl) else float("nan"),
        "excess_vs_placebo_bps": float((ev["ret"].mean() - pl["ret"].mean()) * 1e4) if len(pl) else float("nan"),
    }
    by_year = {}
    for y, g in ev.groupby(ev["ts"].dt.year):
        by_year[int(y)] = {"n": int(len(g)), "net_bps": round(float(g["net"].mean() * 1e4), 1)}
    out["by_year"] = by_year
    pos_years = sum(1 for v in by_year.values() if v["n"] >= 5 and v["net_bps"] > 0)
    out["positive_years"] = f"{pos_years}/{sum(1 for v in by_year.values() if v['n'] >= 5)}"
    return out
