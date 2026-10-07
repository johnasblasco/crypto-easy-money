"""Evaluation statistics: calibration, trading performance, and overfitting control.

Everything here treats the backtest as an experiment that is trying to fail:
confidence intervals come from a stationary block bootstrap (returns are
autocorrelated and heteroskedastic), Sharpe ratios are deflated for the
number of configurations tried, and model comparisons use a HAC
(Newey-West) Diebold-Mariano test on per-sample loss differences.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------- calibration

def log_loss(y, p, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Equal-frequency bins of predicted probability vs observed frequency."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["p_mean", "y_mean", "n"])
    # Bin on the probability values (ties stay together): splitting identical
    # predictions across bins would report pure sampling noise as miscalibration.
    if df["p"].nunique() == 1:
        df["bin"] = 0
    else:
        df["bin"] = pd.qcut(df["p"], q=min(bins, len(df)), labels=False, duplicates="drop")
    t = df.groupby("bin").agg(p_mean=("p", "mean"), y_mean=("y", "mean"), n=("y", "size"))
    return t.reset_index(drop=True)


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error (equal-frequency bins)."""
    t = reliability_table(y, p, bins)
    if t.empty:
        return float("nan")
    return float(np.sum(t["n"] * (t["p_mean"] - t["y_mean"]).abs()) / t["n"].sum())


def calibration_slope(y, p, eps: float = 1e-6) -> tuple[float, float]:
    """Slope/intercept of logit(P(y=1)) on logit(p). Perfect calibration = (1, 0).

    Slope < 1 means over-confident predictions.
    """
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    m = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# ------------------------------------------------------------------ bootstrap

def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix (n_boot x n)."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new_block = rng.random((n_boot, n)) < p
    jumps = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], jumps[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_stat(x, stat, n_boot: int = 2000, mean_block: float | None = None, seed: int = 0):
    """Bootstrap distribution of ``stat(x)`` under the stationary bootstrap."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.array([])
    mean_block = mean_block or max(1.0, n ** (1 / 3))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    # Build indices in batches to bound memory.
    batch = max(1, min(n_boot, int(2e7 // max(n, 1))))
    k = 0
    while k < n_boot:
        b = min(batch, n_boot - k)
        idx = stationary_bootstrap_indices(n, mean_block, b, rng)
        out[k:k + b] = [stat(x[i]) for i in idx]
        k += b
    return out


def mean_pvalue(x, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """One-sided p-value for mean(x) > 0, with a 90% bootstrap CI (lo, hi).

    The null distribution is the bootstrap distribution of the demeaned series.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan"), float("nan"), float("nan")
    m = x.mean()
    boots = bootstrap_stat(x, np.mean, n_boot=n_boot, seed=seed)
    p = float(np.mean(boots - m >= m))
    lo, hi = np.percentile(boots, [5, 95])
    return max(p, 1.0 / n_boot), float(lo), float(hi)


# --------------------------------------------------------------- Sharpe & co.

def sharpe(r) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1))


def sortino(r) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) if len(r) else 0
    return float(r.mean() / downside) if downside > 0 else 0.0


def max_drawdown(r) -> float:
    """Max drawdown of the cumulative log-return path, as a (negative) fraction."""
    eq = np.cumsum(np.nan_to_num(np.asarray(r, float)))
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    dd = np.expm1(eq - peak)
    return float(dd.min()) if len(dd) else 0.0


def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark) given the sample (Bailey & Lopez de Prado)."""
    if n < 3:
        return float("nan")
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum of ``n_trials`` per-period Sharpe ratios under the null of zero skill."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    a = stats.norm.ppf(1 - 1.0 / n_trials)
    b = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(r, n_trials: int, var_trial_sr: float) -> float:
    """Deflated Sharpe ratio: PSR against the best Sharpe expected from luck alone."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    sr = sharpe(r)
    sr0 = expected_max_sharpe(n_trials, var_trial_sr)
    return probabilistic_sharpe(sr, len(r), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)), sr0)


def pbo_cscv(perf: np.ndarray, n_blocks: int = 10, metric=None) -> float:
    """Probability of backtest overfitting via combinatorially symmetric CV.

    ``perf`` is a (T periods x N configurations) matrix of per-period returns.
    For every split of the blocks into equal in-sample/out-of-sample halves,
    pick the best configuration in-sample and record whether it lands below
    the out-of-sample median. PBO is the fraction of splits where it does.
    """
    from itertools import combinations

    metric = metric or (lambda x: x.mean(0) / (x.std(0, ddof=1) + 1e-12))
    T, N = perf.shape
    if N < 2 or T < n_blocks * 2:
        return float("nan")
    blocks = np.array_split(np.arange(T), n_blocks)
    below = []
    for is_blocks in combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[i] for i in is_blocks])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in is_blocks])
        best = int(np.argmax(metric(perf[is_idx])))
        oos = metric(perf[oos_idx])
        rank = stats.rankdata(oos)[best] / (N + 1)
        below.append(rank <= 0.5)
    return float(np.mean(below))


# ---------------------------------------------------------- model comparison

def newey_west_var(x, lags: int | None = None) -> float:
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    lags = lags if lags is not None else int(4 * (n / 100) ** (2 / 9))
    v = np.dot(x, x) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * np.dot(x[k:], x[:-k]) / n
    return float(v)


def diebold_mariano(loss_a, loss_b, lags: int | None = None) -> tuple[float, float]:
    """Test whether model B has lower expected loss than model A.

    Returns (mean loss difference a - b, one-sided p-value for difference > 0).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return float("nan"), float("nan")
    v = newey_west_var(d, lags)
    if v <= 0:
        return float(d.mean()), float("nan")
    z = d.mean() / math.sqrt(v / n)
    return float(d.mean()), float(1 - stats.norm.cdf(z))


def hac_ols(y, X, lags: int):
    """OLS with Newey-West (Bartlett) standard errors. Returns (beta, t, n). X excludes the constant."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return beta, beta / np.sqrt(np.diag(V)), int(len(y))


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def benjamini_hochberg(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj
