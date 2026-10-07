"""Walk-forward prediction, NO-TRADE decision rule, cost-aware simulation, reporting.

Pipeline for one dataset (one symbol, one horizon, one decision grid):

1. ``walk_forward_predict``: for each fold, fit on ``fold.fit``, calibrate on
   ``fold.valid`` (out-of-sample for the fitted model), predict ``fold.test``.
2. ``choose_threshold``: on the same validation slice, pick the confidence
   threshold whose trades had the best *lower confidence bound* of net
   expected value. If no threshold has a positive lower bound, the model
   abstains (NO TRADE) for the whole test fold.
3. ``simulate``: positions on a non-overlapping grid (decision step ==
   horizon), costs charged on every position change, funding while held.
4. ``summarize``: classification, calibration and trading statistics.

Nothing in steps 1-2 ever sees the test fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import metrics as M
from .costs import CostModel
from .validation import Fold

MINUTES_PER_YEAR = 365.25 * 24 * 60


@dataclass
class Dataset:
    symbol: str
    horizon_min: int
    X: pd.DataFrame                # features at decision time (past-only), index = decision times
    fwd_ret: pd.Series             # log return entry -> exit
    label_end: pd.Series           # time the label is fully known
    regimes: pd.DataFrame = field(default_factory=pd.DataFrame)  # ex-ante regime labels for reporting
    sym: np.ndarray | None = None  # per-row symbol for pooled (panel) datasets
    sigma: pd.Series | None = None  # ex-ante std of the label (log-return units), for EV decisions

    def __post_init__(self):
        ok = (self.fwd_ret.notna() & self.X.notna().all(axis=1)).to_numpy()
        self.X = self.X[ok]
        self.fwd_ret = self.fwd_ret[ok]
        self.label_end = self.label_end[ok]
        if self.sym is None:
            self.sym = np.full(len(self.X), self.symbol, dtype=object)
        else:
            self.sym = np.asarray(self.sym, dtype=object)[ok]
        if self.sigma is not None:
            self.sigma = self.sigma[ok] if len(self.sigma) == len(ok) else self.sigma.reindex(self.X.index)
        if not self.regimes.empty:
            self.regimes = self.regimes[ok] if len(self.regimes) == len(ok) else self.regimes.reindex(self.X.index)

    @classmethod
    def pool(cls, datasets: list["Dataset"]) -> "Dataset":
        """Stack per-symbol datasets into one panel, sorted by decision time."""
        X = pd.concat([d.X for d in datasets])
        order = np.argsort(X.index.to_numpy(), kind="stable")
        cat = lambda parts: pd.concat(parts).iloc[order]  # noqa: E731
        regs = [d.regimes for d in datasets]
        sig = [d.sigma for d in datasets]
        return cls(
            symbol="PANEL", horizon_min=datasets[0].horizon_min, X=X.iloc[order],
            fwd_ret=cat([d.fwd_ret for d in datasets]), label_end=cat([d.label_end for d in datasets]),
            regimes=cat(regs) if all(not r.empty for r in regs) else pd.DataFrame(),
            sym=np.concatenate([d.sym for d in datasets])[order],
            sigma=cat(sig) if all(x is not None for x in sig) else None,
        )

    @property
    def times(self) -> pd.DatetimeIndex:
        return self.X.index

    @property
    def y(self) -> pd.Series:
        return (self.fwd_ret > 0).astype(int)


# --------------------------------------------------------------- calibration

class Calibrator:
    """Map raw scores to calibrated probabilities; fitted on validation only."""

    def __init__(self, method: str = "platt"):
        self.method = method
        self.model = None

    def fit(self, p_raw, y):
        p_raw = np.clip(np.asarray(p_raw, float), 1e-6, 1 - 1e-6)
        y = np.asarray(y, int)
        if self.method == "none" or len(np.unique(y)) < 2:
            self.method = "none"
            return self
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(p_raw, y)
        else:  # Platt on the logit: robust with small validation sets
            z = np.log(p_raw / (1 - p_raw)).reshape(-1, 1)
            self.model = LogisticRegression(C=1.0, max_iter=1000).fit(z, y)
        return self

    def transform(self, p_raw):
        p_raw = np.clip(np.asarray(p_raw, float), 1e-6, 1 - 1e-6)
        if self.method == "none":
            return p_raw
        if self.method == "isotonic":
            return self.model.predict(p_raw)
        z = np.log(p_raw / (1 - p_raw)).reshape(-1, 1)
        return self.model.predict_proba(z)[:, 1]


# ------------------------------------------------------------ decision rule

THRESHOLDS = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15])


def positions_from_prob(p, margin: float, allow_short: bool) -> np.ndarray:
    """+1 if P(up) >= 0.5 + margin, -1 if P(up) <= 0.5 - margin (when shorting allowed), else 0."""
    p = np.asarray(p, float)
    pos = np.where(p >= 0.5 + margin, 1, 0)
    if allow_short:
        pos = np.where(p <= 0.5 - margin, -1, pos)
    if margin == 0.0:  # p exactly 0.5 -> flat
        pos = np.where(p == 0.5, 0, pos)
    return pos.astype(int)


def round_trips(symbols, cost: CostModel, horizon_min: int) -> np.ndarray:
    symbols = np.atleast_1d(np.asarray(symbols, dtype=object))
    table = {s: cost.round_trip(s) + cost.holding(horizon_min) for s in set(symbols)}
    return np.array([table[s] for s in symbols])


EV_MARGINS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]) / 1e4
SQRT_2_OVER_PI = np.sqrt(2 / np.pi)


def expected_edge(p, sigma) -> np.ndarray:
    """E[r] under a symmetric return distribution with P(up)=p and scale sigma (log-return units)."""
    return (2 * np.asarray(p, float) - 1) * np.asarray(sigma, float) * SQRT_2_OVER_PI


def positions_from_ev(p, sigma, rt, margin: float, allow_short: bool) -> np.ndarray:
    """Long if expected edge exceeds round-trip cost + margin; short symmetric (if allowed)."""
    e = expected_edge(p, sigma)
    pos = np.where(e > rt + margin, 1, 0)
    if allow_short:
        pos = np.where(-e > rt + margin, -1, pos)
    return pos.astype(int)


def choose_ev_margin(p_valid, sigma_valid, r_valid, symbol, cost: CostModel, horizon_min: int,
                     min_trades: int = 30, z: float = 1.0) -> tuple[float | None, dict]:
    """Like choose_threshold but the rule is 'expected edge > cost + margin'."""
    p_valid, sigma_valid, r_valid = (np.asarray(a, float) for a in (p_valid, sigma_valid, r_valid))
    rt = round_trips(symbol if np.ndim(symbol) else [symbol] * len(p_valid), cost, horizon_min)
    best, best_lcb, table = None, 0.0, {}
    for m in EV_MARGINS:
        pos = positions_from_ev(p_valid, sigma_valid, rt, m, cost.allow_short)
        active = pos != 0
        n = int(active.sum())
        if n < min_trades:
            table[float(m)] = {"n": n}
            continue
        net = pos[active] * r_valid[active] - rt[active]
        ev, se = float(net.mean()), float(net.std(ddof=1) / np.sqrt(n))
        table[float(m)] = {"n": n, "ev": ev, "lcb": ev - z * se}
        if ev - z * se > 0 and (ev - z * se) * n > best_lcb:
            best, best_lcb = float(m), (ev - z * se) * n
    return best, table


def has_skill(p_valid, y_valid, base_rate: float, min_gain: float = 0.0) -> bool:
    """Validation log loss must beat the training base rate (no demonstrated skill -> no trading)."""
    from .metrics import log_loss

    return log_loss(y_valid, p_valid) < log_loss(y_valid, np.full(len(y_valid), base_rate)) - min_gain


def choose_threshold(p_valid, r_valid, symbol, cost: CostModel, horizon_min: int,
                     min_trades: int = 30, z: float = 1.0) -> tuple[float | None, dict]:
    """Pick the margin maximising the lower confidence bound of total net EV on validation.

    Each validation trade is charged a full round trip (conservative: validation
    samples may overlap in time). ``symbol`` may be one symbol or one per row.
    Returns (margin or None for abstain, table).
    """
    p_valid = np.asarray(p_valid, float)
    r_valid = np.asarray(r_valid, float)
    rt = round_trips(symbol if np.ndim(symbol) else [symbol] * len(p_valid), cost, horizon_min)
    best, best_lcb, table = None, 0.0, {}
    for m in THRESHOLDS:
        pos = positions_from_prob(p_valid, m, cost.allow_short)
        active = pos != 0
        n = int(active.sum())
        if n < min_trades:
            table[float(m)] = {"n": n, "ev": float("nan"), "lcb": float("nan")}
            continue
        net = pos[active] * r_valid[active] - rt[active]
        ev = float(net.mean())
        se = float(net.std(ddof=1) / np.sqrt(n))
        lcb_total = (ev - z * se) * n  # total edge, lower bound
        table[float(m)] = {"n": n, "ev": ev, "lcb": ev - z * se}
        if ev - z * se > 0 and lcb_total > best_lcb:
            best, best_lcb = float(m), lcb_total
    return best, table


# ------------------------------------------------------------ walk-forward

def walk_forward_predict(ds: Dataset, folds: list[Fold], model_factory, calibration: str = "platt",
                         decide: bool = True, cost: CostModel | None = None,
                         min_trades: int = 30, decision: str = "prob", skill_gate: bool = True) -> pd.DataFrame:
    """Out-of-sample predictions (and per-fold decisions) for every test sample.

    ``decision``: "prob" (trade when P(up) is far enough from 0.5) or "ev"
    (trade when expected edge, using the ex-ante sigma, beats cost + margin).
    ``skill_gate``: abstain for the whole fold unless the calibrated model beats
    the base rate on validation log loss.
    """
    X, y, r = ds.X.to_numpy(float), ds.y.to_numpy(), ds.fwd_ret.to_numpy(float)
    sig = ds.sigma.to_numpy(float) if ds.sigma is not None else None
    if decision == "ev" and sig is None:
        raise ValueError("EV decisions need ds.sigma")
    out = []
    for k, f in enumerate(folds):
        fit, valid, test = f.fit, f.valid, f.test
        if len(fit) < 100 or len(valid) < 30 or len(test) == 0:
            continue
        model = model_factory()
        model.fit(X[fit], y[fit])
        cal = Calibrator(calibration).fit(model.predict_proba(X[valid])[:, 1], y[valid])
        p_valid = cal.transform(model.predict_proba(X[valid])[:, 1])
        p_test_raw = model.predict_proba(X[test])[:, 1]
        p_test = cal.transform(p_test_raw)
        margin = None
        skilled = has_skill(p_valid, y[valid], float(np.mean(y[fit]))) if skill_gate else True
        if decide and cost is not None and skilled:
            if decision == "ev":
                margin, _ = choose_ev_margin(p_valid, sig[valid], r[valid], ds.sym[valid], cost, ds.horizon_min, min_trades)
            else:
                margin, _ = choose_threshold(p_valid, r[valid], ds.sym[valid], cost, ds.horizon_min, min_trades)
        if margin is None:
            pos = np.zeros(len(test), dtype=int)
        elif decision == "ev":
            pos = positions_from_ev(p_test, sig[test], round_trips(ds.sym[test], cost, ds.horizon_min),
                                    margin, cost.allow_short)
        else:
            pos = positions_from_prob(p_test, margin, cost.allow_short)
        out.append(pd.DataFrame({
            "fold": k,
            "p_raw": p_test_raw,
            "p": p_test,
            "y": y[test],
            "fwd_ret": r[test],
            "margin": np.nan if margin is None else margin,
            "skilled": skilled,
            "pos": pos,
            "sym": ds.sym[test],
            **{f"regime_{c}": ds.regimes[c].to_numpy()[test] for c in ds.regimes.columns},
        }, index=ds.times[test]))
    if not out:
        return pd.DataFrame(columns=["fold", "p_raw", "p", "y", "fwd_ret", "margin", "pos", "sym"])
    return pd.concat(out)


# ---------------------------------------------------------------- simulate

def simulate(pos: pd.Series, fwd_ret: pd.Series, symbol: str, cost: CostModel, horizon_min: int) -> pd.DataFrame:
    """Per-period net log returns of holding ``pos`` from each decision to the next.

    Assumes decisions are spaced exactly ``horizon_min`` apart so holding
    periods tile without overlap. A position change pays ``|delta| * one_side``
    (a flip from +1 to -1 pays two sides). Shorts are zeroed if not allowed.
    """
    pos = pos.astype(float).copy()
    if not cost.allow_short:
        pos[pos < 0] = 0
    r = fwd_ret.reindex(pos.index).fillna(0.0)
    # A missing label means we could not have traded: flat.
    pos[fwd_ret.reindex(pos.index).isna()] = 0
    # Positions are only contiguous within a regular grid; a time gap means flat in between.
    step = pd.Timedelta(minutes=horizon_min)
    prev = pos.shift(1).fillna(0.0)
    gap = pos.index.to_series().diff() != step
    prev[gap.to_numpy()] = 0.0
    turnover = (pos - prev).abs()
    # Closing the last position of each contiguous run is paid when the next period is flat
    # (via turnover there) or at the very end / before a gap:
    next_gap = np.append(gap.to_numpy()[1:], True)
    closing = pos.abs() * next_gap
    costs = (turnover + closing) * cost.one_side(symbol) + pos.abs() * cost.holding(horizon_min)
    gross = pos * r
    return pd.DataFrame({"pos": pos, "gross": gross, "cost": costs, "net": gross - costs})


# --------------------------------------------------------------- summarize

def trading_stats(sim: pd.DataFrame, horizon_min: int, n_boot: int = 1000) -> dict:
    """Per-trade statistics from rows; time-series statistics from the equal-weight portfolio."""
    active = sim["pos"].to_numpy() != 0
    act = sim["net"].to_numpy()[active]
    port = portfolio_returns(sim).to_numpy() if len(sim) else np.array([])
    periods_per_year = MINUTES_PER_YEAR / horizon_min
    gains, losses = act[act > 0].sum(), -act[act < 0].sum()
    p, lo, hi = M.mean_pvalue(act, n_boot=n_boot) if active.sum() >= 10 else (float("nan"),) * 3
    entries = 0
    for _, g in sim.groupby("sym", sort=False):
        pos = g["pos"]
        entries += int(((pos != 0) & (pos != pos.shift(1))).sum())
    return {
        "periods": int(len(port)),
        "rows": int(len(sim)),
        "active_rows": int(active.sum()),
        "coverage": float(active.mean()) if len(sim) else 0.0,
        "entries": entries,
        "hit_rate": float((act > 0).mean()) if len(act) else float("nan"),
        "ev_bps": float(act.mean() * 1e4) if len(act) else float("nan"),
        "ev_ci90_bps": (float(lo * 1e4), float(hi * 1e4)) if np.isfinite(lo) else (float("nan"), float("nan")),
        "ev_pvalue": p,
        "total_logret": float(port.sum()),
        "ann_return": float(np.expm1(port.mean() * periods_per_year)) if len(port) else 0.0,
        "sharpe_ann": float(M.sharpe(port) * np.sqrt(periods_per_year)),
        "sortino_ann": float(M.sortino(port) * np.sqrt(periods_per_year)),
        "max_drawdown": M.max_drawdown(port),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf") if gains > 0 else float("nan"),
        "cost_share": float(sim["cost"].sum() / max(sim["gross"].abs().sum(), 1e-12)),
    }


def classification_stats(pred: pd.DataFrame, base_rate: float | None = None) -> dict:
    y, p = pred["y"].to_numpy(), pred["p"].to_numpy()
    base = base_rate if base_rate is not None else float(np.mean(y))
    slope, intercept = M.calibration_slope(y, p)
    return {
        "n": int(len(y)),
        "auc": M.auc(y, p),
        "log_loss": M.log_loss(y, p),
        "log_loss_base": M.log_loss(y, np.full(len(y), base)),
        "brier": M.brier(y, p),
        "ece": M.ece(y, p),
        "cal_slope": slope,
        "cal_intercept": intercept,
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()) if len(y) else float("nan"),
    }


def simulate_panel(pred: pd.DataFrame, cost: CostModel, horizon_min: int) -> pd.DataFrame:
    """Simulate each symbol separately; rows stay aligned with ``pred`` (same order)."""
    n = len(pred)
    cols = {k: np.zeros(n) for k in ("pos", "gross", "cost", "net")}
    syms = pred["sym"].to_numpy() if "sym" in pred else np.full(n, "?", dtype=object)
    for sym in pd.unique(syms):
        rows = np.where(syms == sym)[0]
        g = pred.iloc[rows]
        s = simulate(g["pos"], g["fwd_ret"], sym, cost, horizon_min)
        for k in cols:
            cols[k][rows] = s[k].to_numpy()
    return pd.DataFrame({**cols, "sym": syms}, index=pred.index)


def portfolio_returns(sim: pd.DataFrame) -> pd.Series:
    """Equal-weight across symbols per decision time (average of per-symbol net returns)."""
    return sim.groupby(level=0)["net"].mean()


def evaluate(ds: Dataset, pred: pd.DataFrame, cost: CostModel, n_boot: int = 1000) -> dict:
    """Full report for one prediction set: classification + trading + per-regime + per-year."""
    sim = simulate_panel(pred, cost, ds.horizon_min)
    rep = {
        "symbol": ds.symbol,
        "horizon_min": ds.horizon_min,
        "cost_model": cost.name,
        "classification": classification_stats(pred),
        "trading": trading_stats(sim, ds.horizon_min, n_boot),
        "abstain_folds": int(pred.groupby("fold")["margin"].first().isna().sum()) if len(pred) else 0,
        "folds": int(pred["fold"].nunique()) if len(pred) else 0,
    }
    by_year = {}
    for year, rows in sim.reset_index(drop=True).groupby(sim.index.year).groups.items():
        s = sim.iloc[np.asarray(rows)]
        act = s["net"][s["pos"] != 0]
        pr = pred.iloc[np.asarray(rows)]
        by_year[int(year)] = {"active": int(len(act)), "net_logret": float(s["net"].sum()),
                              "ev_bps": float(act.mean() * 1e4) if len(act) else float("nan"),
                              "auc": M.auc(pr["y"], pr["p"]), "n": int(len(pr))}
    rep["by_year"] = by_year
    reg_cols = [c for c in pred.columns if c.startswith("regime_")]
    if reg_cols:
        by_regime = {}
        for col in reg_cols:
            for val, rows in sim.reset_index(drop=True).groupby(pred[col].to_numpy()).groups.items():
                s = sim.iloc[np.asarray(rows)]
                act = s["net"][s["pos"] != 0]
                by_regime[f"{col[7:]}={val}"] = {"periods": int(len(s)), "active": int(len(act)),
                                            "ev_bps": float(act.mean() * 1e4) if len(act) else float("nan"),
                                            "net_logret": float(s["net"].sum())}
        rep["by_regime"] = by_regime
    return rep, sim
