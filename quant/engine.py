"""Signal intelligence engine: evidence-gated specialists, NO TRADE by default.

Every specialist carries the out-of-sample evidence that justified it (or the
evidence that it is NOT justified) and a live health monitor. The engine only
emits an actionable signal when:

1. the data feeding it is fresh and complete (``live.data_health``),
2. the live order book is liquid enough to trade at the assumed cost
   (``live.book_check``; optional),
3. the specialist is VALIDATED (passed walk-forward + holdout evaluation after
   costs) and its health monitor is ACTIVE (its edge has not decayed live),
4. the specialist's own decision rule fires (e.g. expected edge > cost + margin).

Otherwise the output is NO TRADE with the reasons. Uncertainty is reported as
information, not hidden.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .costs import COST_MODELS
from .data import ROOT
from .live import LiveStore, book_check, data_health
from .monitor import SignalHealth

ENGINE_DIR = ROOT / "models" / "engine"


@dataclass
class Evidence:
    """What the research found for a specialist (all out-of-sample, after costs)."""

    status: str                     # VALIDATED | NOT_VALIDATED | RISK_OVERLAY
    summary: str
    period: str = ""
    trades: int = 0
    ev_bps: float | None = None     # mean net return per trade
    ev_ci90_bps: tuple | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    benchmark: str = ""
    notes: list = field(default_factory=list)


@dataclass
class Signal:
    symbol: str
    specialist: str
    action: str                     # LONG | SHORT | FLAT | NO TRADE
    horizon: str
    as_of: str
    confidence: float | None = None  # calibrated P(direction) where the specialist has one
    exposure: float | None = None    # target weight in [0, 1] for allocation specialists
    expected_edge_bps: float | None = None
    cost_bps: float | None = None
    reasons: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    health: dict = field(default_factory=dict)
    regime: dict = field(default_factory=dict)
    next_decision: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- specialists

class TrendSpecialist:
    """Daily trend ensemble (Donchian + past-return), long/flat, volatility-sized.

    Research verdict: improves risk (drawdowns roughly halved, Sharpe up) versus
    buy & hold; timing alpha over a volatility-targeted hold is positive but not
    statistically significant at the portfolio level. It is therefore served as
    a RISK OVERLAY / exposure guide, not as a high-confidence alpha signal.
    """

    name = "daily_trend"
    horizon = "1 day (re-evaluated 00:00 UTC)"

    def __init__(self, evidence: Evidence, target_vol: float = 0.5):
        self.evidence = evidence
        self.target_vol = target_vol
        self.health = SignalHealth(self.name, mu0=0.0005, sigma=0.03)

    def evaluate(self, symbol: str, m1: pd.DataFrame, now: pd.Timestamp) -> Signal:
        from .data import resample
        from .studies.trend import ensemble, ewma_vol

        d = resample(m1, "1D")["close"]
        d = d[d.index <= now.floor("D")]           # only completed UTC days
        S = ensemble(d)
        vol = ewma_vol(d)
        s, v = S.iloc[-1], vol.iloc[-1]
        next_dec = (now.floor("D") + pd.Timedelta("1D")).isoformat()
        if not np.isfinite(s) or not np.isfinite(v):
            return Signal(symbol, self.name, "NO TRADE", self.horizon, str(d.index[-1]),
                          reasons=["not enough daily history for the trend ensemble"],
                          evidence=asdict(self.evidence), next_decision=next_dec)
        exposure = float(s * min(1.0, self.target_vol / v))
        action = "LONG" if exposure > 0 else "FLAT"
        comps = int(round(s * 9))
        reasons = [f"{comps}/9 trend components are long", f"annualised volatility {v:.0%} -> size cap {min(1.0, self.target_vol / v):.2f}"]
        return Signal(symbol, self.name, action, self.horizon, str(d.index[-1]), exposure=round(exposure, 3),
                      reasons=reasons, evidence=asdict(self.evidence), health=self.health.snapshot(),
                      next_decision=next_dec)


class ModelSpecialist:
    """A trained, calibrated classifier with an EV-vs-cost decision rule.

    Loaded from an artifact produced by ``quant.train_engine``. If the artifact
    says the configuration did not validate, it can still describe the market
    but always returns NO TRADE.
    """

    def __init__(self, artifact: dict):
        self.a = artifact
        self.name = artifact["name"]
        self.horizon_min = artifact["horizon_min"]
        self.horizon = f"{self.horizon_min // 60}h" if self.horizon_min >= 60 else f"{self.horizon_min}m"
        self.evidence = Evidence(**artifact["evidence"])
        self.cost = COST_MODELS[artifact["cost"]]
        ev = self.evidence
        self.health = SignalHealth(self.name, mu0=(ev.ev_bps or 0) / 1e4, sigma=artifact.get("trade_sigma", 0.01))

    def evaluate(self, symbol: str, m1: pd.DataFrame, now: pd.Timestamp, others: dict | None = None) -> Signal:
        from .experiment import ex_ante_sigma
        from .backtest import expected_edge

        grid = pd.Timedelta(minutes=self.horizon_min)
        t = now.floor(f"{self.horizon_min}min")
        t = min(t, m1["close"].last_valid_index().floor("min"))
        # Compute the full trailing decision grid exactly as in training (some features,
        # e.g. the 1-year volatility percentile, are defined over the grid's history),
        # then keep the last row.
        start = max(m1.index[0], t - pd.Timedelta(days=400)).ceil(f"{self.horizon_min}min")
        times = pd.date_range(start, t, freq=f"{self.horizon_min}min", tz="UTC")
        X = F.build(m1, times, symbol, others=others, families=self.a["families"]).iloc[[-1]]
        X = X.reindex(columns=self.a["columns"])
        sig = ex_ante_sigma(m1, times[-1:], self.horizon_min).to_numpy()[0]
        nxt = (t + grid).isoformat()
        base = dict(symbol=symbol, specialist=self.name, horizon=self.horizon, as_of=t.isoformat(),
                    evidence=asdict(self.evidence), health=self.health.snapshot(), next_decision=nxt)
        if X.isna().any(axis=1).iloc[0] or not np.isfinite(sig):
            missing = [c for c in X.columns if pd.isna(X.iloc[0][c])][:5]
            return Signal(action="NO TRADE", reasons=[f"features unavailable: {missing}"], **base)
        p_raw = self.a["model"].predict_proba(X.to_numpy(float))[:, 1]
        p = float(self.a["calibrator"].transform(p_raw)[0])
        rt = self.cost.round_trip(symbol) + self.cost.holding(self.horizon_min)
        edge = float(expected_edge(p, sig))
        margin = self.a.get("margin")
        direction = "LONG" if p >= 0.5 else "SHORT"
        conf = p if p >= 0.5 else 1 - p
        reasons = [f"calibrated P(up) = {p:.3f}", f"expected move size {sig * 1e4:.0f}bp over {self.horizon}",
                   f"expected edge {edge * 1e4:+.1f}bp vs round-trip cost {rt * 1e4:.1f}bp"]
        action = "NO TRADE"
        if self.evidence.status != "VALIDATED":
            reasons.append("specialist did not pass out-of-sample validation after costs -> informational only")
        elif not self.health.active:
            reasons.append(f"specialist disabled by health monitor: {self.health.reason}")
        elif margin is None:
            reasons.append("no profitable threshold found in validation")
        elif abs(edge) > rt + margin and (direction == "LONG" or self.cost.allow_short):
            action = direction
        else:
            reasons.append(f"edge does not clear cost + margin ({(rt + margin) * 1e4:.1f}bp)")
        return Signal(action=action, confidence=round(conf, 4), expected_edge_bps=round(edge * 1e4, 2),
                      cost_bps=round(rt * 1e4, 2), reasons=reasons, **base)


# ---------------------------------------------------------------- the engine

class Engine:
    def __init__(self, symbols, specialists, store: LiveStore | None = None, use_book: bool = True):
        self.symbols = list(symbols)
        self.specialists = specialists
        self.store = store or LiveStore()
        self.use_book = use_book

    def run(self, now: pd.Timestamp | None = None, sync: bool = True) -> list[dict]:
        now = now or pd.Timestamp.now(tz="UTC")
        others = None
        out = []
        frames = {}
        for sym in self.symbols:
            try:
                if sync:
                    self.store.sync(sym)
                frames[sym] = self.store.frame(sym)
            except Exception as exc:  # network etc.
                frames[sym] = exc
        closes = {s: np.log(f["close"]).ffill() for s, f in frames.items() if isinstance(f, pd.DataFrame)}
        for sym in self.symbols:
            m1 = frames[sym]
            if not isinstance(m1, pd.DataFrame):
                out.append(Signal(sym, "engine", "NO TRADE", "-", str(now), reasons=[f"data error: {m1}"]).to_dict())
                continue
            health = data_health(m1, now)
            book = None
            if self.use_book:
                try:
                    book = book_check(sym)
                except Exception as exc:
                    book = {"ok": False, "problems": [f"order book unavailable: {exc}"]}
            regime = regime_snapshot(m1)
            for spec in self.specialists:
                if isinstance(spec, ModelSpecialist):
                    sig = spec.evaluate(sym, m1, now, others=closes)
                else:
                    sig = spec.evaluate(sym, m1, now)
                sig.regime = regime
                if not health["ok"]:
                    sig.action, sig.reasons = "NO TRADE", health["problems"] + sig.reasons
                elif book is not None and not book["ok"] and sig.action in ("LONG", "SHORT"):
                    sig.action, sig.reasons = "NO TRADE", book["problems"] + sig.reasons
                d = sig.to_dict()
                d["data_health"] = health
                d["book"] = book
                out.append(d)
        return out


def regime_snapshot(m1: pd.DataFrame) -> dict:
    """Ex-ante regime description (same definitions as the research regimes)."""
    r = np.log(m1["close"]).ffill().diff()
    rv_1d = float(np.sqrt((r.iloc[-1440:] ** 2).sum()))
    daily = np.sqrt((r ** 2).resample("1D").sum()).dropna()
    pct = float((daily.iloc[-366:-1] < daily.iloc[-1]).mean()) if len(daily) > 60 else None
    ret_7d = float(np.log(m1["close"].dropna().iloc[-1] / m1["close"].dropna().iloc[-10080])) if len(m1) > 10080 else None
    return {
        "vol_1d_pct": round(rv_1d * 100, 2),
        "vol_percentile_1y": None if pct is None else round(pct, 2),
        "vol_regime": None if pct is None else ("high" if pct > 2 / 3 else "low" if pct < 1 / 3 else "mid"),
        "trend_7d": None if ret_7d is None else ("up" if ret_7d > 0 else "down"),
        "ret_7d_pct": None if ret_7d is None else round(ret_7d * 100, 2),
    }


def load_specialists(engine_dir: Path = ENGINE_DIR) -> list:
    """Trend specialist + every model artifact saved by quant.train_engine."""
    import joblib

    specs = []
    trend_ev = engine_dir / "trend_evidence.json"
    if trend_ev.exists():
        specs.append(TrendSpecialist(Evidence(**json.loads(trend_ev.read_text()))))
    for path in sorted(engine_dir.glob("*.joblib")):
        specs.append(ModelSpecialist(joblib.load(path)))
    return specs
