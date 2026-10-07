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
    ev_p5_bps: float | None = None  # 5th percentile of the day-clustered bootstrap of that mean
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
    view: str | None = None          # directional lean (UP/DOWN) whether or not it is tradeable
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
    horizon_min = 1440
    cost_name = "spot_taker"

    def __init__(self, evidence: Evidence, target_vol: float = 0.5, fetch_daily: bool = True):
        self.evidence = evidence
        self.target_vol = target_vol
        self.fetch_daily = fetch_daily
        self.health = SignalHealth(self.name, mu0=0.0005, sigma=0.03)

    def evaluate(self, symbol: str, m1: pd.DataFrame, now: pd.Timestamp) -> Signal:
        from .data import resample
        from .studies.trend import ensemble, ewma_vol

        d = resample(m1, "1D")["close"]
        d = d[d.index <= now.floor("D")]           # only completed UTC days
        if len(d.dropna()) < 200 and self.fetch_daily:
            try:                                    # not enough local history: one API call
                from .live import fetch_daily

                d = fetch_daily(symbol)["close"]
                d = d[d.index <= now.floor("D")]
            except Exception:
                pass
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
        if self.evidence.status not in ("VALIDATED", "RISK_OVERLAY"):
            action = "NO TRADE"
            reasons.append("trend rule did not pass its pre-registered holdout criteria -> informational only")
        elif not self.health.active:
            action = "NO TRADE"
            reasons.append(f"disabled by health monitor: {self.health.reason}")
        return Signal(symbol, self.name, action, self.horizon, str(d.index[-1]), exposure=round(exposure, 3),
                      view="UP" if s >= 0.5 else "DOWN", reasons=reasons, evidence=asdict(self.evidence), health=self.health.snapshot(),
                      next_decision=next_dec)


class ModelSpecialist:
    """A trained, calibrated classifier with the decision rule it was validated with.

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
        self.cost_name = artifact["cost"]
        ev = self.evidence
        self.health = SignalHealth(self.name, mu0=(ev.ev_bps or 0) / 1e4, sigma=artifact.get("trade_sigma", 0.01))

    ENTRY_WINDOW = pd.Timedelta(minutes=2)   # research entered 1 minute after each decision time

    @staticmethod
    def refresh_fng(max_age_h: float = 12.0) -> None:
        """Keep the Fear & Greed file fresh for live use (stale values become NaN -> NO TRADE)."""
        import time as _time

        from .data import FNG_PATH, fetch_fng

        try:
            if not FNG_PATH.exists() or _time.time() - FNG_PATH.stat().st_mtime > max_age_h * 3600:
                fetch_fng()
        except Exception:
            pass

    def evaluate(self, symbol: str, m1: pd.DataFrame, now: pd.Timestamp, others: dict | None = None) -> Signal:
        from .experiment import ex_ante_sigma
        from .backtest import expected_edge

        if symbol not in self.a.get("symbols", [symbol]):
            return Signal(symbol, self.name, "NO TRADE", self.horizon, str(now), evidence=asdict(self.evidence),
                          reasons=["outside this specialist's tested universe"])
        if "fng" in self.a["families"]:
            self.refresh_fng()

        grid = pd.Timedelta(minutes=self.horizon_min)
        t = now.floor(f"{self.horizon_min}min")
        # If the decision bar has not arrived yet, the latest decision we can compute is the
        # previous grid point (never an off-grid time); the entry window then blocks it.
        t = min(t, m1["close"].last_valid_index().floor(f"{self.horizon_min}min"))
        # Compute the full trailing decision grid exactly as in training (some features,
        # e.g. the 1-year volatility percentile, are defined over the grid's history),
        # then keep the last row.
        start = max(m1.index[0], t - pd.Timedelta(days=400)).ceil(f"{self.horizon_min}min")
        times = pd.date_range(start, t, freq=f"{self.horizon_min}min", tz="UTC")
        X = F.build(m1, times, symbol, others=others, families=self.a["families"]).iloc[[-1]]
        X = X.reindex(columns=self.a["columns"])
        notes = []
        if "xa" in self.a["families"]:
            from .data import KLINE_DIR, UNIVERSE

            expected = self.a.get("xa_universe") or [u for u in UNIVERSE if (KLINE_DIR / f"{u}.parquet").exists()]
            missing_xa = [u for u in expected if u != symbol and u not in (others or {})]
            if missing_xa:
                X.loc[:, [c for c in X.columns if c.startswith("xa__")]] = np.nan   # incomplete market inputs
                notes.append(f"cross-asset inputs missing for {len(missing_xa)} coin(s)")
        if self.horizon_min > 1440 and "regime" in self.a["families"]:
            notes.append("live approximation: the 1-year volatility percentile uses the ~400 days held live")
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
        reasons = notes + [f"calibrated P(up) = {p:.3f}", f"expected move size {sig * 1e4:.0f}bp over {self.horizon}",
                           f"expected edge {edge * 1e4:+.1f}bp vs round-trip cost {rt * 1e4:.1f}bp"]
        action = "NO TRADE"
        if self.evidence.status != "VALIDATED":
            reasons.append("specialist did not pass out-of-sample validation after costs -> informational only")
        elif not self.health.active:
            reasons.append(f"specialist disabled by health monitor: {self.health.reason}")
        elif margin is None:
            reasons.append("no profitable threshold found in validation")
        else:
            # Apply exactly the rule the margin was selected for in validation.
            from .backtest import positions_from_ev, positions_from_prob

            if self.a.get("decision", "ev") == "prob":
                pos = int(positions_from_prob([p], margin, self.cost.allow_short)[0])
                hurdle = f"|P(up) - 0.5| >= {margin:.3f}"
            else:
                pos = int(positions_from_ev(np.array([p]), np.array([sig]), np.array([rt]), margin,
                                            self.cost.allow_short)[0])
                hurdle = f"edge > cost + margin ({(rt + margin) * 1e4:.1f}bp)"
            if pos == 0:
                reasons.append(f"decision rule not met: {hurdle}")
            elif now - t > self.ENTRY_WINDOW:
                reasons.append(f"decision at {t:%Y-%m-%d %H:%M} UTC was {'LONG' if pos > 0 else 'SHORT'}, but its entry "
                               f"window has passed (research entered 1 minute after the decision); next decision {nxt}")
            else:
                action = "LONG" if pos > 0 else "SHORT"
        return Signal(action=action, confidence=round(conf, 4), expected_edge_bps=round(edge * 1e4, 2),
                      cost_bps=round(rt * 1e4, 2), view="UP" if p >= 0.5 else "DOWN", reasons=reasons, **base)


class EventSpecialist:
    """Sparse event rule: flow-driven capitulation reversal (buy after a forced, one-sided sell-off).

    Fires when the last closed 1m bar completes an event: a ``W``-minute move of at
    least ``k`` ex-ante sigmas, in the direction of the taker imbalance, with that
    imbalance in the top 20% of the trailing 30 days. Research entered on the next
    minute, so an event older than ``MAX_EVENT_AGE`` is stale and never traded.
    """

    MAX_EVENT_AGE = pd.Timedelta(seconds=90)
    LOOKBACK = pd.Timedelta(days=40)       # sigma (1-day halflife) + 30-day imbalance percentile

    def __init__(self, artifact: dict):
        self.a = artifact
        self.name = artifact["name"]
        self.h = self.horizon_min = int(artifact["h"])
        self.horizon = f"{self.h // 60}h" if self.h >= 60 else f"{self.h}m"
        self.cost_name = artifact["cost"]
        self.evidence = Evidence(**artifact["evidence"])
        self.cost = COST_MODELS[artifact["cost"]]
        ev = self.evidence
        self.health = SignalHealth(self.name, mu0=(ev.ev_bps or 0) / 1e4, sigma=artifact.get("trade_sigma", 0.01))

    def evaluate(self, symbol: str, m1: pd.DataFrame, now: pd.Timestamp) -> Signal:
        from .events import Bars, select
        from .studies.event_hypotheses import h_flow_driven_reversal

        base = dict(symbol=symbol, specialist=self.name, horizon=self.horizon,
                    evidence=asdict(self.evidence), health=self.health.snapshot())
        if symbol not in self.a["symbols"]:
            return Signal(action="NO TRADE", as_of=str(now), reasons=["outside this specialist's tested universe"], **base)
        tail = m1[m1.index > m1.index[-1] - self.LOOKBACK]
        b = Bars(tail)
        sig = h_flow_driven_reversal(b, int(self.a["W"]), float(self.a["k"]))
        kept = select(sig, cooldown=self.h)            # cooldown over both directions, as in research
        if not self.cost.allow_short:
            kept = np.asarray([i for i in kept if sig[i] > 0], dtype=np.int64)
        last_t = tail.index[-1]
        recent = [i for i in kept if tail.index[i] > last_t - pd.Timedelta(minutes=self.h)]
        reasons = []
        view = None
        action = "NO TRADE"
        if not len(recent):
            reasons.append("no capitulation event in the last holding window")
        else:
            i = recent[-1]
            age = now - tail.index[i]
            view = "UP" if sig[i] > 0 else "DOWN"
            reasons.append(f"capitulation event at {tail.index[i]:%H:%M} UTC ({age.total_seconds() / 60:.0f} min ago): "
                           f"{self.a['W']}m move >= {self.a['k']:g} sigma with one-sided taker flow")
            if i != len(tail) - 1 or age > self.MAX_EVENT_AGE:
                reasons.append("event is stale: research entered within 1 minute -> not tradeable now")
            elif self.evidence.status != "VALIDATED":
                reasons.append("specialist did not pass the pre-registered validation -> informational only")
            elif not self.health.active:
                reasons.append(f"specialist disabled by health monitor: {self.health.reason}")
            else:
                action = "LONG" if sig[i] > 0 else "SHORT"
        rt = self.cost.round_trip(symbol) + self.cost.holding(self.h)
        return Signal(action=action, as_of=str(last_t), view=view, cost_bps=round(rt * 1e4, 2),
                      expected_edge_bps=self.evidence.ev_bps if view else None, reasons=reasons,
                      next_decision=(last_t + pd.Timedelta(minutes=1)).isoformat(), **base)


# ---------------------------------------------------------------- the engine

class Engine:
    def __init__(self, symbols, specialists, store: LiveStore | None = None, use_book: bool = True, paper=None):
        self.symbols = list(symbols)
        self.specialists = specialists
        self.store = store or LiveStore()
        self.use_book = use_book
        self.paper = paper                  # quant.paper.PaperLedger: feeds realised results to each monitor
        if paper is not None:
            paper.load_health(specialists)

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
        if self.paper is not None:          # resolve matured paper trades first, so health is current
            self.paper.resolve({s: f for s, f in frames.items() if isinstance(f, pd.DataFrame)}, self.specialists)
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
                universe = getattr(spec, "a", {}).get("symbols")
                if universe is not None and sym not in universe:
                    continue                       # never applied outside the universe it was tested on
                try:
                    if isinstance(spec, ModelSpecialist):
                        sig = spec.evaluate(sym, m1, now, others=closes)
                    else:
                        sig = spec.evaluate(sym, m1, now)
                except Exception as exc:     # one broken specialist must not take the others down
                    sig = Signal(sym, spec.name, "NO TRADE", spec.horizon, str(now),
                                 evidence=asdict(spec.evidence), reasons=[f"specialist error: {exc}"])
                sig.regime = regime
                veto = None
                if not health["ok"]:
                    veto = health["problems"][0]
                    sig.action, sig.reasons = "NO TRADE", health["problems"] + sig.reasons
                elif book is not None and not book["ok"] and sig.action in ("LONG", "SHORT"):
                    veto = book["problems"][0]
                    sig.action, sig.reasons = "NO TRADE", book["problems"] + sig.reasons
                d = sig.to_dict()
                d["veto"] = veto
                if (self.paper is not None and d["action"] in ("LONG", "SHORT")
                        and spec.evidence.status == "VALIDATED"):
                    self.paper.record(d, spec.horizon_min, spec.cost_name)
                d["data_health"] = health
                d["book"] = book
                out.append(d)
        return out


def consensus(signals: list[dict]) -> list[dict]:
    """One verdict per symbol, reconciling the specialists.

    * Insufficient information (stale or gappy data) -> NO TRADE.
    * Only a VALIDATED, healthy specialist can produce a trade (its action is
      already gated); a RISK_OVERLAY's exposure is allocation guidance, never a trade.
    * Validated specialists pointing opposite ways -> NO TRADE (conflict).
    * ``agreement`` compares the directional views of every specialist, tradeable
      or not. It is context only: research found agreement with the daily trend did
      not make the intraday model more accurate (calibration_regimes.json).
    """
    by_sym: dict[str, list[dict]] = {}
    for s in signals:
        by_sym.setdefault(s["symbol"], []).append(s)
    out = []
    for sym, sigs in by_sym.items():
        health = next((s.get("data_health") for s in sigs if s.get("data_health")), None) or {}
        trades = [s for s in sigs if s["action"] in ("LONG", "SHORT")
                  and s.get("evidence", {}).get("status") == "VALIDATED"]
        views = {s["specialist"]: s.get("view") for s in sigs if s.get("view")}
        ups, downs = sum(v == "UP" for v in views.values()), sum(v == "DOWN" for v in views.values())
        agreement = ("no views" if not views else "single view" if len(views) == 1
                     else "agree" if ups == 0 or downs == 0 else "conflict")
        overlay = next((s for s in sigs if s.get("exposure") is not None), None)
        if not health.get("ok", False):
            verdict, why = "NO TRADE", "insufficient information: " + "; ".join(health.get("problems", ["data unavailable"]))
        elif {s["action"] for s in trades} == {"LONG", "SHORT"}:
            verdict, why = "NO TRADE", "validated specialists conflict"
        elif trades:
            best = max(trades, key=lambda s: s.get("expected_edge_bps") or 0)
            verdict = best["action"]
            why = f"{best['specialist']} ({best['horizon']}): edge {best.get('expected_edge_bps')}bp vs cost {best.get('cost_bps')}bp"
        else:
            verdict, why = "NO TRADE", "no validated edge clears costs right now"
        out.append({"symbol": sym, "verdict": verdict, "why": why, "agreement": agreement, "views": views,
                    "exposure": None if overlay is None else overlay.get("exposure"),
                    "regime": sigs[0].get("regime", {})})
    return out


def regime_snapshot(m1: pd.DataFrame) -> dict:
    """Ex-ante regime description (same definitions as the research regimes)."""
    r = np.log(m1["close"]).ffill().diff()
    rv_1d = float(np.sqrt((r.iloc[-1440:] ** 2).sum()))
    # Rolling 1440-minute windows sampled once a day, ending at the latest bar, so the
    # current value and its reference set are all complete 24-hour windows.
    rolling = np.sqrt((r ** 2).rolling(1440, min_periods=1300).sum())
    ref = rolling.iloc[::-1].iloc[::1440].iloc[1:367].dropna()
    pct = float((ref < rv_1d).mean()) if len(ref) > 60 else None
    closes = m1["close"].dropna()
    ret_7d = (float(np.log(closes.iloc[-1] / closes.asof(closes.index[-1] - pd.Timedelta(days=7))))
              if len(closes) and closes.index[-1] - closes.index[0] > pd.Timedelta(days=7) else None)
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
        art = joblib.load(path)
        specs.append(EventSpecialist(art) if art.get("kind") == "event" else ModelSpecialist(art))
    return specs
