"""Paper-trade ledger: closes the loop between live signals and the degradation monitor.

Every actionable LONG/SHORT from a specialist is recorded once (keyed by symbol,
specialist and decision time). After its horizon has passed, the trade is
resolved exactly like research labels: enter at the VWAP of the bar closing one
minute after the decision, exit at the VWAP ``horizon`` later, minus the
specialist's round-trip cost and long-side funding. The realised net return is
fed to the specialist's ``SignalHealth`` (CUSUM + rolling upper-bound check),
whose state is persisted so a disabled specialist stays disabled across
restarts. A new validation (different evidence) starts a fresh monitor.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT
from .labels import fill_prices

PAPER_DIR = ROOT / "data" / "engine"


def _evidence_key(spec) -> str:
    return hashlib.sha1(json.dumps(spec.evidence.__dict__, sort_keys=True, default=str).encode()).hexdigest()[:12]


class PaperLedger:
    def __init__(self, trades_path: Path = PAPER_DIR / "paper_trades.jsonl",
                 health_path: Path = PAPER_DIR / "health.json", latency_min: int = 1):
        self.trades_path, self.health_path, self.latency = Path(trades_path), Path(health_path), latency_min
        self.trades: list[dict] = []
        if self.trades_path.exists():
            self.trades = [json.loads(line) for line in self.trades_path.read_text().splitlines() if line.strip()]

    # ------------------------------------------------------------ persistence
    def _save_trades(self) -> None:
        self.trades_path.parent.mkdir(parents=True, exist_ok=True)
        self.trades_path.write_text("".join(json.dumps(t) + "\n" for t in self.trades))

    def load_health(self, specialists) -> None:
        """Restore each specialist's monitor if it was saved under the same evidence."""
        if not self.health_path.exists():
            return
        saved = json.loads(self.health_path.read_text())
        for spec in specialists:
            st = saved.get(spec.name)
            if not st or st.get("evidence_key") != _evidence_key(spec):
                continue                                    # new validation -> fresh monitor
            h = spec.health
            h.state, h.cusum, h.n, h.reason = st["state"], st["cusum"], st["n"], st["reason"]
            h.recent = deque(st["recent"], maxlen=h.window)

    def save_health(self, specialists) -> None:
        self.health_path.parent.mkdir(parents=True, exist_ok=True)
        out = {spec.name: {"evidence_key": _evidence_key(spec), "state": spec.health.state,
                           "cusum": spec.health.cusum, "n": spec.health.n, "reason": spec.health.reason,
                           "recent": list(spec.health.recent)} for spec in specialists}
        self.health_path.write_text(json.dumps(out, indent=2))

    # ---------------------------------------------------------------- trading
    def record(self, sig: dict, horizon_min: int, cost_name: str) -> bool:
        """Record an actionable signal once. Returns True if it is new."""
        if sig.get("action") not in ("LONG", "SHORT"):
            return False
        key = f"{sig['symbol']}|{sig['specialist']}|{sig['as_of']}"
        if any(t["key"] == key for t in self.trades):
            return False
        t0 = pd.Timestamp(sig["as_of"])
        entry = t0 + pd.Timedelta(minutes=self.latency)
        self.trades.append({"key": key, "symbol": sig["symbol"], "specialist": sig["specialist"],
                            "side": 1 if sig["action"] == "LONG" else -1, "decision": t0.isoformat(),
                            "entry": entry.isoformat(), "exit": (entry + pd.Timedelta(minutes=horizon_min)).isoformat(),
                            "cost": cost_name, "status": "open", "net": None})
        self._save_trades()
        return True

    def resolve(self, frames: dict, specialists) -> int:
        """Resolve open trades whose exit bar exists; feed each result to its specialist's monitor."""
        from .costs import COST_MODELS

        by_name = {s.name: s for s in specialists}
        done = 0
        for t in self.trades:
            if t["status"] != "open":
                continue
            m1 = frames.get(t["symbol"])
            if not isinstance(m1, pd.DataFrame) or pd.Timestamp(t["exit"]) > m1.index[-1]:
                continue
            lf = fill_prices(m1)
            entry, exit_ = pd.Timestamp(t["entry"]), pd.Timestamp(t["exit"])
            if entry not in lf.index or exit_ not in lf.index or m1.loc[entry:exit_, "gap"].any():
                t["status"], t["net"] = "void (missing bars)", None
                continue
            cost = COST_MODELS[t["cost"]]
            minutes = (exit_ - entry).total_seconds() / 60
            net = t["side"] * float(lf[exit_] - lf[entry]) - cost.round_trip(t["symbol"]) - cost.holding(minutes, t["side"])
            t["status"], t["net"] = "closed", net
            spec = by_name.get(t["specialist"])
            if spec is not None:
                spec.health.update(net, ts=t["exit"])
            done += 1
        if done:
            self._save_trades()
            self.save_health(specialists)
        return done

    def summary(self) -> dict:
        closed = [t["net"] for t in self.trades if t["status"] == "closed"]
        return {"open": sum(t["status"] == "open" for t in self.trades), "closed": len(closed),
                "mean_net_bps": round(float(np.mean(closed)) * 1e4, 2) if closed else None}
