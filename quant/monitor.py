"""Signal health: detect when a previously demonstrated edge has disappeared.

Each signal carries a reference expectation (mean and std of its per-trade net
return, measured out-of-sample when it was validated). Live (or walk-forward)
outcomes are fed in one by one:

* CUSUM (Page 1954), one-sided, for a downward shift of the mean from the
  reference ``mu0`` to ``mu1`` (default: zero edge). Statistic on standardized
  outcomes z = (x - mu0) / sigma with drift k = (mu0 - mu1) / (2 sigma):
  S_t = max(0, S_{t-1} - z_t - k); alarm when S_t > h, with h set (Siegmund's
  approximation) for an in-control average run length of ``arl0`` trades.
  Trading edges are small relative to noise (often ~0.2 sigma per trade), so
  detecting a lost edge honestly takes on the order of 100 trades; a sharp
  break (mean turning clearly negative) is caught in a few dozen.
* Rolling evidence: over the last ``window`` trades, if mean + z*se < 0 the
  signal is confidently losing.

State machine: ACTIVE -> (alarm) -> DISABLED. A disabled signal is only
re-enabled by a fresh validation (``revalidate``), never by waiting.

Defaults are textbook values chosen before looking at our data, so the monitor
itself is not fitted to the history it is evaluated on.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def siegmund_arl(k: float, h: float) -> float:
    """Average run length of a one-sided CUSUM with drift k and threshold h (Siegmund approximation)."""
    if k <= 0:
        return float("inf")
    a = 2 * k * (h + 1.166)
    return (np.exp(a) - a - 1) / (2 * k * k)


def cusum_threshold(k: float, arl0: float = 1000.0) -> float:
    """Smallest h whose in-control average run length is >= arl0 trades."""
    if k <= 0:
        return 10.0
    lo, hi = 0.5, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if siegmund_arl(k, mid) < arl0:
            lo = mid
        else:
            hi = mid
    return hi


@dataclass
class SignalHealth:
    name: str
    mu0: float                  # expected net return per trade at validation
    sigma: float                # std of net return per trade at validation
    mu1: float = 0.0            # "edge is gone" level
    h: float | None = None      # CUSUM threshold (std units); default: ARL0 = arl0 trades
    arl0: float = 3000.0        # in-control average run length -> ~1 false alarm per arl0 trades
    window: int = 50            # rolling-evidence window (trades)
    z: float = 2.0
    min_trades_for_rolling: int = 30
    state: str = "ACTIVE"
    cusum: float = 0.0
    n: int = 0
    reason: str = ""
    recent: deque = field(default_factory=lambda: deque(maxlen=50))
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.recent = deque(maxlen=self.window)
        self._set_reference(self.mu0, self.sigma)

    def _set_reference(self, mu0: float, sigma: float) -> None:
        self.mu0, self.sigma = mu0, sigma
        self.k = max((mu0 - self.mu1) / (2 * sigma), 1e-3) if sigma > 0 else 1e-3
        self._h = self.h if self.h is not None else cusum_threshold(self.k, self.arl0)

    @property
    def active(self) -> bool:
        return self.state == "ACTIVE"

    def update(self, x: float, ts=None) -> str:
        """Feed one realised net trade return; returns the new state."""
        if not np.isfinite(x):
            return self.state
        self.n += 1
        self.recent.append(x)
        zt = (x - self.mu0) / self.sigma if self.sigma > 0 else 0.0
        self.cusum = max(0.0, self.cusum - zt - self.k)
        if self.state == "ACTIVE":
            if self.cusum > self._h:
                self.state, self.reason = "DISABLED", f"CUSUM {self.cusum:.1f} > {self._h:.1f} after {self.n} trades"
            elif len(self.recent) >= self.min_trades_for_rolling:
                r = np.asarray(self.recent)
                ucb = r.mean() + self.z * r.std(ddof=1) / np.sqrt(len(r))
                if ucb < 0:  # recent trades are confidently losing
                    self.state, self.reason = "DISABLED", (
                        f"rolling mean {r.mean() * 1e4:.1f}bp, upper bound {ucb * 1e4:.1f}bp < 0 "
                        f"over last {len(r)} trades")
        self.history.append((ts, x, self.cusum, self.state))
        return self.state

    def revalidate(self, mu0: float, sigma: float) -> None:
        """A fresh out-of-sample validation passed: reset with the new reference."""
        self._set_reference(mu0, sigma)
        self.cusum, self.state, self.reason = 0.0, "ACTIVE", "revalidated"
        self.recent.clear()

    def snapshot(self) -> dict:
        r = np.asarray(self.recent) if self.recent else np.array([])
        return {
            "signal": self.name, "state": self.state, "reason": self.reason, "trades_seen": self.n,
            "cusum": round(self.cusum, 2), "alarm_at": round(self._h, 2),
            "recent_mean_bps": round(float(r.mean() * 1e4), 2) if len(r) else None,
            "reference_bps": round(self.mu0 * 1e4, 2),
        }


def simulate_monitoring(returns: np.ndarray, mu0: float, sigma: float, **kw) -> dict:
    """Replay a return series through a monitor (no re-enabling) and report when it trips."""
    mon = SignalHealth("replay", mu0, sigma, **kw)
    trip = None
    for i, x in enumerate(returns):
        if mon.update(x) == "DISABLED" and trip is None:
            trip = i
    return {"tripped_at": trip, "reason": mon.reason, "n": len(returns)}
