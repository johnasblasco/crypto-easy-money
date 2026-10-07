"""Execution cost models, in log-return units per side.

Defaults are deliberately on the conservative side of retail Binance costs:

* spot: 0.10% taker fee per side, plus half-spread and slippage
* USDT-M perpetual: 0.05% taker fee per side, plus half-spread and slippage,
  plus a funding charge per 8h held by LONG positions (historical average
  funding is positive, so longs usually pay; shorts' funding income is not
  credited)
* perp maker: an UPPER BOUND only -- it assumes every limit order fills at the
  next minute's VWAP, ignoring non-fills and adverse selection. Never used to
  validate a signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Estimated half-spread + slippage per side for a small order (bps). BTC/ETH
# books are ~0.1bp wide; alts are wider and thinner. These are padded.
DEFAULT_IMPACT_BPS = {"BTCUSDT": 1.0, "ETHUSDT": 1.0}
OTHER_IMPACT_BPS = 3.0


@dataclass(frozen=True)
class CostModel:
    name: str
    fee_bps: float              # per side
    impact_bps: dict = field(default_factory=lambda: dict(DEFAULT_IMPACT_BPS))
    other_impact_bps: float = OTHER_IMPACT_BPS
    funding_bps_per_8h: float = 0.0
    allow_short: bool = False

    def one_side(self, symbol: str) -> float:
        """Cost of one fill (entry or exit) as a log-return fraction."""
        return (self.fee_bps + self.impact_bps.get(symbol, self.other_impact_bps)) / 1e4

    def round_trip(self, symbol: str) -> float:
        return 2 * self.one_side(symbol)

    def holding(self, minutes: float, side: int = 1) -> float:
        """Carry cost (funding) for holding a position ``minutes``.

        Historically perp funding is mostly positive (longs pay shorts). Longs are
        charged the assumed rate; shorts are charged nothing (conservative: in
        practice they usually receive funding, which we do not credit).
        """
        if side < 0:
            return 0.0
        return self.funding_bps_per_8h / 1e4 * minutes / 480.0


SPOT = CostModel("spot_taker", fee_bps=10.0, allow_short=False)
SPOT_BNB = CostModel("spot_taker_bnb", fee_bps=7.5, allow_short=False)
PERP = CostModel("perp_taker", fee_bps=5.0, funding_bps_per_8h=1.0, allow_short=True)
PERP_MAKER = CostModel("perp_maker", fee_bps=2.0, funding_bps_per_8h=1.0, allow_short=True)
ZERO = CostModel("zero_cost", fee_bps=0.0, impact_bps={}, other_impact_bps=0.0, allow_short=True)

COST_MODELS = {m.name: m for m in (SPOT, SPOT_BNB, PERP, PERP_MAKER, ZERO)}
