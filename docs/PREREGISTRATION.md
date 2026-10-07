# Holdout pre-registration

Written and committed **before** any holdout data was used for evaluation. The
git commit timestamp of this file is the evidence.

## Data split

| Period | Use |
|---|---|
| 2020-01-01 → 2025-09-30 | Research: walk-forward model fitting, hypothesis tests, variant selection |
| **2025-10-01 → 2026-10-07** | **Final holdout.** Evaluated **once** per specialist, with the configuration frozen below. No re-tuning after seeing it. |

During research, a single exploratory smoke test (default models, BTC 1h, no
selection made from it) ran on data that included the holdout, before the
cutoff was enforced in code. No configuration, threshold or feature was chosen
from it.

## Candidate selection rule (research period only)

A specialist may go to the holdout only if, on the research period:

1. its walk-forward, out-of-sample net EV per trade is positive after the cost model it will be used with, and
2. it beats its simplest baseline (base rate / buy & hold / volatility-targeted hold, as applicable) on the comparison metric named below.

Among variants of the same idea, **at most one** goes to the holdout. It is chosen by its research-period result. Every variant tried is in the trial ledger (`data/ledger/trials.jsonl`) and counts toward the deflated Sharpe ratio.

## Pass criteria on the holdout

### Directional model specialists (intraday / multi-hour)

Status becomes **VALIDATED** only if **all** hold on the holdout:

- at least 30 trades;
- mean net return per trade > 0 after costs (taker fees + spread/impact + funding for perps);
- one-sided day-clustered bootstrap p-value < 0.05 for that mean;
- the research-period deflated Sharpe ratio, computed over all ledger trials of the same study, is > 0.5. This is the probability that the true Sharpe beats the best Sharpe expected from luck.

Otherwise the status is **NOT_VALIDATED**. The engine then shows the specialist's view for information only and always outputs NO TRADE for it.

### Daily trend specialist (exposure / risk overlay)

Equal-weight portfolio across the eligible universe, spot costs, compared with the volatility-targeted hold on the same coins and days:

- **RISK_OVERLAY confirmed** if holdout max drawdown ≤ 0.8 × the vol-targeted hold's max drawdown, **and** holdout Sharpe ≥ vol-targeted hold Sharpe − 0.2.
- **VALIDATED as alpha** additionally needs a positive holdout alpha vs the vol-targeted hold, **and** a research-period alpha t-stat ≥ 2. The research-period t-stat is 1.43, so this is not expected.
- Otherwise: **NOT_VALIDATED**.

## Live monitoring (after deployment)

Every specialist runs a one-sided CUSUM on per-trade net returns:

- The reference is its validated mean and standard deviation.
- The in-control average run length is 3,000 trades.
- A rolling 50-trade upper-confidence-bound check runs alongside.

An alarm disables the specialist (NO TRADE) until a fresh walk-forward validation passes.
