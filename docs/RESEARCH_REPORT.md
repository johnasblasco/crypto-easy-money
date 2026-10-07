# Short-term crypto signals: what survives an adversarial test?

*Research log for the `quant/` signal engine. All results are out-of-sample and after realistic costs unless stated otherwise. Numbers come from `data/results/*.json` produced by `quant/studies/*`.*

**Status: RESULTS_PENDING.** Methods are final; the results sections are filled in as the studies complete.

---

## 1. Question

Can any measurable information produce short-horizon cryptocurrency signals with a **genuine, cost-surviving statistical edge**? And can a system know when it **doesn't** have one, outputting NO TRADE?

The working assumption was that the obvious answers (RSI/MACD crossovers, "buy the dip", candle patterns) are probably wrong, and that a real edge, if any, would be small, conditional and fragile.

## 2. Data

- **Market data:** Binance spot 1-minute klines, 2020-01-01 → 2026-10-07, for 16 liquid USDT pairs:
  - BTC, ETH, BNB, XRP, ADA, DOGE, SOL, LINK, LTC, TRX, AVAX, DOT, ATOM, BCH, ETC, XLM
  - plus USDC/USDT, used for one stablecoin hypothesis.
  - Each bar has open, high, low and close prices, volume, quote volume, trade count, and **taker-buy volume** (exact aggressor side).
- **Data quality:**
  - 0.066% of minutes are missing; the longest outage is 354 minutes. Missing minutes are kept as NaN, never forward-filled.
  - No inconsistent OHLC bars, and no bars with taker-buy volume above total volume.
  - Early zero-volume stretches in thin coins: DOGE, ETC, ATOM.
- **Other data:** the Crypto Fear & Greed index (daily), lagged one day so a decision only uses a published value.
- **Not available in this environment** (proxy-blocked), so not testable:
  - futures (funding, open interest, liquidations, basis)
  - historical order books
  - bulk tick trades
  - on-chain flows
  - delisted coins
- **Survivorship:** the universe is today's survivors. BTC and ETH results are the headline; altcoin results are upper bounds.

## 3. Method: the backtest as an adversarial experiment

| Risk | Control |
|---|---|
| Look-ahead in features | Bars indexed by **close** time. A decision at *t* uses bars with close ≤ *t*. A **perturbation test** changes every price/volume after *t*, rebuilds all features, and asserts nothing at or before *t* changed (`tests/test_quant_harness.py`). |
| Bid-ask bounce / fill fantasy | Entry at the **VWAP of the next 1-minute bar** after a 1-minute reaction delay; exit at the VWAP of the bar *h* later. Never fill at the price that generated the signal. |
| Train/test overlap | Walk-forward 90-day test folds. Training labels that end inside the test fold are **purged**. |
| Selection on test data | Calibration, the NO-TRADE margin and the skill gate are fit on an inner validation slice of the training window, split in two: calibrate on the first half, judge skill and pick thresholds on the second. |
| Overfitting by search | Every configuration is recorded in a **trial ledger** and deflated (deflated Sharpe, PBO). Holm/BH correction across hypothesis cells. A **final holdout** (2025-10-01 → 2026-10-07) is locked; its pass criteria were **pre-registered and committed before use** (`docs/PREREGISTRATION.md`). |
| Unrealistic costs | Spot taker 10 bp/side; perp taker 5 bp/side. Plus 1 bp (BTC/ETH) or 3 bp (alts) per side for spread and impact. Funding 1 bp per 8h charged to both sides. A live order-book check confirmed alt half-spreads of about 2.5 bp. |
| Market beta mistaken for skill | Long-only results are compared with buy & hold, **volatility-targeted hold** and long-only placebos. Trend is compared with random timing at the same time in market. |
| Single-regime luck | Results reported per year and per ex-ante volatility/trend regime. |
| Correlated "independent" evidence | Event studies bootstrap over **days**, since a crash fires on every coin at once. Panel thresholds use per-timestamp clustered standard errors. |
| A broken harness | **Positive control:** the known 15-minute reversal must be recovered. An independent **adversarial review** of the harness (5 reviewers, each finding checked by 2 verifiers) found and fixed real bugs (§7). |

## 4. Results

RESULTS_PENDING

## 5. The engine

RESULTS_PENDING

## 6. What this means for using it

RESULTS_PENDING

## 7. Bugs the adversarial process caught (and what they would have done)

| Bug | Effect if left in |
|---|---|
| Calibrator fit and judged on the same validation slice | Skill gate always passed; NO-TRADE thresholds chosen on in-sample probabilities → more trading on noise |
| Pooled "BTC+ETH" panels silently dropped every BTC row (missing cross-asset column) | "Pooled" results were really ETH only |
| Trend: an outage day blanked the daily close | Donchian components forced flat for up to 90 days → understated trend performance (BTC Sharpe 1.07 vs 1.16) |
| Trend: Donchian component reported "flat" without enough history | Confident signals from incomplete information |
| Resampled last bar included while still forming | Stale/partial bar treated as complete in live use |
| Rolling-window NaN intolerance | 15–25% of range features silently missing |
| ECE split identical predictions across bins | Reported sampling noise as miscalibration |
| CUSUM textbook threshold (h = 5) | 100% false alarms for realistic edge sizes; replaced with an ARL-calibrated threshold |
| Quarter-hour "first minute" off by one bar under close-time indexing | Would have tested the last minute of each quarter instead of the first |
| Fear & Greed array read-only under pandas 3 | Daily ML study crashed |
