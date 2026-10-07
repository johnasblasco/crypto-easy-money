# Short-term crypto signals: what survives an adversarial test?

*Research log for the `quant/` signal engine. All results are out-of-sample and after realistic costs unless stated otherwise. Numbers come from `data/results/*.json`, produced by `quant/studies/*` and `quant/train_engine.py`.*

---

## Summary

**No short-term trading signal survived.** Every candidate that looked promising in research failed the locked, pre-registered holdout (October 2025 → October 2026), or its own validation refused to trade it.

The engine therefore outputs **NO TRADE**, and that answer is backed by evidence. It does not mean the engine is broken.

What the data *does* support:

1. **Short-horizon direction is statistically predictable but not profitably.**
   - At 15 minutes to 4 hours, models reach AUC 0.55–0.56, mostly from order flow and bar shape.
   - That edge is a few basis points: smaller than taker costs, and smaller than its own instability.
2. **Volatility, and whether a move will be large enough to matter, are far more predictable than direction.**
   - HAR volatility forecasts beat EWMA at 4h and 1d with p < 10⁻⁵.
   - "Will the move exceed costs?" scores AUC 0.60–0.69.
3. **A slow daily trend rule is a robust *risk overlay*.** It held up on the holdout:
   - max drawdown −22% versus −54% for a volatility-targeted hold, in a falling market;
   - its timing alpha is suggestive but not statistically established.
4. **The most promising-looking short-term signal was a decaying anomaly.**
   - Buying an altcoin after a forced, one-sided sell-off earned +33 bp per event net in 2020–2025, mostly in 2020–21.
   - On the holdout it lost −13.5 bp per event over 303 events.

Every "edge" the search produced at first glance fell apart under adversarial testing:
- **Market beta:** the daily ML panel's +316 bp per trade.
- **One episode:** the stablecoin premium's t = −3.4 came from the March 2023 USDC de-peg.
- **Selection from many trials:** the capitulation reversal had a deflated Sharpe of 0.07 over 252 cells.

The machinery that caught them is reusable, and it is the most valuable part of this project.

## 1. Question

Can any measurable information produce short-horizon cryptocurrency signals with a **genuine, cost-surviving statistical edge**? And can a system know when it **doesn't** have one, and output NO TRADE?

The working assumption was that the obvious answers are probably wrong, such as RSI/MACD crossovers, "buy the dip" and candle patterns. A real edge, if any, would be small, conditional and fragile.

## 2. Data

- **Market data:** Binance spot 1-minute klines, 2020-01-01 → 2026-10-07, for 16 liquid USDT pairs:
  - BTC, ETH, BNB, XRP, ADA, DOGE, SOL, LINK, LTC, TRX, AVAX, DOT, ATOM, BCH, ETC, XLM
  - plus USDC/USDT for the stablecoin hypothesis.
  - Each bar has open, high, low and close prices, volume, quote volume, trade count, and **taker-buy base and quote volume**. That gives the exact aggressor side, and the average price paid by aggressive buyers and received by aggressive sellers in each minute.
- **Data quality:**
  - 0.066% of minutes are missing; the longest outage is 354 minutes. Missing minutes stay NaN and are never forward-filled into labels.
  - No inconsistent OHLC bars.
- **Other data:** the Crypto Fear & Greed index, lagged one day so a decision only uses a value already published.
- **Not available in this environment** (proxy-blocked), so not testable:
  - futures (funding, open interest, liquidations, basis)
  - historical order books
  - tick trades
  - on-chain flows
  - delisted coins
- **Survivorship:** the universe is today's survivors. BTC and ETH results are the headline; altcoin results are upper bounds.

## 3. Method: the backtest as an adversarial experiment

| Risk | Control |
|---|---|
| Look-ahead in features | Bars are indexed by **close** time; a decision at *t* uses bars with close ≤ *t*. A **perturbation test** changes every price/volume after *t*, rebuilds all features, and asserts nothing at or before *t* changed (`tests/test_quant_harness.py`). |
| Fill fantasy | Entry at the **VWAP of the next 1-minute bar** after a 1-minute reaction delay; exit at the VWAP *h* later. For event signals, also **aggressor-side fills**: buys pay the taker-buy VWAP and sells receive the taker-sell VWAP. That captures state-dependent spreads (§4.4). |
| Train/test overlap | Walk-forward 90-day test folds. Training labels that end inside the test fold are **purged**. |
| Selection on test data | Calibration, the skill gate and the NO-TRADE margin are fit on an inner validation slice of the training window, split in two: calibrate on the first half, judge skill and pick thresholds on the second. |
| Overfitting by search | Every configuration is recorded in a **trial ledger** and deflated (deflated Sharpe, PBO). Holm/BH correction across hypothesis cells. A **final holdout** (2025-10-01 → 2026-10-07) was locked. Its pass criteria were **pre-registered and committed before use** (`docs/PREREGISTRATION.md`, commit `cef24eb`). The candidates were frozen in commits `1ac0e15` and `1ed48d3` before the holdout ran. |
| Unrealistic costs | Spot taker 10 bp/side; perp taker 5 bp/side. Plus 1 bp (BTC/ETH) or 3 bp (alts) per side for spread and impact. Funding of 1 bp per 8 h charged to longs; short funding income is not credited. Maker fills are treated only as an upper bound and never used to validate anything. |
| Market beta mistaken for skill | Long-only results are compared with buy & hold, a **volatility-targeted hold**, and matched placebos. Trend is compared with random timing at the same time in market. Panel trades are decomposed into the same-direction market move plus excess (§4.5). |
| Single-regime luck | Results are reported per year and per ex-ante volatility/trend regime. Leave-one-month-out and episode exclusion are used where a result could be one event. |
| Correlated "independent" evidence | Event studies bootstrap over **days**; panel thresholds use per-timestamp clustered standard errors. The panel's real sample size is its number of distinct decision dates. |
| A broken harness | **Positive control:** the known 15-minute reversal must be recovered. Two rounds of independent **adversarial review** found and fixed real bugs (§7). |

## 4. Results

### 4.1 Overview

| Idea | Research period (2020 → 2025-09) | Holdout (2025-10 → 2026-10) |
|---|---|---|
| Intraday direction model (15m–4h) | Real statistical skill (AUC 0.55–0.56); net EV never significant after costs | ETH 4h candidate abstained (skill gate failed). Forced trading: −34 bp/trade |
| Volatility / tradeability | Strongly predictable (HAR ≫ EWMA; "move > cost" AUC 0.60–0.69) | Used for sizing and risk, not tested as a trade |
| Event and folklore hypotheses | 204 cells, 0 survive Holm/BH | — |
| Flow-driven capitulation reversal (alts) | +33 bp/event net, p = 0.003, but deflated Sharpe 0.07 | **−13.5 bp/event, 303 events, p = 0.89** |
| Daily trend ensemble | Sharpe 1.19 vs 0.82, drawdown −25% vs −56%; alpha t = 1.87 | **Risk overlay confirmed:** drawdown −22% vs −54% |
| Cross-sectional momentum | Not significant (p 0.27–0.58); long-short loses | — |
| Daily ML panel | 3-day LightGBM +316 bp/trade, p = 0.007 (mostly market timing) | Abstained. Forced trading: −249 bp/trade |
| Quarter-hour opening imbalance | Weak continuation (t ≈ 2 at 4h); fails Holm; tiny effect | — |
| Stablecoin premium → BTC | t = −3.4, but driven entirely by the March 2023 USDC de-peg | — |
| Regime specialists / regime calibration | Worse than one global model (DM p ≈ 1) | — |

### 4.2 Short-horizon direction: real information, no profit

**What carries information.**
- Walk-forward models on 1-minute features reach these AUCs:

  | Horizon | AUC (BTC/ETH) |
  |---|---|
  | 15 min | 0.547–0.552 |
  | 1 h | 0.557–0.562 |
  | 4 h | 0.550–0.563 |
  | 1 day | 0.514 |

  Mean calibration error is 1–2%. The information comes mostly from taker order flow and bar shape, i.e. short-term reversal and continuation of aggressive flow.
- Hand-built features beat learning from raw sequences (`sequence_models.json`):
  - raw-sequence MLP: AUC 0.521, versus 0.558 for hand-built features with LightGBM;
  - LightGBM on raw sequences: 0.549;
  - adding raw sequences to the hand-built features: no gain (DM p = 0.54 BTC, 0.67 ETH).

  Deep sequence models are not justified by this data.

**Converting it to profit** (`intraday_conversion.json`). The grid had 24 variants:
- BTC, ETH, or both pooled
- 1 h or 4 h horizon
- spot or perp costs
- a probability margin or an expected-edge-versus-cost decision rule

Results:
- The skill gate passed in 42–95% of folds.
- When it traded, coverage was 1–4% of periods, and net EV was never significant. The best was ETH 4h with the EV rule: 130 trades, +20.6 bp/trade, p = 0.052, deflated Sharpe 0.50, PBO 0.56.
- Most of the positive results come from 2021.

**Holdout.** The frozen ETH 4h candidate's skill gate failed on its most recent validation slice (Sep 2024 → Sep 2025), so it never traded.

A post-hoc check (`holdout_posthoc.json`; not used for any verdict) shows abstaining was right:
- holdout AUC fell to 0.527;
- forced trading would have lost −34 bp/trade, with Sharpe −2.0 and a −46% drawdown.

### 4.3 What *is* predictable: volatility and tradeability

From `volatility_forecast.json`:
- **Volatility.** A HAR model of realised volatility beats EWMA by a wide margin at 4 h and 1 day.
  - BTC 4h QLIKE: 0.29 vs 0.43; ETH 0.28 vs 0.37; SOL 0.21 vs 0.29. All p < 10⁻⁵.
  - Correlation of log forecast with realised volatility: about 0.8.
  - At 1 h, HAR loses to EWMA, because it doesn't model intraday seasonality.
- **Tradeability.** Whether the next move will exceed round-trip costs has AUC 0.60–0.69, far above direction's 0.55.

Implication: the market tells you much more about *how much* price will move than about *which way*. That information belongs in position sizing and risk control (the trend overlay is volatility-targeted), not in directional bets.

### 4.4 Event studies and market folklore

**The hypothesis grid** (`event_hypotheses.json`): 8 hypotheses, each pre-registered from the literature and market lore, tested over parameters × horizons {15m, 1h, 4h} × {BTC/ETH, alts} × three cost models. That made **204 cells**, compared against placebos matched on hour and volatility, with day-clustered inference.

**None survive Holm or Benjamini–Hochberg.** Median net bp per event by hypothesis:

| Hypothesis | Median net (bp/event) |
|---|---|
| Liquidity-sweep reclaim | −14 |
| Breakout continuation | −16 |
| Liquidation-cascade reversal | −16 |
| Market-shock continuation | −20 |
| Weekend reversal | −79 |

Only flow-driven reversal had a positive best cell.

**Flow-driven capitulation reversal** was the strongest research signal in the project. The rule:
- **Trigger:** an altcoin falls at least 4 ex-ante sigmas in 60 minutes;
- **Flow condition:** taker selling is in the top quintile of the last 30 days;
- **Trade:** buy at the next minute and hold 1 hour.

In its favour:
- gross +59 bp/event, 56 bp above the matched placebo;
- a dose-response pattern: weaker trigger thresholds give much smaller returns, which is the shape a real effect has.

**Execution realism** (`execution_realism.json`). Aggressor-side fills:
- **Spreads widen at events:** at event entries, the median effective spread is about 1.3–5 bp per coin, versus 0.5–2 bp at random times. Spreads really do widen exactly when reversal strategies trade.
- **The cost is modest:** fills reduce the alt event returns by 3–4.5 bp. The signal survived: net +33 bp/event, p = 0.003.

Against it:
- **Year by year,** net bp/event was +41 (2020), +115 (2021), +16 (2022), −10 (2023), −9 (2024), +11 (2025, to September).
- **Selection:** it was the best of 252 cells, which gives a deflated Sharpe of 0.07.

**Holdout:** 303 events, mean net **−13.5 bp/event**, hit rate 44.9%, p = 0.89. The anomaly was real in 2020–21 and has since been competed away. The engine still shows these events as information, flagged NOT_VALIDATED.

### 4.5 Daily horizon

**Trend ensemble.** The rule is 9 components (Donchian 10–90 days plus past-return 7–56 days), long/flat, sized to 50% annualised volatility, with spot costs (`trend_daily.json`, `trend_robust.json`).

| Equal-weight portfolio | Trend overlay | Vol-targeted hold | Buy & hold |
|---|---|---|---|
| Sharpe | 1.19 | 0.82 | 0.66 |
| Max drawdown | −25% | −56% | −78% |

- **Timing alpha** versus the vol-targeted hold: +11.4%/yr, t = 1.87.
- **Robustness:** the result holds across 12 variants:
  - half or double lookbacks
  - Donchian-only or past-return-only
  - target volatility 30% or 80%
  - 08:00 or 16:00 UTC day boundaries
  - execution delays up to 12 hours
  - 2× costs

  Across them, Sharpe is 1.10–1.28 and alpha t-stats are 1.56–2.30.
- **Random timing:** beating a random-timing null at the same time in market (p < 0.05) holds for BTC, ETH, SOL, AVAX, DOGE and ADA, but not for the rest.
- **Rejected refinements:**
  - an efficiency-ratio gate (lower return, t = −1.10);
  - a crowding gate (worse, t = −2.25);
  - jump-variance exclusion (no effect, p = 0.74).
- **Holdout:** Sharpe −0.35 versus −0.78 for the vol-targeted hold, and max drawdown **−22% versus −54%**. Holdout alpha is +3.6%/yr (t = 0.35). That is the pre-registered **RISK_OVERLAY** outcome: it reliably cuts drawdowns, and its alpha is unproven.

**Cross-sectional momentum** (`xsection_momentum.json`). Top-quintile coins minus equal weight, over 7/14/28-day lookbacks:
- excess of −1.9% to +7.6%/yr, with p from 0.27 to 0.58;
- long-short Sharpe from −0.26 to −0.51.

Rejected.

**Daily ML panel** (`daily_panel_h1.json`, `daily_panel_h3.json`, `dissect_panel_*.json`).
- One model on all 16 coins pooled, with every feature family plus the trend score.
- At 1 day: AUC 0.50–0.53 and negative EV.
- At 3 days, LightGBM with perp costs looked excellent: +316 bp/trade, 69% hit rate, p = 0.007, deflated Sharpe 0.59.

The dissection found what that result really is:
- **Few independent bets:** 42 decision dates, about 15 coins each, mostly all in the same direction.
- **Mostly beta:** 176 of the 188 bp gross per row is the equal-weight market's own move in that direction. The excess is 12 bp, t = 1.29.
- **Concentrated:** it traded in only 3 of 17 folds, and its top 5 dates supply 74% of the gross.
- **Timing:** its dates beat random dates (p = 0.023). It is a market-timing call on 42 decisions.
- **Spot version:** without shorting, the long-only version shows nothing (date-level t = 0.45).

Holdout: the frozen model's skill gate failed, so it never traded. Post-hoc forced trading would have lost −249 bp/trade (Sharpe −0.71).

Meta-labeling the trend rule (learning *when* to follow the trend) was worse than the plain trend.

### 4.6 Calibration, regimes and agreement

From `calibration_regimes.json` (BTC+ETH, 1h and 4h):

- **Calibration.** ECE is 1.1–1.7%. The calibration slope is 0.71–0.81, so the most confident predictions are somewhat overconfident: a stated 60% is closer to 57%. Calibration holds similarly across volatility and trend regimes (slope 0.62–0.87).
- **Regime-specific calibration** (a separate Platt map per volatility regime) is *worse* than one global map: DM p of 0.78 and 1.00 for "regime-specific is better".
- **Regime specialists** (one model per volatility regime) are *worse* than one global model with regime features. At 4 h: AUC 0.542 versus 0.560, DM p ≈ 1. Simpler wins.
- **Agreement with the daily trend** does *not* make the intraday model more accurate:
  - 4h top-quintile confidence: 57.3% hit rate when agreeing, versus 58.5% in conflict;
  - 1 h: 59.5% versus 58.5%.

  The engine shows agreement or conflict as context only.

### 4.7 New information sources

These come from `flow_regressions.json` and `stablecoin_robust.json`.

**Quarter-hour opening imbalance** (a replication of Kim & Hansen 2026). This is taker imbalance in the first minute of each quarter-hour, controlled for total imbalance and past return.
- **Support:** it predicts continuation at a 4-hour window in both coins (BTC t = 2.60, ETH t = 2.12), weakening at longer horizons.
- **Against:** 3 of 12 regressions have |t| > 2, but none survive Holm across the 12. The effect is about 0.02σ of the forward return.
- **Verdict:** weak, economically negligible support.

**Stablecoin premium.** The USDC/USDT log price at 00:00 predicts the next day's BTC return, t = −3.38 with HAC errors. Every robustness check knocks it down:
- excluding the March 2023 USDC de-peg (23 of 2,093 days): t = −0.73;
- winsorising the premium: t = −1.03;
- by year: 2023 alone gives t = −7.2, and 2025 flips sign;
- a tradeable version adds nothing beyond its exposure (t = 0.54).

Rejected as a single-episode artifact.

### 4.8 Statistical power: why "maybe" is the honest answer for small edges

At the edge sizes found here (Sharpe of a few tenths per year), confirming an edge at 5% significance and 80% power takes thousands of independent trades. Daily or multi-day signals on one market produce a few hundred a year.

Pooling coins does not help much: crypto moves together, and the panel's effective sample size was its number of distinct dates.

This is why the trend overlay's alpha (t ≈ 1.9 over 5.6 years) cannot be confirmed or rejected with data that exists. Its drawdown reduction, a much larger effect, can be, and was.

## 5. The engine

`quant/engine.py`, served at `/api/engine` and shown on the dashboard.

- **Specialists** carry their evidence. Current statuses:

  | Specialist | Status | Role |
  |---|---|---|
  | `daily_trend` | RISK_OVERLAY | Exposure guide only, never a trade |
  | `alt_capitulation_reversal_1h` | NOT_VALIDATED | Detects events live; never trades |
  | `daily_panel_3d` | NOT_VALIDATED | Calibrated directional view |
  | `eth_4h_flow_model` | NOT_VALIDATED | Calibrated directional view |
  | `btc_4h_flow_view` | NOT_VALIDATED | Calibrated directional view |

- **Gates.** A specialist can only output LONG/SHORT when all of these hold:
  1. its status is VALIDATED;
  2. its health monitor is ACTIVE;
  3. its own decision rule fires (expected edge above cost plus margin, or calibrated probability beyond the margin);
  4. the decision is fresh: within 2 minutes of the decision time for models, 90 seconds for events, because research always entered 1 minute later;
  5. the symbol is inside the universe it was tested on;
  6. data health is OK: newest closed bar ≤ 3 minutes old, ≤ 2% of the last day's minutes missing, no implausible jumps;
  7. the live order book can fill the size within spread and impact limits.
- **Consensus.** Each coin gets one verdict:
  - insufficient information → NO TRADE;
  - validated specialists that disagree → NO TRADE;
  - otherwise the validated signal, or NO TRADE.

  Directional views from all specialists are summarised as agree or conflict, as context only (§4.6).
- **Degradation.** Each specialist runs a one-sided CUSUM on its live trade results:
  - the threshold is calibrated to an in-control average run length of 3,000 trades;
  - a rolling 50-trade upper-confidence-bound check runs alongside;
  - an alarm switches the specialist to NO TRADE until it is revalidated.
- **Speed.** The first call seeds 400 days of 1-minute bars per coin from disk and syncs the gap from Binance (about 100 s for 16 coins). Later calls fetch only new bars, and results are cached for 5 minutes.

## 6. What this means for using it

- **Don't expect trade alerts.** None of the short-horizon ideas tested here produced a trade signal that survived costs and the holdout. That covers order flow, bar shape, momentum, events, sweeps, cascades, breakouts, regimes, sequences and pooled ML. An app that alerted anyway would be alerting on noise.
- **What held up is risk control.** Use the trend overlay's *exposure* as a position-size guide for coins you already intend to hold. It cut drawdowns by more than half in research and again in a falling market it had never seen. It is not a promise of higher returns.
- **The holdout is now used.** Any new idea, or any re-tuning of these, must not be judged on 2025-10 → 2026-10 again. The honest path is a new forward test:
  1. freeze the configuration;
  2. paper-trade it from today;
  3. evaluate it with the same pre-registered criteria.
- **What could change the answer:**
  - **Data this environment couldn't reach:** perpetual funding, open interest, liquidations and basis.
  - **Order-book history,** recorded going forward with `python -m quant.record_book` and tested later with the same harness.
  - **Maker execution with real fill data:** costs, not information, are what kill the 15-minute to 4-hour signals.

## 7. Bugs the adversarial process caught (and what they would have done)

| Bug | Effect if left in |
|---|---|
| Calibrator fit and judged on the same validation slice | Skill gate always passed; NO-TRADE thresholds chosen on in-sample probabilities → more trading on noise |
| Pooled "BTC+ETH" panels silently dropped every BTC row (missing cross-asset column) | "Pooled" results were really ETH only |
| Per-trade EV excluded the exit cost | Every trade looked one fee better than it was |
| Portfolio returns averaged log returns across coins | Not the return of an investable equal-weight portfolio (biased low by the Jensen gap, most in volatile periods) |
| Skill gate only required "better than base rate", not significantly better | Folds with no real skill traded anyway |
| Funding charged to shorts | Penalised shorts instead of longs (historical funding favours shorts) |
| Event p-values floored at 1/n_boot | Holm correction could never reject anything |
| Trend: an outage day blanked the daily close | Donchian components forced flat for up to 90 days → understated trend performance |
| Trend: a Donchian component reported "flat" without enough history | Confident signals from incomplete information |
| Resampled last bar included while still forming | Partial bar treated as complete in live use |
| Rolling-window NaN intolerance | 15–25% of range features silently missing |
| ECE split identical predictions across bins | Reported sampling noise as miscalibration |
| CUSUM textbook threshold (h = 5) | 100% false alarms at realistic edge sizes; replaced with an ARL-calibrated threshold |
| Quarter-hour "first minute" off by one bar under close-time indexing | Would have tested the wrong minute |
| Hourly regression grid stepped from a minute-1 row | The on-the-hour filter selected nothing; the study crashed |
| Regime-specialist comparison aligned pooled BTC/ETH rows by timestamp only | Compared one coin's prediction with the other coin's outcome |
| Live engine ignored the decision rule a model was validated with | A probability margin would have been read as a return hurdle |
| Live engine would apply a single-coin model to every coin | Untested predictions on 15 other coins |
| Live signals had no entry window | A 3-day decision could be shown as actionable two days late |
| Experiment runner forked worker processes from a multithreaded parent | Intermittent deadlock (`tests/test_edge.py` hung); now uses spawn |
| Fear & Greed array read-only under pandas 3 | Daily ML study crashed |
