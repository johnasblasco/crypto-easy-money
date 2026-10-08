# Trader-psychology hypotheses: pre-registration

Written and committed **before** any of these hypotheses was run on data. The git commit timestamp of this file is the evidence.

**How they were chosen.** Three research agents searched the published evidence, from three angles:

- behavioural finance;
- market-structure and calendar effects;
- rigorous tests of chart patterns.

A skeptical reviewer then removed ideas that this repository had already tested and rejected, or that cannot be tested with Binance spot data. It ranked the rest and fixed exact, look-ahead-free definitions with one primary test each.

**Decision rule.** Holm correction across the 8 primary cells, family-wise 5%, plus each hypothesis's extra pass condition. A pass leads only to a frozen forward paper test, never directly to trading.

**Data.** Research period only: 2020-01-01 to 2025-09-28. The 2025-10 to 2026-10 holdout was already used in earlier work.

**Deviations from the text below.** These were fixed before running:

- Cells are saved to `data/results/psychology.json` instead of the trial ledger. The ledger stores portfolio return series, which event studies do not have.
- Secondary cells are limited to those marked in `quant/studies/psychology.py`.

## R1 (SELECTED, rank 1). Anchoring to the 52-week high: cross-sectional Nearness52 [merges A-H3 and C-H3; C-H3's time-series and breakout arms dropped because they duplicate the Donchian trend overlay]

**Psychology.** Traders anchor on a coin's 52-week high. Near it, they are reluctant to bid through the anchor and take profits too early, so good news is under-priced until the anchor gives way. Far below it, holders treat the old high as 'fair value' and sell into rallies to break even (the disposition effect). Both groups under-react, so coins near their high keep outperforming coins far from it. Acting earlier than the crowd means owning the coins it is still too anchored to buy.

**Definition and primary test.**

PROTOCOL (applies to R1-R8): (1) Use research data only, 2020-01-01 to 2025-09-28 (event_hypotheses.research_frame). The 2025-10 to 2026-10 holdout is spent. (2) Bars are indexed by close time. A decision at t uses only bars closing at or before t. Entry is the VWAP of the next 1m bar. (3) Each hypothesis has exactly ONE primary cell, fixed below. Holm across the 8 primary cells decides, with BH also reported. Secondary cells are descriptive and cannot rescue a failed primary. Every cell goes in the trial ledger. (4) A primary that passes goes to a frozen forward paper test, not to trading. (5) Costs come from quant.costs; memes use meme_coins.TickCost. (6) Portfolio returns are simple returns averaged across coins, never averaged log returns.

UNIVERSE: the 16 large coins (primary). Secondary universes: the 21 memes alone, and all 37 coins pooled with group-neutral ranks (percentile within the large group or within the meme group).

DATA: daily bars from quant.studies.trend.daily_bars. Bar D closes at D 00:00 UTC. fill(D) = VWAP of the 1m bar (D, D+1m].

DECISION: every Monday D at 00:00 UTC, using only daily bars closing at or before D. First D = 2021-01-04 (365 days after the 2020-01-01 data start). Last D = 2025-09-22.

ELIGIBILITY at D: (a) first finite daily close at or before D-365d; (b) finite closes on all 8 daily bars closing D-7d to D; (c) at least 330 finite daily highs in the 365-bar window.

SIGNAL: N52_i = close_i(D) / max(high_i over the daily bars closing D-364d to D). Requires at least 9 eligible coins. Let n3 = floor(n/3). TOP = the n3 coins with the highest N52; BOT = the n3 lowest; EW = all eligible coins. Ties are broken by symbol name.

PORTFOLIO: equal weights at fill(D), then buy-and-hold to the next Monday's fill, with no rebalancing during the week. Mark to market at every daily fill.

HORIZON: 1d marks; each position is held for 7 consecutive 1d periods. COOLDOWN: re-sort only every 7 days. DIRECTION: +1 for TOP (spot long). BOT is shorted (-1) only in the perp long-short secondary.

COSTS, charged at each re-sort: sum over i of |w_i,new - w_i,drifted| times the one-side cost. Spot taker: 10 bp + 1 bp (BTC/ETH) or 3 bp (others) per side; memes pay 10 bp + max(3 bp, half a tick). Perp taker: 5 bp + the same impact, plus funding of 1 bp per 8h on long weights.

PRIMARY CELL: the daily series s = net(TOP) - net(EW), both at spot costs, on the 16 large coins. H1: mean(s) > 0. p = the larger of (a) the one-sided Newey-West p (lag 7) and (b) the weekly-block bootstrap p (5,000 resamples of whole weeks).

ALSO REQUIRED FOR A PASS (horse race against momentum and trend): a weekly Fama-MacBeth OLS of the next 7-day fill-to-fill log return on cross-sectionally standardised N52 plus four controls: (1) the 28-day return skipping the latest day, ln(close(D-1d)/close(D-29d)); (2) 30-day realised volatility of daily log returns; (3) log of the 30-day median daily quote volume; (4) the coin's 9-component trend score S at D from quant.studies.trend. The mean N52 slope must be > 0 with Newey-West t >= 1.65 (lag 4).

SECONDARY: perp long-short TOP - BOT; memes only; pooled group-neutral; per-year results; side-by-side comparison with xsection_momentum.json.

SURVIVORSHIP: delisted coins would have sat in BOT and dragged EW down. Their absence biases both spreads toward zero, so the bias is conservative.

**Evidence.** List labels: A = first researcher (behavioural H1-H8), B = second (calendar/flow H1-H8), C = third (chart patterns H1-H8). PRIMARY: Jia, Simkins, Yan, Zhang & Zhao, 'Psychological anchoring effect and cross section of cryptocurrency returns', Journal of Banking & Finance, DOI 10.1016/j.jbankfin.2025.107592. Reviewer check: venue, DOI and authors confirmed through a bibliographic listing dated 2025-11-01. The abstract could not be re-read because SSRN and ScienceDirect are blocked from this environment. Reported findings: nearness to the 52-week high positively predicts next-week cross-sectional returns and is robust to standard predictors. The value-weighted long-short portfolio (cANCHOR) earns about 130-140 bp/week, reported as profitable after costs. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5386180 ; https://www.sciencedirect.com/science/article/abs/pii/S0378426625002122 . Quantpedia #1167 (weekly, 2014-2023) reports a 92% indicative return at 78% volatility; not re-verified: https://vvv.quantpedia.com/?p=42289 . EQUITIES (replicated): George & Hwang (2004, JF) find nearness predicts returns better than past returns and subsumes momentum: https://www.bauer.uh.edu/tgeorge/papers/gh4-paper.pdf . Grinblatt & Han (2005, JFE) on capital-gains overhang: https://www.nber.org/system/files/working_papers/w8734/w8734.pdf . Crypto disposition effect at the investor level: Schatzmann & Haslhofer (Digital Finance 2023): https://link.springer.com/article/10.1007/s42521-023-00086-w . CAVEATS: the paper's universe is probably hundreds of coins including micro-caps, where crypto anomalies concentrate (Ficura 2023: https://ideas.repec.org/p/prg/jnlwps/v5y2023id5.003.html). There is no independent replication or out-of-sample test yet. The signal correlates with momentum, which this repo rejected (7/14/28-day, p 0.27-0.58). REPO OVERLAP: the ML panels used only range__pos_{1,7,30}d and range__dd_from_60d_high. There was never a 365-day anchor or a cross-sectional sort, so this is untested. REVIEWER PRIOR: moderate that the effect exists; low that it survives in 16 large survivor coins. It ranks first because it is the only candidate with a peer-reviewed crypto return test that claims after-cost profit.

## R2 (SELECTED, rank 2). Intraday momentum into the session close (Gao / Baltussen / Shen) [merges A-H8, B-H3, C-H4]

**Psychology.** Early traders react first to overnight news. Later participants trade in the same direction near the session close: slow or infrequent rebalancers, short-gamma hedgers, and, since 2024, ETF creation and redemption flow that is priced in the 15:00-16:00 ET benchmark hour. Taking the position before the closing crowd trades means providing what they will demand. Shen et al. attribute part of the effect to liquidity provision, so it is only partly behavioural.

**Definition and primary test.**

PROTOCOL: as in R1 (research period only; one primary cell; Holm across R1-R8; forward paper test before any trading).

CLOCKS.
(a) NY clock (primary). Days: NYSE full trading days from 2020-01-02 to 2025-09-26, i.e. Monday-Friday minus the following hardcoded dates.
Full holidays: 2020-01-01, 01-20, 02-17, 04-10, 05-25, 07-03, 09-07, 11-26, 12-25; 2021-01-01, 01-18, 02-15, 04-02, 05-31, 07-05, 09-06, 11-25, 12-24; 2022-01-17, 02-21, 04-15, 05-30, 06-20, 07-04, 09-05, 11-24, 12-26; 2023-01-02, 01-16, 02-20, 04-07, 05-29, 06-19, 07-04, 09-04, 11-23, 12-25; 2024-01-01, 01-15, 02-19, 03-29, 05-27, 06-19, 07-04, 09-02, 11-28, 12-25; 2025-01-01, 01-09, 01-20, 02-17, 04-18, 05-26, 06-19, 07-04, 09-01.
13:00 ET early closes, also excluded: 2020-11-27, 2020-12-24, 2021-11-26, 2022-11-25, 2023-07-03, 2023-11-24, 2024-07-03, 2024-11-29, 2024-12-24, 2025-07-03.
T_c(D) = 16:00 America/New_York, converted with zoneinfo (20:00 UTC under EDT, 21:00 UTC under EST). T_p = T_c of the previous NYSE trading day.
(b) UTC clock (secondary). Every calendar day; T_c = 00:00 UTC; T_p = T_c - 24h.

EVENT at the 1m bar closing at t = T_c - h, with h in {60 (primary), 15}.
r = lc[t] - lc[T_p], using Bars.lc on the close-time index. m = minutes from T_p to t. z = r / (Bars.sigma[t] * sqrt(m)).
Fire if all hold: |z| >= 1.0; the bars at T_p and t are not gaps; Bars.gaps(60)[t] == 0.
DIRECTION = sign(r) (continuation).
ENTRY = VWAP of bar t+1. EXIT = VWAP of bar t+1+h, i.e. the bar (T_c, T_c+1m].
COOLDOWN: one event per coin per session.

GROUPS: BTC+ETH pooled (primary); the other 14 large coins pooled (secondary). Memes excluded.

PRIMARY CELL: NY clock, h = 60, BTC+ETH, perp_taker, both directions. H1: mean net bp per event > 0 (quant.events.day_bootstrap).

SECONDARY:
- h = 15.
- UTC clock.
- spot_taker, long side only (r > 0, the stronger side in Shen et al.).
- Parameter-free check: Newey-West (lag 5) regression of the fill-to-fill return over [t+1, t+1+h] on z, over ALL sessions with no threshold; slope > 0.
- Gao's first-half-hour signal, r_first = lc[10:00 ET] - lc[T_p], in place of r.
- Pre-ETF (D <= 2024-01-10) versus post-ETF (D >= 2024-01-11) slopes. The BRRNY mechanism predicts post > pre.

NULLS: (i) the identical NY-clock rule on Saturdays, Sundays and weekday NYSE holidays, with T_p = T_c - 24h (no US close, no ETF NAV strike); report the NYSE-day minus non-NYSE-day difference. (ii) The harness placebo matched on hour and volatility.

**Evidence.** Shen, Urquhart & Wang (2022, Financial Review 57(2):319-344, DOI 10.1111/fire.12290). Reviewer-verified abstract: because Bitcoin has no fixed open or close, the authors use trading volume as the market clock. The first half-hour return predicts the last half-hour return. Predictability is greatest in the sessions with the highest volume or volatility. The effect adds economic value for market timing, especially in downturns, and is driven by liquidity provision. Their out-of-sample R2 and cost treatment are NOT verified, because the full text is blocked here. https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/ ; https://centaur.reading.ac.uk/100181/ . Gao, Han, Li & Zhou (2018, JFE 129:394-414), SPY 1993-2013: the first half-hour predicts the last half-hour (R2 1.6% in-sample, 1.2% out-of-sample), more strongly on volatile, high-volume and news days: https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351 . Baltussen, Da, Lammers & Martens (2021, JFE 142:377-403), more than 60 futures, 1974-2020: the rest-of-day return predicts the last 30 minutes and then reverts over the following days; the mechanism is short-gamma hedging: https://www3.nd.edu/~zda/intramom.pdf . MECHANISM FOR THE NY CLOCK: US spot BTC ETFs strike NAV on CF Benchmarks BRRNY, built from spot trades in 15:00-16:00 ET (Bitwise 424B3: https://www.sec.gov/Archives/edgar/data/1763415/000199937124000346/bitcoin-424b3_011024.htm ; https://www.cfbenchmarks.com/data/indices/BRRNY). Crypto volume peaks in US hours (Coinbase Institutional: https://www.coinbase.com/institutional/research-insights/research/market-intelligence/trading-activity-from-a-us-lens), so the NY session is the closest fixed proxy for Shen's volume clock and is pre-registered as primary instead of UTC midnight. CONTRARY: Wen et al. (2022, NAJEF) find momentum in 12 crypto pairs but reversal in 21: https://www.researchgate.net/publication/361202793 . De Nicola (2021, Ledger) finds negative 1-4h autocorrelation in BTC: https://ledger.pitt.edu/ojs/ledger/article/view/213 . REPO OVERLAP: the ML models had hour-of-day and momentum features at 15m-4h, but never this session-anchored signal, so it is untested. EXPECTED SIZE: a few to about 12 bp per event, against a 12 bp perp round trip. A pass is unlikely, but the test is cheap.

## R3 (SELECTED, rank 3). Round-number break continuation: riding the stop-loss cascade [merges A-H1 and C-H1 arm B]

**Psychology.** Traders anchor on round prices. They put take-profits ON the round number and protective stops just BEYOND it (buy-stops above, sell-stops below). A clean break through a fresh round level triggers that cluster of stop market orders. The forced, one-sided flow pushes price further than a break of an arbitrary level would. The trade gets in front of stop flow that other traders' anchoring has already placed in the book.

**Definition and primary test.**

PROTOCOL: as in R1.

GRID (per coin, per UTC day D, fixed for the whole day). C_ref = close of the 1m bar closing at D 00:00 UTC. S = 10^(floor(log10 C_ref) - 1).
ROUND levels: L_j = j * S/2 for integer j. Even j gives 'major' levels k*S; odd j gives 'half' levels (k+0.5)*S.
CONTROL levels: L_j + 0.13*S and L_j + 0.37*S. These are two grids with the same 0.5*S spacing. Quarter levels are avoided because they are semi-round.
Examples: BTC at 63,000 gives S = 1,000, round levels 63,000 and 63,500, and control levels 63,130 / 63,370 / 63,630 / 63,870. ETH at 2,500 gives S = 100. DOGE at 0.15 gives S = 0.01.

TICK FILTER: skip coin-day D if tick_est / C_ref > 5 bp, where tick_est = the smallest positive |close_t - close_(t-1)| among the 1m bars of day D-1 (past data only).

UP-BREAK at 1m bar t, for a level L of the grid under test. All of:
- close_t >= L * 1.0010;
- close_(t-1) < L * 1.0010;
- max(high[t-240 .. t-15]) < L. The level was untraded for the 4h ending 15 minutes before t, so the stop cluster is fresh; the last 14 minutes may approach or touch it.
- Bars.gaps(241)[t] == 0.
Direction +1. If several levels qualify, use the highest; it is one event.

DOWN-BREAK, mirrored: close_t <= L * 0.9990; close_(t-1) > L * 0.9990; min(low[t-240 .. t-15]) > L; no gaps. Direction -1; if several levels qualify, use the lowest.

ENTRY: VWAP of bar t+1. HORIZONS: h in {15, 60, 240} minutes. COOLDOWN: h, per coin per grid (quant.events.select).

GROUPS: the 16 large coins pooled (primary); memes that pass the tick filter (secondary).

PRIMARY CELL: h = 60, 16 large coins, up- and down-breaks together.
Statistic: Delta = mean signed gross return on the ROUND grid minus mean signed gross return on the two CONTROL grids pooled.
Inference: a day-clustered bootstrap that resamples calendar days and recomputes both means. H1: Delta > 0, one-sided.
A pass also requires the ROUND-grid mean net return > 0 under perp_taker.

SECONDARY:
- h = 15 and h = 240.
- spot_taker, long-only (up-breaks).
- Dose-response, reported: major > half > control.
- memes.
- per-year results.
- the harness placebo matched on hour and volatility.

**Evidence.** MECHANISM (FX order data, strong but old). Osler (2003, JF 58:1791-1819), using about 9,700 real stop-loss and take-profit orders at a large dealer: take-profits cluster ON round numbers; stop-buys sit just above and stop-sells just below. https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr125.pdf . Osler (2005, JIMF 24(2), 'Stop-loss orders and price cascades'): trends are faster after rates cross round numbers or stop clusters than after arbitrary levels, significant over hours, not days. Causality is not proven. https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr150.pdf . Osler (2000, FRBNY EPR) finds support and resistance levels interrupt intraday trends more often than arbitrary levels: https://www.newyorkfed.org/medialibrary/media/research/epr/00v06n2/0007osle.pdf . OTHER MARKETS: Donaldson & Kim (1993, JFQA) find the DJIA stalls at 100-point multiples and moves further than warranted after breaching them: https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/price-barriers-in-the-dow-jones-industrial-average/ED089C700C6B674DA8BD1BFB10CA0B85 . De Grauwe & Decupere (1992, CEPR DP621): https://econpapers.repec.org/RePEc:cpr:ceprdp:621 . CRYPTO (clustering only; no return test): Hu, McInish, Miller & Zeng (2019, FRL 28:337-342) find intraday round-number clustering and strategic pricing just above and below round numbers (reviewer-verified listing): https://digitalcommons.memphis.edu/facpubs/11603 . Quiroga-Garcia, Pariente-Martinez & Arenas-Parra (2022, FRL 47, 102811) find clustering at round prices where volume is higher: https://digibuo.uniovi.es/dspace/handle/10651/65509 . Han (2024, JBEF) finds limit orders cluster at .00/.50: https://ideas.repec.org/a/eee/beexfi/v41y2024ics221463502400008x.html . Lee & Seong (ICIS 2019) find barriers at the 1,000 and 10,000 levels: https://aisel.aisnet.org/icis2019/blockchain_fintech/blockchain_fintech/12/ . NEGATIVE: Urquhart (2017, Economics Letters) finds BTC clusters at round numbers but shows NO return pattern after them (daily data): https://ideas.repec.org/a/eee/ecolet/v159y2017icp145-148.html . REPO OVERLAP: breakout_continuation used the prior 24h high, and sweep_reclaim used the 4h high/low. Neither used round levels. The non-round control grid below separates 'roundness' from the generic breakout effect already rejected. REVIEWER PRIOR: low-moderate. The mechanism is well documented, but there is no crypto after-cost test. Expect a few bp, against a 12 bp perp round trip.

## R4 (SELECTED, rank 4). Lottery demand (MAX effect), two-sided weekly cross-section [A-H7]

**Psychology.** Lottery-seeking traders overpay for coins that just printed a huge green day. They are drawn by the small chance of another moonshot, a preference for skewness and the salience of the biggest gain. That over-demand should make high-MAX coins overpriced, so they underperform. In crypto, attention-driven chasing may instead keep pushing them up, so the sign is an open question.

**Definition and primary test.**

PROTOCOL: as in R1.

UNIVERSE: the 16 large coins (primary). Memes are excluded from the primary because they would fill the HIGH tercile and turn the sort into memes versus large coins. Memes alone and pooled group-neutral ranks are secondary.

DATA AND FILLS: daily bars and fill(D) as in R1.

DECISION: every Monday D at 00:00 UTC, from 2020-03-02 to 2025-09-22.

ELIGIBILITY: at least 60 days since the coin's first daily close, and finite closes on the daily bars closing D-7d to D.

SIGNAL: MAX_i = the maximum of the 7 daily log returns ln(close_d / close_(d-1)) for the daily bars closing D-6d to D (Grobys-Junttila). Requires n >= 9. Terciles: LOW = lowest MAX, HIGH = highest MAX, EW = all eligible coins.

PORTFOLIOS, COSTS, MARKING, COOLDOWN: exactly as R1. Equal weight at fill(D), buy-and-hold for 7 days, daily marks in simple returns, turnover costs at each weekly re-sort.

HORIZON: 1d marks. DIRECTION: long LOW, short HIGH (perp).

PRIMARY CELL: the daily series net(LOW) - net(HIGH), perp_taker on both legs with funding charged on the long leg, 16 large coins. TWO-SIDED test, because the literature disagrees on sign: p = the larger of the two-sided Newey-West p (lag 7) and the weekly-block bootstrap p.

ALSO REQUIRED FOR A PASS: a weekly Fama-MacBeth regression of the next 7-day return on standardised MAX, with these controls:
- 7-day return, ln(close(D) / close(D-7d));
- 28-day return skipping the latest day;
- 30-day realised volatility;
- 30-day idiosyncratic volatility: the residual standard deviation of the coin's daily returns regressed on the EW-universe daily return over the prior 30 days.
The MAX slope must have the same sign as the primary spread, with |Newey-West t| >= 1.96.

SECONDARY:
- Spot version: the tercile favoured by the primary's sign minus EW, at spot costs (descriptive only).
- 30-day MAX (Bali et al.) as a sensitivity.
- memes only.
- pooled group-neutral.

NOT TESTED: Yadav's hourly MAX, because its expected gross (about 4 bp) is below any taker cost.

**Evidence.** The papers disagree on sign. Reviewer-verified: Grobys & Junttila (2021, JIFMIM 71, 101289) use MAX = the largest daily return of the prior week on 20 coins, Jan 2016-Dec 2019, with a weekly horizon. Low-minus-high MAX earns more than 1.5%/week raw and risk-adjusted, robust to BTC risk and microstructure. https://ideas.repec.org/a/eee/intfin/v71y2021ics1042443121000081.html ; https://www.sciencedirect.com/science/article/pii/S1042443121000081 . Also reviewer-verified, with the OPPOSITE sign: Özdamar, Akdeniz & Şensoy (2021, Financial Innovation) find high-minus-low MAX deciles earn 3.03%/week raw and 1.99%/week risk-adjusted. The authors suspect MAX proxies idiosyncratic volatility. https://jfin-swufe.springeropen.com/articles/10.1186/s40854-021-00291-9 . MAX-momentum papers (e.g. NAJEF 2021) also find a positive effect: https://www.sciencedirect.com/science/article/abs/pii/S1062940821001625 . Yadav (2025, SEF), top 100 coins, hourly MAX: -0.043% the next hour per 1 SD, far below costs: https://ideas.repec.org/a/eme/sefpps/sef-07-2024-0461.html . VERDICT: real but contradictory crypto evidence, from small samples, mostly before 2021. Grobys-Junttila's 20-large-coin sample is closest to this universe, so their negative MAX effect is the more relevant prior, but the test is two-sided. REPO OVERLAP: correlated with the rejected 7-day cross-sectional momentum and with volatility. MAX itself was never sorted on, so the required Fama-MacBeth controls carry the load. Prior: low-moderate.

## R5 (SELECTED, rank 5). Deribit monthly expiry: pre-settlement dip, post-settlement rebound [B-H1]

**Psychology.** Deribit options settle on the 07:30-08:00 UTC TWAP of its index. When dealers are short gamma, their delta hedging sells into a falling market ahead of settlement, and some holders lean on the TWAP window. This flow carries no information and stops at 08:00, so the push should revert. Traders who act on 'max pain' folklore are late. The trade takes the other side of forced, information-free hedging flow.

**Definition and primary test.**

PROTOCOL: as in R1.

UNIVERSE: BTCUSDT and ETHUSDT only.

CALENDAR (known in advance, so no look-ahead): E = the last Friday of each month from 2020-01 to 2025-09 (69 dates; Deribit monthly expiry). Quarterly subset: months 3, 6, 9 and 12 (23 dates). T = 08:00 UTC all year, with no DST shift.

LEG A (primary; long; tradeable on spot):
- Event at the 1m bar closing 08:00 UTC on each E day. Direction +1.
- Entry: VWAP of the bar (08:00, 08:01].
- Horizons: h = 60 (primary) and h = 15.
- Requires Bars.gaps(60)[t] == 0.
- No condition on the pre-settlement move. The paper's claim is about the average path, and conditioning would leave about 40 events.

LEG B (secondary; perp only; short):
- Event at the bar closing 07:00 UTC on each E day. Direction -1.
- Entry: VWAP of the bar (07:00, 07:01]. h = 60, so the exit is the bar (08:00, 08:01].

COOLDOWN: one event per leg, per coin, per day.

NULLS AND DOSE-RESPONSE (same legs, same clock): (i) other Fridays (weekly expiries); (ii) Monday-Thursday (daily expiries). Both share the 08:00 funding and European-open clock but not the large expiry.

PRIMARY CELL: Leg A, monthly and quarterly E days, BTC+ETH pooled, h = 60.
Statistic: Delta = mean gross return on E days minus mean gross return of Leg A on Monday-Thursday of the same calendar months.
Inference: day-clustered bootstrap, resampling E days and null days separately. BTC and ETH on the same day form one cluster. H1: Delta > 0, one-sided.
A pass also requires the mean net return on E days > 0 under perp_taker. spot_taker is reported.

REPORTED, not tested separately: the predicted ordering quarterly > monthly > other Fridays > Monday-Thursday.

POWER: about 69 independent days. The minimum detectable Delta at the Holm level is roughly 20-25 bp, about the size the paper implies, so a null result is weak evidence of absence.

**Evidence.** Reviewer-verified citation: Weiss, Gaudiosi, Zhou & Webb, 'Bitcoin option expiration, gamma exposure, and intraday price reversals', Finance Research Letters 107 (Sept 2026), article 110340, DOI 10.1016/j.frl.2026.110340: https://www.sciencedirect.com/science/article/pii/S1544612326008688 . It covers 1,059 Deribit expiry days (2021-01 to 2023-12) with 5-minute returns. BTC weakens in the hour before 08:00 UTC and recovers over the following roughly 1.5-2 hours. The effect is strongest when at-the-money open interest is high and dealer gamma is negative. Volume is heavier on the spot venues that feed the settlement index. Reviewer note: 1,059 days over 3 years means the sample is mostly Deribit DAILY expiries, so ordinary Monday-Thursday 08:00 events are 'small expiry' days. That is why a dose-response null is used below. Settlement facts (verified via coverage): the delivery price is the 07:30-08:00 UTC TWAP of the Deribit index, and monthly expiry is the last Friday at 08:00 UTC. Single out-of-sample anecdote, 2026-09-11 expiry: -0.16% to -0.18% from 07:00 to 08:00, then +0.19% to +0.21% by 09:00, then faded by 10:00: https://cryptoslate.com/bitcoins-2-24-billion-friday-options-expiry-teased-a-reversal-then-gave-it-back-within-hours/ . Related: Hoang (2026, FRL), options trading clusters at 08:00-09:00 GMT: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5689945 . Blasco, Corredor & Satrustegui (2023, IREF) find no common return pattern around CME BTC futures expiries: https://zaguan.unizar.es/record/125822/files/texto_completo.pdf . 'Max pain' pinning is NOT supported: https://www.coindesk.com/markets/2026/06/25/forget-max-pain-bitcoin-is-well-below-the-usd72-000-magnet-ahead-of-usd10-billion-options-expiry . CAVEATS: there is a single paper with no replication, and it is in-sample. With no open-interest or gamma data, last-Friday monthly and quarterly expiries stand in for high-OI days, which dilutes the effect. 08:00 UTC is also a Binance funding timestamp and the winter London open; the Monday-Thursday null nets both out. Gross is about 20 bp, roughly one spot round trip. REPO OVERLAP: none. The ML day-of-week and hour features cannot represent a last-Friday-of-month event.

## R6 (SELECTED, rank 6). Round-number first touch: fading the crowd into the take-profit wall [merges A-H2 and C-H1 arm A]

**Psychology.** Traders treat a round price as a reference point (left-digit bias). Liquidity takers pile in with market buys just below a round number, expecting a break. Take-profit sell limits sit exactly on the number and absorb them, so price stalls and those buyers lose. The mirror case applies at support. The trade anticipates where other traders' take-profits rest and leans against crowd flow that runs into them.

**Definition and primary test.**

PROTOCOL: as in R1.

GRID, CONTROL GRIDS, TICK FILTER, GROUPS: identical to R3.

SUPPORT TOUCH at 1m bar t (long). For a level L, all of:
- low_t <= L * 1.0005;
- close_t > L;
- min(low[t-240 .. t-1]) > L * 1.0005 (the first approach to within 5 bp of L in 4h);
- Bars.gaps(241)[t] == 0.
Direction +1. If several levels qualify, use the highest.

RESISTANCE TOUCH at 1m bar t (short; perp only). All of:
- high_t >= L * 0.9995;
- close_t < L;
- max(high[t-240 .. t-1]) < L * 0.9995;
- no gaps.
Direction -1.

A bar that closes beyond the level is not a touch; it may later qualify as an R3 break.

ENTRY: VWAP of bar t+1. HORIZONS: h in {15, 60} minutes. COOLDOWN: h, per coin per grid.

PRIMARY CELL: h = 60, 16 large coins, support and resistance touches together.
Statistic: Delta = mean signed gross return on the ROUND grid minus mean signed gross return on the CONTROL grids pooled.
Inference: day-clustered bootstrap. H1: Delta > 0.
A pass also requires the ROUND-grid mean net return > 0 under perp_taker.

SINGLE PRE-REGISTERED FLOW SECONDARY (the Bhattacharya-Holden-Jacobsen crowd-flow condition):
- For resistance, add Bars.trailing_pctile(Bars.imbalance(15))[t] >= 0.8 (heavy aggressive buying into the level).
- For support, add the same percentile <= 0.2.
- Report Delta with and without the condition.

OTHER SECONDARY: h = 15; spot long-only (support touches); major versus half levels; memes.

**Evidence.** Bhattacharya, Holden & Jacobsen (2012, Management Science 58(2):413-431), more than 100M NYSE trades. Liquidity takers show excess buying 1 cent BELOW and excess selling 1 cent ABOVE round numbers. The imbalance is monotonic in roundness and strongest when price stalls at the level. Through these imbalances, 24-hour returns differ by price point, and the losing trades cost about $1bn per year. https://ideas.repec.org/a/inm/ormnsc/v58y2012i2p413-431.html ; https://host.kelley.iu.edu/cholden/Bhattacharya%20Holden%20and%20Jacobsen%20(2012).pdf . Osler (2003, JF): take-profits cluster exactly ON round numbers: https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr125.pdf . Osler (2000, FRBNY EPR): published support and resistance levels interrupt intraday trends more often than arbitrary levels: https://www.newyorkfed.org/medialibrary/media/research/epr/00v06n2/0007osle.pdf . Tsinaslanidis (2024) on BTC/ETH/BNB daily data: price bounces more often at support than at resistance, but trading rules were mixed: https://ideas.repec.org/h/spr/prbchp/978-3-031-49105-4_62.html . Crypto pricing just above and below round numbers: Hu et al. (2019, FRL): https://digitalcommons.memphis.edu/facpubs/11603 . UNVERIFIED: the cited QREF 2026 crypto replication of buy-sell imbalances (https://www.sciencedirect.com/science/article/abs/pii/S106297692600058X) could not be found by reviewer searches; treat it as unconfirmed. COUNTER-EVIDENCE: Albert Lee (SSRN 3331198) finds HFT liquidity takers show the reversed pattern, so the aggregate imbalance vanishes: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3331198 . Urquhart (2017) finds no BTC return pattern after round numbers. REPO OVERLAP: sweep_reclaim used 4h extremes, not round levels. The control grid isolates roundness. Prior: low-moderate. The taker-buy column makes the crowd-flow version directly testable, which most published studies could not do.

## R7 (SELECTED, rank 7). Turn-of-month long, days -1 to +3 [B-H5]

**Psychology.** Salary and dollar-cost-averaging buys cluster on the 1st of the month. Fund subscriptions and ETF inflows arrive at month start. Month-end rebalancing and window dressing trim positions, which are rebuilt early in the new month. Buying the day before puts the position in place before this predictable calendar flow arrives.

**Definition and primary test.**

PROTOCOL: as in R1.

UNIVERSE: an equal-weight portfolio of the 16 large coins (primary); each coin counts from its first full day. Secondary: BTC alone, ETH alone, and an equal-weight meme portfolio as a cross-asset check.

DAILY RETURNS: fill_i(D) = VWAP of the 1m bar (D 00:00, 00:01] (the 'fill' column of quant.studies.trend.daily_bars). r_EW(D) = mean over coins i of (fill_i(D+1d) / fill_i(D) - 1), using coins that have both fills. These are simple returns.

TOM DAYS (calendar, known in advance): for each month m, D is in {the last calendar day of m, day 1, day 2, day 3 of m+1}. In trading terms: buy at the 00:00 UTC fill on the last day of m and sell at the 00:00 UTC fill on day 4 of m+1.

HORIZON: 1d (four consecutive 1d holds; one round trip per month). COOLDOWN: one window per month.

PRIMARY CELL: OLS r_EW(D) = a + b * TOM(D) + e over all research days from 2020-01-01 to 2025-09-28, with Newey-West standard errors (lag 5). H1: b > 0, one-sided.
The regression measures excess return over the average day, so a bull market alone cannot pass it.
A pass also requires 4*b > the equal-weight round-trip spot cost (about 25 bp), i.e. the window must pay for its own trade.

SECONDARY: quarter-end months only; BTC, ETH and the meme portfolio separately; a placebo distribution from the same 4-day-window statistic for every possible start day of the month, with TOM's rank reported; per-year results.

**Evidence.** Mixed, and the largest recent study is negative. FOR: Turn-of-the-month effect in cryptocurrencies (Managerial Finance 2022), BTC/ETH/LTC 2015-2021, dummy regressions. Turn-of-month returns are significantly higher, not explained by day-of-week or January effects. The timing rule beats buy-and-hold by 21.8%/yr for BTC and 47.1%/yr for LTC; costs unclear. https://www.researchgate.net/publication/359327404_Turn-of-the-month_effect_in_cryptocurrencies ; https://www.emerald.com/insight/content/doi/10.1108/mf-02-2022-0084/full/pdf?title=turn-of-the-month-effect-in-cryptocurrencies . Qadan, Aharon & Eichel (2022, FRL 46) find only a within-month effect that is common across coins: https://www.sciencedirect.com/science/article/abs/pii/S1544612321003597 . AGAINST: Kaiser (2019, FRL) finds no significant turn-of-month effect: https://www.sciencedirect.com/science/article/abs/pii/S1544612318304513 . 'Revisiting seasonality in cryptocurrencies' (2024, FRL) finds no robust seasonality across about 500 coins: https://www.sciencedirect.com/science/article/pii/S1544612324004598 . Baur et al. (2019) find calendar effects that are time-varying and do not persist: https://www.researchgate.net/publication/332404798 . The equity origin is Lakonishok & Smidt (1988). Flow anecdote, with no price study: about USD 900M of quarter-end spot-ETF redemptions in Q3 2025: https://decrypt.co/341782/bitcoin-etfs-four-week-streak-quarter-end-rebalancing . REPO OVERLAP: the ML calendar features covered only hour and day-of-week, never day-of-month, so this is untested. A 4-day hold spreads the costs. POWER: low, about 69 months. Prior: low-moderate.

## R8 (SELECTED, rank 8). Kernel-detected chart patterns: head-and-shoulders, inverse H&S, double top / double bottom (Lo-Mamaysky-Wang) [C-H2]

**Psychology.** Many discretionary traders watch the same textbook shapes. They sell when a head-and-shoulders completes and buy an inverse head-and-shoulders or a double bottom. If enough of them act on it, the pattern partly fulfils itself. Detecting the shape mechanically at completion means entering before the neckline traders arrive.

**Definition and primary test.**

PROTOCOL: as in R1.

BARS: 4h bars from the 1m data (quant.data.resample; UTC boundaries 00/04/.../20; close-time index). A 4h bar with more than 5 missing minutes is NaN, and any window containing it is skipped.

SMOOTHING, at each 4h close t:
- x = the log closes of the 38 bars t-37 .. t.
- Fit a Nadaraya-Watson Gaussian kernel regression on the bar index. Bandwidth = 0.3 * h_CV, where h_CV minimises the leave-one-out CV error over the grid {0.5, 0.6, ..., 10.0} bars, computed inside this window only.
- Extrema are the points tau where the first difference of the smoothed curve m(tau) changes sign. Map each to the highest close (for a maximum) or lowest close (for a minimum) in [tau-1, tau+1].
- Keep only extrema at index <= t-3 (a 3-bar confirmation lag, LMW's d = 3).
- Let E1..En be these extrema in time order, with their closes as prices.

PATTERNS on the five most recent alternating extrema E1..E5 (E5 the most recent):
- HS: E1 is a maximum; E3 > E1; E3 > E5; |E1 - E5| <= 0.015 * mean(E1, E5); |E2 - E4| <= 0.015 * mean(E2, E4). Direction -1.
- IHS: E1 is a minimum; E3 < E1; E3 < E5; the same two tolerances. Direction +1.

PATTERNS on all extrema in the window:
- DTOP: E1 (the oldest) is a maximum. Ea = the highest later maximum with t_a - t_1 > 22 bars, and |E1 - Ea| <= 0.015 * mean(E1, Ea). Direction -1.
- DBOT: the mirror, using minima. Direction +1.

FIRING: a pattern type fires on the first 4h close at which it is detected (i.e. not detected at the previous 4h close). COOLDOWN: 38 4h-bars per (coin, pattern type). If a bullish and a bearish pattern fire on the same bar, there is no event.

EVENT TIME: the 1m bar closing at the 4h boundary. ENTRY: VWAP of the next 1m bar. HORIZONS: h in {240, 1440} minutes.

UNIVERSE: the 16 large coins (primary).

PRIMARY CELL: all four patterns pooled by direction, h = 1440, 16 large coins, perp_taker. H1: mean net return > 0 (day bootstrap), and the excess over the placebo matched on hour and volatility must be > 0.

SECONDARY:
- h = 240.
- each pattern separately.
- spot long-only (IHS + DBOT).
- LMW's Kolmogorov-Smirnov test of normalised post-pattern returns against unconditional returns.
- Incremental test: regress event returns on the coin's trend-overlay weight that day; the pattern must add to the trend.

**Evidence.** Lo, Mamaysky & Wang (2000, JF 55:1705-1765; NBER w7613) detected patterns with kernel regression on US stocks 1962-1996. Return distributions after several patterns, including H&S and double bottoms, differ significantly from unconditional returns. This is NOT a profitability test, and the authors say so. https://www.nber.org/papers/w7613 . Savin, Weller & Zvingelis (2007, J. Financial Econometrics 5(2):243-265): H&S tops predict risk-adjusted excess returns of 5-7%/yr on the S&P 500 and Russell 2000 (1990-99), but give little support for a stand-alone strategy: https://ideas.repec.org/a/oup/jfinec/v5yi2p243-265.html . Chang & Osler (1999, Economic Journal 109:636-661): H&S was profitable on daily USD/JPY and USD/DEM with bootstrap significance, but simple moving-average and filter rules dominated it: https://ideas.repec.org/a/ecj/econjl/v109y1999i458p636-61.html . Chong & Poon (MPRA 60825): https://mpra.ub.uni-muenchen.de/60825 . CRYPTO: Rink (2025, Financial Innovation) applied the LMW detector to hourly Mt.Gox data 2011-13. Buy signals coincided with +53% abnormal volume, which shows traders acted on these patterns in an immature market; no modern out-of-sample edge was shown. https://jfin-swufe.springeropen.com/articles/10.1186/s40854-025-00763-2 . VERDICT: statistically informative in equities and FX, never shown profitable after costs, and untested in modern crypto. The main risk is that any effect is just trend or reversal already captured, so the incremental test against the trend overlay is reported. REPO OVERLAP: the ML used single-bar shape, sweep and breakout features, but no multi-extremum pattern detector, so this is untested. Every parameter is fixed at LMW's published values. Prior: low-moderate. It is included because it is the canonical, falsifiable form of the chart reading the user asked about.

## Not selected / removed

- **R9 (NOT SELECTED, rank 9). Pre-FOMC drift [B-H6]**: NOT SELECTED; frozen in case it is ever run. Uses a hardcoded calendar of scheduled FOMC statement dates (federalreserve.gov), about 46 between 2020-01 and 2025-09, excluding the unscheduled actions of 2020-03-03 and 2020-03-15. T = 14:00 America/New_York, converted to UTC. Event at the 1m bar closing T - 1445 min; direction +1; entry VWAP of bar t+1; h = 1440, so the exit bar closes at T - 4 min,
- **R10 (NOT SELECTED, rank 10). Candle streaks: gambler's fallacy versus hot hand [A-H6]**: NOT SELECTED; frozen in case it is ever run. Daily UTC closes. s_d = sign(close_d - close_(d-1)), where 0 breaks the streak. Event at 00:00 UTC of day d+1 when the streak length first reaches exactly 4. Direction = -s_d (reversal); entry 00:01 VWAP; h = 1d; one event per streak. Up-streaks and down-streaks are reported separately. The result must hold with 08:00 and 16:00 UTC day boundaries too. p
- **R11 (NOT SELECTED, rank 11). US equity-open reversal of the pre-open drift ('10am dump') [B-H2]**: NOT SELECTED; frozen in case it is ever run. NYSE trading days, using the R2 calendar. T_o = 09:30 America/New_York, converted to UTC. r_pre = lc[T_o] - lc[T_o - 240]; z = r_pre / (Bars.sigma * sqrt(240)). If |z| >= 1, direction = -sign(r_pre); entry VWAP of bar (T_o, T_o+1m]; h = 60; one event per day. Nulls: the same rule at the wrong UTC hour under DST (falsification), and the same rule on non-
- **R12 (NOT SELECTED, rank 12). Seller exhaustion: RSI plus taker-flow divergence at a lower low [C-H5]**: NOT SELECTED; frozen in case it is ever run. 1h bars. A swing low A at bar j is confirmed at bar j+3 when low_j = min(low[j-3 .. j+3]). At each 1h close t, take the latest confirmed A with t-48 <= j_A <= t-6. Long if all hold: low_t < low_A; RSI14(A) < 30; RSI14(t) >= RSI14(A) + 5; 1h taker imbalance at t > imbalance at A; close_t in the upper half of bar t. h in {4h, 1d}; cooldown 24h. Control: t
- **X1 (REMOVED: already tested). First cross of the 365-day high/low [A-H4]**: NOT TO BE RUN. Duplicates the Donchian trend overlay (quant/studies/trend.py) and the rejected h_breakout_continuation (quant/studies/event_hypotheses.py). The 52-week anchor survives only as the cross-sectional R1.
- **X2 (REMOVED: already tested). Post-extreme-day drift, Caporale-Plastun [A-H5]**: NOT TO BE RUN. A renamed version of the rejected market_shock_continuation, flow_driven_reversal and cascade_reversal (event_hypotheses.py) and of the daily ML momentum features (daily_panel.py).
- **X3 (REMOVED: already tested). Post-US-close / Asian pre-open drift, long 21:00-23:00 UTC [B-H4]**: NOT TO BE RUN as a hypothesis. A renamed hour-of-day calendar effect (features.calendar). Optional engineering check only: move the trend overlay's fills to 21:00 UTC (buys) and 23:00 UTC (sells) and measure the change in net return.
- **X4 (REMOVED: already tested). CME gap fill and weekly-open 'magnet' [B-H8]**: NOT TO BE RUN. Duplicates h_weekend_reversal (event_hypotheses.py), and the mechanism no longer exists.
- **X5 (REMOVED: already tested). Wyckoff spring/upthrust, hammer at range low, selling climax [C-H8]**: NOT TO BE RUN. Duplicates h_sweep_reclaim, h_flow_driven_reversal and h_cascade_reversal (event_hypotheses.py).
- **X6 (REMOVED: already tested). Multi-bar candlestick reversals: engulfing, three outside up, morning star, hammer [C-H6]**: NOT TO BE RUN. A renamed bar-shape signal (features.py shape__clv/upwick/lowwick/sweep/breakout), which was rejected.
- **X7 (REMOVED: already tested). Anchored-VWAP deviation reversion [C-H7]**: NOT TO BE RUN. A renamed short-horizon reversal (h_reversal_15m_control, h_flow_driven_reversal, ML momentum features).
- **X8 (REMOVED: not testable). Funding-settlement clock: crowded longs sell before 00/08/16 UTC, then price rebounds [B-H7]**: NOT TO BE RUN as an edge test. Optional engineering check only, non-directional: compare the aggressor-side effective spread (execution_realism.py) in T-15m to T+15m around 00/08/16 UTC with the off-cycle 04/12/20 UTC times. If spreads are reliably wider, add 'avoid entering within 15 minutes of funding timestamps' as an execution rule.
