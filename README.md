# Cryptocurrency Market Direction Classification Using Machine Learning and Technical Indicators

> **Research question:** Can historical cryptocurrency market data and technical
> indicators be used to classify whether the next trading interval will see an
> upward or downward price movement?

This project trains several classifiers to predict whether the **next candle
closes UP or DOWN**, compares them against naive baselines, and serves the best
one in a web dashboard with a live candlestick chart.

![Dashboard](docs/dashboard.png)
<sub>Screenshot uses the built-in synthetic data, so the numbers in it say nothing about real markets.</sub>

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Download ~7 months of BTC/USDT hourly candles from Binance and train
python -m cryptopredict.train --source binance --symbol BTCUSDT --interval 1h --limit 5000

# 2. One-off prediction in the terminal
python -m cryptopredict.predict

# 3. Dashboard at http://127.0.0.1:8000
uvicorn app.server:app --reload
```

No internet, or Binance blocked where you are? Use `--source synthetic` to run
the whole pipeline on generated candles, or `--source csv --csv path/to/file.csv`
with any CSV that has `timestamp,open,high,low,close,volume` columns.

Run the tests with `pytest`.

## How it works

```
Binance klines ─► indicators & features ─► UP/DOWN target ─► chronological split
                                                                 │
                ┌────────────── walk-forward CV picks the best ◄─┤ train (80%)
                ▼                                                 │
   held-out evaluation vs baselines ◄──────────────────────────── test (20%)
                │
                ▼
   best model refit on all data ─► models/model.joblib ─► dashboard / predict
```

### 1. Data — `cryptopredict/data.py`
OHLCV candles from Binance's public API (no API key). It pages back past the
1000-candle limit, drops the still-forming candle, and falls back to
`api.binance.com` and `api.binance.us` if `data-api.binance.vision` is unreachable.
Downloads are cached in `data/<SYMBOL>_<interval>.csv`.

### 2. Features — `cryptopredict/features.py`
| Indicator | Feature(s) given to the model |
|---|---|
| Returns | 1, 3, 6 and 12-candle % change |
| SMA 20 / 50 | % distance of close from each average |
| EMA 12 / 26 | % gap between the two EMAs (trend) |
| RSI 14 | RSI value (Wilder smoothing) |
| MACD 12/26/9 | MACD, signal and histogram, divided by price |
| Bollinger Bands 20, 2σ | %B (position inside the bands) and band width |
| Volume | % change, and ratio to its 20-candle average |
| Candle shape | high−low range, candle body, 12-candle volatility |

All features are **ratios or oscillators, not raw prices**, so a model trained
when BTC was $60k still makes sense at $100k.

**Target:** `1 (UP)` if `close[t+horizon] > close[t]`, otherwise `0 (DOWN)`.

### 3. Models — `cryptopredict/models.py`
Logistic Regression (baseline ML model), Random Forest, Gradient Boosting
(scikit-learn) and XGBoost. Hyperparameters are deliberately conservative
(shallow trees, big leaves) because noisy market data overfits easily.

### 4. Evaluation — `cryptopredict/train.py`
These are the points a reviewer is most likely to probe:

- **No shuffling.** The data is split by time: first 80% train, last 20% test.
  A random split would let the model "see the future".
- **No look-ahead.** Each feature at time *t* uses only candles up to *t*. A
  unit test (`test_features_do_not_look_ahead`) changes future prices and checks
  that past features stay the same.
- **A gap between train and test.** The last `horizon` training rows are dropped
  so no training label overlaps the test period.
- **Model selection without peeking.** The best model is chosen by
  **walk-forward cross-validation** (`TimeSeriesSplit`) on the training set, not
  by its test score.
- **Baselines.** Every model is compared with *always predict the majority class*
  and *repeat the last move*. A model is only interesting if it beats them.
- **Metrics:** accuracy, precision, recall, F1 and ROC AUC, plus a toy long-only
  backtest with 0.1% fees compared against buy & hold.
- **Explainability:** permutation importance, measured as the drop in test ROC
  AUC when a feature is shuffled.

Outputs in `models/`:
- `report.md` — results table, ready to paste into the paper
- `metrics.json` — every number
- `test_predictions.csv` — out-of-sample predictions
- `model.joblib` — the deployed model

### 5. Dashboard — `app/`
FastAPI backend with a [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts)
frontend (vendored under `app/static/vendor`, Apache-2.0). It shows:

- Candlesticks with EMA 12/26, plus synced volume and RSI panes
- The **next-candle call** (▲ UP / ▼ DOWN), its confidence and P(UP)
- Signals for RSI, MACD, EMA trend, Bollinger %B, volume and SMA
- **Past predictions on the chart** (▲/▼ markers, ✓ right / ✗ wrong). Only
  out-of-sample candles are marked: the held-out test set plus candles that
  arrived after training. "Recent hit rate" is computed from these.
- The model comparison table

The dashboard refreshes every minute. Settings come from environment variables:
- `DATA_SOURCE` — `binance`, `csv` or `synthetic`
- `CANDLES` — number of candles on the chart
- `MODEL_PATH` — trained model to load

## Experiments to run for the paper

The CLI flags map directly to the research variables:

```bash
# Different coins
python -m cryptopredict.train --symbol ETHUSDT
python -m cryptopredict.train --symbol SOLUSDT
# Different time intervals
python -m cryptopredict.train --interval 15m --limit 10000
python -m cryptopredict.train --interval 4h
python -m cryptopredict.train --interval 1d --limit 2000
# Different prediction horizons (e.g. direction 4 candles ahead)
python -m cryptopredict.train --horizon 4
# Separate output folders keep runs side by side
python -m cryptopredict.train --symbol ETHUSDT --interval 4h --out models/eth_4h
```

To study which indicators matter, edit `FEATURE_COLUMNS` in
`cryptopredict/features.py` and compare runs (ablation study).

## What results to expect

Expect accuracy of roughly **50–58%**. Crypto prices are close to a random
walk at short horizons. Results well above that usually point to a bug (most
often look-ahead leakage), not a discovery. Also check the test set's UP share:
if 55% of test candles went up, 55% accuracy is just "always UP". Report
**ROC AUC** and the comparison with the baselines, not accuracy alone.

## Limitations and future work

- **Market regimes change.** A model trained in a bull market may fail in a bear market. Retrain periodically or use rolling-window training.
- **The backtest is a toy.** It ignores slippage and spread and only holds one candle at a time (horizon = 1).
- **Possible extensions:**
  - An LSTM or Temporal CNN for comparison
  - Sentiment or on-chain features
  - A "no trade" band that only acts when confidence passes a threshold
  - Probability calibration
  - Hyperparameter search inside the walk-forward CV

## Project structure

```
cryptopredict/
  data.py        Binance / CSV / synthetic loaders
  features.py    indicators, features, target
  models.py      classifiers compared in the study
  train.py       evaluation pipeline + CLI
  predict.py     next-candle prediction + CLI
app/
  server.py      FastAPI backend (/api/dashboard)
  static/        dashboard HTML/CSS/JS + vendored chart library
tests/           unit and end-to-end tests
```

*Educational project — not financial advice.*
