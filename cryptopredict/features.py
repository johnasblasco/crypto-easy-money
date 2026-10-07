"""Technical indicators, model features and the UP/DOWN target.

Every feature at row ``t`` uses only candles ``<= t``. The target at row
``t`` looks ``horizon`` candles ahead and is the only forward-looking column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Features fed to the models. They are ratios and oscillators rather than raw
# prices so that a model trained at $30k still makes sense at $100k.
FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "close_vs_sma20",
    "close_vs_sma50",
    "ema12_vs_ema26",
    "rsi_14",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "bb_pct_b",
    "bb_width",
    "volume_change",
    "volume_ratio_20",
    "hl_range",
    "candle_body",
    "volatility_12",
]


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # No losses in the window means maximum strength.
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd line, signal line, histogram)."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Return (middle, upper, lower) Bollinger Bands."""
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return mid, mid + num_std * std, mid - num_std * std


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add raw indicator columns (in price units) used for display and features."""
    out = df.copy()
    close = out["close"]
    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)
    out["ema_12"] = ema(close, 12)
    out["ema_26"] = ema(close, 26)
    out["rsi_14"] = rsi(close, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close)
    out["bb_mid"], out["bb_upper"], out["bb_lower"] = bollinger(close)
    out["volume_change"] = out["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators plus the scale-free model features in ``FEATURE_COLUMNS``."""
    out = add_indicators(df)
    close = out["close"]

    for n in (1, 3, 6, 12):
        out[f"ret_{n}"] = close.pct_change(n)
    out["close_vs_sma20"] = close / out["sma_20"] - 1
    out["close_vs_sma50"] = close / out["sma_50"] - 1
    out["ema12_vs_ema26"] = out["ema_12"] / out["ema_26"] - 1
    out["macd_norm"] = out["macd"] / close
    out["macd_signal_norm"] = out["macd_signal"] / close
    out["macd_hist_norm"] = out["macd_hist"] / close
    band = out["bb_upper"] - out["bb_lower"]
    out["bb_pct_b"] = ((close - out["bb_lower"]) / band).where(band > 0, 0.5)
    out["bb_width"] = band / out["bb_mid"]
    out["volume_ratio_20"] = out["volume"] / sma(out["volume"], 20)
    out["hl_range"] = (out["high"] - out["low"]) / close
    out["candle_body"] = (close - out["open"]) / out["open"]
    out["volatility_12"] = out["ret_1"].rolling(12, min_periods=12).std()

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add ``future_return`` and ``target`` (1 = UP, 0 = DOWN).

    target = 1 if close[t + horizon] > close[t] else 0. The last ``horizon``
    rows have no future yet, so their target is NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    out = df.copy()
    future_close = out["close"].shift(-horizon)
    out["future_return"] = future_close / out["close"] - 1
    out["target"] = (future_close > out["close"]).astype(float).where(future_close.notna())
    return out


def build_dataset(df: pd.DataFrame, horizon: int = 1):
    """Return (X, y, frame) with every row that has full features and a target."""
    frame = add_target(add_features(df), horizon)
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    return frame[FEATURE_COLUMNS], frame["target"].astype(int), frame
