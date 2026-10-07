import numpy as np
import pandas as pd

from cryptopredict.data import generate_synthetic
from cryptopredict.features import FEATURE_COLUMNS, add_features, add_target, build_dataset, rsi


def test_rsi_is_bounded():
    df = generate_synthetic(1000)
    values = rsi(df["close"]).dropna()
    assert values.between(0, 100).all()


def test_rsi_extremes():
    rising = pd.Series(np.arange(1, 50, dtype=float))
    assert rsi(rising).dropna().eq(100).all()
    falling = rising[::-1].reset_index(drop=True)
    assert rsi(falling).dropna().eq(0).all()


def test_target_marks_next_close_direction():
    df = pd.DataFrame({"close": [10.0, 11.0, 10.5, 10.5, 12.0]})
    out = add_target(df, horizon=1)
    assert out["target"].tolist()[:4] == [1, 0, 0, 1]  # equal close counts as DOWN
    assert np.isnan(out["target"].iloc[-1])  # no future for the last candle


def test_target_horizon():
    df = pd.DataFrame({"close": [10.0, 9.0, 12.0, 8.0]})
    out = add_target(df, horizon=2)
    assert out["target"].tolist()[:2] == [1, 0]
    assert out["target"].iloc[2:].isna().all()


def test_features_do_not_look_ahead():
    """Changing the future must not change any feature in the past."""
    df = generate_synthetic(600)
    cut = 400
    altered = df.copy()
    altered.loc[cut:, ["open", "high", "low", "close"]] *= 1.5
    altered.loc[cut:, "volume"] *= 3

    a = add_features(df).loc[: cut - 1, FEATURE_COLUMNS]
    b = add_features(altered).loc[: cut - 1, FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(a, b)


def test_build_dataset_is_clean():
    X, y, frame = build_dataset(generate_synthetic(800))
    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isna().any().any()
    assert np.isfinite(X.to_numpy()).all()
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == len(frame)
    assert frame["timestamp"].is_monotonic_increasing
