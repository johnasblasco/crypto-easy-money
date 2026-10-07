"""Chronological walk-forward splits with purging and embargo.

Every test fold is strictly later than its training data. A training sample
is *purged* if its label window (decision time -> label end) reaches into the
test fold, because its target would then be computed from test-period prices.
An *embargo* additionally drops training samples whose decision time falls
within ``embargo`` before the test start, to absorb serial correlation in
features (e.g. long rolling windows).

The same purging is applied to the inner validation slice that is carved out
of each training window for calibration / threshold selection, so nothing
fitted on "validation" has seen test data either.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    train: np.ndarray        # integer positions
    valid: np.ndarray        # inner validation (later part of the training window)
    test: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def fit(self) -> np.ndarray:
        """Training positions excluding the inner validation slice."""
        return np.setdiff1d(self.train, self.valid, assume_unique=True)


def _purge(times: pd.DatetimeIndex, label_end: pd.DatetimeIndex, candidates: np.ndarray,
           boundary: pd.Timestamp, embargo: pd.Timedelta) -> np.ndarray:
    keep = (label_end[candidates] < boundary) & (times[candidates] < boundary - embargo)
    return candidates[keep]


def walk_forward(times: pd.DatetimeIndex, label_end: pd.DatetimeIndex, test_size: str = "90D",
                 min_train: str = "365D", train_window: str | None = None, valid_frac: float = 0.2,
                 embargo: str = "0D", start: str | None = None) -> list[Fold]:
    """Expanding (or rolling, if ``train_window``) walk-forward folds.

    ``times`` must be sorted decision times; ``label_end`` the time each
    sample's label is fully known. Test folds tile the period after
    ``min_train`` (or from ``start``) in ``test_size`` blocks.
    """
    times = pd.DatetimeIndex(times)
    label_end = pd.DatetimeIndex(label_end)
    if not times.is_monotonic_increasing:
        raise ValueError("times must be sorted")
    test_size_td, embargo_td = pd.Timedelta(test_size), pd.Timedelta(embargo)
    first_test = pd.Timestamp(start, tz="UTC") if start else times[0] + pd.Timedelta(min_train)
    folds = []
    t0 = first_test
    all_pos = np.arange(len(times))
    while t0 < times[-1]:
        t1 = t0 + test_size_td
        test = all_pos[(times >= t0) & (times < t1)]
        lo = t0 - pd.Timedelta(train_window) if train_window else times[0]
        cand = all_pos[(times >= lo) & (times < t0)]
        train = _purge(times, label_end, cand, t0, embargo_td)
        if len(test) and len(train):
            # Inner validation = last valid_frac of the training window (by time),
            # purged against its own boundary so the fit part can't see it.
            v_start = times[train[0]] + (times[train[-1]] - times[train[0]]) * (1 - valid_frac)
            valid = train[times[train] >= v_start]
            fit_cand = train[times[train] < v_start]
            fit = _purge(times, label_end, fit_cand, v_start, embargo_td)
            train = np.concatenate([fit, valid])
            folds.append(Fold(train=train, valid=valid, test=test, test_start=t0, test_end=t1))
        t0 = t1
    return folds


def check_no_leakage(folds: list[Fold], times: pd.DatetimeIndex, label_end: pd.DatetimeIndex) -> None:
    """Raise if any training/fit label is known only after its evaluation boundary."""
    times = pd.DatetimeIndex(times)
    label_end = pd.DatetimeIndex(label_end)
    for f in folds:
        if len(f.train) and label_end[f.train].max() >= f.test_start:
            raise AssertionError(f"train label overlaps test starting {f.test_start}")
        if len(f.test) and times[f.test].min() < f.test_start:
            raise AssertionError("test sample before test start")
        fit = f.fit
        if len(fit) and len(f.valid) and label_end[fit].max() >= times[f.valid].min():
            raise AssertionError(f"fit label overlaps validation in fold starting {f.test_start}")
