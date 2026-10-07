"""Model ladder, simplest first. A more complex rung must beat the one below it.

All models expose ``fit(X, y)`` and ``predict_proba(X)``; inputs are numpy
arrays. Hyperparameters are fixed in advance (not tuned on test data) and
deliberately conservative: financial labels are mostly noise.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BaseRate:
    """Predicts the training up-frequency for everything (the no-information model)."""

    def fit(self, X, y):
        self.p = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.column_stack([1 - p, p])


class Clipped:
    """Winsorize features at training quantiles before the wrapped model (robust to fat tails)."""

    def __init__(self, model, q: float = 0.005):
        self.model, self.q = model, q

    def fit(self, X, y):
        self.lo = np.nanquantile(X, self.q, axis=0)
        self.hi = np.nanquantile(X, 1 - self.q, axis=0)
        self.model.fit(np.clip(X, self.lo, self.hi), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(np.clip(X, self.lo, self.hi))


def logistic(C: float = 0.05):
    return lambda: Clipped(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)))


def lightgbm(n_estimators: int = 300, learning_rate: float = 0.02, num_leaves: int = 15,
             min_child_samples: int = 400, n_jobs: int = 2, seed: int = 0):
    import lightgbm as lgb

    return lambda: lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
        min_child_samples=min_child_samples, subsample=0.7, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=10.0, n_jobs=n_jobs, random_state=seed, verbose=-1,
    )


def base_rate():
    return BaseRate
