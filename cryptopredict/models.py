"""The classifiers compared in the study."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost is optional
    XGBClassifier = None


def get_models(random_state: int = 42) -> dict:
    """Return name -> unfitted estimator, simplest model first.

    Hyperparameters are deliberately conservative (shallow trees, large
    leaves): with noisy financial data, flexible models overfit quickly.
    """
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=50,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            reg_lambda=1.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )
    return models
