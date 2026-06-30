"""
ML pipeline for the Insurance Charges dashboard.

Pure Python (no Streamlit) so it can be imported, reused, and unit-tested
independently of the UI. Handles data loading, feature engineering,
model definitions, training/evaluation, and cross-validation.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
    AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
)

# Optional gradient-boosting libraries — the app degrades gracefully if a
# wheel is unavailable on the deployment target.
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:                       # pragma: no cover
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:                       # pragma: no cover
    HAS_LGBM = False

RANDOM_STATE = 42
REGIONS = ["northeast", "northwest", "southeast", "southwest"]
LINEAR_MODELS = {"Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet"}

# Canonical feature order — training and single-row inference must align to this.
FEATURE_ORDER = [
    "age", "sex", "bmi", "children", "smoker",
    "region_northwest", "region_southeast", "region_southwest",
    "age_sq", "bmi_obese", "smoker_bmi", "smoker_obese", "smoker_age",
]

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "InsuranceLR.csv")


# ----------------------------------------------------------------------
# Data + feature engineering
# ----------------------------------------------------------------------
def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and drop the non-predictive row index."""
    df = pd.read_csv(path)
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    return df


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering and return columns aligned to FEATURE_ORDER.

    Works for both the full training frame and a single-row inference frame.
    """
    fe = df.copy()
    fe["sex"] = (fe["sex"].astype(str).str.lower() == "male").astype(int)      # female=0, male=1
    fe["smoker"] = (fe["smoker"].astype(str).str.lower() == "yes").astype(int)  # no=0, yes=1

    fe = pd.get_dummies(fe, columns=["region"], dtype=int)
    for r in REGIONS:                                   # guarantee every region dummy exists
        col = f"region_{r}"
        if col not in fe.columns:
            fe[col] = 0
    fe = fe.drop(columns=["region_northeast"])          # northeast = baseline

    # domain-driven engineered features (smoker x bmi is the dominant driver)
    fe["age_sq"] = fe["age"] ** 2
    fe["bmi_obese"] = (fe["bmi"] >= 30).astype(int)
    fe["smoker_bmi"] = fe["smoker"] * fe["bmi"]
    fe["smoker_obese"] = fe["smoker"] * fe["bmi_obese"]
    fe["smoker_age"] = fe["smoker"] * fe["age"]

    return fe.reindex(columns=FEATURE_ORDER, fill_value=0)


def engineer_features(df: pd.DataFrame):
    """Return (X, y) for the full dataset. Expects a `charges` column."""
    X = _build_features(df)
    y = df["charges"].astype(float)
    return X, y


def build_input_features(record: dict) -> pd.DataFrame:
    """Build a single aligned feature row from raw user inputs (for prediction)."""
    return _build_features(pd.DataFrame([record]))


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def get_models() -> dict:
    """Return the full model roster. Linear models are scaled (penalties need it);
    tree models are scale-invariant and used as-is."""
    alphas = np.logspace(-3, 3, 50)

    def scaled(est):
        return Pipeline([("scaler", StandardScaler()), ("model", est)])

    models = {
        # linear / regularized family
        "Linear Regression": scaled(LinearRegression()),
        "Ridge Regression": scaled(RidgeCV(alphas=alphas)),
        "Lasso Regression": scaled(
            LassoCV(alphas=alphas, max_iter=20000, cv=5, random_state=RANDOM_STATE)),
        "ElasticNet": scaled(
            ElasticNetCV(alphas=alphas, l1_ratio=[.1, .5, .7, .9, .95, 1.0],
                         max_iter=20000, cv=5, random_state=RANDOM_STATE)),
        # decision-tree family
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Bagging (DT)": BaggingRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "AdaBoost (DT)": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=4),
            n_estimators=300, learning_rate=0.5, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9,
            colsample_bytree=0.9, random_state=RANDOM_STATE, verbosity=0)
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=RANDOM_STATE, verbose=-1)
    return models


# ----------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------
def split_data(X, y, test_size=0.20, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Fit every model and score it on the test set.

    Returns (results_df, fitted_models, predictions) where results_df is
    sorted by R^2 descending and carries R2/MAE/RMSE columns.
    """
    models = get_models()
    fitted, preds, rows = {}, {}, []
    for name, est in models.items():
        est.fit(X_train, y_train)
        p = est.predict(X_test)
        fitted[name] = est
        preds[name] = p
        rows.append({
            "Model": name,
            "Family": "Linear" if name in LINEAR_MODELS else "Tree",
            "R2": r2_score(y_test, p),
            "MAE": mean_absolute_error(y_test, p),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, p))),
        })
    results = (pd.DataFrame(rows)
               .sort_values("R2", ascending=False)
               .reset_index(drop=True))
    return results, fitted, preds


def cross_validate_models(X_train, y_train, n_splits=5, random_state=RANDOM_STATE):
    """5-fold cross-validated R^2 for every model (robustness check)."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    out = {}
    for name, est in get_models().items():
        scores = cross_val_score(est, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        out[name] = float(scores.mean())
    return out


def feature_importance(est, feature_names=FEATURE_ORDER):
    """Return a Series of importances (tree models) or |coefficients| (linear)."""
    model = est.named_steps["model"] if hasattr(est, "named_steps") else est
    if hasattr(model, "feature_importances_"):
        vals, kind = model.feature_importances_, "importance"
    elif hasattr(model, "coef_"):
        vals, kind = np.abs(np.ravel(model.coef_)), "|coefficient| (scaled)"
    else:
        return None, None
    return pd.Series(vals, index=feature_names).sort_values(), kind
