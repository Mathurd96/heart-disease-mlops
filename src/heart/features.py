"""Preprocessing / feature engineering as a reusable sklearn transformer.

The whole preprocessing lives INSIDE the model Pipeline, so the exact same
transformations are applied at training time, in cross-validation and at
inference time in the API: no training/serving skew.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Adds a few domain-inspired features on top of the raw columns.

    * ``hr_reserve``      : fraction of the age-predicted max heart rate reached
                            (thalach / (220 - age)). Low values suggest poor
                            cardiac response to exercise.
    * ``bp_chol_load``    : trestbps * chol / 1000, a crude combined
                            cardiovascular load indicator.
    * ``st_depression``   : oldpeak > 1 flag (clinically meaningful ST drop).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["hr_reserve"] = X["thalach"] / (220 - X["age"]).clip(lower=1)
        X["bp_chol_load"] = X["trestbps"] * X["chol"] / 1000.0
        X["st_depression"] = (X["oldpeak"] > 1.0).astype(int)
        return X

    def set_output(self, *, transform=None):  # keep DataFrame output
        return self


ENGINEERED_NUMERIC = ["hr_reserve", "bp_chol_load"]
ENGINEERED_BINARY = ["st_depression"]


def build_preprocessor(engineer: bool = True) -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categoricals."""
    numeric = config.NUMERIC_FEATURES + (ENGINEERED_NUMERIC if engineer else [])
    categorical = config.CATEGORICAL_FEATURES + (ENGINEERED_BINARY if engineer else [])

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def build_pipeline(estimator, engineer: bool = True) -> Pipeline:
    """Full model pipeline: engineered features -> preprocessing -> estimator."""
    steps = []
    if engineer:
        steps.append(("features", ClinicalFeatures()))
    steps.append(("preprocess", build_preprocessor(engineer)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the columns produced by the preprocessor (after one-hot)."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
