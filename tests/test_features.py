import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from heart import config
from heart.features import ClinicalFeatures, build_pipeline, build_preprocessor, feature_names


def test_clinical_features_adds_expected_columns(synthetic_df):
    out = ClinicalFeatures().transform(synthetic_df[config.FEATURES])
    for col in ("hr_reserve", "bp_chol_load", "st_depression"):
        assert col in out.columns
    assert out["st_depression"].isin([0, 1]).all()
    row = synthetic_df.iloc[0]
    expected = row["thalach"] / (220 - row["age"])
    assert np.isclose(out.iloc[0]["hr_reserve"], expected)


def test_clinical_features_does_not_mutate_input(synthetic_df):
    X = synthetic_df[config.FEATURES].copy()
    before = X.copy()
    ClinicalFeatures().transform(X)
    pd.testing.assert_frame_equal(X, before)


def test_preprocessor_handles_missing_values(synthetic_df):
    X = synthetic_df[config.FEATURES]
    assert X.isna().any().any()  # fixture really has gaps
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, synthetic_df[config.TARGET])
    Xt = pipe.named_steps["preprocess"].transform(pipe.named_steps["features"].transform(X))
    dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert not np.isnan(dense).any()


def test_numeric_columns_are_standardised(synthetic_df):
    X = synthetic_df[config.FEATURES]
    pre = build_preprocessor(engineer=False).fit(X)
    Xt = pre.transform(X)
    dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    n_num = len(config.NUMERIC_FEATURES)
    assert np.allclose(dense[:, :n_num].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(dense[:, :n_num].std(axis=0), 1, atol=1e-6)


def test_unseen_category_does_not_crash(synthetic_df):
    X = synthetic_df[config.FEATURES]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, synthetic_df[config.TARGET])
    odd = X.head(2).copy()
    odd["cp"] = 99  # never seen during training
    assert pipe.predict_proba(odd).shape == (2, 2)


def test_feature_names_match_transformed_width(synthetic_df):
    X = synthetic_df[config.FEATURES]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, synthetic_df[config.TARGET])
    Xt = pipe.named_steps["preprocess"].transform(pipe.named_steps["features"].transform(X))
    assert len(feature_names(pipe)) == Xt.shape[1]
