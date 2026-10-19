import numpy as np
import pandas as pd
import pytest

from heart import config
from heart.data import clean, load_raw, prepare_dataset, split_xy, validate


def _raw_frame():
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, 3, 0],
        [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, "?", 3, 0],
        [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, 2, 3, 3],
        [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, 2, 3, 3],  # duplicate
    ]
    return pd.DataFrame(rows, columns=config.RAW_COLUMNS)


def test_clean_binarises_target_and_drops_duplicates():
    out = clean(_raw_frame())
    assert "num" not in out.columns
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert len(out) == 5  # duplicate removed
    # num 2 and 3 -> 1; num 0 -> 0
    assert out[config.TARGET].tolist() == [0, 1, 0, 0, 1]


def test_clean_converts_question_marks_to_nan():
    out = clean(_raw_frame())
    assert out["ca"].isna().sum() == 1
    assert pd.api.types.is_numeric_dtype(out["ca"])


def test_validate_accepts_clean_data_and_flags_problems():
    ok = clean(_raw_frame())
    assert validate(ok) == []

    bad = ok.copy()
    bad.loc[0, "age"] = 500
    bad.loc[1, "cp"] = 9
    problems = validate(bad)
    assert any("age" in p for p in problems)
    assert any("cp" in p for p in problems)


def test_validate_reports_missing_columns():
    assert "missing columns" in validate(pd.DataFrame({"age": [1]}))[0]


def test_load_raw_and_prepare_dataset_roundtrip(tmp_path):
    raw = tmp_path / "raw.data"
    _raw_frame().to_csv(raw, header=False, index=False)
    assert load_raw(raw).shape[1] == 14
    out = tmp_path / "clean.csv"
    df = prepare_dataset(raw, out)
    assert out.exists() and len(df) == 5


def test_prepare_dataset_raises_on_invalid_data(tmp_path):
    frame = _raw_frame()
    frame.loc[0, "age"] = 999
    raw = tmp_path / "raw.data"
    frame.to_csv(raw, header=False, index=False)
    with pytest.raises(ValueError, match="validation failed"):
        prepare_dataset(raw, tmp_path / "clean.csv")


def test_split_is_stratified_and_reproducible(synthetic_df):
    a = split_xy(synthetic_df)
    b = split_xy(synthetic_df)
    pd.testing.assert_frame_equal(a[0], b[0])
    X_tr, X_te, y_tr, y_te = a
    assert len(X_tr) + len(X_te) == len(synthetic_df)
    assert abs(y_tr.mean() - y_te.mean()) < 0.05
    assert list(X_tr.columns) == config.FEATURES


def test_synthetic_matches_schema(synthetic_df):
    assert set(config.FEATURES + [config.TARGET]) <= set(synthetic_df.columns)
    assert np.isin(synthetic_df[config.TARGET], [0, 1]).all()
