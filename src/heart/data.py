"""Data acquisition, cleaning and splitting."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

logger = logging.getLogger(__name__)


def download_raw(dest: Path = config.RAW_FILE, url: str = config.UCI_URL) -> Path:
    """Download the Cleveland subset of the UCI Heart Disease dataset.

    Falls back to the `ucimlrepo` package if the direct download fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=30) as resp:  # nosec B310
            dest.write_bytes(resp.read())
    except Exception as exc:  # network errors vary widely
        logger.warning("Direct download failed (%s); trying ucimlrepo", exc)
        from ucimlrepo import fetch_ucirepo  # optional dependency

        ds = fetch_ucirepo(id=config.UCI_REPO_ID)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.columns = config.RAW_COLUMNS
        df.to_csv(dest, header=False, index=False, na_rep="?")
    logger.info("Raw data saved to %s", dest)
    return dest


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the headerless raw file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning that is safe to do BEFORE the train/test split.

    * binarise the 0-4 `num` label into `target` (0 = no disease, 1 = disease)
    * drop exact duplicates
    * coerce every column to numeric
    * leave missing feature values (`ca`, `thal`) in place: they are imputed
      inside the sklearn pipeline so the imputer is fitted on training data only
      (no leakage).
    """
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "num" in out.columns:
        out[config.TARGET] = (out["num"] > 0).astype(int)
        out = out.drop(columns="num")
    out = out.dropna(subset=[config.TARGET]).drop_duplicates().reset_index(drop=True)
    return out


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of data-quality problems (empty list means OK)."""
    problems: list[str] = []
    missing_cols = set(config.FEATURES + [config.TARGET]) - set(df.columns)
    if missing_cols:
        problems.append(f"missing columns: {sorted(missing_cols)}")
        return problems
    for col, (lo, hi) in config.VALID_RANGES.items():
        bad = df[col].dropna()
        bad = bad[(bad < lo) | (bad > hi)]
        if len(bad):
            problems.append(f"{col}: {len(bad)} values outside [{lo}, {hi}]")
    for col, allowed in config.VALID_CATEGORIES.items():
        bad = df[col].dropna()
        bad = bad[~bad.isin(allowed)]
        if len(bad):
            problems.append(f"{col}: {len(bad)} values not in {allowed}")
    if not set(df[config.TARGET].unique()) <= {0, 1}:
        problems.append("target must be binary 0/1")
    return problems


def prepare_dataset(
    raw_path: Path = config.RAW_FILE, out_path: Path = config.CLEAN_FILE
) -> pd.DataFrame:
    """Load raw -> clean -> validate -> save cleaned CSV."""
    df = clean(load_raw(raw_path))
    problems = validate(df)
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Cleaned data (%d rows) saved to %s", len(df), out_path)
    return df


def load_clean(path: Path = config.CLEAN_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame, test_size: float = 0.2, random_state: int = config.RANDOM_STATE):
    """Stratified train/test split returning X_train, X_test, y_train, y_test."""
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def make_synthetic(n: int = 303, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Synthetic stand-in with the same schema as the real data.

    ONLY for tests / offline development. It is NOT the UCI dataset and must
    never be used to report results.
    """
    rng = np.random.default_rng(seed)
    target = rng.binomial(1, 0.46, n)
    t = target
    df = pd.DataFrame(
        {
            "age": np.clip(rng.normal(52 + 5 * t, 9), 29, 77).round(),
            "sex": rng.binomial(1, 0.55 + 0.2 * t),
            "cp": np.where(
                t == 1,
                rng.choice([1, 2, 3, 4], n, p=[0.05, 0.1, 0.15, 0.7]),
                rng.choice([1, 2, 3, 4], n, p=[0.1, 0.25, 0.35, 0.3]),
            ),
            "trestbps": np.clip(rng.normal(130 + 5 * t, 17), 94, 200).round(),
            "chol": np.clip(rng.normal(244 + 8 * t, 50), 126, 564).round(),
            "fbs": rng.binomial(1, 0.15, n),
            "restecg": rng.choice([0, 1, 2], n, p=[0.5, 0.01, 0.49]),
            "thalach": np.clip(rng.normal(158 - 20 * t, 19), 71, 202).round(),
            "exang": rng.binomial(1, 0.15 + 0.4 * t),
            "oldpeak": np.clip(rng.gamma(1.2, 0.6 + 0.8 * t), 0, 6.2).round(1),
            "slope": np.where(
                t == 1,
                rng.choice([1, 2, 3], n, p=[0.2, 0.7, 0.1]),
                rng.choice([1, 2, 3], n, p=[0.6, 0.3, 0.1]),
            ),
            "ca": np.where(
                t == 1,
                rng.choice([0, 1, 2, 3], n, p=[0.35, 0.3, 0.2, 0.15]),
                rng.choice([0, 1, 2, 3], n, p=[0.8, 0.12, 0.06, 0.02]),
            ),
            "thal": np.where(
                t == 1,
                rng.choice([3, 6, 7], n, p=[0.3, 0.08, 0.62]),
                rng.choice([3, 6, 7], n, p=[0.8, 0.05, 0.15]),
            ),
            config.TARGET: target,
        }
    )
    df.loc[rng.choice(n, 4, replace=False), "ca"] = np.nan
    df.loc[rng.choice(n, 2, replace=False), "thal"] = np.nan
    return df
