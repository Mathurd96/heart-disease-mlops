"""Shared fixtures. Tests use a small SYNTHETIC dataset so they run offline in
seconds; they validate code behaviour, not clinical performance."""

import pytest

from heart import config
from heart.data import make_synthetic


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic(n=240, seed=7)


@pytest.fixture(scope="session")
def sample_patient():
    return {
        "age": 63,
        "sex": 1,
        "cp": 4,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory, synthetic_df):
    """Run the real training entry-point end-to-end on synthetic data."""
    from heart.train import run_training

    root = tmp_path_factory.mktemp("train")
    data_path = root / "clean.csv"
    synthetic_df.to_csv(data_path, index=False)
    meta = run_training(
        data_path=data_path,
        experiment="pytest-experiment",
        tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
        cv_folds=3,
        only=["logreg", "random_forest"],
        model_path=root / "model.joblib",
        metadata_path=root / "meta.json",
        fig_dir=root / "figs",
        register=False,
    )
    return {
        "root": root,
        "meta": meta,
        "model_path": root / "model.joblib",
        "meta_path": root / "meta.json",
        "features": config.FEATURES,
    }
