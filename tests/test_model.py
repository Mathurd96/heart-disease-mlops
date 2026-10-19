import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from heart import config
from heart.predict import predict_records, risk_band
from heart.train import candidate_models, holdout_metrics


def test_candidate_models_cover_required_families():
    zoo = candidate_models()
    assert {"logreg", "random_forest"} <= set(zoo)
    assert len(zoo) >= 2


def test_training_creates_artifacts(trained_artifacts):
    assert trained_artifacts["model_path"].exists()
    meta = json.loads(trained_artifacts["meta_path"].read_text())
    assert meta["model_name"] in {"logreg", "random_forest"}
    assert meta["features"] == config.FEATURES
    for key in ("cv_roc_auc_mean", "test_roc_auc", "test_recall", "test_precision"):
        assert 0.0 <= meta["metrics"][key] <= 1.0


def test_trained_model_beats_chance(trained_artifacts, synthetic_df):
    model = joblib.load(trained_artifacts["model_path"])
    X = synthetic_df[config.FEATURES]
    y = synthetic_df[config.TARGET]
    assert holdout_metrics(model, X, y)["test_roc_auc"] > 0.7


def test_predictions_are_valid_probabilities(trained_artifacts, sample_patient):
    model = joblib.load(trained_artifacts["model_path"])
    out = predict_records(model, [sample_patient])[0]
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1.0
    assert 0.0 <= out["probability_disease"] <= 1.0
    assert out["risk_band"] in {"low", "moderate", "high"}


def test_prediction_is_deterministic(trained_artifacts, sample_patient):
    model = joblib.load(trained_artifacts["model_path"])
    a = predict_records(model, [sample_patient])
    b = predict_records(model, [sample_patient])
    assert a == b


def test_missing_optional_fields_still_predict(trained_artifacts, sample_patient):
    model = joblib.load(trained_artifacts["model_path"])
    patient = {**sample_patient, "ca": None, "thal": None}
    assert predict_records(model, [patient])[0]["prediction"] in (0, 1)


def test_risk_band_thresholds():
    assert risk_band(0.1) == "low"
    assert risk_band(0.5) == "moderate"
    assert risk_band(0.9) == "high"


def test_mlflow_run_logged_params_metrics_and_artifacts(trained_artifacts):
    root = trained_artifacts["root"]
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    exp = mlflow.get_experiment_by_name("pytest-experiment")
    runs = mlflow.search_runs([exp.experiment_id], output_format="pandas")
    children = runs[runs["tags.mlflow.runName"].isin(["logreg", "random_forest"])]
    assert len(children) == 2
    for col in ("metrics.cv_roc_auc_mean", "metrics.test_roc_auc", "metrics.test_recall"):
        assert children[col].notna().all()
    client = mlflow.MlflowClient()
    arts = {a.path for a in client.list_artifacts(children.iloc[0]["run_id"])}
    assert {"plots", "tuning"} <= arts
    # MLflow 3 stores models as first-class logged models linked to the run
    logged = mlflow.search_logged_models(experiment_ids=[exp.experiment_id], output_format="list")
    assert len(logged) == 2


def test_holdout_metrics_perfect_classifier():
    class Perfect:
        def predict_proba(self, X):
            p = X["y"].to_numpy().astype(float)
            return np.column_stack([1 - p, p])

    X = pd.DataFrame({"y": [0, 1, 0, 1]})
    m = holdout_metrics(Perfect(), X, pd.Series([0, 1, 0, 1]))
    assert m["test_accuracy"] == 1.0 and m["test_roc_auc"] == 1.0
