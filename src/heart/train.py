"""Train, tune, evaluate and track the heart disease classifiers.

Usage:
    python -m heart.train                      # uses data/processed/heart_clean.csv
    python -m heart.train --data some.csv --experiment my-exp

What happens:
  1. stratified 80/20 split (test set is never used for tuning or selection)
  2. for every candidate model: GridSearchCV (5-fold stratified, ROC-AUC)
  3. 5-fold cross-validation of the tuned pipeline on the training split
     (accuracy / precision / recall / F1 / ROC-AUC, mean and std)
  4. hold-out test evaluation + plots
  5. everything is logged to MLflow (one child run per model)
  6. the model with the best CV ROC-AUC is saved to models/model.joblib
     together with a metadata JSON, and registered in the MLflow registry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate  # noqa: E402

from . import config  # noqa: E402
from .data import load_clean, split_xy  # noqa: E402
from .features import build_pipeline, feature_names  # noqa: E402

logger = logging.getLogger("heart.train")

REGISTERED_MODEL_NAME = "heart-disease-classifier"
CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def candidate_models() -> dict[str, dict]:
    """Model zoo: estimator, hyper-parameter grid and whether to engineer features."""
    rs = config.RANDOM_STATE
    return {
        "logreg_baseline": {
            "estimator": LogisticRegression(max_iter=2000, random_state=rs),
            "grid": {"model__C": [0.01, 0.1, 1, 10]},
            "engineer": False,
        },
        "logreg": {
            "estimator": LogisticRegression(max_iter=2000, random_state=rs),
            "grid": {"model__C": [0.01, 0.1, 1, 10], "model__class_weight": [None, "balanced"]},
            "engineer": True,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=rs, n_jobs=-1),
            "grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
            "engineer": True,
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=rs),
            "grid": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
            },
            "engineer": True,
        },
    }


def holdout_metrics(pipe, X_test, y_test) -> dict[str, float]:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_recall": recall_score(y_test, pred, zero_division=0),
        "test_f1": f1_score(y_test, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def _save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_roc(pipe, X_test, y_test, name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_title(f"ROC curve: {name}")
    ax.legend(loc="lower right")
    return _save_fig(fig, out_dir / f"roc_{name}.png")


def plot_confusion(pipe, X_test, y_test, name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipe,
        X_test,
        y_test,
        ax=ax,
        cmap="Blues",
        display_labels=["No disease", "Disease"],
        colorbar=False,
    )
    ax.set_title(f"Confusion matrix: {name}")
    return _save_fig(fig, out_dir / f"confusion_{name}.png")


def plot_importance(pipe, name: str, out_dir: Path) -> Path | None:
    """Coefficients (linear models) or impurity importances (tree models)."""
    model = pipe.named_steps["model"]
    names = feature_names(pipe)
    if hasattr(model, "coef_"):
        values = model.coef_.ravel()
        label = "coefficient"
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        label = "importance"
    else:
        return None
    order = np.argsort(np.abs(values))[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(np.array(names)[order], values[order], color="#2b6cb0")
    ax.set_xlabel(label)
    ax.set_title(f"Top features: {name}")
    return _save_fig(fig, out_dir / f"importance_{name}.png")


def plot_comparison(results: list[dict], out_dir: Path) -> Path:
    df = pd.DataFrame(results).set_index("model")
    metrics = ["cv_roc_auc_mean", "test_roc_auc", "test_recall", "test_precision"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df[metrics].plot.bar(ax=ax, rot=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison")
    ax.legend(
        ["CV ROC-AUC", "Test ROC-AUC", "Test recall", "Test precision"],
        loc="lower right",
        fontsize=8,
    )
    return _save_fig(fig, out_dir / "model_comparison.png")


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of the training data so a run can be tied to its dataset."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[
        :16
    ]


def train_one(
    name: str, spec: dict, X_train, X_test, y_train, y_test, cv: StratifiedKFold, fig_dir: Path
) -> tuple[dict, object]:
    """Tune, cross-validate, evaluate and log a single candidate model."""
    pipe = build_pipeline(spec["estimator"], engineer=spec["engineer"])
    with mlflow.start_run(run_name=name, nested=True) as run:
        mlflow.set_tags({"model_family": name, "engineered_features": spec["engineer"]})

        search = GridSearchCV(pipe, spec["grid"], scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        cv_metrics = {}
        for m in CV_SCORING:
            cv_metrics[f"cv_{m}_mean"] = float(np.mean(cv_res[f"test_{m}"]))
            cv_metrics[f"cv_{m}_std"] = float(np.std(cv_res[f"test_{m}"]))
        test_metrics = holdout_metrics(best, X_test, y_test)

        # ---- MLflow logging -------------------------------------------------
        mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_param("grid_size", len(search.cv_results_["params"]))
        mlflow.log_param("cv_folds", cv.get_n_splits())
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_metric("grid_best_cv_roc_auc", float(search.best_score_))
        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics(test_metrics)

        mlflow.log_dict(
            pd.DataFrame(search.cv_results_).drop(columns=["params"]).to_dict("list"),
            "tuning/cv_results.json",
        )
        for fig in (
            plot_roc(best, X_test, y_test, name, fig_dir),
            plot_confusion(best, X_test, y_test, name, fig_dir),
            plot_importance(best, name, fig_dir),
        ):
            if fig is not None:
                mlflow.log_artifact(str(fig), artifact_path="plots")

        # Float dtypes so the signature tolerates missing values (ca / thal) at inference.
        example = X_train.astype(float)
        signature = infer_signature(example, best.predict_proba(X_train)[:, 1])
        # cloudpickle: the pipeline contains our own ClinicalFeatures transformer,
        # which MLflow's default skops serializer does not trust.
        mlflow.sklearn.log_model(
            best,
            name="model",
            signature=signature,
            input_example=example.head(3),
            serialization_format="cloudpickle",
        )

        row = {
            "model": name,
            "run_id": run.info.run_id,
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            **cv_metrics,
            **test_metrics,
        }
    logger.info(
        "%-18s CV AUC %.3f +/- %.3f | test AUC %.3f | test recall %.3f",
        name,
        row["cv_roc_auc_mean"],
        row["cv_roc_auc_std"],
        row["test_roc_auc"],
        row["test_recall"],
    )
    return row, best


def run_training(
    data_path: Path = config.CLEAN_FILE,
    experiment: str = "heart-disease-classification",
    tracking_uri: str | None = None,
    cv_folds: int = 5,
    only: list[str] | None = None,
    model_path: Path = config.MODEL_FILE,
    metadata_path: Path = config.METADATA_FILE,
    fig_dir: Path = config.FIGURES_DIR,
    register: bool = True,
) -> dict:
    mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{config.PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(experiment)

    df = load_clean(data_path)
    X_train, X_test, y_train, y_test = split_xy(df)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

    zoo = candidate_models()
    if only:
        zoo = {k: v for k, v in zoo.items() if k in only}

    results, fitted = [], {}
    with mlflow.start_run(run_name="training-session") as parent:
        mlflow.log_params(
            {
                "data_file": str(data_path.name),
                "n_rows": len(df),
                "random_state": config.RANDOM_STATE,
                "sklearn_version": sklearn.__version__,
                "python_version": platform.python_version(),
            }
        )
        mlflow.log_param("data_fingerprint", data_fingerprint(df))
        mlflow.log_param("class_balance_positive", round(float(df[config.TARGET].mean()), 4))

        for name, spec in zoo.items():
            row, best = train_one(name, spec, X_train, X_test, y_train, y_test, cv, fig_dir)
            results.append(row)
            fitted[name] = best

        summary = pd.DataFrame(results).drop(columns=["best_params"])
        summary_path = fig_dir.parent / "model_comparison.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(plot_comparison(results, fig_dir)), artifact_path="plots")

        # Select on cross-validated ROC-AUC (NOT on the test set).
        best_row = max(results, key=lambda r: r["cv_roc_auc_mean"])
        best_name = best_row["model"]
        mlflow.set_tag("best_model", best_name)
        mlflow.log_metric("best_cv_roc_auc", best_row["cv_roc_auc_mean"])
        mlflow.log_metric("best_test_roc_auc", best_row["test_roc_auc"])

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted[best_name], model_path)
        metadata = {
            "model_name": best_name,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mlflow_run_id": best_row["run_id"],
            "mlflow_parent_run_id": parent.info.run_id,
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "features": config.FEATURES,
            "best_params": best_row["best_params"],
            "metrics": {
                k: round(v, 4) for k, v in best_row.items() if k.startswith(("cv_", "test_"))
            },
            "data_fingerprint": data_fingerprint(df),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        mlflow.log_artifact(str(metadata_path))

        if register:
            try:
                mlflow.register_model(f"runs:/{best_row['run_id']}/model", REGISTERED_MODEL_NAME)
            except Exception as exc:  # registry unavailable (e.g. plain file store)
                logger.warning("Model registry step skipped: %s", exc)

    logger.info(
        "Best model: %s (CV ROC-AUC %.3f). Saved to %s",
        best_name,
        best_row["cv_roc_auc_mean"],
        model_path,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=config.CLEAN_FILE)
    parser.add_argument("--experiment", default="heart-disease-classification")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--only", nargs="*", help="train only these model names")
    parser.add_argument("--model-path", type=Path, default=config.MODEL_FILE)
    parser.add_argument("--metadata-path", type=Path, default=config.METADATA_FILE)
    parser.add_argument("--fig-dir", type=Path, default=config.FIGURES_DIR)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training(
        args.data,
        args.experiment,
        args.tracking_uri,
        args.cv_folds,
        args.only,
        args.model_path,
        args.metadata_path,
        args.fig_dir,
        not args.no_register,
    )


if __name__ == "__main__":
    main()
