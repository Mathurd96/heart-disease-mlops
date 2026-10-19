---
title: "End-to-End MLOps Pipeline for Heart Disease Risk Prediction"
author: "Deepanshu Kumar Mathur (2025AE05095)"
date: "2026-10-20"
---

# 1. Introduction

This report documents an end-to-end MLOps solution that predicts the presence of heart disease from routine clinical measurements and serves the prediction as a monitored, containerised API. The work covers the whole lifecycle required by the assignment: data acquisition and exploratory analysis, feature engineering and model development, experiment tracking with MLflow, reproducible packaging, automated testing and CI/CD with GitHub Actions, Docker containerisation, Kubernetes deployment, and monitoring with Prometheus and Grafana.

**Code repository:** https://github.com/Mathurd96/heart-disease-mlops

**Problem.** Given 13 patient attributes (age, sex, chest-pain type, blood pressure, cholesterol, ECG and exercise-test results, and so on), classify whether the patient has angiographic heart disease (`target = 1`) or not (`target = 0`). The model returns a prediction together with a confidence value, so a downstream consumer can decide how much to trust it.

**Scope note.** This is an educational project. The model is trained on a small public dataset and is **not** a medical device; its output must not be used for clinical decisions.

# 2. Setup and installation

**Prerequisites:** Python 3.12, Git, Docker; for the deployment part additionally Minikube (or Docker Desktop Kubernetes) and `kubectl`.

```bash
git clone https://github.com/Mathurd96/heart-disease-mlops && cd heart-disease-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .

python scripts/download_data.py       # download + clean the UCI data
python scripts/eda.py                 # EDA figures -> reports/figures/eda
python -m heart.train                 # tune, evaluate, track, save model
pytest                                # unit tests
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse experiments
uvicorn heart.api.app:app --port 8000 # run the API locally
```

All scripts run from a clean checkout using only `requirements.txt`; versions are pinned so that results are reproducible. A `Makefile` wraps each step (`make data`, `make eda`, `make train`, `make test`, `make docker-build`, `make k8s-deploy`).

# 3. Data acquisition and cleaning

The data is the Cleveland subset of the *Heart Disease* dataset from the UCI Machine Learning Repository. `scripts/download_data.py` downloads `processed.cleveland.data` (falling back to the `ucimlrepo` package, with manual instructions in the script header) and writes a cleaned CSV to `data/processed/heart_clean.csv`. The raw file has no header, so column names come from the UCI data dictionary stored in `src/heart/config.py`.

Cleaning steps (`src/heart/data.py`):

1. Missing values are encoded as `?` in the raw file and converted to `NaN`.
2. The original label `num` (0 = healthy, 1-4 = increasing severity) is **binarised** to `target` (0 = no disease, 1 = disease), matching the binary task in the assignment.
3. Exact duplicate rows are removed.
4. A validation function checks every feature against clinically plausible ranges and allowed category codes; the pipeline stops with a clear error if anything is out of range.

The cleaned dataset has **303 patients** and **13 features**. Columns with missing values: `ca` (4), `thal` (2). These are *not* imputed during cleaning; imputation happens inside the model pipeline so that the imputer is fitted on training data only, which avoids data leakage into cross-validation folds and the test set.

# 4. Exploratory data analysis

All figures are produced by `scripts/eda.py`.

**Class balance.** 139 patients (45.9%) have heart disease and 164 do not. The classes are close to balanced, so accuracy is meaningful, but recall and ROC-AUC are still reported because missing a sick patient is costlier than a false alarm.

![Class balance](reports/figures/eda/01_class_balance.png){ width=40% }


**Distributions.** The histograms below overlay the two classes for each numeric feature. Maximum heart rate achieved (`thalach`) is visibly lower in patients with disease, while ST depression (`oldpeak`) is higher and more right-skewed. Cholesterol and resting blood pressure overlap heavily between the classes, so they carry little signal on their own.

![Numeric distributions by class](reports/figures/eda/02_numeric_histograms.png){ width=82% }


**Categorical features.** The chart shows the share of patients with disease within each level of every categorical feature (with the group size). Asymptomatic chest pain (`cp = 4`), exercise-induced angina and a reversible thalassemia defect (`thal = 7`) are associated with much higher disease rates.

![Disease rate per category](reports/figures/eda/03_categorical_rates.png){ width=82% }


**Correlations.** The strongest linear associations with the target are: `thal` (+0.53), `ca` (+0.46), `exang` (+0.43), `oldpeak` (+0.42), `thalach` (-0.42). Correlations among the predictors themselves are moderate, so multicollinearity is not a serious concern for the regularised linear model.

![Correlation heatmap](reports/figures/eda/04_correlation_heatmap.png){ width=62% }


**Outliers.** The box plots show a few extreme cholesterol and `oldpeak` values. They are medically possible, so they are kept; the use of standard scaling together with a regularised model and a tree ensemble limits their influence.

![Numeric features vs target](reports/figures/eda/05_boxplots.png){ width=75% }


# 5. Feature engineering and model development

## 5.1 Preprocessing and features

All preprocessing lives inside one scikit-learn `Pipeline` (`src/heart/features.py`), so exactly the same transformations are applied during training, cross-validation and online inference:

* **Numeric** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`): median imputation, then standardisation.
* **Categorical** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`): most-frequent imputation, then one-hot encoding with `handle_unknown="ignore"`, so an unseen category at serving time cannot crash the API.
* **Engineered features** (custom `ClinicalFeatures` transformer): `hr_reserve = thalach / (220 - age)` (fraction of the age-predicted maximum heart rate reached), `bp_chol_load = trestbps * chol / 1000` (a crude combined load indicator) and `st_depression = 1[oldpeak > 1]`.

To check whether the engineered features are worth keeping, the experiment includes an ablation: the same logistic regression with raw features only (`logreg_baseline`) versus with engineered features (`logreg`).

## 5.2 Models and tuning

Four candidates were trained: a regularised logistic regression (interpretable baseline), a random forest, gradient boosting, and the raw-feature logistic regression used for the ablation. Hyper-parameters were tuned with `GridSearchCV` using 5-fold stratified cross-validation and ROC-AUC as the objective, on the training split only.

| Model | Features | Hyper-parameter grid |
|---|---|---|
| `logreg_baseline` | raw only | `C` in [0.01, 0.1, 1, 10] |
| `logreg` | engineered + raw | `C` in [0.01, 0.1, 1, 10]; `class_weight` in [None, 'balanced'] |
| `random_forest` | engineered + raw | `n_estimators` in [200, 400]; `max_depth` in [3, 5, None]; `min_samples_leaf` in [1, 3, 5] |
| `gradient_boosting` | engineered + raw | `n_estimators` in [100, 200]; `learning_rate` in [0.03, 0.1]; `max_depth` in [2, 3] |

## 5.3 Evaluation protocol

* The data is split once into **242 training and 61 test** patients (stratified, fixed seed 42). The test set is never used for tuning or model selection.
* For each tuned model, 5-fold stratified cross-validation on the training split reports accuracy, precision, recall, F1 and ROC-AUC (mean and standard deviation).
* The final model is chosen by **cross-validated ROC-AUC**, and only then scored once on the held-out test set.

## 5.4 Results

Cross-validated metrics (training split, mean over 5 folds):

| Model | Acc | Prec | Recall | F1 | ROC-AUC | AUC std |
|---|---|---|---|---|---|---|
| logreg_baseline | 0.851 | 0.878 | 0.792 | 0.830 | 0.911 | 0.018 |
| logreg | 0.851 | 0.889 | 0.783 | 0.828 | 0.909 | 0.019 |
| random_forest | 0.818 | 0.834 | 0.755 | 0.789 | 0.894 | 0.029 |
| gradient_boosting | 0.802 | 0.806 | 0.765 | 0.779 | 0.887 | 0.021 |

Hold-out test metrics (threshold 0.5):

| Model | Acc | Prec | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| logreg_baseline | 0.885 | 0.839 | 0.929 | 0.881 | 0.961 |
| logreg | 0.885 | 0.839 | 0.929 | 0.881 | 0.962 |
| random_forest | 0.869 | 0.857 | 0.857 | 0.857 | 0.953 |
| gradient_boosting | 0.902 | 0.867 | 0.929 | 0.897 | 0.952 |

![Model comparison](reports/figures/model_comparison.png){ width=70% }


**Selected model: `logreg_baseline`** (best parameters: `C=0.1`). It reached a cross-validated ROC-AUC of 0.911 (± 0.018) and, on the untouched test set, ROC-AUC 0.961, accuracy 0.885, precision 0.839 and recall 0.929.

With only a few hundred patients, the test set is small, so the difference between the top models is within the noise indicated by the cross-validation standard deviation. For that reason the selection rule (CV ROC-AUC) and the simplicity of the model were both considered; a simpler, better-calibrated model is preferable when performance is statistically indistinguishable.

![ROC curve of logreg_baseline](reports/figures/roc_logreg_baseline.png){ width=45% }


![Confusion matrix of logreg_baseline](reports/figures/confusion_logreg_baseline.png){ width=40% }


![Top features of logreg_baseline](reports/figures/importance_logreg_baseline.png){ width=55% }


# 6. Experiment tracking

Experiment tracking uses **MLflow** (`src/heart/train.py`) with a local SQLite backend (`mlflow.db`). One *parent* run (`training-session`) records dataset-level information, with one *child* run per model:

* **Parameters:** tuned hyper-parameters, grid size, CV folds, train/test sizes; on the parent run the dataset fingerprint (hash), class balance, Python and scikit-learn versions.
* **Metrics:** CV mean and standard deviation for accuracy, precision, recall, F1 and ROC-AUC; the same metrics on the test set; best grid-search score.
* **Artifacts and plots:** ROC curve, confusion matrix, feature importance or coefficients, the full grid-search results (`tuning/cv_results.json`), the model comparison chart and table, and the model itself with an input/output **signature** and an input example.
* **Model registry:** the selected model is registered as `heart-disease-classifier`.

The deployed model carries the MLflow run id (`bab4caab16f240919c8e1c567103471d`) and a data fingerprint (`06f4eb999c06107a`) in `models/model_metadata.json`, so any prediction can be traced back to the exact run and data.

![MLflow UI: runs of the experiment with metric columns](docs/screenshots/03_mlflow_experiments.png){ width=62% }

![MLflow UI: one run with parameters, metrics and artifacts](docs/screenshots/04_mlflow_run_detail.png){ width=62% }


# 7. Model packaging and reproducibility

* **Model format.** The complete fitted pipeline (preprocessing + estimator) is serialised with `joblib` to `models/model.joblib`, and also logged to MLflow. Because preprocessing is part of the artefact, callers send raw feature values and no training/serving skew is possible.
* **Metadata.** `models/model_metadata.json` stores the model name, parameters, metrics, feature list, library versions and data fingerprint.
* **Environments.** `requirements.txt` pins every direct dependency (tested with Python 3.12.9 and scikit-learn 1.9.1); `requirements-serve.txt` is the slim subset used in the Docker image; `requirements-dev.txt` adds test and lint tools.
* **Determinism.** All random seeds are fixed (`RANDOM_STATE = 42`), the split is stratified, and the same code path is used locally, in CI and in the container.

# 8. CI/CD pipeline and automated testing

## 8.1 Unit tests

The `tests/` folder contains **38 tests** (pytest) that cover data cleaning and validation, the preprocessing and feature-engineering transformers (including missing values and unseen categories), the training entry point end to end (artefacts, metrics in range, MLflow runs logged), prediction logic, and the API (valid and invalid payloads, optional fields, batch endpoint, Prometheus metrics, JSON request logging, degraded mode without a model). Tests use a small synthetic dataset so they run offline in about 30 seconds, and the suite achieves roughly 90% line coverage.

## 8.2 GitHub Actions workflow

`.github/workflows/ci.yml` defines four dependent jobs, so any failure stops the pipeline and the logs show the failing step:

1. **Lint**: `flake8` and `black --check`.
2. **Unit tests**: `pytest` with coverage (build fails below 80%), uploading JUnit and coverage XML as artifacts.
3. **Train**: downloads the data, generates EDA figures, trains and tracks the models, writes a model comparison table to the job summary, and uploads the trained model, reports and MLflow store as artifacts.
4. **Docker**: builds the image with the model trained in the same run, starts the container, waits for the health check, sends a real `/predict` request, validates the response, uploads the container log, and (on `main`) pushes the image to GitHub Container Registry.

![GitHub Actions: successful workflow run (all four jobs green)](docs/screenshots/01_github_actions_run.png){ width=62% }

![Artifacts uploaded by the workflow (model, reports, test results)](docs/screenshots/02_actions_artifacts.png){ width=62% }


# 9. Containerisation

The API (`src/heart/api/app.py`, FastAPI) exposes:

| Endpoint | Purpose |
|---|---|
| `POST /predict` | one patient (JSON) returns prediction, label, confidence, disease probability, risk band, model name and version |
| `POST /predict/batch` | up to 500 patients per call |
| `GET /health` | readiness/liveness, HTTP 503 if no model is loaded |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | interactive OpenAPI documentation |

Input is validated with Pydantic (types, ranges, allowed category codes); invalid requests receive HTTP 422 with a clear message. The `ca` and `thal` fields are optional because they are missing in some real records and the pipeline imputes them.

The `Dockerfile` uses `python:3.12-slim`, installs only the slim serving requirements, copies the code and the trained model, runs as a non-root user and defines a `HEALTHCHECK`. Build and run:

```bash
docker build -t heart-disease-api:1.0.0 .
docker run -p 8000:8000 heart-disease-api:1.0.0
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":4,"trestbps":145,"chol":233,"fbs":1,"restecg":2,
       "thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

![docker build finishing successfully](docs/screenshots/05_docker_build.png){ width=62% }

![Container running and answering a curl /predict request](docs/screenshots/06_docker_run_curl.png){ width=62% }

![Interactive API documentation (/docs)](docs/screenshots/12_swagger_ui.png){ width=62% }


# 10. Production deployment on Kubernetes

The manifests in `k8s/` deploy the container to Kubernetes (tested target: Minikube / Docker Desktop; they are cloud-neutral and work on GKE, EKS and AKS by changing the image reference to the GHCR image):

* `deployment.yaml`: 2 replicas, rolling updates with zero unavailable pods, readiness and liveness probes on `/health`, CPU/memory requests and limits, non-root security context.
* `service.yaml`: `LoadBalancer` service on port 80 (`minikube tunnel` provides the external address locally).
* `ingress.yaml`: optional NGINX Ingress alternative with host `heart.local`.
* `hpa.yaml`: horizontal pod autoscaler (2 to 5 pods at 70% CPU).
* `kustomization.yaml`: `kubectl apply -k k8s/` deploys everything, including monitoring.

```bash
minikube start && eval $(minikube docker-env)
docker build -t heart-disease-api:1.0.0 .
kubectl apply -k k8s/
kubectl -n heart-mlops get all
minikube tunnel                      # in a second terminal
curl http://127.0.0.1/health
```

![kubectl get all -n heart-mlops (pods Running)](docs/screenshots/07_kubectl_get_all.png){ width=62% }

![Service of type LoadBalancer with external address](docs/screenshots/08_service_external_ip.png){ width=62% }

![curl /predict against the Kubernetes deployment](docs/screenshots/09_predict_on_cluster.png){ width=62% }


# 11. Monitoring and logging

**Logging.** Every request produces one structured JSON log line (request id, method, path, status, latency in ms, client). The request id is also returned in the `x-request-id` header so a client report can be matched to a log line. Logs go to stdout, where Kubernetes (`kubectl logs`) or any log shipper can collect them.

**Metrics.** `/metrics` exposes Prometheus metrics: request count by endpoint and status, latency histogram, requests in flight, prediction count by class, a histogram of prediction confidence, a model-loaded gauge and an error counter. Metric labels use the route template so cardinality stays bounded.

**Dashboards.** Prometheus scrapes each API pod (discovered through pod annotations) and Grafana displays a provisioned dashboard (`k8s/monitoring/heart-api-dashboard.json`) with request rate, p50/p95/p99 latency, error ratio, status codes, prediction mix and mean confidence. The latter two act as simple drift signals: a sustained shift in the share of positive predictions or in mean confidence suggests the incoming data has changed. `scripts/load_test.py` generates traffic (including some invalid requests) for demonstrations. A local alternative is `docker compose -f monitoring/docker-compose.yml up`.

![Prometheus targets page: API pods UP](docs/screenshots/10_prometheus_targets.png){ width=62% }

![Grafana dashboard with live traffic](docs/screenshots/11_grafana_dashboard.png){ width=62% }


# 12. Architecture

![System architecture](docs/architecture.png){ width=100% }


Data flows from the UCI repository through cleaning, EDA and training; MLflow records every experiment; the selected model and its metadata are produced as artefacts. GitHub Actions lints, tests, trains and builds the container on every push, and publishes it to a registry; Kubernetes pulls the image and serves it behind a load balancer, while Prometheus and Grafana observe it.

# 13. Discussion, limitations and future work

* **Small data.** About 303 patients from one hospital: confidence intervals are wide and the model may not generalise to other populations. Performance numbers should be read with that uncertainty in mind.
* **Not a clinical tool.** No calibration study, fairness analysis or clinical validation was done. The risk bands (low, moderate, high) are simple probability thresholds.
* **Threshold.** The default 0.5 threshold treats both error types equally; a screening use case would favour higher recall, which could be achieved by lowering the threshold after a cost analysis.
* **Next steps.** Probability calibration, drift detection with a reference-data comparison, a model-promotion gate in CI (deploy only if the new model beats the production one), canary deployments, authentication on the API, and secrets management for Grafana credentials.

# 14. Deliverables checklist and access

| Deliverable | Location |
|---|---|
| Source code, Dockerfile, requirements | repository root, `src/`, `Dockerfile`, `requirements*.txt` |
| Cleaned dataset and download script | `data/processed/heart_clean.csv`, `scripts/download_data.py` |
| EDA / training / inference scripts and notebook | `scripts/`, `src/heart/`, `notebooks/` |
| Unit tests | `tests/` |
| CI workflow | `.github/workflows/ci.yml` |
| Kubernetes manifests | `k8s/` |
| Screenshots | `docs/screenshots/` |
| Pipeline video | linked in the repository README |

**Access for local testing:** run the commands in Section 10 and open `http://127.0.0.1/docs` (or `http://localhost:8000/docs` when running the container directly).
