# Pipeline demo video: script (target 4 to 5 minutes)

Record the screen with voice-over (OBS, QuickTime, or Zoom "record to computer").
Keep a terminal, VS Code, a browser and the GitHub repo ready.

| Time | Show | Say (in your own words) |
|---|---|---|
| 0:00 | Repo README | Goal: heart disease risk API with a full MLOps pipeline. One sentence on the dataset. |
| 0:25 | `python scripts/download_data.py`, then `heart_clean.csv` | Download script, cleaning, binary target, missing values handled in the pipeline. |
| 0:50 | `reports/figures/eda/` | Class balance, histograms, correlation heatmap; one insight (e.g. `thalach`, `oldpeak`, `ca`, `thal`). |
| 1:20 | `python -m heart.train` (speed up the wait), then `mlflow ui` | Models tried, grid search, 5-fold CV, params/metrics/plots logged; compare runs; best model by CV ROC-AUC. |
| 2:10 | `pytest` | 38 tests: data, features, model, API. |
| 2:30 | GitHub Actions run | lint > test > train > docker; artifacts; show that it fails on a broken test (optional: a red run). |
| 3:00 | `docker build` + `docker run` + curl `/predict` | JSON in, prediction + confidence out. |
| 3:30 | `kubectl apply -k k8s/`, `get all`, curl through the service | 2 replicas, probes, LoadBalancer. |
| 4:00 | `make load-test`, then Grafana dashboard + a log line | Request rate, latency, error ratio, prediction mix; JSON logs. |
| 4:40 | Wrap-up | Summary and one limitation/future-work item. |

Upload to Google Drive / YouTube (unlisted) and put the link in README.md under "Demo video".
