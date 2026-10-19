# Screenshot checklist

Save each screenshot in this folder with EXACTLY these names; `scripts/build_report.py`
embeds them automatically (missing ones appear as marked placeholders).

| File | What to capture | How |
|---|---|---|
| `01_github_actions_run.png` | Workflow run with all 4 jobs green | GitHub > Actions > latest run |
| `02_actions_artifacts.png` | Artifacts section (trained-model, training-reports, test-reports) | bottom of the same run page |
| `03_mlflow_experiments.png` | Runs table with metric columns | `make mlflow-ui`, open http://localhost:5000, tick the runs |
| `04_mlflow_run_detail.png` | One run: params, metrics, artifacts (plots) | click a run, e.g. `random_forest` |
| `05_docker_build.png` | `docker build` finishing | terminal |
| `06_docker_run_curl.png` | container running + curl `/predict` answer | terminal, two panes |
| `07_kubectl_get_all.png` | `kubectl -n heart-mlops get all` (pods Running) | terminal |
| `08_service_external_ip.png` | `kubectl -n heart-mlops get svc` with EXTERNAL-IP | terminal (`minikube tunnel` running) |
| `09_predict_on_cluster.png` | curl `/predict` against the cluster | terminal |
| `10_prometheus_targets.png` | Status > Targets, API pods UP | `kubectl -n heart-mlops port-forward svc/prometheus 9090` |
| `11_grafana_dashboard.png` | dashboard with traffic | `port-forward svc/grafana 3000`, run `make load-test` first |
| `12_swagger_ui.png` | `/docs` page with an executed request | browser |

Tip: run `make load-test` for ~1 minute before taking screenshot 10/11 so the graphs are populated.
