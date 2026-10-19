.PHONY: help install data eda train test lint format run mlflow-ui docker-build docker-run \
        k8s-deploy k8s-status k8s-delete monitor-up load-test report all

PY ?= python
IMAGE ?= heart-disease-api:1.0.0

help:           ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:        ## install dev dependencies and the package
	pip install -r requirements-dev.txt && pip install -e .

data:           ## download and clean the UCI dataset
	$(PY) scripts/download_data.py

eda:            ## EDA figures -> reports/figures/eda
	$(PY) scripts/eda.py

train:          ## tune, evaluate, track (MLflow) and save the model
	$(PY) -m heart.train

test:           ## unit tests with coverage
	pytest --cov=src/heart --cov-report=term-missing

lint:           ## flake8 + black check
	flake8 . && black --check .

format:         ## auto-format
	black .

run:            ## run the API locally on :8000
	uvicorn heart.api.app:app --reload --port 8000

mlflow-ui:      ## browse experiments on :5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:   ## build the serving image (needs models/model.joblib)
	docker build -t $(IMAGE) .

docker-run:     ## run the image on :8000
	docker run --rm -p 8000:8000 $(IMAGE)

k8s-deploy:     ## deploy API + Prometheus + Grafana to the current cluster
	kubectl apply -k k8s/
	kubectl -n heart-mlops rollout status deploy/heart-api

k8s-status:
	kubectl -n heart-mlops get all

k8s-delete:
	kubectl delete -k k8s/

monitor-up:     ## local API + Prometheus + Grafana with docker compose
	docker compose -f monitoring/docker-compose.yml up --build

load-test:      ## generate demo traffic
	$(PY) scripts/load_test.py --requests 300 --rps 10

report:         ## build docs/report.{md,docx,pdf} from the latest run
	$(PY) scripts/draw_architecture.py
	$(PY) scripts/build_report.py --formats docx pdf

all: data eda train test
