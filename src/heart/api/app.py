"""FastAPI service for heart disease risk prediction.

Endpoints
    GET  /            service info
    GET  /health      liveness/readiness (model loaded?)
    POST /predict     single patient -> prediction + confidence
    POST /predict/batch
    GET  /metrics     Prometheus metrics
    GET  /docs        interactive OpenAPI UI

Every request is logged as one JSON line (method, path, status, latency,
request id) and counted in Prometheus.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from heart import __version__, config
from heart.predict import load_metadata, load_model, predict_records

from .schemas import BatchRequest, BatchResponse, PatientFeatures, PredictionResponse

MODEL_PATH = Path(os.getenv("MODEL_PATH", config.MODEL_FILE))
METADATA_PATH = Path(os.getenv("METADATA_PATH", config.METADATA_FILE))

# ---------------------------------------------------------------- logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
access_log = logging.getLogger("heart.access")
app_log = logging.getLogger("heart.app")

# ---------------------------------------------------------------- metrics
REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Counter("model_predictions_total", "Predictions by class", ["label"])
CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Confidence of predictions",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
IN_PROGRESS = Gauge("api_requests_in_progress", "Requests being processed")
MODEL_LOADED = Gauge("model_loaded", "1 if a model is loaded")
ERRORS = Counter("api_errors_total", "Unhandled errors", ["endpoint"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.meta = {}
    try:
        app.state.model = load_model(MODEL_PATH)
        app.state.meta = load_metadata(METADATA_PATH)
        MODEL_LOADED.set(1)
        app_log.info(
            json.dumps(
                {
                    "event": "model_loaded",
                    "path": str(MODEL_PATH),
                    "model": app.state.meta.get("model_name"),
                }
            )
        )
    except FileNotFoundError:
        MODEL_LOADED.set(0)
        app_log.error(json.dumps({"event": "model_missing", "path": str(MODEL_PATH)}))
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    version=__version__,
    description="Predicts the presence of heart disease from 13 clinical "
    "features. Educational project, not a medical device.",
    lifespan=lifespan,
)


def _endpoint_label(request: Request) -> str:
    """Use the route template so metric cardinality stays bounded."""
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    IN_PROGRESS.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    except Exception:
        ERRORS.labels(request.url.path).inc()
        app_log.exception("unhandled error")
        raise
    finally:
        IN_PROGRESS.dec()
        elapsed = time.perf_counter() - start
        endpoint = _endpoint_label(request)
        if endpoint != "/metrics":
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            access_log.info(
                json.dumps(
                    {
                        "event": "request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "latency_ms": round(elapsed * 1000, 2),
                        "client": request.client.host if request.client else None,
                    }
                )
            )


def _require_model(request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def _to_response(result: dict, meta: dict) -> PredictionResponse:
    return PredictionResponse(
        **result,
        model_name=meta.get("model_name", "unknown"),
        model_version=meta.get("trained_at", __version__),
    )


def _record_prediction(result: dict) -> None:
    PREDICTIONS.labels(result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])


@app.get("/")
def root():
    return {
        "service": "heart-disease-risk-api",
        "version": __version__,
        "docs": "/docs",
        "predict": "POST /predict",
    }


@app.get("/health")
def health(request: Request):
    loaded = request.app.state.model is not None
    body = {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_name": request.app.state.meta.get("model_name"),
    }
    return Response(
        content=json.dumps(body), media_type="application/json", status_code=200 if loaded else 503
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    model = _require_model(request)
    result = predict_records(model, [patient.model_dump()])[0]
    _record_prediction(result)
    return _to_response(result, request.app.state.meta)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    model = _require_model(request)
    results = predict_records(model, [p.model_dump() for p in batch.patients])
    for r in results:
        _record_prediction(r)
    return BatchResponse(predictions=[_to_response(r, request.app.state.meta) for r in results])


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
