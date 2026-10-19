import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(trained_artifacts, monkeypatch):
    import heart.api.app as api

    monkeypatch.setattr(api, "MODEL_PATH", trained_artifacts["model_path"])
    monkeypatch.setattr(api, "METADATA_PATH", trained_artifacts["meta_path"])
    with TestClient(api.app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["risk_band"] in {"low", "moderate", "high"}
    assert body["model_name"] in {"logreg", "random_forest"}
    assert "x-request-id" in r.headers


def test_predict_accepts_missing_optional_fields(client, sample_patient):
    payload = {k: v for k, v in sample_patient.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", -5),
        ("age", "old"),
        ("sex", 2),
        ("cp", 9),
        ("trestbps", 1000),
        ("thal", 5),
        ("oldpeak", -1),
    ],
)
def test_predict_rejects_invalid_values(client, sample_patient, field, value):
    r = client.post("/predict", json={**sample_patient, field: value})
    assert r.status_code == 422


def test_predict_rejects_missing_required_field(client, sample_patient):
    payload = dict(sample_patient)
    payload.pop("age")
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_predict(client, sample_patient):
    r = client.post("/predict/batch", json={"patients": [sample_patient, sample_patient]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_metrics_endpoint_counts_requests(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert 'endpoint="/predict"' in text
    assert "model_predictions_total" in text
    assert "api_request_latency_seconds_bucket" in text


def test_request_is_logged_as_json(client, sample_patient, caplog):
    import json
    import logging

    with caplog.at_level(logging.INFO, logger="heart.access"):
        client.post("/predict", json=sample_patient)
    records = [json.loads(r.message) for r in caplog.records if r.name == "heart.access"]
    assert any(
        r["path"] == "/predict" and r["status"] == 200 and "latency_ms" in r for r in records
    )


def test_service_degraded_without_model(tmp_path, monkeypatch, sample_patient):
    import heart.api.app as api

    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(api, "METADATA_PATH", tmp_path / "missing.json")
    with TestClient(api.app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=sample_patient).status_code == 503
