"""Inference helpers shared by the CLI and the API."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from . import config


def load_model(path: Path = config.MODEL_FILE):
    return joblib.load(path)


def load_metadata(path: Path = config.METADATA_FILE) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def risk_band(p: float) -> str:
    """Map disease probability to a coarse, human-readable band."""
    if p < 0.33:
        return "low"
    if p < 0.66:
        return "moderate"
    return "high"


def predict_records(model, records: list[dict]) -> list[dict]:
    """Predict for a list of feature dicts.

    ``confidence`` is the probability of the *predicted* class, i.e.
    max(p, 1 - p); ``probability_disease`` is P(target = 1).
    """
    df = pd.DataFrame(records)[config.FEATURES]
    proba = model.predict_proba(df)[:, 1]
    out = []
    for p in proba:
        p = float(p)
        pred = int(p >= 0.5)
        out.append(
            {
                "prediction": pred,
                "label": "disease" if pred else "no disease",
                "confidence": round(max(p, 1 - p), 4),
                "probability_disease": round(p, 4),
                "risk_band": risk_band(p),
            }
        )
    return out


def main() -> None:
    """CLI: python -m heart.predict '{"age": 63, ...}'"""
    import argparse

    parser = argparse.ArgumentParser(description="Predict heart disease risk for one patient.")
    parser.add_argument("json_record", help="JSON object with all 13 features")
    parser.add_argument("--model", type=Path, default=config.MODEL_FILE)
    args = parser.parse_args()
    result = predict_records(load_model(args.model), [json.loads(args.json_record)])[0]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
