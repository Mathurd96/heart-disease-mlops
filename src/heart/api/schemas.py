"""Request / response schemas (validation derives from heart.config)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from heart import config

R = config.VALID_RANGES


class PatientFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    age: float = Field(..., ge=R["age"][0], le=R["age"][1], description="Age in years")
    sex: Literal[0, 1] = Field(..., description="0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic"
    )
    trestbps: float = Field(
        ..., ge=R["trestbps"][0], le=R["trestbps"][1], description="Resting blood pressure (mm Hg)"
    )
    chol: float = Field(
        ..., ge=R["chol"][0], le=R["chol"][1], description="Serum cholesterol (mg/dl)"
    )
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(
        ..., ge=R["thalach"][0], le=R["thalach"][1], description="Max heart rate achieved"
    )
    exang: Literal[0, 1] = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(
        ..., ge=R["oldpeak"][0], le=R["oldpeak"][1], description="ST depression induced by exercise"
    )
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: float | None = Field(
        None,
        ge=R["ca"][0],
        le=R["ca"][1],
        description="Major vessels colored by fluoroscopy (optional)",
    )
    thal: Literal[3, 6, 7] | None = Field(
        None, description="3 normal, 6 fixed defect, 7 reversible defect (optional)"
    )


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float
    risk_band: Literal["low", "moderate", "high"]
    model_name: str
    model_version: str


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
