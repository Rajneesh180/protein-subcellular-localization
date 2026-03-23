"""Pydantic schemas for inference API."""

from pydantic import BaseModel


class PredictionResult(BaseModel):
    labels: list[str]
    probabilities: dict[str, float]
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
