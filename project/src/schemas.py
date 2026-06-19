from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JiraIssueFeatures(BaseModel):
    issue_type: Literal["Bug", "Suggestion"]
    priority: Literal["Low", "Medium", "High", "Highest", "Unknown"]
    has_priority: bool
    component_present: bool
    summary_length: int = Field(ge=0, le=1000)
    summary_word_count: int = Field(ge=0, le=200)
    description_length: int = Field(ge=0, le=50000)
    description_word_count: int = Field(ge=0, le=10000)


class PredictionRequest(BaseModel):
    issues: list[JiraIssueFeatures] = Field(min_length=1, max_length=100)


class PredictionItem(BaseModel):
    prediction: int
    probability: float
    is_delayed: int
    delay_probability: float
    delay_risk: str


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: list[PredictionItem]
