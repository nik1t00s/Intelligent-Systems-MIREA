from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TaskFeatures(BaseModel):
    priority: Literal["low", "medium", "high", "critical"]
    assignee_experience: float = Field(ge=0.0, le=20.0)
    estimated_hours: float = Field(gt=0.0, le=300.0)
    actual_progress: float = Field(ge=0.0, le=100.0)
    days_since_created: int = Field(ge=0, le=365)
    comments_count: int = Field(ge=0, le=200)
    status: Literal["todo", "in_progress", "review", "blocked", "done"]
    team_size: int = Field(ge=1, le=50)
    task_type: Literal["feature", "bug", "maintenance", "research", "documentation"]
    has_blockers: bool
    blockers_count: int = Field(ge=0, le=10)
    recent_reassignments: int = Field(ge=0, le=10)
    sprint_phase: Literal["planning", "mid_sprint", "release_week"]
    priority_score: int = Field(ge=1, le=4)
    workload_ratio: float = Field(ge=0.0, le=20.0)
    is_customer_facing: bool
    requires_review: bool

    @field_validator("priority_score")
    @classmethod
    def validate_priority_score(cls, value: int, info) -> int:
        priority = info.data.get("priority")
        expected_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        if priority is not None and value != expected_map[priority]:
            raise ValueError("priority_score must match the selected priority")
        return value


class PredictionRequest(BaseModel):
    tasks: list[TaskFeatures] = Field(min_length=1, max_length=100)


class PredictionItem(BaseModel):
    is_overdue: int
    overdue_probability: float
    overdue_risk: str


class PredictionResponse(BaseModel):
    model_name: str
    predictions: list[PredictionItem]
