from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRIORITIES = ["low", "medium", "high", "critical"]
STATUSES = ["todo", "in_progress", "review", "blocked", "done"]
TASK_TYPES = ["feature", "bug", "maintenance", "research", "documentation"]
SPRINT_PHASES = ["planning", "mid_sprint", "release_week"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_task_dataset(rows: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    priority = rng.choice(PRIORITIES, size=rows, p=[0.18, 0.38, 0.29, 0.15])
    task_type = rng.choice(TASK_TYPES, size=rows, p=[0.34, 0.24, 0.16, 0.12, 0.14])
    status = rng.choice(STATUSES, size=rows, p=[0.22, 0.33, 0.18, 0.12, 0.15])
    sprint_phase = rng.choice(SPRINT_PHASES, size=rows, p=[0.22, 0.56, 0.22])

    assignee_experience = np.clip(rng.normal(loc=3.8, scale=2.2, size=rows), 0.2, 12.0).round(1)
    estimated_hours = np.clip(rng.gamma(shape=2.5, scale=7.0, size=rows), 1.0, 120.0).round(1)
    actual_progress = np.clip(rng.beta(a=2.1, b=1.8, size=rows) * 100.0, 0.0, 100.0).round(1)
    days_since_created = np.clip(rng.integers(1, 65, size=rows), 1, 64)
    comments_count = np.clip(rng.poisson(lam=4.5, size=rows), 0, 35)
    team_size = rng.integers(3, 13, size=rows)
    blockers_count = np.clip(rng.poisson(lam=0.8, size=rows), 0, 6)
    recent_reassignments = np.clip(rng.poisson(lam=0.6, size=rows), 0, 4)

    priority_score_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    priority_score = np.array([priority_score_map[item] for item in priority])
    has_blockers = blockers_count > 0
    is_customer_facing = rng.random(rows) < np.where(np.isin(task_type, ["feature", "bug"]), 0.44, 0.18)
    requires_review = np.logical_or(task_type == "feature", rng.random(rows) < 0.35)

    workload_ratio = np.clip(days_since_created / np.maximum(estimated_hours / 6.5, 1.0), 0.2, 10.0).round(2)

    risk_score = (
        0.9 * (priority_score - 2.0)
        + 0.05 * estimated_hours
        + 0.06 * days_since_created
        + 1.1 * blockers_count
        + 0.45 * recent_reassignments
        + 0.14 * comments_count
        + 0.08 * (team_size - 6)
        - 0.07 * actual_progress
        - 0.4 * assignee_experience
        + 0.65 * (status == "blocked").astype(int)
        + 0.35 * (status == "review").astype(int)
        - 1.2 * (status == "done").astype(int)
        + 0.55 * (task_type == "bug").astype(int)
        + 0.35 * (task_type == "research").astype(int)
        + 0.42 * (sprint_phase == "release_week").astype(int)
        + 0.25 * is_customer_facing.astype(int)
        + 0.15 * requires_review.astype(int)
        + rng.normal(0, 1.2, size=rows)
    )
    overdue_probability = _sigmoid((risk_score - 3.2) / 2.8)
    is_overdue = (rng.random(rows) < overdue_probability).astype(int)

    overdue_risk = np.where(
        overdue_probability >= 0.66,
        "high",
        np.where(overdue_probability >= 0.38, "medium", "low"),
    )

    dataframe = pd.DataFrame(
        {
            "task_id": [f"TASK-{index:05d}" for index in range(1, rows + 1)],
            "priority": priority,
            "assignee_experience": assignee_experience,
            "estimated_hours": estimated_hours,
            "actual_progress": actual_progress,
            "days_since_created": days_since_created,
            "comments_count": comments_count,
            "status": status,
            "team_size": team_size,
            "task_type": task_type,
            "has_blockers": has_blockers.astype(int),
            "blockers_count": blockers_count,
            "recent_reassignments": recent_reassignments,
            "sprint_phase": sprint_phase,
            "priority_score": priority_score,
            "workload_ratio": workload_ratio,
            "is_customer_facing": is_customer_facing.astype(int),
            "requires_review": requires_review.astype(int),
            "overdue_risk": overdue_risk,
            "is_overdue": is_overdue,
        }
    )
    return dataframe


def save_dataset(output_path: Path, rows: int = 5000, random_seed: int = 42) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_task_dataset(rows=rows, random_seed=random_seed)
    dataset.to_csv(output_path, index=False)
    return output_path
