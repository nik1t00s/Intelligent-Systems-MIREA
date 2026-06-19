from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_COLUMNS = [
    "Issue key",
    "Issue Type",
    "Priority",
    "Created",
    "Resolved",
    "Component/s",
    "Summary",
    "Description",
]


def parse_jira_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%d/%b/%Y %I:%M %p", errors="coerce")


def load_jira_issue_dataset(raw_dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(raw_dataset_path, usecols=lambda column: column in RAW_COLUMNS)


def prepare_jira_delay_dataset(raw_dataset_path: Path, output_path: Path, delay_quantile: float = 0.75) -> Path:
    raw = load_jira_issue_dataset(raw_dataset_path)
    raw["Created"] = parse_jira_datetime(raw["Created"])
    raw["Resolved"] = parse_jira_datetime(raw["Resolved"])

    resolved = raw.dropna(subset=["Created", "Resolved"]).copy()
    resolved["resolution_time_days"] = (resolved["Resolved"] - resolved["Created"]).dt.total_seconds() / 86400
    resolved = resolved[resolved["resolution_time_days"] >= 0].copy()
    if resolved.empty:
        raise ValueError("No resolved Jira issues with valid Created/Resolved dates were found.")

    delay_threshold_days = float(resolved["resolution_time_days"].quantile(delay_quantile))
    resolved["is_delayed"] = (resolved["resolution_time_days"] > delay_threshold_days).astype(int)

    summary = resolved["Summary"].fillna("").astype(str)
    description = resolved["Description"].fillna("").astype(str)
    component = resolved["Component/s"].fillna("").astype(str).str.strip()
    priority = resolved["Priority"].fillna("Unknown").astype(str)

    prepared = pd.DataFrame(
        {
            "issue_key": resolved["Issue key"].astype(str),
            "issue_type": resolved["Issue Type"].fillna("Unknown").astype(str),
            "priority": priority,
            "has_priority": (priority != "Unknown").astype(int),
            "component_present": (component != "").astype(int),
            "summary_length": summary.str.len(),
            "summary_word_count": summary.str.split().str.len(),
            "description_length": description.str.len(),
            "description_word_count": description.str.split().str.len(),
            "created_year": resolved["Created"].dt.year,
            "created_month": resolved["Created"].dt.month,
            "created_dayofweek": resolved["Created"].dt.dayofweek,
            "created_hour": resolved["Created"].dt.hour,
            "resolution_time_days": resolved["resolution_time_days"].round(4),
            "delay_threshold_days": delay_threshold_days,
            "is_delayed": resolved["is_delayed"],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    return output_path
