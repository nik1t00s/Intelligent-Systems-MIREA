from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from src.eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [10, 20, 30, None],
        "height": [140, 150, 160, 170],
        "city": ["A", "B", "A", None],
    })

def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)
    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)
    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns

def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)
    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1
    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0

def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False
    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

def test_constant_column_detection():
    """Тест для новой эвристики has_constant_columns"""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "constant_col": [5, 5, 5, 5],  # Константная колонка
        "var_col": [1, 2, 3, 4]
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    assert flags["has_constant_columns"] is True

def test_zero_values_detection():
    """Тест для новой эвристики has_many_zero_values"""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "zero_col": [0, 0, 0, 0],  # Колонка с нулями
        "normal_col": [10, 20, 30, 40]
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    assert flags["has_many_zero_values"] is True