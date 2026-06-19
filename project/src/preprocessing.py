from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_feature_preprocessor(config: dict) -> ColumnTransformer:
    """Build the preprocessing pipeline shared by training and inference."""
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]
    boolean_features = config["features"]["boolean"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
            ("boolean", "passthrough", boolean_features),
        ]
    )


def normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight request-time cleanup before the saved sklearn pipeline."""
    cleaned = frame.copy()
    for column in ["has_priority", "component_present"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(int)
    for column in ["issue_type", "priority"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("Unknown").astype(str)
    return cleaned
