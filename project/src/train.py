from __future__ import annotations

import json
import logging

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROJECT_ROOT, get_settings, load_train_config
from src.data_generation import save_dataset
from src.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)


def build_preprocessor(config: dict) -> ColumnTransformer:
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


def build_models(config: dict) -> dict[str, object]:
    training_config = config["training"]["models"]
    return {
        "logistic_regression": LogisticRegression(**training_config["logistic_regression"]),
        "random_forest": RandomForestClassifier(**training_config["random_forest"]),
        "gradient_boosting": GradientBoostingClassifier(**training_config["gradient_boosting"]),
    }


def evaluate_model(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": round(accuracy_score(target, predictions), 4),
        "precision": round(precision_score(target, predictions), 4),
        "recall": round(recall_score(target, predictions), 4),
        "f1": round(f1_score(target, predictions), 4),
        "roc_auc": round(roc_auc_score(target, probabilities), 4),
    }


def main() -> None:
    settings = get_settings()
    config = load_train_config()
    configure_logging(settings.log_level)

    data_config = config["data"]
    random_seed = config["project"]["random_seed"]
    dataset_path = PROJECT_ROOT / data_config["raw_dataset_path"]
    dataset_path = save_dataset(dataset_path, rows=data_config["rows"], random_seed=random_seed)
    LOGGER.info("Synthetic dataset saved to %s", dataset_path)

    dataframe = pd.read_csv(dataset_path)
    target_column = data_config["target_column"]
    id_column = data_config["id_column"]
    feature_columns = (
        config["features"]["numeric"] + config["features"]["categorical"] + config["features"]["boolean"]
    )

    x = dataframe[feature_columns]
    y = dataframe[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=1.0 - data_config["train_split"],
        random_state=random_seed,
        stratify=y,
    )

    preprocessor = build_preprocessor(config)
    model_candidates = build_models(config)
    leaderboard: dict[str, dict[str, float]] = {}
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in model_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test)
        leaderboard[model_name] = metrics
        trained_pipelines[model_name] = pipeline
        LOGGER.info("%s metrics: %s", model_name, metrics)

    primary_metric = config["training"]["metrics"]["primary"]
    best_model_name = max(leaderboard, key=lambda name: leaderboard[name][primary_metric])
    best_model = trained_pipelines[best_model_name]

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, settings.model_path)

    metadata = {
        "project_name": config["project"]["name"],
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(dataframe)),
        "feature_columns": feature_columns,
        "id_column": id_column,
        "target_column": target_column,
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,
        "best_metrics": leaderboard[best_model_name],
        "class_balance": dataframe[target_column].value_counts(normalize=True).round(4).to_dict(),
    }
    with settings.model_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    leaderboard_path = artifacts_dir / "leaderboard.json"
    with leaderboard_path.open("w", encoding="utf-8") as file:
        json.dump(leaderboard, file, ensure_ascii=False, indent=2)

    sample_predictions = dataframe[[id_column] + feature_columns + [target_column]].head(10)
    sample_predictions.to_csv(artifacts_dir / "sample_tasks.csv", index=False)
    LOGGER.info("%s selected as final model", best_model_name)


if __name__ == "__main__":
    main()
