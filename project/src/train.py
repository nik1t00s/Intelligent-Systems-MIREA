from __future__ import annotations

import json
import logging

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT, get_settings, load_train_config
from src.data_generation import prepare_jira_delay_dataset
from src.logging_utils import configure_logging
from src.preprocessing import build_feature_preprocessor


LOGGER = logging.getLogger(__name__)


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
    raw_dataset_path = PROJECT_ROOT / data_config["raw_dataset_path"]
    dataset_path = PROJECT_ROOT / data_config["prepared_dataset_path"]
    dataset_path = prepare_jira_delay_dataset(
        raw_dataset_path,
        dataset_path,
        delay_quantile=data_config["delay_quantile"],
    )
    LOGGER.info("Prepared Jira dataset saved to %s", dataset_path)

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

    preprocessor = build_feature_preprocessor(config)
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
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, settings.model_path)

    canonical_models_dir = PROJECT_ROOT / "models"
    canonical_models_dir.mkdir(parents=True, exist_ok=True)
    canonical_model_path = canonical_models_dir / "model.joblib"
    if canonical_model_path != settings.model_path:
        joblib.dump(best_model, canonical_model_path)

    artifact_model_path = artifacts_dir / "model.joblib"
    if artifact_model_path != settings.model_path and artifact_model_path != canonical_model_path:
        joblib.dump(best_model, artifact_model_path)

    metadata = {
        "project_name": config["project"]["name"],
        "source_dataset_path": str(raw_dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(dataframe)),
        "delay_threshold_days": float(dataframe["delay_threshold_days"].iloc[0]),
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

    canonical_metadata_path = canonical_models_dir / "model_metadata.json"
    if canonical_metadata_path != settings.model_metadata_path:
        with canonical_metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    artifact_metadata_path = artifacts_dir / "model_metadata.json"
    if artifact_metadata_path != settings.model_metadata_path and artifact_metadata_path != canonical_metadata_path:
        with artifact_metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    leaderboard_path = artifacts_dir / "leaderboard.json"
    with leaderboard_path.open("w", encoding="utf-8") as file:
        json.dump(leaderboard, file, ensure_ascii=False, indent=2)

    sample_predictions = dataframe[[id_column] + feature_columns + [target_column]].head(10)
    sample_predictions.to_csv(artifacts_dir / "sample_tasks.csv", index=False)
    LOGGER.info("%s selected as final model", best_model_name)


if __name__ == "__main__":
    main()
