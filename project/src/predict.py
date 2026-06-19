from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.config import PROJECT_ROOT, get_settings
from src.features import ensure_dataframe
from src.service import _risk_bucket, load_metadata


def predict_from_records(records: list[dict]) -> dict:
    settings = get_settings()
    model_path = settings.model_path if settings.model_path.is_absolute() else PROJECT_ROOT / settings.model_path
    metadata_path = (
        settings.model_metadata_path
        if settings.model_metadata_path.is_absolute()
        else PROJECT_ROOT / settings.model_metadata_path
    )
    model = joblib.load(model_path)
    metadata = load_metadata(metadata_path)
    probabilities = model.predict_proba(ensure_dataframe(records))[:, 1]
    return {
        "model_name": metadata["best_model_name"],
        "model_version": settings.model_version,
        "predictions": [
            {
                "prediction": int(probability >= 0.5),
                "probability": round(float(probability), 4),
                "is_delayed": int(probability >= 0.5),
                "delay_probability": round(float(probability), 4),
                "delay_risk": _risk_bucket(float(probability)),
            }
            for probability in probabilities
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local predictions with the saved Jira delay model.")
    parser.add_argument("--input", default="src/demo_request.json", help="Path to JSON with an 'issues' list.")
    args = parser.parse_args()

    payload_path = Path(args.input)
    if not payload_path.is_absolute():
        payload_path = PROJECT_ROOT / payload_path
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    result = predict_from_records(payload["issues"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
