from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class AppSettings:
    def __init__(self) -> None:
        _load_dotenv(PROJECT_ROOT / ".env")
        self.app_name = os.getenv("APP_NAME", "task-overdue-service")
        self.app_env = os.getenv("APP_ENV", "development")
        self.app_host = os.getenv("APP_HOST", "0.0.0.0")
        self.app_port = int(os.getenv("APP_PORT", "8000"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.model_path = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "artifacts" / "model.joblib")))
        self.model_metadata_path = Path(
            os.getenv("MODEL_METADATA_PATH", str(PROJECT_ROOT / "artifacts" / "model_metadata.json"))
        )
        self.config_path = Path(os.getenv("CONFIG_PATH", str(PROJECT_ROOT / "configs" / "train_config.yaml")))


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def load_train_config(config_path: Path | None = None) -> dict:
    settings = get_settings()
    target_path = config_path or settings.config_path
    with target_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
