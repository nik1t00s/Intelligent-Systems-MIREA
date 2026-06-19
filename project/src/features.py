from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing import normalize_prediction_frame


def ensure_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    return normalize_prediction_frame(frame)
