from __future__ import annotations

from typing import Any

import pandas as pd


def ensure_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    boolean_columns = ["has_blockers", "is_customer_facing", "requires_review"]
    for column in boolean_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype(int)
    return frame
