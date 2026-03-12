from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_dataset_ids(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "dataset_id" not in df.columns:
        raise ValueError("CSV must contain a 'dataset_id' column")

    dataset_ids = [str(v).strip() for v in df["dataset_id"].tolist() if str(v).strip()]
    unique_dataset_ids = list(dict.fromkeys(dataset_ids))
    return unique_dataset_ids

