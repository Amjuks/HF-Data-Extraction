from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

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


def normalize_hf_dataset_link(link: str) -> str:
    value = str(link).strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        path = parsed.path.strip("/")
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "datasets":
            return "/".join(parts[1:3])
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return path
    return value.strip("/")


def read_dataset_links(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "link" not in df.columns:
        raise ValueError("CSV must contain a 'link' column")

    dataset_ids = [normalize_hf_dataset_link(v) for v in df["link"].tolist()]
    dataset_ids = [dataset_id for dataset_id in dataset_ids if dataset_id]
    return list(dict.fromkeys(dataset_ids))

