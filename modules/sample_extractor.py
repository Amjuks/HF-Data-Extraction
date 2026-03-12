from __future__ import annotations

from typing import Any

from datasets import IterableDataset


def extract_columns_and_samples(
    stream: IterableDataset,
    max_sample_rows: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    columns = list(stream.features.keys()) if hasattr(stream, "features") and stream.features else []
    samples: list[dict[str, Any]] = []

    for idx, row in enumerate(stream):
        if not columns and isinstance(row, dict):
            columns = list(row.keys())
        if isinstance(row, dict):
            samples.append(row)
        if idx + 1 >= max_sample_rows:
            break
    return columns, samples

