from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable


class CsvWriter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = ["conversation", "reasoning", "metadata", "dataset_id"]

    def append_records(
        self,
        records: Iterable[dict[str, Any]],
        *,
        flush_each_record: bool = True,
        on_record: Callable[[int], None] | None = None,
    ) -> int:
        count = 0
        file_exists = self.output_path.exists() and self.output_path.stat().st_size > 0

        with self.output_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()

            for record in records:
                metadata = record.get("metadata", {}) or {}
                writer.writerow(
                    {
                        "conversation": json.dumps(record.get("conversation", []), ensure_ascii=False),
                        "reasoning": record.get("reasoning"),
                        "metadata": json.dumps(metadata, ensure_ascii=False),
                        "dataset_id": metadata.get("dataset_id"),
                    }
                )
                count += 1
                if flush_each_record:
                    f.flush()
                if on_record:
                    on_record(count)
        return count

