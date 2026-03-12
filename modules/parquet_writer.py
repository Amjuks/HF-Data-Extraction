from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


class ParquetDatasetWriter:
    def __init__(self, output_path: Path, batch_size: int = 1000) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(1, batch_size)
        self._schema = pa.schema(
            [
                ("conversation", pa.string()),
                ("reasoning", pa.string()),
                ("metadata", pa.string()),
                ("dataset_id", pa.string()),
            ]
        )
        self._writer: pq.ParquetWriter | None = None

    def _ensure_writer(self) -> pq.ParquetWriter:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.output_path.as_posix(), self._schema, compression="snappy")
        return self._writer

    def append_records(
        self,
        records: Iterable[dict[str, Any]],
        *,
        on_record: Callable[[int], None] | None = None,
    ) -> int:
        count = 0
        batch: list[dict[str, str | None]] = []

        for record in records:
            metadata = record.get("metadata", {}) or {}
            batch.append(
                {
                    "conversation": json.dumps(record.get("conversation", []), ensure_ascii=False),
                    "reasoning": record.get("reasoning"),
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                    "dataset_id": metadata.get("dataset_id"),
                }
            )
            count += 1
            if on_record:
                on_record(count)
            if len(batch) >= self.batch_size:
                table = pa.Table.from_pylist(batch, schema=self._schema)
                self._ensure_writer().write_table(table)
                batch = []

        if batch:
            table = pa.Table.from_pylist(batch, schema=self._schema)
            self._ensure_writer().write_table(table)

        return count

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

