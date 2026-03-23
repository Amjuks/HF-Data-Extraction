from __future__ import annotations

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path


class PipelineRunLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.fieldnames = [
            "domain",
            "timestamp_utc",
            "dataset_id",
            "status",
            "config_name",
            "split_name",
            "records_written",
            "message",
            "error",
        ]
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            with self.log_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(
        self,
        *,
        domain: str = "",
        dataset_id: str,
        status: str,
        config_name: str | None = None,
        split_name: str | None = None,
        records_written: int | None = None,
        message: str = "",
        error: str = "",
    ) -> None:
        row = {
            "domain": domain,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_id": dataset_id,
            "status": status,
            "config_name": config_name or "",
            "split_name": split_name or "",
            "records_written": records_written if records_written is not None else "",
            "message": message,
            "error": error,
        }
        with self._lock:
            with self.log_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row)
                f.flush()

