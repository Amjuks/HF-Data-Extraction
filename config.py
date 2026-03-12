from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _int_or_none(name: str, default: str = "") -> int | None:
    raw = os.getenv(name, default).strip()
    if raw == "":
        return None
    return int(raw)


@dataclass(frozen=True)
class Settings:
    max_sample_rows: int = int(os.getenv("MAX_SAMPLE_ROWS", "12"))
    max_rows_per_dataset: int | None = _int_or_none("MAX_ROWS_PER_DATASET", "")
    output_file: str = os.getenv("OUTPUT_FILE", "combined_dataset.csv")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_base_url: str | None = os.getenv("LLM_BASE_URL")
    request_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "40"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_backoff_seconds: float = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.5"))
    streaming: bool = os.getenv("HF_STREAMING", "true").lower() == "true"
    default_split: str = os.getenv("DEFAULT_SPLIT", "train")
    csv_path: str = os.getenv("CSV_PATH", "datasets.csv")
    progress_log_every: int = int(os.getenv("PROGRESS_LOG_EVERY", "100"))
    flush_each_record: bool = os.getenv("FLUSH_EACH_RECORD", "true").lower() == "true"

    def output_path(self, project_root: Path) -> Path:
        return project_root / self.output_file

    def csv_file_path(self, project_root: Path) -> Path:
        return project_root / self.csv_path


SETTINGS = Settings()
