from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    output_csv_file: str = os.getenv("OUTPUT_CSV_FILE", "combined_dataset.csv")
    output_parquet_file: str = os.getenv("OUTPUT_PARQUET_FILE", "combined_dataset.parquet")
    run_log_file: str = os.getenv("RUN_LOG_FILE", "pipeline_progress_log.csv")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_base_url: str | None = os.getenv("LLM_BASE_URL")
    llm_use_chat_completions: bool = os.getenv("LLM_USE_CHAT_COMPLETIONS", "false").lower() == "true"
    llm_reasoning_hint: str = os.getenv("LLM_REASONING_HINT", "Reasoning: Low")
    llm_enable_thinking: bool = os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true"
    hf_api_key: str = os.getenv("HF_API_KEY", os.getenv("HUGGINGFACE_API_KEY", os.getenv("HF_TOKEN", "")))
    request_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "40"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_backoff_seconds: float = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.5"))
    streaming: bool = os.getenv("HF_STREAMING", "true").lower() == "true"
    default_split: str = os.getenv("DEFAULT_SPLIT", "train")
    csv_path: str = os.getenv("CSV_PATH", "datasets.csv")
    progress_log_every: int = int(os.getenv("PROGRESS_LOG_EVERY", "100"))
    parquet_batch_size: int = int(os.getenv("PARQUET_BATCH_SIZE", "1000"))
    write_batch_size: int = int(os.getenv("WRITE_BATCH_SIZE", "500"))
    domains_config_path: str = os.getenv("DOMAINS_CONFIG_PATH", "domains.json")
    max_parallel_domains: int = int(os.getenv("MAX_PARALLEL_DOMAINS", "3"))
    llm_max_concurrency: int = int(os.getenv("LLM_MAX_CONCURRENCY", "2"))
    llm_min_interval_seconds: float = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "0.0"))
    hf_max_concurrency: int = int(os.getenv("HF_MAX_CONCURRENCY", "2"))
    hf_min_interval_seconds: float = float(os.getenv("HF_MIN_INTERVAL_SECONDS", "0.2"))

    def output_csv_path(self, project_root: Path) -> Path:
        return project_root / self.output_csv_file

    def output_parquet_path(self, project_root: Path) -> Path:
        return project_root / self.output_parquet_file

    def csv_file_path(self, project_root: Path) -> Path:
        return project_root / self.csv_path

    def run_log_path(self, project_root: Path) -> Path:
        return project_root / self.run_log_file

    def domains_file_path(self, project_root: Path) -> Path:
        return project_root / self.domains_config_path


@dataclass(frozen=True)
class DomainPipelineConfig:
    name: str
    task_type: str
    csv_path: str | None = None
    dataset_ids: list[str] | None = None
    output_csv_file: str | None = None
    output_parquet_file: str | None = None
    run_log_file: str | None = None
    max_rows_per_dataset: int | None = None

    @property
    def slug(self) -> str:
        return self.name.strip().lower().replace(" ", "_")


def _coerce_dataset_ids(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError("'dataset_ids' must be a list of dataset IDs")
    return list(dict.fromkeys(str(v).strip() for v in raw if str(v).strip()))


def load_domain_configs(config_path: Path) -> list[DomainPipelineConfig]:
    if not config_path.exists():
        raise FileNotFoundError(f"Domain config file not found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    domains_raw = data.get("domains")
    if not isinstance(domains_raw, list) or not domains_raw:
        raise ValueError("Domain config must contain a non-empty 'domains' list")

    domains: list[DomainPipelineConfig] = []
    for idx, item in enumerate(domains_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Domain entry #{idx} must be an object")
        name = str(item.get("name", "")).strip()
        task_type = str(item.get("task_type", "")).strip().lower()
        if not name:
            raise ValueError(f"Domain entry #{idx} is missing 'name'")
        if task_type not in {"code_generation", "math", "natural_language"}:
            raise ValueError(
                f"Domain '{name}' has unsupported task_type '{task_type}'. "
                "Expected one of: code_generation, math, natural_language"
            )
        dataset_ids_raw = item.get("dataset_ids")
        dataset_ids = _coerce_dataset_ids(dataset_ids_raw) if dataset_ids_raw is not None else None
        csv_path = str(item["csv_path"]).strip() if item.get("csv_path") else None
        if not csv_path and dataset_ids is None:
            raise ValueError(f"Domain '{name}' must define either 'csv_path' or 'dataset_ids'")
        max_rows = item.get("max_rows_per_dataset")
        domains.append(
            DomainPipelineConfig(
                name=name,
                task_type=task_type,
                csv_path=csv_path,
                dataset_ids=dataset_ids,
                output_csv_file=str(item["output_csv_file"]).strip() if item.get("output_csv_file") else None,
                output_parquet_file=str(item["output_parquet_file"]).strip() if item.get("output_parquet_file") else None,
                run_log_file=str(item["run_log_file"]).strip() if item.get("run_log_file") else None,
                max_rows_per_dataset=int(max_rows) if max_rows is not None and str(max_rows).strip() != "" else None,
            )
        )
    return domains


SETTINGS = Settings()
