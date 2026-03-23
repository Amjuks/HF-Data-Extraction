from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
from typing import Literal

import pyarrow.parquet as pq
from loguru import logger

from config import DomainPipelineConfig, SETTINGS, load_domain_configs
from modules.csv_reader import normalize_hf_dataset_link, read_dataset_ids, read_dataset_links
from modules.csv_writer import CsvWriter
from modules.dataset_classifier import is_relevant_dataset
from modules.dataset_converter import convert_rows
from modules.dataset_loader import inspect_dataset, load_dataset_stream
from modules.parquet_writer import ParquetDatasetWriter
from modules.run_logger import PipelineRunLogger
from modules.sample_extractor import extract_columns_and_samples
from modules.schema_agent import SchemaAgent
from modules.utils import setup_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Hugging Face datasets into a unified conversation schema.")
    parser.add_argument(
        "--output-format",
        choices=["csv", "parquet", "both"],
        default=None,
        help="Output format to write. Default: csv.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="default",
        default=None,
        help="Resume from previous progress. Optionally provide a progress name.",
    )
    parser.add_argument(
        "--progress-name",
        default="default",
        help="Progress name for logs/checkpoints when not using --resume.",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable built-in deduplication for output records.",
    )
    parser.add_argument(
        "--domains-config",
        default=None,
        help="JSON config describing one or more domains and their dataset IDs.",
    )
    return parser.parse_args()


def _suffix_path(base_path: Path, *parts: str) -> Path:
    suffix = "_".join(part for part in parts if part and part != "default")
    if not suffix:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def _checkpoint_path(root: Path, name: str) -> Path:
    return root / f".pipeline_checkpoint_{name}.json"


def _dataset_registry_path(root: Path) -> Path:
    return root / ".dataset_registry.json"


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x) for x in data}
    except Exception:  # noqa: BLE001
        pass
    return set()


def _save_checkpoint(path: Path, processed_ids: set[str]) -> None:
    path.write_text(json.dumps(sorted(processed_ids), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_done_from_run_log(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    done: set[str] = set()
    try:
        with log_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset_id = (row.get("dataset_id") or "").strip()
                status = (row.get("status") or "").strip().lower()
                if dataset_id and dataset_id != "__system__" and status in {"added", "skipped"}:
                    done.add(dataset_id)
    except Exception:  # noqa: BLE001
        return set()
    return done


def _domain_file_path(root: Path, file_name: str, domain_slug: str, progress_name: str) -> Path:
    return _suffix_path(root / file_name, domain_slug, progress_name)


def _resolve_domain_dataset_ids(root: Path, domain: DomainPipelineConfig) -> list[str]:
    if domain.csv_path:
        csv_path = Path(domain.csv_path)
        if not csv_path.is_absolute():
            csv_path = root / csv_path
        return read_dataset_links(csv_path)
    return domain.dataset_ids or []


def _load_pipeline_domains(root: Path, domains_config_arg: str | None) -> list[DomainPipelineConfig]:
    config_path = Path(domains_config_arg) if domains_config_arg else SETTINGS.domains_file_path(root)
    if config_path.exists():
        domains = load_domain_configs(config_path)
        resolved_domains: list[DomainPipelineConfig] = []
        for domain in domains:
            resolved_domains.append(
                DomainPipelineConfig(
                    name=domain.name,
                    task_type=domain.task_type,
                    csv_path=domain.csv_path,
                    dataset_ids=_resolve_domain_dataset_ids(root, domain),
                    output_csv_file=domain.output_csv_file,
                    output_parquet_file=domain.output_parquet_file,
                    run_log_file=domain.run_log_file,
                    max_rows_per_dataset=domain.max_rows_per_dataset,
                )
            )
        return resolved_domains

    csv_path = SETTINGS.csv_file_path(root)
    logger.warning(
        f"Domain config not found at {config_path}. Falling back to legacy CSV input at {csv_path} as code_generation."
    )
    dataset_ids = read_dataset_ids(csv_path)
    return [DomainPipelineConfig(name="code_generation", task_type="code_generation", dataset_ids=dataset_ids)]


class DatasetRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def should_skip(self, domain_name: str, dataset_id: str) -> bool:
        with self._lock:
            domain_states = self._data.get(dataset_id, {})
            status = domain_states.get(domain_name)
            return status in {"added", "skipped", "resumed_skip", "duplicate_input"}

    def mark(self, domain_name: str, dataset_id: str, status: str) -> None:
        with self._lock:
            domain_states = self._data.setdefault(dataset_id, {})
            domain_states[domain_name] = status
            self._save()

    def has_seen_any_domain(self, dataset_id: str) -> bool:
        with self._lock:
            return dataset_id in self._data


def _fingerprint(conversation: str, reasoning: str | None) -> str:
    payload = f"{conversation}\n<SEP>\n{reasoning or ''}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _record_fingerprint(record: dict) -> str:
    conversation = json.dumps(record.get("conversation", []), ensure_ascii=False)
    reasoning = record.get("reasoning")
    return _fingerprint(conversation, reasoning if isinstance(reasoning, str) or reasoning is None else str(reasoning))


def _load_seen_fingerprints_from_csv(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add(_fingerprint(row.get("conversation", "") or "", row.get("reasoning")))
    return seen


def _load_seen_fingerprints_from_parquet(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    parquet = pq.ParquetFile(path.as_posix())
    for batch in parquet.iter_batches(columns=["conversation", "reasoning"], batch_size=50_000):
        table = batch.to_pydict()
        convs = table.get("conversation", [])
        reas = table.get("reasoning", [])
        for conv, rea in zip(convs, reas):
            seen.add(_fingerprint(conv or "", rea if isinstance(rea, str) or rea is None else str(rea)))
    return seen


def _process_domain(
    root: Path,
    domain: DomainPipelineConfig,
    progress_name: str,
    output_mode: Literal["csv", "parquet", "both"],
    enable_dedup: bool,
    resume_enabled: bool,
    dataset_registry: DatasetRegistry,
) -> dict[str, int | str]:
    output_csv_file = domain.output_csv_file or SETTINGS.output_csv_file
    output_parquet_file = domain.output_parquet_file or SETTINGS.output_parquet_file
    run_log_file = domain.run_log_file or SETTINGS.run_log_file
    output_csv_path = _domain_file_path(root, output_csv_file, domain.slug, progress_name)
    output_parquet_path = _domain_file_path(root, output_parquet_file, domain.slug, progress_name)
    run_log_path = _domain_file_path(root, run_log_file, domain.slug, progress_name)
    checkpoint_path = _checkpoint_path(root, f"{domain.slug}_{progress_name}")

    csv_writer = CsvWriter(output_csv_path) if output_mode in {"csv", "both"} else None
    parquet_writer = (
        ParquetDatasetWriter(output_parquet_path, batch_size=SETTINGS.parquet_batch_size)
        if output_mode in {"parquet", "both"}
        else None
    )
    run_logger = PipelineRunLogger(run_log_path)
    agent = SchemaAgent()

    if not SETTINGS.llm_api_key:
        warning_msg = "LLM_API_KEY is missing. Schema inference will use fallback detection and may be less accurate."
        logger.warning(f"[{domain.name}] {warning_msg}")
        run_logger.log(domain=domain.name, dataset_id="__system__", status="warning", message=warning_msg)

    processed_ids = set()
    if resume_enabled:
        processed_ids = _load_checkpoint(checkpoint_path) | _load_done_from_run_log(run_log_path)
        logger.info(f"[{domain.name}] Resume enabled. Loaded {len(processed_ids)} already-processed dataset IDs.")
        run_logger.log(
            domain=domain.name,
            dataset_id="__system__",
            status="resume",
            records_written=len(processed_ids),
            message=f"Resuming with {len(processed_ids)} processed dataset IDs",
        )
    else:
        _save_checkpoint(checkpoint_path, set())

    seen_fingerprints: set[str] = set()
    if enable_dedup:
        if output_mode in {"csv", "both"}:
            seen_fingerprints |= _load_seen_fingerprints_from_csv(output_csv_path)
        if output_mode in {"parquet", "both"}:
            seen_fingerprints |= _load_seen_fingerprints_from_parquet(output_parquet_path)
        logger.info(f"[{domain.name}] Dedup enabled. Loaded {len(seen_fingerprints)} existing output fingerprints.")

    total_written = 0
    total_duplicates_filtered = 0
    duplicate_datasets_skipped = 0
    max_rows = domain.max_rows_per_dataset if domain.max_rows_per_dataset is not None else SETTINGS.max_rows_per_dataset
    try:
        for dataset_id in domain.dataset_ids:
            dataset_id = normalize_hf_dataset_link(dataset_id)
            if not dataset_id:
                continue
            if dataset_registry.should_skip(domain.name, dataset_id):
                duplicate_datasets_skipped += 1
                logger.info(f"[{domain.name}::{dataset_id}] skipped: already tracked as processed for this domain.")
                run_logger.log(
                    domain=domain.name,
                    dataset_id=dataset_id,
                    status="duplicate_input",
                    records_written=0,
                    message="Skipped because dataset was already processed in an earlier run for this domain",
                )
                continue
            if dataset_id in processed_ids:
                logger.info(f"[{domain.name}::{dataset_id}] skipped: already processed in resume state.")
                run_logger.log(
                    domain=domain.name,
                    dataset_id=dataset_id,
                    status="resumed_skip",
                    records_written=0,
                    message="Skipped because already processed in previous run",
                )
                dataset_registry.mark(domain.name, dataset_id, "resumed_skip")
                continue
            info = None
            try:
                run_logger.log(
                    domain=domain.name,
                    dataset_id=dataset_id,
                    status="started",
                    message=f"Dataset processing started for domain {domain.name}",
                )
                info = inspect_dataset(dataset_id)
                sample_stream = load_dataset_stream(dataset_id, info.config_name, info.split_name)
                columns, samples = extract_columns_and_samples(sample_stream, SETTINGS.max_sample_rows)
                if not columns:
                    columns = info.columns

                schema = agent.infer_schema(
                    dataset_id=dataset_id,
                    task_type=domain.task_type,
                    columns=columns,
                    sample_rows=samples,
                )
                if not is_relevant_dataset(schema):
                    msg = f"Skipped: dataset not relevant for domain '{domain.task_type}'."
                    logger.info(f"[{domain.name}::{dataset_id}] {msg}")
                    run_logger.log(
                        domain=domain.name,
                        dataset_id=dataset_id,
                        status="skipped",
                        config_name=info.config_name,
                        split_name=info.split_name,
                        records_written=0,
                        message=msg,
                    )
                    dataset_registry.mark(domain.name, dataset_id, "skipped")
                    continue

                limit_display = max_rows if max_rows is not None else "ALL"
                mode_label = "CSV+Parquet" if output_mode == "both" else output_mode.upper()
                logger.info(
                    f"[{domain.name}::{dataset_id}] converting rows for task={domain.task_type} "
                    f"(limit={limit_display}) and appending to {mode_label}..."
                )
                convert_stream = load_dataset_stream(dataset_id, info.config_name, info.split_name)
                records = convert_rows(
                    rows=convert_stream,
                    schema=schema,
                    dataset_id=dataset_id,
                    max_rows=max_rows,
                )

                written = 0
                duplicates_filtered = 0
                batch: list[dict] = []
                for record in records:
                    if enable_dedup:
                        fp = _record_fingerprint(record)
                        if fp in seen_fingerprints:
                            duplicates_filtered += 1
                            continue
                        seen_fingerprints.add(fp)
                    batch.append(record)
                    written += 1
                    if written == 1 or written % SETTINGS.progress_log_every == 0:
                        logger.info(f"[{domain.name}::{dataset_id}] appended {written} records so far...")
                    if len(batch) >= SETTINGS.write_batch_size:
                        if csv_writer is not None:
                            csv_writer.append_records(batch, flush_each_record=False)
                        if parquet_writer is not None:
                            parquet_writer.append_records(batch)
                        batch = []

                if batch:
                    if csv_writer is not None:
                        csv_writer.append_records(batch, flush_each_record=True)
                    if parquet_writer is not None:
                        parquet_writer.append_records(batch)

                total_written += written
                total_duplicates_filtered += duplicates_filtered
                logger.info(f"[{domain.name}::{dataset_id}] wrote {written} normalized rows.")
                run_logger.log(
                    domain=domain.name,
                    dataset_id=dataset_id,
                    status="added",
                    config_name=info.config_name,
                    split_name=info.split_name,
                    records_written=written,
                    message=f"Dataset converted and appended. Duplicates filtered: {duplicates_filtered}",
                )
                processed_ids.add(dataset_id)
                _save_checkpoint(checkpoint_path, processed_ids)
                dataset_registry.mark(domain.name, dataset_id, "added")
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"[{domain.name}::{dataset_id}] failed and skipped: {exc}")
                run_logger.log(
                    domain=domain.name,
                    dataset_id=dataset_id,
                    status="failed",
                    config_name=info.config_name if info else None,
                    split_name=info.split_name if info else None,
                    records_written=0,
                    message="Dataset failed during processing",
                    error=str(exc),
                )
                dataset_registry.mark(domain.name, dataset_id, "failed")
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    run_logger.log(
        domain=domain.name,
        dataset_id="__system__",
        status="completed",
        records_written=total_written,
        message=f"Domain pipeline execution completed. Duplicates filtered: {total_duplicates_filtered}",
    )
    return {
        "domain": domain.name,
        "written": total_written,
        "duplicates_filtered": total_duplicates_filtered,
        "duplicate_datasets_skipped": duplicate_datasets_skipped,
        "csv_path": str(output_csv_path),
        "parquet_path": str(output_parquet_path),
        "run_log_path": str(run_log_path),
        "checkpoint_path": str(checkpoint_path),
    }


def run() -> None:
    args = _parse_args()
    setup_logging()
    root = Path(__file__).resolve().parent
    progress_name = args.resume if args.resume is not None else args.progress_name
    output_mode: Literal["csv", "parquet", "both"] = args.output_format or "csv"
    enable_dedup = not args.no_deduplicate
    domains = _load_pipeline_domains(root, args.domains_config)
    dataset_registry = DatasetRegistry(_dataset_registry_path(root))

    if args.output_format is None:
        logger.info("No output format specified. Defaulting to CSV output.")
    logger.info(f"Output mode selected: {output_mode}")
    logger.info(f"Progress name: {progress_name}")
    logger.info(f"Loaded {len(domains)} domain configuration(s).")

    if not any(domain.dataset_ids for domain in domains):
        logger.warning("No dataset IDs found in domain configuration. Exiting.")
        return

    max_workers = min(max(1, SETTINGS.max_parallel_domains), len(domains))
    logger.info(
        f"Processing {len(domains)} domain(s) in parallel with max_workers={max_workers} "
        f"and llm_max_concurrency={SETTINGS.llm_max_concurrency}."
    )

    results: list[dict[str, int | str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _process_domain,
                root,
                domain,
                progress_name,
                output_mode,
                enable_dedup,
                args.resume is not None,
                dataset_registry,
            ): domain
            for domain in domains
        }
        for future in as_completed(future_map):
            domain = future_map[future]
            result = future.result()
            results.append(result)
            logger.info(
                f"[{domain.name}] completed with {result['written']} records and "
                f"{result['duplicates_filtered']} record duplicates filtered. "
                f"Dataset duplicates skipped: {result['duplicate_datasets_skipped']}."
            )

    total_written = sum(int(result["written"]) for result in results)
    total_duplicates_filtered = sum(int(result["duplicates_filtered"]) for result in results)
    total_duplicate_datasets_skipped = sum(int(result["duplicate_datasets_skipped"]) for result in results)
    logger.info(f"Done. Total normalized records written across domains: {total_written}")
    if enable_dedup:
        logger.info(f"Duplicates filtered across domains: {total_duplicates_filtered}")
    logger.info(f"Duplicate datasets skipped from registry/resume tracking: {total_duplicate_datasets_skipped}")
    for result in results:
        if output_mode in {"csv", "both"}:
            logger.info(f"[{result['domain']}] CSV output file: {result['csv_path']}")
        if output_mode in {"parquet", "both"}:
            logger.info(f"[{result['domain']}] Parquet output file: {result['parquet_path']}")
        logger.info(f"[{result['domain']}] Progress log file: {result['run_log_path']}")
        logger.info(f"[{result['domain']}] Checkpoint file: {result['checkpoint_path']}")


if __name__ == "__main__":
    run()
