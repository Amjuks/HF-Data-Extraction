from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from config import SETTINGS
from modules.csv_reader import read_dataset_ids
from modules.csv_writer import CsvWriter
from modules.dataset_classifier import is_relevant_codegen_dataset
from modules.dataset_converter import convert_rows
from modules.dataset_loader import inspect_dataset, load_dataset_stream
from modules.parquet_writer import ParquetDatasetWriter
from modules.run_logger import PipelineRunLogger
from modules.sample_extractor import extract_columns_and_samples
from modules.schema_agent import SchemaAgent
from modules.utils import setup_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Hugging Face codegen datasets.")
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
    return parser.parse_args()


def _named_path(base_path: Path, name: str) -> Path:
    if name == "default":
        return base_path
    return base_path.with_name(f"{base_path.stem}_{name}{base_path.suffix}")


def _checkpoint_path(root: Path, name: str) -> Path:
    return root / f".pipeline_checkpoint_{name}.json"


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


def run() -> None:
    args = _parse_args()
    setup_logging()
    root = Path(__file__).resolve().parent
    csv_path = SETTINGS.csv_file_path(root)
    progress_name = args.resume if args.resume is not None else args.progress_name
    output_csv_path = _named_path(SETTINGS.output_csv_path(root), progress_name)
    output_parquet_path = _named_path(SETTINGS.output_parquet_path(root), progress_name)
    run_log_path = _named_path(SETTINGS.run_log_path(root), progress_name)
    checkpoint_path = _checkpoint_path(root, progress_name)
    output_mode: Literal["csv", "parquet", "both"] = args.output_format or "csv"
    enable_dedup = not args.no_deduplicate

    if args.output_format is None:
        logger.info("No output format specified. Defaulting to CSV output.")
    logger.info(f"Output mode selected: {output_mode}")
    logger.info(f"Progress name: {progress_name}")

    logger.info(f"Reading dataset list from {csv_path}")
    dataset_ids = read_dataset_ids(csv_path)
    if not dataset_ids:
        logger.warning("No dataset IDs found. Exiting.")
        return

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
        logger.warning(warning_msg)
        run_logger.log(dataset_id="__system__", status="warning", message=warning_msg)

    processed_ids = set()
    if args.resume is not None:
        processed_ids = _load_checkpoint(checkpoint_path) | _load_done_from_run_log(run_log_path)
        logger.info(f"Resume enabled. Loaded {len(processed_ids)} already-processed dataset IDs.")
        run_logger.log(
            dataset_id="__system__",
            status="resume",
            records_written=len(processed_ids),
            message=f"Resuming with {len(processed_ids)} processed dataset IDs",
        )
    else:
        # New run mode: clear checkpoint for this progress name.
        _save_checkpoint(checkpoint_path, set())

    seen_fingerprints: set[str] = set()
    if enable_dedup:
        if output_mode in {"csv", "both"}:
            seen_fingerprints |= _load_seen_fingerprints_from_csv(output_csv_path)
        if output_mode in {"parquet", "both"}:
            seen_fingerprints |= _load_seen_fingerprints_from_parquet(output_parquet_path)
        logger.info(f"Dedup enabled. Loaded {len(seen_fingerprints)} existing output fingerprints.")

    total_written = 0
    total_duplicates_filtered = 0
    try:
        for dataset_id in tqdm(dataset_ids, desc="Datasets"):
            if dataset_id in processed_ids:
                logger.info(f"[{dataset_id}] skipped: already processed in resume state.")
                run_logger.log(
                    dataset_id=dataset_id,
                    status="resumed_skip",
                    records_written=0,
                    message="Skipped because already processed in previous run",
                )
                continue
            info = None
            try:
                run_logger.log(dataset_id=dataset_id, status="started", message="Dataset processing started")
                info = inspect_dataset(dataset_id)
                sample_stream = load_dataset_stream(dataset_id, info.config_name, info.split_name)
                columns, samples = extract_columns_and_samples(sample_stream, SETTINGS.max_sample_rows)
                if not columns:
                    columns = info.columns

                schema = agent.infer_schema(dataset_id=dataset_id, columns=columns, sample_rows=samples)
                if not is_relevant_codegen_dataset(schema):
                    msg = "Skipped: not a code-generation dataset."
                    logger.info(f"[{dataset_id}] {msg}")
                    run_logger.log(
                        dataset_id=dataset_id,
                        status="skipped",
                        config_name=info.config_name,
                        split_name=info.split_name,
                        records_written=0,
                        message=msg,
                    )
                    continue

                limit_display = SETTINGS.max_rows_per_dataset if SETTINGS.max_rows_per_dataset is not None else "ALL"
                mode_label = "CSV+Parquet" if output_mode == "both" else output_mode.upper()
                logger.info(f"[{dataset_id}] converting rows (limit={limit_display}) and appending to {mode_label}...")
                convert_stream = load_dataset_stream(dataset_id, info.config_name, info.split_name)
                records = convert_rows(
                    rows=convert_stream,
                    schema=schema,
                    dataset_id=dataset_id,
                    max_rows=SETTINGS.max_rows_per_dataset,
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
                        logger.info(f"[{dataset_id}] appended {written} records so far...")
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
                logger.info(f"[{dataset_id}] wrote {written} normalized rows.")
                run_logger.log(
                    dataset_id=dataset_id,
                    status="added",
                    config_name=info.config_name,
                    split_name=info.split_name,
                    records_written=written,
                    message=f"Dataset converted and appended. Duplicates filtered: {duplicates_filtered}",
                )
                processed_ids.add(dataset_id)
                _save_checkpoint(checkpoint_path, processed_ids)
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"[{dataset_id}] failed and skipped: {exc}")
                run_logger.log(
                    dataset_id=dataset_id,
                    status="failed",
                    config_name=info.config_name if info else None,
                    split_name=info.split_name if info else None,
                    records_written=0,
                    message="Dataset failed during processing",
                    error=str(exc),
                )
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    logger.info(f"Done. Total normalized records written: {total_written}")
    if enable_dedup:
        logger.info(f"Duplicates filtered: {total_duplicates_filtered}")
    if output_mode in {"csv", "both"}:
        logger.info(f"CSV output file: {output_csv_path}")
    if output_mode in {"parquet", "both"}:
        logger.info(f"Parquet output file: {output_parquet_path}")
    logger.info(f"Progress log file: {run_log_path}")
    logger.info(f"Checkpoint file: {checkpoint_path}")
    run_logger.log(
        dataset_id="__system__",
        status="completed",
        records_written=total_written,
        message=f"Pipeline execution completed. Duplicates filtered: {total_duplicates_filtered}",
    )


if __name__ == "__main__":
    run()
