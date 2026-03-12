from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

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
    return parser.parse_args()


def run() -> None:
    args = _parse_args()
    setup_logging()
    root = Path(__file__).resolve().parent
    csv_path = SETTINGS.csv_file_path(root)
    output_csv_path = SETTINGS.output_csv_path(root)
    output_parquet_path = SETTINGS.output_parquet_path(root)
    run_log_path = SETTINGS.run_log_path(root)
    output_mode: Literal["csv", "parquet", "both"] = args.output_format or "csv"

    if args.output_format is None:
        logger.info("No output format specified. Defaulting to CSV output.")
    logger.info(f"Output mode selected: {output_mode}")

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

    total_written = 0
    try:
        for dataset_id in tqdm(dataset_ids, desc="Datasets"):
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
                batch: list[dict] = []
                for record in records:
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
                logger.info(f"[{dataset_id}] wrote {written} normalized rows.")
                run_logger.log(
                    dataset_id=dataset_id,
                    status="added",
                    config_name=info.config_name,
                    split_name=info.split_name,
                    records_written=written,
                    message="Dataset converted and appended",
                )
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
    if output_mode in {"csv", "both"}:
        logger.info(f"CSV output file: {output_csv_path}")
    if output_mode in {"parquet", "both"}:
        logger.info(f"Parquet output file: {output_parquet_path}")
    logger.info(f"Progress log file: {run_log_path}")
    run_logger.log(
        dataset_id="__system__",
        status="completed",
        records_written=total_written,
        message="Pipeline execution completed",
    )


if __name__ == "__main__":
    run()
