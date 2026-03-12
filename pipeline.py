from __future__ import annotations

from pathlib import Path

from loguru import logger
from tqdm import tqdm

from config import SETTINGS
from modules.csv_reader import read_dataset_ids
from modules.csv_writer import CsvWriter
from modules.dataset_classifier import is_relevant_codegen_dataset
from modules.dataset_converter import convert_rows
from modules.dataset_loader import inspect_dataset, load_dataset_stream
from modules.sample_extractor import extract_columns_and_samples
from modules.schema_agent import SchemaAgent
from modules.utils import setup_logging


def run() -> None:
    setup_logging()
    root = Path(__file__).resolve().parent
    csv_path = SETTINGS.csv_file_path(root)
    output_path = SETTINGS.output_path(root)

    logger.info(f"Reading dataset list from {csv_path}")
    dataset_ids = read_dataset_ids(csv_path)
    if not dataset_ids:
        logger.warning("No dataset IDs found. Exiting.")
        return

    writer = CsvWriter(output_path)
    agent = SchemaAgent()

    total_written = 0
    for dataset_id in tqdm(dataset_ids, desc="Datasets"):
        try:
            info = inspect_dataset(dataset_id)
            sample_stream = load_dataset_stream(dataset_id, info.config_name, info.split_name)
            columns, samples = extract_columns_and_samples(sample_stream, SETTINGS.max_sample_rows)
            if not columns:
                columns = info.columns

            schema = agent.infer_schema(dataset_id=dataset_id, columns=columns, sample_rows=samples)
            if not is_relevant_codegen_dataset(schema):
                logger.info(f"[{dataset_id}] skipped: not a code-generation dataset.")
                continue

            limit_display = SETTINGS.max_rows_per_dataset if SETTINGS.max_rows_per_dataset is not None else "ALL"
            logger.info(f"[{dataset_id}] converting rows (limit={limit_display}) and appending to {output_path.name}...")
            convert_stream = load_dataset_stream(dataset_id, info.config_name, info.split_name)
            records = convert_rows(
                rows=convert_stream,
                schema=schema,
                dataset_id=dataset_id,
                max_rows=SETTINGS.max_rows_per_dataset,
            )

            def _on_record(count: int) -> None:
                if count == 1 or count % SETTINGS.progress_log_every == 0:
                    logger.info(f"[{dataset_id}] appended {count} records so far...")

            written = writer.append_records(
                records,
                flush_each_record=SETTINGS.flush_each_record,
                on_record=_on_record,
            )
            total_written += written
            logger.info(f"[{dataset_id}] wrote {written} normalized rows.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{dataset_id}] failed and skipped: {exc}")

    logger.info(f"Done. Total normalized records written: {total_written}")
    logger.info(f"Output file: {output_path}")


if __name__ == "__main__":
    run()
