from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import IterableDataset, get_dataset_config_names, get_dataset_split_names, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger
import pyarrow.parquet as pq

from config import SETTINGS
from modules.utils import RateLimiter, is_feature_schema_unsupported_error, is_hf_token_required_error, retry


_HF_RATE_LIMITER = RateLimiter(
    max_concurrency=SETTINGS.hf_max_concurrency,
    min_interval_seconds=SETTINGS.hf_min_interval_seconds,
)
_HF_API = HfApi(token=SETTINGS.hf_api_key.strip() or None)


@dataclass
class DatasetLoadInfo:
    dataset_id: str
    config_name: str | None
    split_names: list[str]
    columns: list[str]


class DatasetDeferredError(RuntimeError):
    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


RAW_CONFIG_SENTINEL = "__raw__"


def _with_optional_token(func, *args, **kwargs):
    with _HF_RATE_LIMITER.acquire():
        token = SETTINGS.hf_api_key.strip()
        if not token:
            return func(*args, **kwargs)
        try:
            return func(*args, token=token, **kwargs)
        except TypeError:
            # Backward compatibility for older datasets versions.
            return func(*args, use_auth_token=token, **kwargs)


def _list_repo_files(dataset_id: str) -> list[str]:
    with _HF_RATE_LIMITER.acquire():
        return _HF_API.list_repo_files(repo_id=dataset_id, repo_type="dataset")


def _download_repo_file(dataset_id: str, filename: str) -> str:
    with _HF_RATE_LIMITER.acquire():
        return hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=filename,
            token=SETTINGS.hf_api_key.strip() or None,
        )


def _pick_splits(dataset_id: str, config_name: str | None) -> list[str]:
    splits = _with_optional_token(get_dataset_split_names, path=dataset_id, config_name=config_name)
    return list(dict.fromkeys(str(split).strip() for split in splits if str(split).strip()))


def _pick_config(dataset_id: str) -> str | None:
    try:
        configs = _with_optional_token(get_dataset_config_names, path=dataset_id)
    except Exception:  # noqa: BLE001
        return None
    if not configs:
        return None
    return configs[0]


def _load_stream(dataset_id: str, config_name: str | None, split_name: str) -> IterableDataset:
    if config_name:
        ds = _with_optional_token(
            load_dataset,
            dataset_id,
            name=config_name,
            split=split_name,
            streaming=SETTINGS.streaming,
        )
    else:
        ds = _with_optional_token(load_dataset, dataset_id, split=split_name, streaming=SETTINGS.streaming)
    if not isinstance(ds, IterableDataset) and SETTINGS.streaming:
        ds = ds.to_iterable_dataset()
    return ds


def _merge_streams(streams: list[IterableDataset]) -> Iterable[dict]:
    for stream in streams:
        for row in stream:
            yield row


def _load_all_splits_stream(dataset_id: str, config_name: str | None, split_names: list[str]) -> Iterable[dict]:
    streams = [_load_stream(dataset_id, config_name, split_name) for split_name in split_names]
    return _merge_streams(streams)


def _iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _iter_json_rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        for key in ("data", "rows", "items", "examples", "records"):
            items = payload.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload


def _iter_csv_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def _iter_parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    parquet = pq.ParquetFile(path.as_posix())
    for batch in parquet.iter_batches(batch_size=1000):
        for row in batch.to_pylist():
            if isinstance(row, dict):
                yield row


def _raw_file_iter(path: Path) -> Iterable[dict[str, Any]]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        yield from _iter_jsonl_rows(path)
        return
    if suffixes.endswith(".json"):
        yield from _iter_json_rows(path)
        return
    if suffixes.endswith(".csv"):
        yield from _iter_csv_rows(path)
        return
    if suffixes.endswith(".parquet"):
        yield from _iter_parquet_rows(path)


def _candidate_data_files(files: list[str]) -> list[str]:
    preferred_suffixes = (".jsonl", ".json", ".csv", ".parquet")
    filtered = [
        file for file in files
        if file.lower().endswith(preferred_suffixes)
        and "/." not in file
        and not file.lower().endswith("dataset_infos.json")
        and not file.lower().endswith("state.json")
    ]
    return sorted(filtered)


def _load_raw_repo_stream(dataset_id: str) -> tuple[list[str], Iterable[dict[str, Any]]]:
    files = retry(_list_repo_files, SETTINGS.max_retries, SETTINGS.retry_backoff_seconds, dataset_id)
    candidates = _candidate_data_files(files)
    if not candidates:
        raise ValueError(f"No supported raw data files found in dataset repo {dataset_id}")

    local_paths = [Path(retry(_download_repo_file, SETTINGS.max_retries, SETTINGS.retry_backoff_seconds, dataset_id, filename)) for filename in candidates]

    def _iter_rows() -> Iterable[dict[str, Any]]:
        for local_path in local_paths:
            yield from _raw_file_iter(local_path)

    return [Path(file).name for file in candidates], _iter_rows()


def inspect_dataset(dataset_id: str) -> DatasetLoadInfo:
    try:
        config_name = retry(_pick_config, SETTINGS.max_retries, SETTINGS.retry_backoff_seconds, dataset_id)
        split_names = retry(_pick_splits, SETTINGS.max_retries, SETTINGS.retry_backoff_seconds, dataset_id, config_name)
        if not split_names:
            raise ValueError(f"No splits found for dataset {dataset_id}")
        stream = retry(
            _load_all_splits_stream,
            SETTINGS.max_retries,
            SETTINGS.retry_backoff_seconds,
            dataset_id,
            config_name,
            split_names,
        )
    except Exception as exc:  # noqa: BLE001
        if is_hf_token_required_error(exc):
            raise DatasetDeferredError(
                "Dataset access deferred because Hugging Face requested authentication after rate limiting. "
                "Set HF_API_KEY / HF_TOKEN and rerun.",
                reason="hf_auth_required",
            ) from exc
        if is_feature_schema_unsupported_error(exc) or "JSON parse error" in str(exc):
            logger.warning(f"[{dataset_id}] datasets loader failed; falling back to raw repo files: {exc}")
            try:
                split_names, stream = _load_raw_repo_stream(dataset_id)
                config_name = RAW_CONFIG_SENTINEL
            except Exception as raw_exc:  # noqa: BLE001
                raise DatasetDeferredError(
                    "Dataset access deferred because both the datasets loader and raw-file fallback failed.",
                    reason="raw_fallback_failed",
                ) from raw_exc
        else:
            raise

    columns: list[str] = []
    if hasattr(stream, "features") and stream.features:
        columns = list(stream.features.keys())
    logger.info(f"[{dataset_id}] config={config_name or '<default>'}, splits={split_names}, columns={columns}")
    return DatasetLoadInfo(
        dataset_id=dataset_id,
        config_name=config_name,
        split_names=split_names,
        columns=columns,
    )


def load_dataset_stream(dataset_id: str, config_name: str | None, split_names: list[str]) -> Iterable[dict]:
    try:
        if config_name == RAW_CONFIG_SENTINEL:
            _, raw_stream = _load_raw_repo_stream(dataset_id)
            return raw_stream
        return retry(
            _load_all_splits_stream,
            SETTINGS.max_retries,
            SETTINGS.retry_backoff_seconds,
            dataset_id,
            config_name,
            split_names,
        )
    except Exception as exc:  # noqa: BLE001
        if is_hf_token_required_error(exc):
            raise DatasetDeferredError(
                "Dataset stream deferred because Hugging Face requested authentication after rate limiting. "
                "Set HF_API_KEY / HF_TOKEN and rerun.",
                reason="hf_auth_required",
            ) from exc
        if is_feature_schema_unsupported_error(exc) or "JSON parse error" in str(exc):
            try:
                _, raw_stream = _load_raw_repo_stream(dataset_id)
                return raw_stream
            except Exception as raw_exc:  # noqa: BLE001
                raise DatasetDeferredError(
                    "Dataset stream deferred because both the datasets loader and raw-file fallback failed.",
                    reason="raw_fallback_failed",
                ) from raw_exc
        raise
