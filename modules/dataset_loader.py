from __future__ import annotations

from dataclasses import dataclass

from datasets import IterableDataset, get_dataset_config_names, get_dataset_split_names, load_dataset
from loguru import logger

from config import SETTINGS
from modules.utils import retry


@dataclass
class DatasetLoadInfo:
    dataset_id: str
    config_name: str | None
    split_name: str
    columns: list[str]


def _pick_split(dataset_id: str, config_name: str | None) -> str:
    splits = get_dataset_split_names(path=dataset_id, config_name=config_name)
    if SETTINGS.default_split in splits:
        return SETTINGS.default_split
    if "train" in splits:
        return "train"
    return splits[0]


def _pick_config(dataset_id: str) -> str | None:
    try:
        configs = get_dataset_config_names(path=dataset_id)
    except Exception:  # noqa: BLE001
        return None
    if not configs:
        return None
    return configs[0]


def _load_stream(dataset_id: str, config_name: str | None, split_name: str) -> IterableDataset:
    if config_name:
        ds = load_dataset(dataset_id, name=config_name, split=split_name, streaming=SETTINGS.streaming)
    else:
        ds = load_dataset(dataset_id, split=split_name, streaming=SETTINGS.streaming)
    if not isinstance(ds, IterableDataset) and SETTINGS.streaming:
        ds = ds.to_iterable_dataset()
    return ds


def inspect_dataset(dataset_id: str) -> DatasetLoadInfo:
    config_name = retry(_pick_config, SETTINGS.max_retries, SETTINGS.retry_backoff_seconds, dataset_id)
    split_name = retry(_pick_split, SETTINGS.max_retries, SETTINGS.retry_backoff_seconds, dataset_id, config_name)
    stream = retry(_load_stream, SETTINGS.max_retries, SETTINGS.retry_backoff_seconds, dataset_id, config_name, split_name)

    columns: list[str] = []
    if hasattr(stream, "features") and stream.features:
        columns = list(stream.features.keys())
    logger.info(f"[{dataset_id}] config={config_name or '<default>'}, split={split_name}, columns={columns}")
    return DatasetLoadInfo(
        dataset_id=dataset_id,
        config_name=config_name,
        split_name=split_name,
        columns=columns,
    )


def load_dataset_stream(dataset_id: str, config_name: str | None, split_name: str) -> IterableDataset:
    return retry(_load_stream, SETTINGS.max_retries, SETTINGS.retry_backoff_seconds, dataset_id, config_name, split_name)

