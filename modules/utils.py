from __future__ import annotations

import json
import random
import time
from typing import Any, Callable, Iterable

from loguru import logger


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "prompt": "user",
    "assistant": "assistant",
    "model": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
}


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )


def safe_get(row: dict[str, Any], field: str | None, default: Any = None) -> Any:
    if not field:
        return default
    return row.get(field, default)


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def parse_json_if_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    parsed = parse_json_if_string(raw_messages)
    if not isinstance(parsed, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        role = to_text(item.get("role") or item.get("from") or item.get("speaker")).lower()
        role = ROLE_ALIASES.get(role, role)
        content = to_text(item.get("content") or item.get("value") or item.get("text"))
        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def keep_non_empty_strings(values: Iterable[str]) -> list[str]:
    return [v for v in values if isinstance(v, str) and v.strip()]


def retry(
    func: Callable[..., Any],
    retries: int,
    backoff_seconds: float,
    *args: Any,
    **kwargs: Any,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            msg = str(exc).lower()
            is_rate_limited = "rate limit" in msg or "too many requests" in msg or "429" in msg
            reason = "rate-limited" if is_rate_limited else "failed"
            logger.warning(f"Attempt {attempt}/{retries} {reason}: {exc}")
            if attempt < retries:
                delay = backoff_seconds * (2 ** (attempt - 1))
                if is_rate_limited:
                    delay *= 2
                delay += random.uniform(0, 0.3)
                time.sleep(delay)
    if last_error is not None:
        raise last_error

