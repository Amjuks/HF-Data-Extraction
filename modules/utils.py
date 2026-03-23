from __future__ import annotations

import json
import random
import re
import threading
import time
from contextlib import contextmanager
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

LANGUAGE_ALIASES = {
    "en": "english",
    "eng": "english",
    "english": "english",
    "en-us": "english",
    "en-gb": "english",
    "hindi": "hindi",
    "hi": "hindi",
    "hin": "hindi",
    "spanish": "spanish",
    "es": "spanish",
    "spa": "spanish",
    "french": "french",
    "fr": "french",
    "fra": "french",
    "de": "german",
    "deu": "german",
    "german": "german",
    "zh": "chinese",
    "zho": "chinese",
    "chinese": "chinese",
    "ja": "japanese",
    "jpn": "japanese",
    "japanese": "japanese",
    "ko": "korean",
    "kor": "korean",
    "korean": "korean",
    "multilingual": "multilingual",
    "multiple": "multilingual",
    "mixed": "multilingual",
    "unknown": "unknown",
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
    current: Any = row
    for part in field.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current.get(part)
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return default
            current = current[index]
            continue
        return default
    return current


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


def normalize_messages(
    raw_messages: Any,
    *,
    role_path: str = "",
    content_path: str = "",
    role_mapping: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    parsed = parse_json_if_string(raw_messages)
    messages = parsed if isinstance(parsed, list) else _extract_message_list(parsed)
    if not isinstance(messages, list):
        return []

    mapping = {k.lower(): v for k, v in (role_mapping or ROLE_ALIASES).items()}
    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role_value = safe_get(item, role_path) if role_path else item.get("role") or item.get("from") or item.get("speaker")
        content_value = safe_get(item, content_path) if content_path else item.get("content") or item.get("value") or item.get("text")
        role = to_text(role_value).lower()
        role = mapping.get(role, role)
        content = to_text(content_value)
        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def keep_non_empty_strings(values: Iterable[str]) -> list[str]:
    return [v for v in values if isinstance(v, str) and v.strip()]


def normalize_language(value: Any) -> str:
    raw = to_text(value).lower().replace("_", "-")
    if not raw:
        return "unknown"
    if raw in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[raw]
    if "-" in raw:
        primary = raw.split("-", 1)[0]
        if primary in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[primary]
    cleaned = "".join(ch if ch.isalpha() else "_" for ch in raw).strip("_")
    if not cleaned:
        return "unknown"
    if cleaned in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[cleaned]
    return cleaned


def _extract_message_list(value: Any) -> list[Any] | None:
    parsed = parse_json_if_string(value)
    if isinstance(parsed, list):
        if parsed and any(isinstance(item, dict) for item in parsed):
            return parsed
        for item in parsed:
            nested = _extract_message_list(item)
            if nested:
                return nested
        return parsed
    if isinstance(parsed, dict):
        for key in ("messages", "conversation", "conversations", "chat", "chats", "dialog", "dialogue"):
            nested = parsed.get(key)
            if isinstance(nested, list):
                return nested
        for nested_value in parsed.values():
            nested = _extract_message_list(nested_value)
            if nested:
                return nested
    return None


def find_message_path(value: Any) -> str | None:
    def _walk(node: Any, prefix: str) -> str | None:
        parsed = parse_json_if_string(node)
        if isinstance(parsed, list):
            if parsed and any(isinstance(item, dict) for item in parsed):
                first = parsed[0]
                if isinstance(first, dict) and any(k in first for k in ("role", "content", "from", "value", "speaker", "text")):
                    return prefix
            for index, item in enumerate(parsed):
                child_prefix = f"{prefix}.{index}" if prefix else str(index)
                found = _walk(item, child_prefix)
                if found is not None:
                    return found
        if isinstance(parsed, dict):
            for key, item in parsed.items():
                child_prefix = f"{prefix}.{key}" if prefix else key
                found = _walk(item, child_prefix)
                if found is not None:
                    return found
        return None

    found_path = _walk(value, "")
    return found_path or None


class RateLimiter:
    def __init__(self, max_concurrency: int, min_interval_seconds: float = 0.0) -> None:
        self._semaphore = threading.Semaphore(max(1, max_concurrency))
        self._lock = threading.Lock()
        self._min_interval_seconds = max(0.0, min_interval_seconds)
        self._last_started = 0.0

    @contextmanager
    def acquire(self) -> Any:
        self._semaphore.acquire()
        try:
            if self._min_interval_seconds > 0:
                with self._lock:
                    now = time.monotonic()
                    wait_time = self._min_interval_seconds - (now - self._last_started)
                    if wait_time > 0:
                        time.sleep(wait_time)
                    self._last_started = time.monotonic()
            yield
        finally:
            self._semaphore.release()


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
            is_rate_limited = is_rate_limit_error(exc)
            reason = "rate-limited" if is_rate_limited else "failed"
            logger.warning(f"Attempt {attempt}/{retries} {reason}: {exc}")
            if attempt < retries:
                delay = backoff_seconds * (2 ** (attempt - 1))
                if is_rate_limited:
                    delay *= 2
                    retry_after = extract_retry_after_seconds(exc)
                    if retry_after is not None:
                        delay = max(delay, retry_after)
                delay += random.uniform(0, 0.3)
                time.sleep(delay)
    if last_error is not None:
        raise last_error


def is_rate_limit_error(exc: Exception | str) -> bool:
    msg = str(exc).lower()
    return (
        "rate limit" in msg
        or "too many requests" in msg
        or "429" in msg
        or "we had to rate limit your ip" in msg
    )


def is_hf_token_required_error(exc: Exception | str) -> bool:
    msg = str(exc).lower()
    return "pass a hf_token" in msg or "login to your existing account" in msg


def is_feature_schema_unsupported_error(exc: Exception | str) -> bool:
    msg = str(exc)
    return "Feature type 'Json' not found" in msg or "Available feature types" in msg


def extract_retry_after_seconds(exc: Exception | str) -> float | None:
    candidates = [
        getattr(exc, "response", None),
        getattr(exc, "__cause__", None),
    ]
    for candidate in candidates:
        headers = getattr(candidate, "headers", None)
        if headers:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass

    msg = str(exc)
    patterns = [
        r"retry[- ]after[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"Retry-After[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"please try again in\s*([0-9]+(?:\.[0-9]+)?)\s*(seconds?|secs?|s)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, msg, flags=re.IGNORECASE)
        if match:
            try:
                return max(0.0, float(match.group(1)))
            except ValueError:
                continue
    return None

