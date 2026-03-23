from __future__ import annotations

import json
from typing import Any, Iterable

from modules.schema_agent import SchemaDetection
from modules.utils import keep_non_empty_strings, normalize_language, normalize_messages, safe_get, to_text


def _build_user_prompt(row: dict[str, Any], schema: SchemaDetection) -> str:
    primary = to_text(safe_get(row, schema.user_path))
    additional_raw = safe_get(row, schema.input_path)
    additional = to_text(additional_raw)
    # Drop auxiliary fields that look like vectors/masks/labels instead of prompt context.
    if _is_noisy_aux_field(additional_raw, additional):
        additional = ""
    chunks = keep_non_empty_strings([primary, additional])
    return "\n\n".join(chunks)


def _is_noisy_aux_field(raw_value: Any, text_value: str) -> bool:
    if raw_value is None:
        return False
    if isinstance(raw_value, (list, dict)):
        return True
    if len(text_value) > 1200:
        return True
    stripped = text_value.strip()
    if not stripped:
        return False
    if stripped.startswith("[") and stripped.endswith("]"):
        return True
    if stripped.startswith("{") and stripped.endswith("}"):
        return True
    return False


def _sanitize_metadata_value(value: Any) -> Any | None:
    # Keep metadata lightweight and avoid huge arrays/masks/tensors.
    if value is None:
        return None
    if isinstance(value, (list, dict, tuple, set)):
        return None
    if isinstance(value, (bool, int, float)):
        return value
    text = to_text(value)
    if not text:
        return None
    if len(text) > 500:
        return None
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (list, dict)):
                return None
        except json.JSONDecodeError:
            pass
    return text


def _build_single_turn_record(
    row: dict[str, Any],
    schema: SchemaDetection,
    dataset_id: str,
) -> dict[str, Any] | None:
    user_content = _build_user_prompt(row, schema)
    assistant_content = to_text(safe_get(row, schema.assistant_path))

    if not user_content or not assistant_content:
        return None

    language = schema.language
    if schema.language_path:
        language = normalize_language(safe_get(row, schema.language_path))

    metadata = {
        "dataset_id": dataset_id,
        "task_type": schema.task_type,
        "language_source": schema.language_path or "schema.language",
        "source_fields": [schema.user_path, schema.input_path, schema.assistant_path],
    }
    for field in schema.metadata_paths:
        sanitized = _sanitize_metadata_value(safe_get(row, field))
        if sanitized is not None:
            metadata[field] = sanitized

    reasoning = to_text(safe_get(row, schema.reasoning_path)) or None
    return {
        "conversation": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "language": language,
        "reasoning": reasoning,
        "metadata": metadata,
    }


def _build_multi_turn_record(
    row: dict[str, Any],
    schema: SchemaDetection,
    dataset_id: str,
) -> dict[str, Any] | None:
    if schema.message_parse is None:
        return None
    raw_messages = safe_get(row, schema.message_parse.messages_path)
    conversation = normalize_messages(
        raw_messages,
        role_path=schema.message_parse.role_path,
        content_path=schema.message_parse.content_path,
        role_mapping=schema.message_parse.role_mapping,
    )
    if len(conversation) < 2:
        return None
    if not any(m["role"] == "assistant" for m in conversation):
        return None
    if not any(m["role"] == "user" for m in conversation):
        return None

    language = schema.language
    if schema.language_path:
        language = normalize_language(safe_get(row, schema.language_path))

    metadata = {
        "dataset_id": dataset_id,
        "task_type": schema.task_type,
        "language_source": schema.language_path or "schema.language",
        "source_fields": [schema.message_parse.messages_path],
    }
    for field in schema.metadata_paths:
        sanitized = _sanitize_metadata_value(safe_get(row, field))
        if sanitized is not None:
            metadata[field] = sanitized

    reasoning = to_text(safe_get(row, schema.reasoning_path)) or None
    return {
        "conversation": conversation,
        "language": language,
        "reasoning": reasoning,
        "metadata": metadata,
    }


def convert_rows(
    rows: Iterable[dict[str, Any]],
    schema: SchemaDetection,
    dataset_id: str,
    max_rows: int | None,
) -> Iterable[dict[str, Any]]:
    converted = 0
    for row in rows:
        if max_rows is not None and converted >= max_rows:
            break
        if not isinstance(row, dict):
            continue
        record: dict[str, Any] | None = None
        if schema.conversation_type == "single_turn":
            record = _build_single_turn_record(row, schema, dataset_id)
        elif schema.conversation_type == "multi_turn":
            record = _build_multi_turn_record(row, schema, dataset_id)
        if record:
            converted += 1
            yield record
