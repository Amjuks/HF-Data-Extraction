from __future__ import annotations

from typing import Any, Iterable

from modules.schema_agent import SchemaDetection
from modules.utils import keep_non_empty_strings, normalize_messages, safe_get, to_text


def _build_user_prompt(row: dict[str, Any], schema: SchemaDetection) -> str:
    primary = to_text(safe_get(row, schema.user_field))
    additional = to_text(safe_get(row, schema.input_field))
    chunks = keep_non_empty_strings([primary, additional])
    return "\n\n".join(chunks)


def _build_single_turn_record(
    row: dict[str, Any],
    schema: SchemaDetection,
    dataset_id: str,
) -> dict[str, Any] | None:
    user_content = _build_user_prompt(row, schema)
    assistant_content = to_text(safe_get(row, schema.assistant_field))

    if not user_content or not assistant_content:
        return None

    metadata = {"dataset_id": dataset_id, "source_fields": [schema.user_field, schema.input_field, schema.assistant_field]}
    for field in schema.metadata_fields:
        metadata[field] = row.get(field)

    reasoning = to_text(safe_get(row, schema.reasoning_field)) or None
    return {
        "conversation": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "reasoning": reasoning,
        "metadata": metadata,
    }


def _build_multi_turn_record(
    row: dict[str, Any],
    schema: SchemaDetection,
    dataset_id: str,
) -> dict[str, Any] | None:
    conversation = normalize_messages(safe_get(row, schema.messages_field))
    if len(conversation) < 2:
        return None
    if not any(m["role"] == "assistant" for m in conversation):
        return None
    if not any(m["role"] == "user" for m in conversation):
        return None

    metadata = {"dataset_id": dataset_id, "source_fields": [schema.messages_field]}
    for field in schema.metadata_fields:
        metadata[field] = row.get(field)

    reasoning = to_text(safe_get(row, schema.reasoning_field)) or None
    return {
        "conversation": conversation,
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

