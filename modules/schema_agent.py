from __future__ import annotations

import json
from typing import Any, Literal

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from config import SETTINGS


class SchemaDetection(BaseModel):
    is_codegen_dataset: bool
    conversation_type: Literal["single_turn", "multi_turn"] | None = None
    user_field: str | None = None
    input_field: str | None = None
    assistant_field: str | None = None
    messages_field: str | None = None
    reasoning_field: str | None = None
    metadata_fields: list[str] = Field(default_factory=list)


def _heuristic_schema(columns: list[str]) -> SchemaDetection:
    lower_map = {c.lower(): c for c in columns}

    messages_candidate = None
    for key in ("messages", "conversation", "conversations", "chat"):
        if key in lower_map:
            messages_candidate = lower_map[key]
            break
    if messages_candidate:
        return SchemaDetection(
            is_codegen_dataset=True,
            conversation_type="multi_turn",
            messages_field=messages_candidate,
            metadata_fields=[],
        )

    user_field = None
    for key in ("instruction", "prompt", "question", "input"):
        if key in lower_map:
            user_field = lower_map[key]
            break
    assistant_field = None
    for key in ("output", "response", "answer", "completion", "assistant"):
        if key in lower_map:
            assistant_field = lower_map[key]
            break
    input_field = lower_map.get("input") if user_field and user_field.lower() != "input" else None

    if user_field and assistant_field:
        return SchemaDetection(
            is_codegen_dataset=True,
            conversation_type="single_turn",
            user_field=user_field,
            input_field=input_field,
            assistant_field=assistant_field,
            metadata_fields=[],
        )

    return SchemaDetection(is_codegen_dataset=False)


class SchemaAgent:
    def __init__(self) -> None:
        self.client: OpenAI | None = None
        if SETTINGS.llm_api_key:
            self.client = OpenAI(api_key=SETTINGS.llm_api_key, base_url=SETTINGS.llm_base_url)

    def infer_schema(
        self,
        dataset_id: str,
        columns: list[str],
        sample_rows: list[dict[str, Any]],
    ) -> SchemaDetection:
        if not self.client:
            logger.warning(f"[{dataset_id}] LLM_API_KEY not set. Falling back to heuristic schema detection.")
            return _heuristic_schema(columns)

        prompt_payload = {
            "dataset_id": dataset_id,
            "columns": columns,
            "sample_rows": sample_rows[: min(len(sample_rows), 8)],
            "task": (
                "Determine if this is a code-generation dataset and map fields to unified conversation schema."
                " Return ONLY valid JSON."
            ),
            "expected_json_schema": {
                "is_codegen_dataset": "boolean",
                "conversation_type": "single_turn | multi_turn | null",
                "user_field": "string | null",
                "input_field": "string | null",
                "assistant_field": "string | null",
                "messages_field": "string | null",
                "reasoning_field": "string | null",
                "metadata_fields": "string[]",
            },
        }

        system_prompt = (
            "You are a strict dataset schema analyzer. "
            "Only classify as code-generation if user prompts request code OR assistant outputs are code. "
            "Return JSON only; no markdown."
        )

        try:
            response = self.client.responses.create(
                model=SETTINGS.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
                ],
                temperature=0,
                max_output_tokens=400,
            )
            text = response.output_text.strip()
            parsed = json.loads(text)
            schema = SchemaDetection.model_validate(parsed)
            if not schema.is_codegen_dataset:
                return schema
            if schema.conversation_type == "single_turn" and not (schema.user_field and schema.assistant_field):
                logger.warning(f"[{dataset_id}] Invalid single_turn mapping from LLM. Falling back to heuristic.")
                return _heuristic_schema(columns)
            if schema.conversation_type == "multi_turn" and not schema.messages_field:
                logger.warning(f"[{dataset_id}] Invalid multi_turn mapping from LLM. Falling back to heuristic.")
                return _heuristic_schema(columns)
            return schema
        except (json.JSONDecodeError, ValidationError, Exception) as exc:  # noqa: BLE001
            logger.warning(f"[{dataset_id}] Schema inference failed: {exc}. Falling back to heuristic.")
            return _heuristic_schema(columns)

