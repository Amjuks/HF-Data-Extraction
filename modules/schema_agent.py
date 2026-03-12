from __future__ import annotations

import json
import re
from typing import Any, Literal

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from config import SETTINGS
from modules.utils import retry


class SchemaDetection(BaseModel):
    is_codegen_dataset: bool
    conversation_type: Literal["single_turn", "multi_turn"] | None = None
    user_field: str | None = None
    input_field: str | None = None
    assistant_field: str | None = None
    messages_field: str | None = None
    reasoning_field: str | None = None
    metadata_fields: list[str] = Field(default_factory=list)


def _try_json_extract(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return json.loads(match.group(0))


def _looks_like_messages(value: Any) -> bool:
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") or value.startswith("{"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return False
        else:
            return False
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and any(k in first for k in ("role", "content", "from", "value", "speaker"))


def _content_score_as_code(text: str) -> float:
    if not text:
        return 0.0
    code_markers = [
        "def ",
        "class ",
        "import ",
        "return ",
        "public ",
        "private ",
        "function ",
        "=>",
        "{",
        "}",
        ";",
        "```",
    ]
    score = sum(1 for marker in code_markers if marker in text)
    # Dense punctuation and many newlines often indicate code blocks.
    score += text.count("\n") * 0.05
    score += (text.count("{") + text.count("}")) * 0.1
    return score


def _content_score_as_prompt(text: str) -> float:
    if not text:
        return 0.0
    prompt_markers = [
        "?",
        "write",
        "implement",
        "create",
        "solve",
        "problem",
        "task",
        "question",
        "given",
        "input",
        "output",
        "explain",
    ]
    lower = text.lower()
    score = sum(1 for marker in prompt_markers if marker in lower)
    # Prefer natural-language instructions over raw code.
    score += 1.0 if lower.count(" ") > 5 else 0.0
    score -= 0.5 if _content_score_as_code(text) > 3 else 0.0
    return score


def _is_probably_vectorish_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")):
        return True
    if len(stripped) > 300 and stripped.count(",") > 30:
        return True
    digit_ratio = sum(ch.isdigit() for ch in stripped) / max(1, len(stripped))
    if digit_ratio > 0.35 and stripped.count(" ") < 10:
        return True
    return False


def _is_candidate_context_column(values: list[str]) -> bool:
    if not values:
        return False
    non_vector = [v for v in values if not _is_probably_vectorish_text(v)]
    if len(non_vector) < max(1, len(values) // 2):
        return False
    avg_len = sum(len(v) for v in non_vector) / len(non_vector)
    if avg_len > 1200:
        return False
    return True


def _content_based_schema(columns: list[str], sample_rows: list[dict[str, Any]]) -> SchemaDetection:
    if not columns or not sample_rows:
        return SchemaDetection(is_codegen_dataset=False)

    # 1) Multi-turn detection from sample values.
    for col in columns:
        values = [row.get(col) for row in sample_rows if isinstance(row, dict)]
        if values and sum(1 for v in values if _looks_like_messages(v)) >= max(1, len(values) // 2):
            return SchemaDetection(
                is_codegen_dataset=True,
                conversation_type="multi_turn",
                messages_field=col,
                metadata_fields=[],
            )

    # 2) Single-turn detection using content statistics (not fixed column names).
    text_scores: dict[str, dict[str, float]] = {}
    col_samples: dict[str, list[str]] = {}
    for col in columns:
        col_values: list[str] = []
        for row in sample_rows:
            if not isinstance(row, dict):
                continue
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                col_values.append(val.strip())
        if not col_values:
            continue
        col_samples[col] = col_values
        prompt_score = sum(_content_score_as_prompt(v) for v in col_values) / len(col_values)
        code_score = sum(_content_score_as_code(v) for v in col_values) / len(col_values)
        avg_len = sum(len(v) for v in col_values) / len(col_values)
        text_scores[col] = {"prompt": prompt_score, "code": code_score, "avg_len": avg_len}

    if len(text_scores) < 2:
        return SchemaDetection(is_codegen_dataset=False)

    assistant_field = max(text_scores, key=lambda c: text_scores[c]["code"])
    user_field = max(
        (c for c in text_scores if c != assistant_field),
        key=lambda c: text_scores[c]["prompt"],
        default=None,
    )
    if not user_field:
        return SchemaDetection(is_codegen_dataset=False)

    # Weak confidence guard.
    if text_scores[assistant_field]["code"] < 0.8 or text_scores[user_field]["prompt"] < 0.8:
        return SchemaDetection(is_codegen_dataset=False)

    # Optional secondary context: second-best prompt column.
    prompt_ranked = sorted(
        [c for c in text_scores if c != user_field and c != assistant_field],
        key=lambda c: text_scores[c]["prompt"],
        reverse=True,
    )
    input_field = None
    if prompt_ranked:
        candidate = prompt_ranked[0]
        if text_scores[candidate]["prompt"] >= 0.8 and _is_candidate_context_column(col_samples.get(candidate, [])):
            input_field = candidate

    return SchemaDetection(
        is_codegen_dataset=True,
        conversation_type="single_turn",
        user_field=user_field,
        input_field=input_field,
        assistant_field=assistant_field,
        metadata_fields=[],
    )


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
            return _content_based_schema(columns, sample_rows)

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
            "You are a strict dataset schema analyzer. Decide mapping by dataset content, not column names alone. "
            "Only classify as code-generation if prompts ask for code or outputs are code. "
            "Return JSON only; no markdown."
        )

        try:
            text = retry(
                self._invoke_llm,
                SETTINGS.max_retries,
                SETTINGS.retry_backoff_seconds,
                system_prompt,
                prompt_payload,
            ).strip()
            parsed = _try_json_extract(text)
            schema = SchemaDetection.model_validate(parsed)
            if not schema.is_codegen_dataset:
                return schema
            if schema.conversation_type == "single_turn" and not (schema.user_field and schema.assistant_field):
                logger.warning(f"[{dataset_id}] Invalid single_turn mapping from LLM. Falling back to content-based detection.")
                return _content_based_schema(columns, sample_rows)
            if schema.conversation_type == "multi_turn" and not schema.messages_field:
                logger.warning(f"[{dataset_id}] Invalid multi_turn mapping from LLM. Falling back to content-based detection.")
                return _content_based_schema(columns, sample_rows)
            # Ensure chosen fields are valid dataset columns.
            if schema.user_field and schema.user_field not in columns:
                raise ValueError(f"user_field not in dataset columns: {schema.user_field}")
            if schema.assistant_field and schema.assistant_field not in columns:
                raise ValueError(f"assistant_field not in dataset columns: {schema.assistant_field}")
            if schema.messages_field and schema.messages_field not in columns:
                raise ValueError(f"messages_field not in dataset columns: {schema.messages_field}")
            return schema
        except (json.JSONDecodeError, ValidationError, Exception) as exc:  # noqa: BLE001
            logger.warning(f"[{dataset_id}] Schema inference failed: {exc}. Falling back to content-based detection.")
            return _content_based_schema(columns, sample_rows)

    def _invoke_llm(self, system_prompt: str, payload: dict[str, Any]) -> str:
        assert self.client is not None
        user_content = json.dumps(payload, ensure_ascii=False)

        # Newer OpenAI SDK interface.
        if hasattr(self.client, "responses"):
            response = self.client.responses.create(
                model=SETTINGS.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_output_tokens=400,
            )
            return response.output_text

        # Backward-compatible fallback for older OpenAI SDKs.
        completion = self.client.chat.completions.create(
            model=SETTINGS.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        return content or "{}"
