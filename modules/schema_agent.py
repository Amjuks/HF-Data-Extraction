from __future__ import annotations

import json
import re
from typing import Any, Literal

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from config import SETTINGS
from modules.utils import find_message_path, normalize_language, retry, safe_get


TARGET_DOMAINS = {"code_generation", "math", "natural_language"}
_LLM_RATE_LIMITER = None
DEFAULT_ROLE_MAPPING = {
    "human": "user",
    "user": "user",
    "prompt": "user",
    "assistant": "assistant",
    "model": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
}


class MessageParseSpec(BaseModel):
    messages_path: str
    role_path: str
    content_path: str
    role_mapping: dict[str, str] = Field(default_factory=dict)


class SchemaDetection(BaseModel):
    is_target_dataset: bool
    task_type: Literal["code_generation", "math", "natural_language"]
    conversation_type: Literal["single_turn", "multi_turn"] | None = None
    user_path: str | None = None
    input_path: str | None = None
    assistant_path: str | None = None
    reasoning_path: str | None = None
    language: str = "unknown"
    language_path: str | None = None
    metadata_paths: list[str] = Field(default_factory=list)
    message_parse: MessageParseSpec | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "SchemaDetection":
        if not self.is_target_dataset:
            return self
        if self.conversation_type == "single_turn":
            if not self.user_path or not self.assistant_path:
                raise ValueError("single_turn schema requires user_path and assistant_path")
        if self.conversation_type == "multi_turn" and self.message_parse is None:
            raise ValueError("multi_turn schema requires message_parse")
        self.language = normalize_language(self.language)
        return self


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
    return find_message_path(value) is not None


def _message_spec_from_value(value: Any) -> MessageParseSpec | None:
    message_path = find_message_path(value)
    if message_path is None:
        return None
    candidate = value if not message_path else safe_get({"root": value}, f"root.{message_path}")
    if not isinstance(candidate, list) or not candidate:
        return None
    first = next((item for item in candidate if isinstance(item, dict)), None)
    if not isinstance(first, dict):
        return None
    role_key = next((key for key in ("role", "from", "speaker") if key in first), None)
    content_key = next((key for key in ("content", "value", "text") if key in first), None)
    if not role_key or not content_key:
        return None
    return MessageParseSpec(
        messages_path=message_path,
        role_path=role_key,
        content_path=content_key,
        role_mapping=DEFAULT_ROLE_MAPPING.copy(),
    )


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


def _content_score_as_math(text: str) -> float:
    if not text:
        return 0.0
    lower = text.lower()
    markers = [
        "solve",
        "equation",
        "proof",
        "derivative",
        "integral",
        "algebra",
        "geometry",
        "fraction",
        "calculate",
        "simplify",
        "math",
    ]
    score = sum(1 for marker in markers if marker in lower)
    score += sum(1 for ch in text if ch in "=+-*/^") * 0.03
    score += sum(1 for ch in text if ch.isdigit()) * 0.01
    return score


def _content_score_as_natural_language(text: str) -> float:
    if not text:
        return 0.0
    lower = text.lower()
    markers = [
        "summarize",
        "translate",
        "rewrite",
        "classify",
        "answer",
        "question",
        "respond",
        "write",
        "explain",
        "article",
        "story",
    ]
    score = sum(1 for marker in markers if marker in lower)
    score += 1.0 if lower.count(" ") > 7 else 0.0
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


def _task_score(text: str, task_type: str) -> float:
    if task_type == "code_generation":
        return _content_score_as_prompt(text) + (_content_score_as_code(text) * 0.25)
    if task_type == "math":
        return _content_score_as_math(text)
    return _content_score_as_natural_language(text)


def _content_based_schema(
    columns: list[str],
    sample_rows: list[dict[str, Any]],
    task_type: str,
) -> SchemaDetection:
    if not columns or not sample_rows:
        return SchemaDetection(is_target_dataset=False, task_type=task_type)

    language_path = None
    language_value = "unknown"
    for candidate in columns:
        if candidate.lower() in {"language", "lang", "locale"}:
            for row in sample_rows:
                if not isinstance(row, dict):
                    continue
                value = row.get(candidate)
                if value is not None:
                    language_path = candidate
                    language_value = normalize_language(value)
                    break
        if language_path:
            break

    # 1) Multi-turn detection from sample values, including nested message payloads.
    for col in columns:
        values = [row.get(col) for row in sample_rows if isinstance(row, dict)]
        if values and sum(1 for v in values if _looks_like_messages(v)) >= max(1, len(values) // 2):
            message_spec = None
            for value in values:
                message_spec = _message_spec_from_value(value)
                if message_spec is not None:
                    break
            if message_spec is None:
                continue
            return SchemaDetection(
                is_target_dataset=True,
                task_type=task_type,
                conversation_type="multi_turn",
                language=language_value,
                language_path=language_path,
                message_parse=MessageParseSpec(
                    messages_path=col if not message_spec.messages_path else f"{col}.{message_spec.messages_path}",
                    role_path=message_spec.role_path,
                    content_path=message_spec.content_path,
                    role_mapping=message_spec.role_mapping.copy(),
                ),
                metadata_paths=[],
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
        prompt_score = sum(_task_score(v, task_type) for v in col_values) / len(col_values)
        code_score = sum(_content_score_as_code(v) for v in col_values) / len(col_values)
        math_score = sum(_content_score_as_math(v) for v in col_values) / len(col_values)
        nl_score = sum(_content_score_as_natural_language(v) for v in col_values) / len(col_values)
        avg_len = sum(len(v) for v in col_values) / len(col_values)
        text_scores[col] = {
            "prompt": prompt_score,
            "code": code_score,
            "math": math_score,
            "natural_language": nl_score,
            "avg_len": avg_len,
        }

    if len(text_scores) < 2:
        return SchemaDetection(is_target_dataset=False, task_type=task_type)

    if task_type == "code_generation":
        assistant_field = max(text_scores, key=lambda c: text_scores[c]["code"])
        assistant_confidence = text_scores[assistant_field]["code"]
    elif task_type == "math":
        assistant_field = max(text_scores, key=lambda c: text_scores[c]["math"])
        assistant_confidence = text_scores[assistant_field]["math"]
    else:
        assistant_field = max(text_scores, key=lambda c: text_scores[c]["natural_language"])
        assistant_confidence = text_scores[assistant_field]["natural_language"]
    user_field = max(
        (c for c in text_scores if c != assistant_field),
        key=lambda c: text_scores[c]["prompt"],
        default=None,
    )
    if not user_field:
        return SchemaDetection(is_target_dataset=False, task_type=task_type)

    # Weak confidence guard.
    if assistant_confidence < 0.8 or text_scores[user_field]["prompt"] < 0.8:
        return SchemaDetection(is_target_dataset=False, task_type=task_type)

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
        is_target_dataset=True,
        task_type=task_type,
        conversation_type="single_turn",
        user_path=user_field,
        input_path=input_field,
        assistant_path=assistant_field,
        language=language_value,
        language_path=language_path,
        metadata_paths=[],
    )


class SchemaAgent:
    def __init__(self) -> None:
        global _LLM_RATE_LIMITER
        self.client: OpenAI | None = None
        if SETTINGS.llm_api_key:
            self.client = OpenAI(api_key=SETTINGS.llm_api_key, base_url=SETTINGS.llm_base_url)
        if _LLM_RATE_LIMITER is None:
            from modules.utils import RateLimiter

            _LLM_RATE_LIMITER = RateLimiter(
                max_concurrency=SETTINGS.llm_max_concurrency,
                min_interval_seconds=SETTINGS.llm_min_interval_seconds,
            )

    def infer_schema(
        self,
        dataset_id: str,
        task_type: str,
        columns: list[str],
        sample_rows: list[dict[str, Any]],
    ) -> SchemaDetection:
        if task_type not in TARGET_DOMAINS:
            raise ValueError(f"Unsupported task_type: {task_type}")
        if not self.client:
            logger.warning(f"[{dataset_id}] LLM_API_KEY not set. Falling back to heuristic schema detection.")
            return _content_based_schema(columns, sample_rows, task_type)

        prompt_payload = {
            "dataset_id": dataset_id,
            "task_type": task_type,
            "columns": columns,
            "sample_rows": sample_rows[: min(len(sample_rows), 8)],
            "task": (
                "Determine if this dataset is relevant to the requested task type and map fields to the unified conversation schema."
                " Multi-turn conversations may be nested inside objects such as conversations/messages/chats."
                " Return ONLY valid JSON."
            ),
            "expected_json_schema": {
                "is_target_dataset": "boolean",
                "task_type": "code_generation | math | natural_language",
                "conversation_type": "single_turn | multi_turn | null",
                "user_path": "string | null",
                "input_path": "string | null",
                "assistant_path": "string | null",
                "reasoning_path": "string | null",
                "language": "normalized lowercase language name such as english | hindi | spanish | multilingual | unknown",
                "language_path": "string | null",
                "metadata_paths": "string[]",
                "message_parse": {
                    "messages_path": "string",
                    "role_path": "string",
                    "content_path": "string",
                    "role_mapping": {"<raw-role>": "user | assistant"}
                },
            },
        }

        system_prompt = (
            "You are a strict dataset schema analyzer. Decide mapping by dataset content, not column names alone. "
            f"Only classify as relevant if the dataset primarily matches the target task type '{task_type}'. "
            "Arbitrary top-level keys and nested JSON keys are allowed. "
            "Return an exact parse specification using dot paths from the row root. "
            "For multi_turn datasets, message_parse.messages_path must point to the exact list of message objects, and role_path/content_path must be relative to each message object. "
            "Always return a normalized human-language label in lowercase English words, for example english, hindi, spanish, multilingual, or unknown. "
            "If language varies row-by-row, return language_path as the exact dot path and still normalize the language values to that same format at extraction time. "
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
            if not schema.is_target_dataset:
                return schema
            if schema.task_type != task_type:
                raise ValueError(f"LLM returned mismatched task_type '{schema.task_type}' for requested '{task_type}'")
            if schema.conversation_type == "single_turn":
                for row in sample_rows:
                    if not isinstance(row, dict):
                        continue
                    if safe_get(row, schema.user_path) is not None and safe_get(row, schema.assistant_path) is not None:
                        break
                else:
                    raise ValueError("single_turn paths not found in sample rows")
            if schema.conversation_type == "multi_turn" and schema.message_parse is not None:
                for row in sample_rows:
                    if not isinstance(row, dict):
                        continue
                    messages = safe_get(row, schema.message_parse.messages_path)
                    if isinstance(messages, str):
                        try:
                            messages = json.loads(messages)
                        except json.JSONDecodeError:
                            pass
                    if isinstance(messages, list) and messages:
                        first = next((item for item in messages if isinstance(item, dict)), None)
                        if first and safe_get(first, schema.message_parse.role_path) is not None and safe_get(first, schema.message_parse.content_path) is not None:
                            break
                else:
                    raise ValueError(f"multi_turn parse spec not found in sample rows: {schema.message_parse.messages_path}")
            if schema.language_path:
                for row in sample_rows:
                    if not isinstance(row, dict):
                        continue
                    if safe_get(row, schema.language_path) is not None:
                        break
                else:
                    raise ValueError(f"language_path not found in sample rows: {schema.language_path}")
            return schema
        except (json.JSONDecodeError, ValidationError, Exception) as exc:  # noqa: BLE001
            logger.warning(f"[{dataset_id}] Schema inference failed: {exc}. Falling back to content-based detection.")
            return _content_based_schema(columns, sample_rows, task_type)

    def _invoke_llm(self, system_prompt: str, payload: dict[str, Any]) -> str:
        assert self.client is not None
        user_content = json.dumps(payload, ensure_ascii=False)
        assert _LLM_RATE_LIMITER is not None

        with _LLM_RATE_LIMITER.acquire():
            # Newer OpenAI SDK interface.
            if hasattr(self.client, "responses"):
                response = self.client.responses.create(
                    model=SETTINGS.model_name,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                    max_output_tokens=500,
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
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            return content or "{}"
