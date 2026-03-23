from __future__ import annotations

import json
import re
from typing import Any, Literal

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from config import SETTINGS
from modules.utils import RateLimiter, normalize_language, retry


TARGET_DOMAINS = {"code_generation", "math", "natural_language"}
_LLM_RATE_LIMITER = None


class SchemaDetection(BaseModel):
    is_target_dataset: bool
    task_type: Literal["code_generation", "math", "natural_language"]
    conversation_type: Literal["single_turn", "multi_turn"] | None = None
    user_field: str | None = None
    input_field: str | None = None
    assistant_field: str | None = None
    messages_field: str | None = None
    reasoning_field: str | None = None
    language: str = "unknown"
    language_field: str | None = None
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
    return isinstance(first, dict) and any(k in first for k in ("role", "content", "from", "value", "speaker", "text"))


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
    score += 1.0 if lower.count(" ") > 5 else 0.0
    score -= 0.5 if _content_score_as_code(text) > 3 else 0.0
    return score


def _content_score_as_math(text: str) -> float:
    if not text:
        return 0.0
    lower = text.lower()
    markers = ["solve", "equation", "proof", "derivative", "integral", "algebra", "geometry", "calculate", "simplify", "math"]
    score = sum(1 for marker in markers if marker in lower)
    score += sum(1 for ch in text if ch in "=+-*/^") * 0.03
    return score


def _content_score_as_natural_language(text: str) -> float:
    if not text:
        return 0.0
    lower = text.lower()
    markers = ["summarize", "translate", "rewrite", "classify", "answer", "question", "respond", "write", "explain"]
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


def _task_prompt_score(text: str, task_type: str) -> float:
    if task_type == "code_generation":
        return _content_score_as_prompt(text)
    if task_type == "math":
        return _content_score_as_math(text)
    return _content_score_as_natural_language(text)


def _task_answer_score(text: str, task_type: str) -> float:
    if task_type == "code_generation":
        return _content_score_as_code(text)
    if task_type == "math":
        return _content_score_as_math(text)
    return _content_score_as_natural_language(text)


def _detect_language(columns: list[str], sample_rows: list[dict[str, Any]]) -> tuple[str, str | None]:
    for col in columns:
        if col.lower() in {"language", "lang", "locale"}:
            for row in sample_rows:
                if not isinstance(row, dict):
                    continue
                value = row.get(col)
                if value is not None:
                    return normalize_language(value), col
    return "unknown", None


def _content_based_schema(columns: list[str], sample_rows: list[dict[str, Any]], task_type: str) -> SchemaDetection:
    if not columns or not sample_rows:
        return SchemaDetection(is_target_dataset=False, task_type=task_type)

    language, language_field = _detect_language(columns, sample_rows)

    for col in columns:
        values = [row.get(col) for row in sample_rows if isinstance(row, dict)]
        if values and sum(1 for v in values if _looks_like_messages(v)) >= max(1, len(values) // 2):
            return SchemaDetection(
                is_target_dataset=True,
                task_type=task_type,
                conversation_type="multi_turn",
                messages_field=col,
                language=language,
                language_field=language_field,
                metadata_fields=[],
            )

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
        prompt_score = sum(_task_prompt_score(v, task_type) for v in col_values) / len(col_values)
        answer_score = sum(_task_answer_score(v, task_type) for v in col_values) / len(col_values)
        avg_len = sum(len(v) for v in col_values) / len(col_values)
        text_scores[col] = {"prompt": prompt_score, "answer": answer_score, "avg_len": avg_len}

    if len(text_scores) < 2:
        return SchemaDetection(is_target_dataset=False, task_type=task_type)

    assistant_field = max(text_scores, key=lambda c: text_scores[c]["answer"])
    user_field = max((c for c in text_scores if c != assistant_field), key=lambda c: text_scores[c]["prompt"], default=None)
    if not user_field:
        return SchemaDetection(is_target_dataset=False, task_type=task_type)

    if text_scores[assistant_field]["answer"] < 0.8 or text_scores[user_field]["prompt"] < 0.8:
        return SchemaDetection(is_target_dataset=False, task_type=task_type)

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
        user_field=user_field,
        input_field=input_field,
        assistant_field=assistant_field,
        language=language,
        language_field=language_field,
        metadata_fields=[],
    )


class SchemaAgent:
    def __init__(self) -> None:
        global _LLM_RATE_LIMITER
        self.client: OpenAI | None = None
        if SETTINGS.llm_api_key:
            self.client = OpenAI(api_key=SETTINGS.llm_api_key, base_url=SETTINGS.llm_base_url)
        if _LLM_RATE_LIMITER is None:
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
                "Look at the sample rows and determine whether this dataset matches the requested task type. "
                "Then decide whether conversion is a simple top-level column mapping or a single top-level column containing a list of chat messages. "
                "Return ONLY valid JSON."
            ),
            "expected_json_schema": {
                "is_target_dataset": "boolean",
                "task_type": "code_generation | math | natural_language",
                "conversation_type": "single_turn | multi_turn | null",
                "user_field": "string | null",
                "input_field": "string | null",
                "assistant_field": "string | null",
                "messages_field": "string | null",
                "reasoning_field": "string | null",
                "language": "string",
                "language_field": "string | null",
                "metadata_fields": "string[]",
            },
        }

        system_prompt = (
            "You are a strict dataset schema analyzer. Keep the mapping simple and only use top-level fields. "
            f"Only classify as relevant if the dataset primarily matches the target task type '{task_type}'. "
            "If one top-level field contains chat messages, use conversation_type='multi_turn' and set messages_field to that field. "
            "Otherwise map top-level prompt/response fields for conversation_type='single_turn'. "
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
            schema.language = normalize_language(schema.language)
            if not schema.is_target_dataset:
                return schema
            if schema.task_type != task_type:
                raise ValueError(f"LLM returned mismatched task_type '{schema.task_type}' for requested '{task_type}'")
            if schema.conversation_type == "single_turn" and not (schema.user_field and schema.assistant_field):
                logger.warning(f"[{dataset_id}] Invalid single_turn mapping from LLM. Falling back to content-based detection.")
                return _content_based_schema(columns, sample_rows, task_type)
            if schema.conversation_type == "multi_turn" and not schema.messages_field:
                logger.warning(f"[{dataset_id}] Invalid multi_turn mapping from LLM. Falling back to content-based detection.")
                return _content_based_schema(columns, sample_rows, task_type)
            if schema.user_field and schema.user_field not in columns:
                raise ValueError(f"user_field not in dataset columns: {schema.user_field}")
            if schema.assistant_field and schema.assistant_field not in columns:
                raise ValueError(f"assistant_field not in dataset columns: {schema.assistant_field}")
            if schema.messages_field and schema.messages_field not in columns:
                raise ValueError(f"messages_field not in dataset columns: {schema.messages_field}")
            if schema.language_field and schema.language_field not in columns:
                raise ValueError(f"language_field not in dataset columns: {schema.language_field}")
            return schema
        except (json.JSONDecodeError, ValidationError, Exception) as exc:  # noqa: BLE001
            logger.warning(f"[{dataset_id}] Schema inference failed: {exc}. Falling back to content-based detection.")
            return _content_based_schema(columns, sample_rows, task_type)

    def _invoke_llm(self, system_prompt: str, payload: dict[str, Any]) -> str:
        assert self.client is not None
        user_content = json.dumps(payload, ensure_ascii=False)
        assert _LLM_RATE_LIMITER is not None

        with _LLM_RATE_LIMITER.acquire():
            use_chat_completions = SETTINGS.llm_use_chat_completions or bool(
                SETTINGS.llm_base_url and "sglang" in SETTINGS.llm_base_url.lower()
            )

            if hasattr(self.client, "responses") and not use_chat_completions:
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

            completion = self.client.chat.completions.create(
                model=SETTINGS.model_name,
                messages=[
                    {"role": "system", "content": SETTINGS.llm_reasoning_hint},
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=400,
                extra_body={"chat_template_kwargs": {"enable_thinking": SETTINGS.llm_enable_thinking}},
            )
            content = completion.choices[0].message.content
            return content or "{}"
