"""Structured JSON extraction and validation without tool calling.

Avoids with_structured_output / tool binding hallucinations (e.g. hallucinated open_file calls)
by using plain chat calls, extracting balanced JSON, validating against Pydantic models,
and performing targeted single repairs on validation failure.
"""

import json
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from app.utils.logger import get_logger

_log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_balanced_json(text: str) -> str | None:
    """Finds and extracts the first balanced JSON object {...} from text, ignoring markdown fences."""
    if not text:
        return None

    # Strip triple-backtick markdown blocks if they wrap everything
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    cleaned = cleaned.removesuffix("```")
    cleaned = cleaned.strip()

    start_idx = cleaned.find("{")
    if start_idx == -1:
        return None

    depth = 0
    in_str = False
    escape = False

    for i in range(start_idx, len(cleaned)):
        ch = cleaned[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            if in_str:
                escape = True
            continue

        if ch == '"':
            in_str = not in_str
            continue

        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start_idx : i + 1]

    return None


def invoke_json(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    model_cls: type[T],
    agent_name: str = "unknown",
) -> tuple[T | None, str, str]:
    """Invokes LLM with plain chat call, extracts balanced JSON, and validates against model_cls.

    On validation error, sends the error text back once for repair.
    If repair fails or produces the identical error, returns (None, error_text, raw_text).

    Returns:
        (parsed_instance_or_None, error_text, raw_text)
    """
    raw_text = ""
    try:
        response = llm.invoke(messages)
        raw_text = str(getattr(response, "content", ""))
    except Exception as e:
        from app.core.model_router import LLMRateLimitError
        if isinstance(e, LLMRateLimitError):
            raise
        err_msg = f"LLM invocation error: {e!s}"
        _log.error("[%s] %s", agent_name, err_msg)
        return None, err_msg, raw_text

    # First attempt: parse and validate
    json_str = extract_balanced_json(raw_text)
    if not json_str:
        first_err = "No balanced JSON object found in response"
    else:
        try:
            parsed_dict = json.loads(json_str)
            instance = model_cls.model_validate(parsed_dict)
            return instance, "", raw_text
        except json.JSONDecodeError as jde:
            first_err = f"JSON decode error: {jde!s}"
        except ValidationError as ve:
            first_err = f"Schema validation error: {ve!s}"
        except Exception as ex:
            first_err = f"Parse error: {ex!s}"

    _log.warning(
        "[%s] JSON validation failed on first attempt: %s. Requesting single repair.",
        agent_name,
        first_err,
    )

    # Attempt repair
    repair_prompt = (
        f"Your previous response had the following JSON validation error:\n{first_err}\n"
        f"Please fix the error and output ONLY the corrected, valid JSON object conforming to the required schema."
    )
    repair_messages = list(messages) + [
        AIMessage(content=raw_text),
        HumanMessage(content=repair_prompt),
    ]

    repair_raw = ""
    try:
        repair_resp = llm.invoke(repair_messages)
        repair_raw = str(getattr(repair_resp, "content", ""))
    except Exception as e:
        from app.core.model_router import LLMRateLimitError
        if isinstance(e, LLMRateLimitError):
            raise
        repair_err = f"Repair LLM invocation error: {e!s}"
        _log.error("[%s] %s", agent_name, repair_err)
        return None, repair_err, repair_raw

    repaired_json_str = extract_balanced_json(repair_raw)
    if not repaired_json_str:
        final_err = "Repair failed: No balanced JSON object found in repair response"
        _log.error("[%s] %s", agent_name, final_err)
        return None, final_err, repair_raw

    try:
        repaired_dict = json.loads(repaired_json_str)
        instance = model_cls.model_validate(repaired_dict)
        _log.info("[%s] Repair succeeded.", agent_name)
        return instance, "", repair_raw
    except (json.JSONDecodeError, ValidationError, Exception) as ex:
        final_err = f"Repair failed validation: {ex!s}"
        _log.error("[%s] %s", agent_name, final_err)
        return None, final_err, repair_raw
