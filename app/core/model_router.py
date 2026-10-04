"""Router abstraction providing LLM instances for agent roles with execution logging."""

import os
import threading
import time
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from app.config import LLM_TIMEOUT_S, OLLAMA_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX
from app.utils.logger import get_logger

_log = get_logger(__name__)

# Global lock to ensure only ONE LLM request is in flight at a time per process
_LLM_LOCK = threading.Lock()


class LLMRateLimitError(Exception):
    """Raised when an LLM provider returns HTTP 429 after retries are exhausted."""
    pass


class LLMCircuitBreaker:
    """Tracks consecutive LLMRateLimitError instances. Tripped after threshold consecutive errors."""

    def __init__(self, threshold: int = 2):
        self.threshold = threshold
        self.consecutive_rate_limits = 0
        self.last_detail = ""

    @property
    def tripped(self) -> bool:
        return self.consecutive_rate_limits >= self.threshold

    def record_success(self) -> None:
        self.consecutive_rate_limits = 0

    def record_rate_limit(self, detail: str) -> None:
        self.consecutive_rate_limits += 1
        self.last_detail = str(detail)

    def reset(self) -> None:
        self.consecutive_rate_limits = 0
        self.last_detail = ""


circuit_breaker = LLMCircuitBreaker()


def is_rate_limit_error(exc: Exception) -> bool:
    """Detects HTTP 429 status code or rate limit messages."""
    if hasattr(exc, "status_code") and getattr(exc, "status_code") == 429:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def extract_retry_after(exc: Exception, default: float = 20.0) -> float:
    """Extracts Retry-After header from an HTTP exception if present, else returns default."""
    resp = getattr(exc, "response", None)
    if resp is not None and hasattr(resp, "headers"):
        retry_val = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
        if retry_val is not None:
            try:
                return max(0.0, float(retry_val))
            except (ValueError, TypeError):
                pass
    return default


class GuardedChatModel:
    """Wraps a ChatModel to enforce single-flight execution via a global lock,
    automatic HTTP 429 retries (up to 2 retries), and circuit-breaker recording.
    """

    def __init__(self, model: Any, sleep_fn: Callable[[float], None] = time.sleep):
        self.model = model
        self.sleep_fn = sleep_fn

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        with _LLM_LOCK:
            max_retries = 2
            attempt = 0
            while True:
                try:
                    res = self.model.invoke(*args, **kwargs)
                    circuit_breaker.record_success()
                    return res
                except Exception as exc:
                    if not is_rate_limit_error(exc):
                        raise
                    if attempt >= max_retries:
                        circuit_breaker.record_rate_limit(str(exc))
                        raise LLMRateLimitError(
                            f"HTTP 429 rate limit exceeded after {max_retries} retries: {exc}"
                        ) from exc

                    wait_s = extract_retry_after(exc, default=20.0)
                    _log.warning(
                        "[LLM GUARD] HTTP 429 encountered (attempt %d/%d). Waiting %.1fs before retry...",
                        attempt + 1,
                        max_retries,
                        wait_s,
                    )
                    self.sleep_fn(wait_s)
                    attempt += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


class LLMLoggingCallback(BaseCallbackHandler):
    """Logs per-LLM call metrics: agent, prompt chars, estimated tokens, output chars, seconds."""

    def __init__(self, role: str, num_ctx: int = 16384):
        self.role = role
        self.num_ctx = num_ctx
        self.t0 = time.monotonic()
        self.prompt_chars = 0

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self.t0 = time.monotonic()
        self.prompt_chars = sum(len(p) for p in prompts)

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        self.t0 = time.monotonic()
        self.prompt_chars = sum(
            len(str(getattr(m, "content", ""))) for msg_list in messages for m in msg_list
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        dur = time.monotonic() - self.t0
        output_chars = 0
        if response and hasattr(response, "generations"):
            for gen_list in response.generations:
                for gen in gen_list:
                    text_val = getattr(gen, "text", "") or ""
                    if not text_val and hasattr(gen, "message"):
                        text_val = str(getattr(gen.message, "content", ""))
                    output_chars += len(text_val)

        est_tokens = int(self.prompt_chars / 3.5)
        _log.info(
            "[LLM] agent=%s prompt_chars=%d est_tokens=%d output_chars=%d seconds=%.2fs",
            self.role, self.prompt_chars, est_tokens, output_chars, dur,
        )
        if est_tokens > 0.8 * self.num_ctx:
            _log.warning(
                "[LLM] WARNING: agent=%s estimated prompt tokens (%d) exceeds 80%% of num_ctx (%d)",
                self.role, est_tokens, self.num_ctx,
            )


class ModelRouter:
    """Router abstraction providing LLM instances for agent roles."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = OLLAMA_MODEL,
        api_key: str = OLLAMA_API_KEY,
        num_ctx: int = OLLAMA_NUM_CTX,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.num_ctx = num_ctx
        self._models: dict[str, Any] = {}

    def get_model(self, role: str, temperature: float = 0.0) -> Any:
        """Returns a configured guarded chat model for the requested agent role.

        All roles default to the configured Ollama model.
        num_ctx is passed so the model has enough context for large briefs.
        Caching the instance per role ensures clean separation and flexibility.
        """
        cache_key = f"{role}_{temperature}"
        if cache_key in self._models:
            return self._models[cache_key]

        client_kwargs: dict[str, Any] = {}
        timeout_s = float(os.getenv("LLM_TIMEOUT_S", str(LLM_TIMEOUT_S)))
        client_kwargs["timeout"] = timeout_s
        if self.api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}

        raw_model = ChatOllama(
            base_url=self.base_url,
            model=self.model_name,
            temperature=temperature,
            num_ctx=self.num_ctx,
            client_kwargs=client_kwargs if client_kwargs else None,
            callbacks=[LLMLoggingCallback(role, num_ctx=self.num_ctx)],
        )
        guarded_model = GuardedChatModel(raw_model)
        self._models[cache_key] = guarded_model
        return guarded_model
