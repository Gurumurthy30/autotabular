"""Router abstraction providing LLM instances for agent roles with execution logging."""

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from app.config import OLLAMA_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX
from app.utils.logger import get_logger

_log = get_logger(__name__)


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
        self._models: dict[str, BaseChatModel] = {}

    def get_model(self, role: str, temperature: float = 0.0) -> BaseChatModel:
        """Returns a configured chat model for the requested agent role.

        All roles default to the configured Ollama model (gpt-oss:120b).
        num_ctx is passed so the model has enough context for large briefs.
        Caching the instance per role ensures clean separation and flexibility.
        """
        cache_key = f"{role}_{temperature}"
        if cache_key in self._models:
            return self._models[cache_key]

        client_kwargs = {}
        if self.api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}

        model = ChatOllama(
            base_url=self.base_url,
            model=self.model_name,
            temperature=temperature,
            num_ctx=self.num_ctx,
            client_kwargs=client_kwargs if client_kwargs else None,
            callbacks=[LLMLoggingCallback(role, num_ctx=self.num_ctx)],
        )
        self._models[cache_key] = model
        return model
