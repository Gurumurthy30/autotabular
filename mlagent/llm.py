"""Single-model wrapper (default gemma4:31b via Ollama): one call at a time, 429 backoff, JSON/code extraction.

Env: MLAGENT_MODEL (verify the exact tag!), OLLAMA_HOST, OLLAMA_API_KEY.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time

from pydantic import BaseModel

from . import ui


class RateLimitError(RuntimeError):
    """Raised after backoff is exhausted; the graph turns it into stop_reason='rate_limit'."""


class LLMFormatError(RuntimeError):
    """The model could not produce valid JSON for the schema after retries."""


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by some models (e.g. Gemma)."""
    import re as _re
    return _re.sub(r"<think>.*?</think>", "", text, flags=_re.S).strip()


def extract_json(text: str) -> dict:
    text = _strip_think(text)
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object found in reply")
    return json.loads(text[i:j + 1])


def extract_code(text: str) -> str:
    text = _strip_think(text)
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return re.sub(r"^```\w*\n", "", text.strip()).rstrip("`").strip() + "\n"   # unterminated fence


_defaults: dict = {"model": None, "host": None, "temperature": 0.2}
_injected = False


class LLM:
    def __init__(self, model: str | None = None, host: str | None = None, temperature: float | None = None):
        self.model = model or _defaults["model"] or os.getenv("MLAGENT_MODEL") or os.getenv("OLLAMA_MODEL") or "gemma4:31b"
        self.host = host or _defaults["host"] or os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL")
        self.temperature = temperature if temperature is not None else _defaults["temperature"]
        self._lock = threading.Lock()          # free tier: 1 concurrent call
        self._client = None

    def _c(self):
        if self._client is None:
            from ollama import Client
            kw: dict = {}
            if self.host:
                kw["host"] = self.host
            if os.getenv("OLLAMA_API_KEY"):
                kw["headers"] = {"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]}
            self._client = Client(**kw)
        return self._client

    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float | None = None) -> str:
        temperature = self.temperature if temperature is None else temperature
        delays, conn_tries = [5, 15, 45, 90, 180], 0
        for attempt in range(len(delays) + 1):
            try:
                kw = {"format": "json"} if json_mode else {}
                with self._lock:
                    r = self._c().chat(model=self.model, options={"temperature": temperature}, **kw,
                                       messages=[{"role": "system", "content": system},
                                                 {"role": "user", "content": user}])
                return r["message"]["content"]
            except Exception as e:                                   # noqa: BLE001
                if getattr(e, "status_code", None) == 429 or "429" in str(e)[:120]:
                    if attempt == len(delays):
                        raise RateLimitError(str(e)) from e
                    ui.warn(f"rate limited, waiting {delays[attempt]}s")
                    time.sleep(delays[attempt])
                    continue
                if isinstance(e, (ConnectionError, OSError)) and conn_tries < 3:
                    conn_tries += 1
                    time.sleep(5 * conn_tries)
                    continue
                raise
        raise RateLimitError("retries exhausted")

    def json(self, system: str, user: str, model: type[BaseModel], retries: int = 2) -> BaseModel:
        schema = json.dumps(model.model_json_schema().get("properties", {}))
        sys_p = f"{system}\n\nReply with ONLY one JSON object, no prose. Fields: {schema}"
        err = None
        for _ in range(retries + 1):
            prompt = user if err is None else f"{user}\n\nYour previous reply was invalid ({err}). Return ONLY valid JSON."
            text = self.chat(sys_p, prompt, json_mode=True)
            try:
                return model.model_validate(extract_json(text))
            except ValueError as e:                                  # JSONDecodeError + ValidationError
                err = str(e)[:300]
        raise LLMFormatError(err)

    def code(self, system: str, user: str, retries: int = 1) -> str:
        err = None
        for _ in range(retries + 1):
            prompt = user if err is None else f"{user}\n\nYour previous script had a syntax error: {err}\nReturn the full corrected script."
            code = extract_code(self.chat(system, prompt))
            try:
                compile(code, "<generated>", "exec")
                return code
            except SyntaxError as e:
                err = f"{e.msg} (line {e.lineno})"
        return code


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


def set_llm(obj) -> None:
    """Swap the model (tests use a fake); configure() will not replace an injected model."""
    global _llm, _injected
    _llm, _injected = obj, obj is not None


def configure(model: str | None = None, host: str | None = None, temperature: float | None = None) -> None:
    """Apply the resolved config (env/CLI already merged into it by config.resolve)."""
    global _llm
    _defaults.update(model=model, host=host)
    if temperature is not None:
        _defaults["temperature"] = temperature
    if not _injected:
        _llm = None
