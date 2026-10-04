import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECTS_ROOT = os.getenv("PROJECTS_ROOT", str(BASE_DIR / "projects"))
PROJECTS_DIR = Path(PROJECTS_ROOT).resolve()
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# App Database URL (SQLite)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'app_metadata.db'}")

# Ollama Cloud configuration
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")

# Context window size for the LLM (tokens) — passed to ChatOllama as num_ctx
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))

# Client timeout for LLM requests (seconds)
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "180"))

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# LangSmith / Tracing configuration (optional)
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ML_agent")


def validate_config() -> None:
    """Fails fast with clear descriptive messages if required configuration is missing or invalid."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable must be set (e.g. sqlite:///app_metadata.db)")

    from app.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("[CONFIG] OLLAMA_MODEL=%s OLLAMA_NUM_CTX=%d", OLLAMA_MODEL, OLLAMA_NUM_CTX)

    # If pointing to Ollama Cloud API (https://ollama.com), ensure API key is present
    if "ollama.com" in OLLAMA_BASE_URL.lower() and not OLLAMA_API_KEY.strip():
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Ollama Cloud (https://ollama.com) requires an API key in .env.\n"
            "Please add OLLAMA_API_KEY=your_key to your .env file."
        )

    # Ensure projects root directory exists and is writable
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
