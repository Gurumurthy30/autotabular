"""mlagent doctor: diagnoses the environment, dependencies, GPU, Ollama, Kaggle, etc."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from rich.console import Console
from rich.table import Table

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from . import hw, llm, ui

# Reconfigure stdout/stderr on Windows to avoid charmap / cp1252 crashes
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

def _can_encode(s: str) -> bool:
    try:
        enc = sys.stdout.encoding or "utf-8"
        s.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False

USE_UNICODE = _can_encode("● ✓ ✗")
SYM_OK = "[bold green]✓[/]" if USE_UNICODE else "[bold green]OK[/]"
SYM_FAIL = "[bold red]✗[/]" if USE_UNICODE else "[bold red]FAIL[/]"
SYM_INFO = "[bold cyan]INFO[/]"

REQUIRED_PACKAGES = [
    "langgraph",
    "pydantic",
    "pandas",
    "numpy",
    "scikit-learn",
    "rich",
    "ollama",
    "pyyaml",
]

OPTIONAL_PACKAGES = [
    "lightgbm",
    "xgboost",
    "catboost",
    "optuna",
]


def check_python() -> tuple[bool, str]:
    ver = sys.version_info
    ok = (ver.major, ver.minor) >= (3, 10)
    return ok, f"Python {ver.major}.{ver.minor}.{ver.micro} ({sys.executable})"


def check_virtualenv() -> tuple[bool, str]:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return in_venv, f"Virtualenv: {'active (' + sys.prefix + ')' if in_venv else 'NOT in a virtual environment'}"


def check_mlagent_pkg() -> tuple[bool, str]:
    try:
        ver = metadata.version("mlagent")
        return True, f"mlagent v{ver} (installed)"
    except Exception as e:
        return False, f"mlagent not installed in current environment ({e})"


def check_package(pkg: str) -> tuple[bool, str]:
    dist_name = "scikit-learn" if pkg == "sklearn" else pkg
    import_name = "yaml" if pkg == "pyyaml" else ("sklearn" if pkg == "scikit-learn" else pkg)
    try:
        mod = importlib.import_module(import_name)
        try:
            ver = metadata.version(dist_name)
        except Exception:
            ver = getattr(mod, "__version__", "unknown")
        return True, f"{pkg} ({ver})"
    except Exception as e:
        return False, f"{pkg} missing: {e}"


def check_ollama(model_tag: str | None = None, host: str | None = None) -> list[tuple[str, bool, str]]:
    results = []
    # 1. CLI presence
    ollama_bin = shutil.which("ollama")
    if ollama_bin:
        results.append(("Ollama CLI", True, f"Found at {ollama_bin}"))
    else:
        results.append(("Ollama CLI", False, "ollama binary not found in PATH"))

    # 2. Ollama service / models
    host = host or os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL")
    try:
        from ollama import Client
        kw: dict = {}
        if host:
            kw["host"] = host
        if os.getenv("OLLAMA_API_KEY"):
            kw["headers"] = {"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]}
        client = Client(**kw)
        models_resp = client.list()
        model_names = []
        if hasattr(models_resp, "models"):
            model_names = [getattr(m, "model", "") or getattr(m, "name", "") for m in models_resp.models]
        elif isinstance(models_resp, dict):
            model_names = [m.get("model") or m.get("name") for m in models_resp.get("models", [])]
        loc = host or "local daemon"
        results.append(("Ollama Service", True, f"Responding at {loc}"))
        if model_tag:
            found = any(model_tag in str(name) for name in model_names)
            results.append((f"Model '{model_tag}'", found,
                            "Found in models list" if found else f"Not found in models list ({', '.join(filter(None, model_names)) or 'none'})"))
    except Exception as e:
        # Fallback to local CLI
        try:
            p = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if p.returncode == 0:
                results.append(("Ollama Service", True, "Local daemon responding"))
                if model_tag:
                    found = model_tag in p.stdout
                    results.append((f"Model '{model_tag}'", found,
                                    "Found in ollama list" if found else f"Not found in 'ollama list' (run 'ollama pull {model_tag}')"))
            else:
                results.append(("Ollama Service", False, f"ollama list failed: {p.stderr.strip() or p.stdout.strip()}"))
        except Exception:
            results.append(("Ollama Service", False, f"Could not contact Ollama: {e}"))
    return results


def check_llm_json_roundtrip(model_tag: str) -> tuple[bool, str]:
    from pydantic import BaseModel
    class TestSchema(BaseModel):
        status: str
        number: int
    try:
        client = llm.LLM(model=model_tag)
        res = client.json("You are a test helper. Return JSON: status='ok', number=42", "ping", TestSchema)
        if res.status == "ok" and res.number == 42:
            return True, f"Model '{model_tag}' responded with valid JSON"
        return False, f"Unexpected response from '{model_tag}': {res}"
    except Exception as e:
        return False, f"JSON round-trip failed with '{model_tag}': {e}"


def check_gpu() -> tuple[bool, str]:
    gpus = hw.gpus()
    if gpus:
        desc = ", ".join(f"{g.get('name', 'GPU')} ({g.get('free_mb', 0)}MB free)" for g in gpus)
        return True, f"Found {len(gpus)} GPU(s): {desc}"
    return True, "No GPU detected (CPU mode)"


def check_kaggle_cli() -> tuple[bool, str]:
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        try:
            p = subprocess.run(["kaggle", "--version"], capture_output=True, text=True, timeout=5)
            ver = p.stdout.strip() or "available"
            return True, f"Kaggle CLI found ({ver})"
        except Exception:
            return True, "Kaggle CLI found"
    return False, "kaggle CLI not found (optional for submission)"


def check_console_encoding() -> tuple[bool, str]:
    try:
        test_str = "● ✓ ✗"
        # Test encode
        enc = sys.stdout.encoding or "utf-8"
        test_str.encode(enc)
        return True, f"Encoding {enc} supports UTF-8 symbols"
    except Exception as e:
        return False, f"Console encoding cannot render UTF-8 glyphs: {e}"


def check_write_permission(folder: Path) -> tuple[bool, str]:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        test_file = folder / ".mlagent_doctor_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        return True, f"Write permission verified on {folder}"
    except Exception as e:
        return False, f"Cannot write to {folder}: {e}"


def run_doctor(target_folder: str | None = None, model: str | None = None, check_json: bool = False) -> bool:
    console = Console(safe_box=True)
    table = Table(title="mlagent Doctor Diagnostics", show_header=True, header_style="bold magenta", safe_box=True)
    table.add_column("Category", style="dim", width=16)
    table.add_column("Check", width=24)
    table.add_column("Status", width=8)
    table.add_column("Details")

    all_ok = True

    def add_row(cat: str, name: str, ok: bool, details: str, info_only: bool = False):
        nonlocal all_ok
        if not ok and not info_only:
            all_ok = False
        sym = SYM_INFO if info_only else (SYM_OK if ok else SYM_FAIL)
        table.add_row(cat, name, sym, details)

    # Environment
    ok, msg = check_python()
    add_row("Environment", "Python >= 3.10", ok, msg)
    ok, msg = check_virtualenv()
    add_row("Environment", "Virtualenv", ok, msg)
    ok, msg = check_console_encoding()
    add_row("Environment", "Console Encoding", ok, msg)
    ok, msg = check_mlagent_pkg()
    add_row("Environment", "mlagent Package", ok, msg)

    # Core Dependencies
    for pkg in REQUIRED_PACKAGES:
        ok, msg = check_package(pkg)
        add_row("Core Libs", pkg, ok, msg)

    # Optional Dependencies
    for pkg in OPTIONAL_PACKAGES:
        ok, msg = check_package(pkg)
        add_row("Boost Libs", pkg, ok, msg, info_only=True)

    # Hardware & External
    ok, msg = check_gpu()
    add_row("Hardware", "GPU", ok, msg, info_only=True)
    ok, msg = check_kaggle_cli()
    add_row("External", "Kaggle CLI", ok, msg, info_only=True)

    # Ollama
    m_tag = model or os.environ.get("MLAGENT_MODEL", "gemma4:31b")
    for name, ok, msg in check_ollama(m_tag):
        add_row("Ollama", name, ok, msg, info_only=(name == "Ollama CLI" and not ok))

    if check_json:
        ok, msg = check_llm_json_roundtrip(m_tag)
        add_row("Ollama", "JSON Round-trip", ok, msg)

    # Data folder permissions
    if target_folder:
        p = Path(target_folder)
        ok, msg = check_write_permission(p)
        add_row("Filesystem", "Write Permission", ok, msg)

    console.print()
    console.print(table)
    console.print()
    if all_ok:
        console.print("[bold green]All required checks passed! mlagent is ready.[/]")
    else:
        console.print("[bold red]Some required checks failed. Please address the issues above.[/]")
    return all_ok
