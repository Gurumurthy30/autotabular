"""Shared Executor: a function, not an agent. Runs a script in a subprocess inside the workspace."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .ledger import Workspace

RESULT_PREFIX = "RESULT_JSON:"


@dataclass
class ExecResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False


def _s(x) -> str:
    if x is None:
        return ""
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else x


def tail(text: str, n: int = 1800) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else "..." + text[-n:]


def write_script(ws: Workspace, name: str, code: str) -> Path:
    ws.scripts.mkdir(parents=True, exist_ok=True)
    p = ws.scripts / f"{name}.py"
    p.write_text(code, encoding="utf-8")
    return p


def run_script(ws: Workspace, script: Path, timeout: int = 900, env_extra: dict | None = None) -> ExecResult:
    env = {**os.environ,
           "PYTHONPATH": str(ws.root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "PYTHONHASHSEED": "0", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore", "MPLBACKEND": "Agg",
           **{k: str(v) for k, v in (env_extra or {}).items()}}
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, str(script)], cwd=ws.root, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        res = ExecResult(p.returncode == 0, p.returncode, p.stdout, p.stderr, time.time() - t0)
    except subprocess.TimeoutExpired as e:
        res = ExecResult(False, -1, _s(e.stdout), _s(e.stderr) + f"\nTimeoutExpired: script exceeded {timeout}s",
                         time.time() - t0, timed_out=True)
    ws.logs.mkdir(parents=True, exist_ok=True)
    (ws.logs / f"{Path(script).stem}.log").write_text(f"# rc={res.returncode} {res.seconds:.1f}s\n"
                                                      f"## stdout\n{res.stdout}\n## stderr\n{res.stderr}\n", encoding="utf-8")
    return res


def parse_result(stdout: str) -> dict | None:
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                out = json.loads(line[len(RESULT_PREFIX):])
                return out if isinstance(out, dict) else None
            except json.JSONDecodeError:
                return None
    return None
