import os
import shutil
import subprocess
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

from app.config import PROJECTS_DIR


class ExecutionResult:
    def __init__(self, exit_code: int, stdout: str, stderr: str, workspace_dir: str):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.workspace_dir = workspace_dir
        self.success = (exit_code == 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "workspace_dir": self.workspace_dir,
        }


def normalize_script_unicode(code: str) -> str:
    """Replaces non-standard unicode dashes, spaces, and quotes with standard ASCII equivalents."""
    replacements = {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201a": "'",  # single low-9 quote
        "\u201b": "'",  # single high-reversed-9 quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u201e": '"',  # double low-9 quote
        "\u00a0": " ",  # non-breaking space
        "\u202f": " ",  # narrow no-break space
        "\ufeff": "",   # zero width no-break space (BOM)
    }
    for char, rep in replacements.items():
        code = code.replace(char, rep)
    return code


class ExecutionManager:
    """Manages Python script execution in isolated per-run workspace directories without timeouts."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.workspace_base = PROJECTS_DIR / project_id / "workspace"
        self.workspace_base.mkdir(parents=True, exist_ok=True)

    def _prepare_workspace(self) -> Path:
        """Cleans and re-prepares the workspace directory for a fresh execution."""
        if self.workspace_base.exists():
            shutil.rmtree(self.workspace_base, ignore_errors=True)
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        return self.workspace_base

    def run_script(
        self,
        script_content: str,
        script_name: str = "run_task.py",
        env_vars: dict[str, str] | None = None,
        stage: str | None = None,
        run_id: str | None = None,
        task_description: str | None = None,
        attempt: int | None = None,
    ) -> ExecutionResult:
        """Writes script_content into the clean workspace, executes it via subprocess, and records execution history."""
        import json
        import time
        import uuid
        from datetime import datetime

        ws = self._prepare_workspace()
        script_file = ws / script_name
        sanitized_script = normalize_script_unicode(script_content)
        script_file.write_text(sanitized_script, encoding="utf-8")

        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        # Ensure project root is in PYTHONPATH so imports work if needed
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        # Force UTF-8 encoding for standard streams and child Python process
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        python_executable = sys.executable

        start_time = time.time()
        # Execute subprocess in UTF-8 mode without timeout
        process = subprocess.Popen(
            [python_executable, "-X", "utf8", str(script_file.resolve())],
            cwd=str(ws.resolve()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        self.current_process = process
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    if run_id:
                        from app.core.run_memory import RunMemory
                        mem = RunMemory(self.project_id, run_id)
                        if mem.is_cancelled():
                            process.kill()
                            stdout, stderr = process.communicate()
                            return ExecutionResult(
                                exit_code=-9,
                                stdout=stdout or "",
                                stderr="Execution cancelled by user",
                                workspace_dir=str(ws),
                            )
        finally:
            self.current_process = None

        duration_ms = int((time.time() - start_time) * 1000)

        # Persist execution history for monitoring the coder agent
        try:
            exec_dir = PROJECTS_DIR / self.project_id / "code_executions"
            exec_dir.mkdir(parents=True, exist_ok=True)
            now_dt = datetime.now(UTC)
            record_id = f"exec_{int(now_dt.timestamp())}_{uuid.uuid4().hex[:6]}"
            record = {
                "id": record_id,
                "project_id": self.project_id,
                "run_id": run_id,
                "stage": stage or "coder",
                "script_name": script_name,
                "task_description": task_description,
                "attempt": attempt or 1,
                "code": script_content,
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "success": (process.returncode == 0),
                "executed_at": now_dt.isoformat(),
                "duration_ms": duration_ms,
            }
            rec_file = exec_dir / f"{record_id}.json"
            rec_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

            # Broadcast event to SSE subscribers if run_id provided
            if run_id:
                from app.core.events import event_manager
                event_manager.emit_event(
                    project_id=self.project_id,
                    run_id=run_id,
                    event_type="CODE_EXECUTION_COMPLETED",
                    stage=stage or "coder",
                    message=f"Coder executed '{script_name}' (exit code: {process.returncode}, {duration_ms}ms)",
                    data=record,
                )
        except Exception as e:
            # Execution recording should never crash the workflow
            print(f"[ExecutionManager] Warning: failed to save code execution log: {e}", flush=True)

        return ExecutionResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            workspace_dir=str(ws),
        )
