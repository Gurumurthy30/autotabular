"""Coder agent responsible for writing, executing, and repairing standalone Python scripts."""

import hashlib
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.prompts import CODER_SYSTEM_PROMPT
from app.core.model_router import ModelRouter
from app.core.run_memory import RunMemory
from app.tools.execution_manager import ExecutionManager
from app.tools.file_tools import FileTools
from app.utils.logger import get_logger

_log = get_logger(__name__)


def extract_error_summary(stderr: str, stdout: str) -> str:
    """Extracts a short, normalized error signature (e.g. 'ValueError: Unknown label') for logging and deduplication."""
    if not (stderr or "").strip() and not (stdout or "").strip():
        return "Unknown execution error"

    combined = f"{stderr}\n{stdout}"
    # Look for Python traceback exception line (e.g. 'TypeError: ...', 'ValueError: ...')
    matches = re.findall(r"^([A-Z][a-zA-Z0-9_.]*(?:Error|Exception|Warning|Interrupt|Exit)): (.*)$", combined, re.MULTILINE)
    if matches:
        exc_type, exc_msg = matches[-1]
        # Normalize msg: remove specific paths/addresses
        norm_msg = re.sub(r"0x[0-9a-fA-F]+", "0x...", exc_msg.strip())
        return f"{exc_type}: {norm_msg[:120]}"

    # Fallback to last non-empty line of stderr
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    if lines:
        return lines[-1][:120]
    return "Non-zero exit code"


class CoderSubAgent:
    """Sub-agent responsible for writing, executing, debugging, and retrying Python scripts."""

    def __init__(self, project_id: str, model_router: ModelRouter, file_tools: FileTools, execution_manager: ExecutionManager):
        self.project_id = project_id
        self.router = model_router
        self.file_tools = file_tools
        self.execution = execution_manager
        self.llm = self.router.get_model("coder", temperature=0.1)

    def _extract_code(self, response_text: str) -> str | None:
        """Extracts python code from markdown block or returns None if no code block found."""
        code_match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()
        code_match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return None

    def run_task(
        self,
        task_description: str,
        context: dict[str, Any] | str,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Iteratively writes, runs, and debugs a Python script to satisfy the given task.

        No hard numeric retry cap: stops when the code succeeds, when a stuck state is detected
        (same error signature twice in a row without progress), when the coder explains the task
        cannot be done, or when the run is cancelled cooperatively.
        """
        context_str = json.dumps(context, default=str) if isinstance(context, dict) else str(context)
        stage_name = context.get("stage", "coder") if isinstance(context, dict) else "coder"
        run_id = context.get("run_id") if isinstance(context, dict) else None

        attempt = 0
        last_error = None
        last_code = ""
        last_stdout = ""
        last_stderr = ""
        last_matched_hint = ""
        error_summaries: list[str] = []
        seen_code_hashes: set[str] = set()

        while True:
            # Check cancellation before every step
            if run_id:
                mem = RunMemory(self.project_id, run_id)
                if mem.is_cancelled():
                    _log.info("[CODER] Run cancelled cooperatively during attempt %d", attempt + 1)
                    return {
                        "status": "CANCELLED",
                        "error": "Execution cancelled by user",
                        "stdout_summary": "",
                        "full_stdout": "",
                        "error_summaries": error_summaries,
                        "attempts": attempt,
                        "executed_code": last_code,
                    }

            # Check for stuck state: if same error signature appears twice in a row
            stuck_note = ""
            if len(error_summaries) >= 2 and error_summaries[-1] == error_summaries[-2]:
                stuck_note = (
                    f"\n\nSTUCK STATE DETECTED: The exact error '{error_summaries[-1]}' occurred twice consecutively. "
                    "You MUST change your strategy (simplify, use a different library, or take a different approach). "
                    "If you conclude this task cannot be accomplished in this environment, output a concise explanation "
                    "without code blocks explaining why."
                )

            # Build messages
            if attempt == 0:
                messages = [
                    SystemMessage(content=CODER_SYSTEM_PROMPT),
                    HumanMessage(content=f"Task Description:\n{task_description}\n\nTask Context:\n{context_str}"),
                ]
            else:
                hint_block = f"\n\nCRITICAL FIX HINT:\n{last_matched_hint}" if last_matched_hint else ""
                feedback = (
                    f"Execution of your previous code failed.\n\n"
                    f"PREVIOUS CODE:\n```python\n{last_code}\n```\n\n"
                    f"STDERR:\n{last_stderr}\n\n"
                    f"STDOUT (last 3000 chars):\n{last_stdout[-3000:] if last_stdout else '(No stdout)'}\n"
                    f"{hint_block}{stuck_note}\n\n"
                    "Please debug the error and rewrite the entire Python script inside a single ```python ``` block."
                )
                messages = [
                    SystemMessage(content=CODER_SYSTEM_PROMPT),
                    HumanMessage(content=f"Task Description:\n{task_description}\n\nTask Context:\n{context_str}\n\n{feedback}"),
                ]

            # 1. Ask LLM to generate or repair the code
            ai_msg = self.llm.invoke(messages)
            raw_content = ai_msg.content
            if isinstance(raw_content, list):
                raw_content = "\n".join(str(part) for part in raw_content)

            code = self._extract_code(raw_content)

            # If the Coder decided the task cannot be done, it returned plain text without a code block
            if not code:
                _log.warning("[CODER] Coder returned plain text explanation without code: %s", raw_content[:300])
                return {
                    "status": "FAILED",
                    "error": f"Coder concluded task cannot be performed: {raw_content[:300]}",
                    "stdout_summary": "",
                    "full_stdout": raw_content,
                    "error_summaries": error_summaries,
                    "attempts": attempt + 1,
                    "executed_code": last_code,
                }

            last_code = code

            # Check if identical code was repeated
            code_hash = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()
            if code_hash in seen_code_hashes:
                attempt += 1
                recent_err = error_summaries[-1] if error_summaries else "execution failure"
                err_text = f"IdenticalCodeError: You returned identical code without changes. Prior error: {recent_err}"
                last_error = err_text
                last_stderr = err_text
                error_summaries.append(err_text)
                _log.warning("[CODER] Identical code detected on attempt %d", attempt)
                if (max_retries is not None and attempt >= max_retries) or (len(error_summaries) >= 3 and error_summaries[-1] == error_summaries[-2]):
                    return {
                        "status": "FAILED",
                        "error": "Coder repeatedly returned identical failing code.",
                        "stdout_summary": "",
                        "full_stdout": "",
                        "error_summaries": error_summaries,
                        "attempts": attempt,
                        "executed_code": last_code,
                    }
                continue

            seen_code_hashes.add(code_hash)

            # Run script
            res = self.execution.run_script(
                code,
                stage=stage_name,
                run_id=run_id,
                task_description=task_description,
                attempt=attempt + 1,
            )

            # Detect cancelled during execution
            if res.exit_code == -9 or "cancelled by user" in (res.stderr or "").lower():
                return {
                    "status": "CANCELLED",
                    "error": "Execution cancelled by user",
                    "stdout_summary": res.stdout[-2000:] if len(res.stdout) > 2000 else res.stdout,
                    "full_stdout": res.stdout,
                    "error_summaries": error_summaries,
                    "attempts": attempt + 1,
                    "executed_code": code,
                }

            # Detect soft failures in execution output
            if res.success and "<MODEL_RESULTS>" in res.stdout:
                if '"models": []' in res.stdout or '"best_model_name": null' in res.stdout:
                    res.success = False
                    res.stderr = (res.stderr or "") + "\nExecution resulted in empty models list. Check data types or pipeline errors."

            if res.success:
                return {
                    "status": "SUCCESS",
                    "stdout_summary": res.stdout[-2000:] if len(res.stdout) > 2000 else res.stdout,
                    "full_stdout": res.stdout,
                    "error": None,
                    "error_summaries": error_summaries,
                    "attempts": attempt + 1,
                    "executed_code": code,
                }

            # On failure, prepare feedback for retry
            attempt += 1
            err_text = (res.stderr or "").strip()
            out_text = (res.stdout or "").strip()
            last_error = err_text or (out_text[-1000:] if out_text else f"Non-zero exit code: {res.exit_code}")
            last_stderr = err_text if err_text else f"Non-zero exit code: {res.exit_code}"
            last_stdout = res.stdout

            err_summary = extract_error_summary(res.stderr, res.stdout)
            error_summaries.append(err_summary)

            _log.warning(
                "[CODER] Script execution failed | stage=%s attempt=%d exit_code=%d | error=%s",
                stage_name, attempt, res.exit_code, last_error[:300],
            )

            if max_retries is not None and attempt >= max_retries:
                return {
                    "status": "FAILED",
                    "error": f"Execution failed after {attempt} attempts: {last_error}",
                    "stdout_summary": res.stdout[-2000:] if len(res.stdout) > 2000 else res.stdout,
                    "full_stdout": res.stdout,
                    "error_summaries": error_summaries,
                    "attempts": attempt,
                    "executed_code": code,
                }

            # If the same error signature has occurred 3 times in a row, break stuck loop
            if len(error_summaries) >= 3 and error_summaries[-1] == error_summaries[-2] == error_summaries[-3]:
                _log.error("[CODER] Aborting after 3 consecutive identical failures: %s", err_summary)
                return {
                    "status": "FAILED",
                    "error": f"Coder repeatedly failed with identical error: {err_summary}",
                    "stdout_summary": res.stdout[-2000:] if len(res.stdout) > 2000 else res.stdout,
                    "full_stdout": res.stdout,
                    "error_summaries": error_summaries,
                    "attempts": attempt,
                    "executed_code": code,
                }

            hints = []
            combined_err = f"{err_text}\n{out_text}"
            if "Can't pickle" in combined_err or "lambda" in combined_err or "pickling" in combined_err.lower():
                hints.append(
                    "CRITICAL PICKLE FIX: Can't pickle local function/lambda! "
                    "DO NOT use lambda functions or nested local functions inside Pipeline or FunctionTransformer. "
                    "Define transformer classes at the top module level using SafeTransformer from app.ml_harness.base."
                )

            last_matched_hint = "\n".join(hints)
