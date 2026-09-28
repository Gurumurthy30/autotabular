import re
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.model_router import ModelRouter
from app.tools.execution_manager import ExecutionManager
from app.tools.file_tools import FileTools


CODER_SYSTEM_PROMPT = """You are an expert Python machine learning software engineer.
Your job is to write clean, robust, and standalone Python scripts to execute narrow tasks.

CRITICAL RULES:
1. ONLY write pure Python code. Wrap your code inside a single ```python ... ``` block.
2. NEVER import or use plotting or visualization libraries (NO matplotlib, NO seaborn, NO plotly, NO bokeh). Everything must be outputted as structured text, numbers, JSON, or saved to Parquet/Pickle/CSV files.
3. Handle exceptions gracefully and print informative stdout summarizing results.
4. When saving artifacts, use exact paths as instructed.
5. All code must be self-contained and run from top to bottom.
6. When outputting JSON, use `json.dumps(obj, default=str)` to prevent NumPy int64/float64 serialization errors.
7. Use pure ASCII characters only in code, comments, strings, and print statements (use standard hyphen '-' instead of non-breaking hyphen '\u2011', standard quotes, no fancy unicode symbols).
8. When scaling continuous columns or assigning 2D arrays back into DataFrames, assign column by column (e.g. `for idx, col in enumerate(cols): df[col] = scaled[:, idx]`) or cast columns beforehand (e.g. `df[cols] = df[cols].astype(float)`) to avoid pandas LossySetitemError/TypeError when float arrays are assigned into integer columns.
"""


class CoderSubAgent:
    """Sub-agent responsible for writing, executing, debugging, and retrying Python scripts."""

    def __init__(self, project_id: str, model_router: ModelRouter, file_tools: FileTools, execution_manager: ExecutionManager):
        self.project_id = project_id
        self.router = model_router
        self.file_tools = file_tools
        self.execution = execution_manager
        self.llm = self.router.get_model("coder", temperature=0.1)

    def _extract_code(self, response_text: str) -> str:
        """Extracts python code from markdown block or returns raw text."""
        code_match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()
        # Fallback for generic code block
        code_match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response_text.strip()

    def run_task(
        self,
        task_description: str,
        context: dict[str, Any],
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Iteratively writes, runs, and debugs a Python script to satisfy the given task."""
        history = [
            SystemMessage(content=CODER_SYSTEM_PROMPT),
            HumanMessage(content=f"Task Description:\n{task_description}\n\nTask Context:\n{context}"),
        ]

        attempt = 0
        last_error = None
        last_code = ""
        last_stdout = ""

        while attempt <= max_retries:
            # 1. Ask LLM to generate or repair the code
            ai_msg = self.llm.invoke(history)
            raw_content = ai_msg.content
            if isinstance(raw_content, list):
                # Handle possible multimodality or structured blocks if returned as list
                raw_content = "\n".join(str(part) for part in raw_content)

            code = self._extract_code(raw_content)
            last_code = code

            stage_name = context.get("stage", "coder") if isinstance(context, dict) else "coder"
            run_id = context.get("run_id") if isinstance(context, dict) else None
            res = self.execution.run_script(
                code,
                stage=stage_name,
                run_id=run_id,
                task_description=task_description,
                attempt=attempt + 1,
            )

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
                    "attempts": attempt + 1,
                    "executed_code": code,
                }

            # On failure, prepare feedback for retry
            attempt += 1
            last_error = res.stderr or f"Non-zero exit code: {res.exit_code}"
            last_stdout = res.stdout

            if attempt <= max_retries:
                history.append(ai_msg)
                retry_feedback = (
                    f"Execution failed with exit code {res.exit_code}.\n"
                    f"STDERR:\n{res.stderr[-2000:]}\n\n"
                    f"STDOUT:\n{res.stdout[-1000:]}\n\n"
                    "Please debug the error and rewrite the entire Python script inside a single ```python ``` block."
                )
                history.append(HumanMessage(content=retry_feedback))

        return {
            "status": "FAILED",
            "stdout_summary": last_stdout[-2000:],
            "full_stdout": last_stdout,
            "error": last_error,
            "attempts": attempt,
            "executed_code": last_code,
        }
