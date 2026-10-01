"""Run-level and project-level memory management for autonomous ML runs."""

import json
from pathlib import Path
from typing import Any

from app.config import PROJECTS_DIR
from app.core.schemas import LedgerRow, NotebookRow


class RunMemory:
    """Manages run-isolated and cross-run memory structures under projects/<project_id>/runs/<run_id>/memory/."""

    def __init__(self, project_id: str, run_id: str):
        self.project_id = project_id
        self.run_id = run_id
        self.run_dir = PROJECTS_DIR / project_id / "runs" / run_id
        self.memory_dir = self.run_dir / "memory"
        self.notebooks_dir = self.memory_dir / "notebooks"
        self.project_memory_dir = PROJECTS_DIR / project_id / "memory"

        # Ensure directory structure exists
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        self.project_memory_dir.mkdir(parents=True, exist_ok=True)

        self.mission_file = self.memory_dir / "mission.json"
        self.plan_file = self.memory_dir / "plan.json"
        self.ledger_file = self.memory_dir / "ledger.jsonl"
        self.best_file = self.memory_dir / "best.json"
        self.lessons_file = self.project_memory_dir / "lessons.json"

    @property
    def cancel_file(self) -> Path:
        return self.run_dir / ".cancelled"

    def set_cancelled(self) -> None:
        self.cancel_file.write_text("cancelled", encoding="utf-8")

    def is_cancelled(self) -> bool:
        return self.cancel_file.exists()

    def init_mission(self, mission_data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initializes the fixed mission.json file for this run."""
        data = dict(mission_data or {})
        data.update(kwargs)
        with open(self.mission_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_mission(self) -> dict[str, Any]:
        """Reads the mission.json file if present, else returns empty dict."""
        if not self.mission_file.exists():
            return {}
        try:
            with open(self.mission_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def append_ledger(self, row: LedgerRow | dict[str, Any]) -> None:
        """Appends a single verified step record to the run ledger."""
        if isinstance(row, dict):
            row_obj = LedgerRow(**row)
        else:
            row_obj = row
        line = json.dumps(row_obj.model_dump(), default=str)
        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_ledger(self, last_n: int | None = None) -> list[LedgerRow]:
        """Reads ledger records in chronological order, optionally returning only the last_n."""
        if not self.ledger_file.exists():
            return []
        rows: list[LedgerRow] = []
        try:
            with open(self.ledger_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(LedgerRow(**json.loads(line)))
        except Exception:
            return []
        if last_n is not None and last_n > 0:
            return rows[-last_n:]
        return rows

    def render_ledger(self, max_chars: int = 4000) -> str:
        """Renders ledger as a compact table, collapsing rows older than the last 12 into a single summary."""
        all_rows = self.read_ledger()
        if not all_rows:
            return "No actions recorded in ledger yet."

        older_rows = all_rows[:-12] if len(all_rows) > 12 else []
        recent_rows = all_rows[-12:] if len(all_rows) > 12 else all_rows

        lines: list[str] = []

        if older_rows:
            oldest_step = older_rows[0].step
            newest_old_step = older_rows[-1].step
            actions_summary = ", ".join(f"{r.action}:{r.status}" for r in older_rows[:8])
            if len(older_rows) > 8:
                actions_summary += f", ... (+{len(older_rows) - 8} more)"
            lines.append(
                f"[Steps {oldest_step}-{newest_old_step} collapsed: {len(older_rows)} steps executed ({actions_summary})]"
            )

        header = "| Step | Agent | Action | Status | Score (Mean±Std) | NoiseFloor | Sig | Delta | Summary/Error |"
        sep = "|---|---|---|---|---|---|---|---|---|"
        table_lines = [header, sep]

        for r in recent_rows:
            if r.mean is not None and r.std is not None:
                score_str = f"{r.mean:.4f}±{r.std:.4f}"
            elif r.score is not None:
                score_str = f"{r.score:.4f}"
            else:
                score_str = "-"
            nf_str = f"{r.noise_floor:.4f}" if r.noise_floor is not None else "-"
            sig_str = "yes" if r.significant else ("no" if r.significant is False else "-")
            delta_str = f"{r.delta_vs_best:+.4f}" if r.delta_vs_best is not None else "-"
            summary_clean = r.result_summary.replace("|", "/").replace("\n", " ")[:60]
            if r.status == "failed" and r.error_text:
                summary_clean += f" [Error: {r.error_text[-120:].replace('|', '/')}]"
            table_lines.append(
                f"| {r.step} | {r.agent} | {r.action} | {r.status} | {score_str} | {nf_str} | {sig_str} | {delta_str} | {summary_clean} |"
            )

        lines.extend(table_lines)
        rendered = "\n".join(lines)
        if len(rendered) > max_chars:
            return rendered[:max_chars - 3] + "..."
        return rendered

    def append_notebook(self, agent: str, row: NotebookRow | dict[str, Any]) -> None:
        """Appends a specialist worker learning or error record to its isolated notebook."""
        if isinstance(row, dict):
            row_obj = NotebookRow(**row)
        else:
            row_obj = row
        file_path = self.notebooks_dir / f"{agent}.jsonl"
        line = json.dumps(row_obj.model_dump(), default=str)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def top_lessons(self, agent: str, n: int = 5) -> list[str]:
        """Extracts the top n distinct, non-empty lessons recorded by a specialist worker."""
        file_path = self.notebooks_dir / f"{agent}.jsonl"
        if not file_path.exists():
            return []
        lessons: list[str] = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        lesson = data.get("lesson", "").strip()
                        if lesson and lesson not in lessons:
                            lessons.append(lesson)
        except Exception:
            return []
        return lessons[-n:]

    def get_plan(self) -> Any:
        """Reads the dynamic task plan."""
        target_file = self.plan_file if self.plan_file.exists() else (self.run_dir / "plan.json")
        if not target_file.exists():
            return []
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def set_plan(self, plan: Any) -> None:
        """Saves an updated task plan."""
        if hasattr(plan, "model_dump"):
            serializable_plan = plan.model_dump()
        elif isinstance(plan, list):
            serializable_plan = [
                p.model_dump() if hasattr(p, "model_dump") else p for p in plan
            ]
        elif isinstance(plan, dict):
            serializable_plan = plan
        else:
            serializable_plan = str(plan)

        with open(self.plan_file, "w", encoding="utf-8") as f:
            json.dump(serializable_plan, f, indent=2)

        root_plan = self.run_dir / "plan.json"
        if root_plan != self.plan_file:
            try:
                with open(root_plan, "w", encoding="utf-8") as f:
                    json.dump(serializable_plan, f, indent=2)
            except Exception:
                pass

    def get_best(self) -> dict[str, Any] | None:
        """Reads current best model version and score metadata."""
        if not self.best_file.exists():
            return None
        try:
            with open(self.best_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set_best(self, best_data: dict[str, Any]) -> None:
        """Writes new best model version and score metadata."""
        with open(self.best_file, "w", encoding="utf-8") as f:
            json.dump(best_data, f, indent=2)

    def add_project_lesson(self, text: str) -> None:
        """Appends a deduplicated lesson to the project-level lessons list, capped at 10 items."""
        clean = text.strip()
        if not clean:
            return
        lessons = self.get_project_lessons()
        if clean in lessons:
            return
        lessons.append(clean)
        # Cap at 10
        if len(lessons) > 10:
            lessons = lessons[-10:]
        with open(self.lessons_file, "w", encoding="utf-8") as f:
            json.dump(lessons, f, indent=2)

    def get_project_lessons(self) -> list[str]:
        """Retrieves cross-run project lessons."""
        if not self.lessons_file.exists():
            return []
        try:
            with open(self.lessons_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []
