"""Judge agent performing rigorous, holistic reviews of the entire ML pipeline."""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.prompts import JUDGE_SYSTEM_PROMPT
from app.config import PROJECTS_DIR
from app.core.briefs import (
    JUDGE_CONTEXT_LIMIT,
    build_worker_brief,
    check_and_log_budget,
)
from app.core.llm_json import invoke_json
from app.core.model_router import ModelRouter
from app.core.run_memory import RunMemory
from app.core.schemas import JudgeReport, WorkerReport
from app.core.state import ProjectState
from app.tools.registry import ToolRegistry
from app.utils.logger import get_logger

_log = get_logger(__name__)


def run_judge(
    state: ProjectState,
    router: ModelRouter,
    registry: ToolRegistry,
    brief: str | None = None,
) -> dict[str, Any]:
    """Reviews the pipeline using JudgeReport schema via invoke_json. Never defaults to ship on failure."""
    stage_start = time.monotonic()
    project_id = state["project_id"]
    run_id = state.get("run_id", "default")
    mem = RunMemory(project_id, run_id=run_id)

    if brief is None:
        decision = {
            "action": "judge",
            "reason": "Holistic pipeline review",
            "brief": {
                "objective": "Evaluate pipeline health, stage quality, and next direction",
                "focus_points": ["EDA finding relevance", "Data leakage & validity", "Overfitting gap", "Headroom vs noise"],
                "constraints": ["Cite evidence for each verdict"],
                "mode": "standard",
            }
        }
        brief = build_worker_brief("judge", decision, mem, state)

    check_and_log_budget("Judge Prompt Brief", brief, JUDGE_CONTEXT_LIMIT)

    llm = router.get_model("judge", temperature=0.0)

    judge_output, last_err, raw_resp = invoke_json(
        llm,
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=brief),
        ],
        JudgeReport,
        agent_name="judge",
    )

    if judge_output is None:
        judge_output = JudgeReport(
            stage_verdicts={},
            blame_stage="none",
            overall="blocked",
            advice=[f"Judge failed to produce valid evaluation: {last_err}"],
        )
        worker_status = "failed"
        result_summary = f"Judge failed: {last_err}"[:300]
    else:
        worker_status = "ok"
        verdicts_summary = ", ".join(
            f"{stage}:{v.status}" for stage, v in judge_output.stage_verdicts.items()
        ) if judge_output.stage_verdicts else "No specific verdicts"

        result_summary = (
            f"Verdict: {judge_output.overall.upper()} | Blame: {judge_output.blame_stage} | "
            f"Verdicts: [{verdicts_summary}]"
        )[:300]

    # Save summary report
    eval_dir = PROJECTS_DIR / project_id / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "judge_verdict.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(judge_output.model_dump(), fp, indent=2)

    # WorkerReport
    worker_report = WorkerReport(
        status=worker_status,
        result_summary=result_summary,
        evidence=judge_output.model_dump(),
        concern=None,
        suggestion=judge_output.advice[0] if judge_output.advice else None,
        notebook={
            "step": state.get("step", 0),
            "tried": "Judge pipeline evaluation",
            "outcome": judge_output.overall,
            "lesson": f"Blame: {judge_output.blame_stage}. Advice: {'; '.join(judge_output.advice[:2])}",
            "score_impact": None,
            "errors": [last_err] if last_err else [],
        },
        artifacts=[str(summary_path)],
    )

    elapsed = time.monotonic() - stage_start
    _log.info(
        "[JUDGE] Review complete | status=%s verdict=%s blame=%s duration=%.2fs",
        worker_status, judge_output.overall, judge_output.blame_stage, elapsed,
    )

    return {
        "judge_report": judge_output.model_dump(),
        "current_stage": "judge",
        "status": "SUCCESS" if worker_status == "ok" else "FAILED",
        "report": worker_report.model_dump(),
    }


# Backward-compatible alias
def run_evaluator(state: ProjectState, router: ModelRouter, registry: ToolRegistry, brief: str | None = None) -> dict[str, Any]:
    """Backward compatibility wrapper delegating to run_judge."""
    return run_judge(state, router, registry, brief=brief)
