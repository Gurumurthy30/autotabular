"""Technical Reporting Agent generating executive final_report.md and summary.json."""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import PROJECTS_DIR
from app.core.briefs import (
    build_worker_brief,
)
from app.core.model_router import ModelRouter
from app.core.run_memory import RunMemory
from app.core.schemas import WorkerReport
from app.core.state import ProjectState, ReportSummaryOutput
from app.tools.registry import ToolRegistry
from app.utils.logger import get_logger

_log = get_logger(__name__)

from app.agents.prompts import REPORT_SYSTEM_PROMPT
from app.core.llm_json import invoke_json


def run_report(
    state: ProjectState,
    router: ModelRouter,
    registry: ToolRegistry,
    brief: str | None = None,
) -> dict[str, Any]:
    """Generates final_report.md and summary.json from completed pipeline artifacts."""
    stage_start = time.monotonic()
    project_id = state["project_id"]
    run_id = state.get("run_id", "unknown")
    _log.info("[REPORT] Stage started | project=%s run=%s", project_id, run_id)

    tools = registry.get_tools_for_role("report")
    mem = RunMemory(project_id, run_id)

    if not brief:
        decision = state.get("supervisor_decision") or {
            "brief": {
                "objective": "Assemble executive report documenting best model, pipeline lineage, and ledger",
                "focus_points": ["Executive summary", "Ledger audit table", "Decision log", "Limitations and risks"],
                "constraints": ["Include best version score", "Zero charts"],
                "mode": "final",
            }
        }
        brief = build_worker_brief("report", decision, mem, state)

    llm = router.get_model("report", temperature=0.0)
    best_meta = mem.get_best() or {}
    best_score = float(best_meta.get("score") or state.get("best_metric_value") or 0.0)
    best_model_name = str(best_meta.get("best_model") or state.get("model_summary", {}).get("best_model_name") or "Best Model")
    target_metric = str(best_meta.get("metric") or state.get("target_metric") or "metric")

    # LLM Call 1: Structured Report Summary
    summary_prompt = f"""Synthesize a structured summary for the final project report:

Worker Brief:
{brief}

Best Version Meta:
{json.dumps(best_meta, default=str)}

Return valid JSON conforming to ReportSummaryOutput.
"""

    report_summary, err_msg, raw_text = invoke_json(
        llm,
        [
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=summary_prompt),
        ],
        ReportSummaryOutput,
        agent_name="report",
    )

    if report_summary is None:
        report_summary = ReportSummaryOutput(
            project_id=project_id,
            objective=state.get("user_goal", f"Predict {state.get('target_column')}"),
            dataset_summary=state.get("profile_summary", {}),
            best_model_name=best_model_name,
            best_score=best_score,
            metric_name=target_metric,
            iterations_run=state.get("step", 1),
            key_findings=["Autonomous ML engineering run completed successfully."],
            recommendations=["Pipeline is reproducible via features/feature_pipeline.py."],
            limitations=["Trained strictly using scikit-learn models."],
        )

    summary_dict = report_summary.model_dump()

    # LLM Call 2: Final Markdown Report
    ledger_table = mem.render_ledger(max_chars=2500)
    rows = mem.read_ledger()
    decision_log_lines = [f"- **Step {r.step} ({r.action})**: {r.decision_reason}" for r in rows if r.decision_reason]
    decision_log_str = "\n".join(decision_log_lines) if decision_log_lines else "- No decisions logged."

    markdown_prompt = f"""Write the complete, publication-grade final_report.md for Project `{project_id}`.

Worker Brief:
{brief}

Key Structured Elements to Embed:
- Objective: {report_summary.objective}
- Best Model: {best_model_name}
- Headline Metric: {target_metric} = {best_score:.4f}
- Best Version: {best_meta.get('version_id', 'v1')}

LEDGER TABLE (Include Verbatim):
{ledger_table}

DECISION LOG (Include in dedicated section):
{decision_log_str}

Ensure the report includes:
# Project Final Report
1. Executive Summary
2. Dataset Profile & Data Quality
3. Exploratory Data Analysis & Leakage Check
4. Feature Engineering Pipeline & Lineage
5. Model Evaluation & Best Model Performance
6. Run Ledger & Decision History
7. Risks, Limitations & Production Deployment Guidance
"""

    try:
        report_md_msg = llm.invoke([
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=markdown_prompt),
        ])
        final_report_md = report_md_msg.content
        if isinstance(final_report_md, list):
            final_report_md = "\n".join(str(p) for p in final_report_md)
    except Exception as exc:
        _log.warning("[REPORT] Markdown generation failed: %s", exc)
        final_report_md = (
            f"# Final Report: Project `{project_id}`\n\n"
            f"## Executive Summary\n"
            f"Autonomous pipeline completed. Best Model: `{best_model_name}` ({target_metric}={best_score:.4f}).\n\n"
            f"## Run Ledger\n{ledger_table}\n\n"
            f"## Decision Log\n{decision_log_str}\n"
        )

    reports_dir = PROJECTS_DIR / project_id / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "summary.json"
    md_path = reports_dir / "final_report.md"

    tools.files.write_file("reports/summary.json", json.dumps(summary_dict, indent=2))
    tools.files.write_file("reports/final_report.md", str(final_report_md))

    # Register artifacts
    art_sum = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="final_summary_json",
        path=str(json_path),
        run_id=state.get("run_id"),
    )
    art_rep = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="final_report_md",
        path=str(md_path),
        run_id=state.get("run_id"),
    )

    artifacts_list = list(state.get("artifacts", []))
    artifacts_list.extend([art_sum, art_rep])

    elapsed = time.monotonic() - stage_start
    _log.info("[REPORT] Stage completed | project=%s run=%s | duration=%.2fs", project_id, run_id, elapsed)

    worker_report = WorkerReport(
        status="ok",
        result_summary=f"Final report generated. Best model: {best_model_name} with {target_metric}={best_score:.4f}.",
        evidence={"best_model": best_model_name, "best_score": best_score, "metric": target_metric},
        concern=None,
        suggestion="Workflow complete; proceed to finish.",
        notebook={
            "step": state.get("step", 0),
            "tried": "Final markdown report synthesis",
            "outcome": "Report written to reports/final_report.md",
            "lesson": f"Documented {best_model_name} ({best_score:.4f})",
            "score_impact": None,
            "errors": [],
        },
        artifacts=[str(json_path), str(md_path)],
    )

    return {
        "report_summary": summary_dict,
        "current_stage": "report",
        "artifacts": artifacts_list,
        "status": "SUCCESS",
        "report": worker_report.model_dump(),
    }
