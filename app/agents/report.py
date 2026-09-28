import json
import time
from pathlib import Path
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.state import ProjectState, ReportSummaryOutput
from app.core.model_router import ModelRouter
from app.core.memory import get_stage_context
from app.tools.registry import ToolRegistry
from app.config import PROJECTS_DIR
from app.utils.logger import get_logger

_log = get_logger(__name__)


REPORT_SYSTEM_PROMPT = """You are an expert Technical ML Reporting Agent.
Your responsibility is to synthesize a comprehensive, executive-grade Final Report and structured JSON summary from existing workflow artifacts.

CRITICAL RULES:
1. Synthesize strictly from existing findings and numbers. Do NOT re-analyze or invent new statistics.
2. Structure the Markdown report professionally:
   - Executive Summary & Objective
   - Dataset Profile & Quality Assessment
   - Key EDA Insights
   - Feature Engineering Highlights & Lineage
   - Model Evaluation & Best Model Performance
   - Known Limitations, Risks & Recommendations
3. ZERO PLOTS, CHARTS, OR IMAGES. All tables, text, and numbers.
"""


def run_report(state: ProjectState, router: ModelRouter, registry: ToolRegistry) -> dict[str, Any]:
    """Generates final_report.md and summary.json from completed pipeline artifacts."""
    stage_start = time.monotonic()
    project_id = state["project_id"]
    run_id = state.get("run_id", "unknown")
    _log.info("[REPORT] Stage started | project=%s run=%s", project_id, run_id)

    tools = registry.get_tools_for_role("report")
    ctx = get_stage_context(state, "report")

    llm = router.get_model("report", temperature=0.0)
    structured_llm = llm.with_structured_output(ReportSummaryOutput)

    summary_prompt = f"""Assemble the project summary from the completed stage records:
Context:
{json.dumps(ctx, indent=2)[:3500]}

Generate a structured executive report summary covering the objective, best model, score, iterations, key findings, recommendations, and limitations.
"""

    report_summary = None
    try:
        report_summary = structured_llm.invoke([
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=summary_prompt),
        ])
    except Exception as exc:
        _log.exception("[REPORT] Structured summary failed: %s", exc)

    if report_summary is None:
        report_summary = ReportSummaryOutput(
            project_id=project_id,
            objective=state.get("user_goal", f"Predict {state.get('target_column')}"),
            dataset_summary=state.get("profile_summary", {}),
            best_model_name=state.get("model_summary", {}).get("best_model_name", "Best Model"),
            best_score=float(state.get("best_metric_value", 0.0) or 0.0),
            metric_name=state.get("target_metric") or "metric",
            iterations_run=state.get("iteration", 1),
            key_findings=["Tabular ML model trained and evaluated successfully."],
            recommendations=["Pipeline is reproducible via features/feature_pipeline.py."],
            limitations=["Trained strictly using scikit-learn algorithms."],
        )

    summary_dict = report_summary.model_dump()

    # Now generate comprehensive markdown report
    markdown_prompt = f"""Write the complete, highly detailed final_report.md for Project `{project_id}` using this summary:
Summary:
{json.dumps(summary_dict, indent=2)}

Detailed State:
- Task Type: {state.get('task_type')}
- Target Column: {state.get('target_column')}
- Target Metric: {state.get('target_metric')}
- Best Experiment ID: {state.get('best_experiment_id')}
- Best Score: {state.get('best_metric_value')}
- Profile Summary: {json.dumps(state.get('profile_summary', {}))[:1000]}
- EDA Findings: {json.dumps(state.get('eda_findings', {}))[:1000]}
- Feature Summary: {json.dumps(state.get('feature_summary', {}))[:1000]}
- Model Summary: {json.dumps(state.get('model_summary', {}))[:1000]}
- Evaluation Summary: {json.dumps(state.get('evaluation_summary', {}))[:1000]}

Format clearly with markdown headings, bullet points, and tables. ZERO PLOTS.
"""

    report_md_msg = llm.invoke([
        SystemMessage(content=REPORT_SYSTEM_PROMPT),
        HumanMessage(content=markdown_prompt),
    ])
    final_report_md = report_md_msg.content
    if isinstance(final_report_md, list):
        final_report_md = "\n".join(str(p) for p in final_report_md)

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

    return {
        "current_stage": "report",
        "artifacts": artifacts_list,
        "status": "SUCCESS",
    }
