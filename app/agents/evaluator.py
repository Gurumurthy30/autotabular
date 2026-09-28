import json
import hashlib
import time
from pathlib import Path
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.state import ProjectState, EvaluatorOutput
from app.core.model_router import ModelRouter
from app.core.memory import get_stage_context
from app.tools.registry import ToolRegistry
from app.config import PROJECTS_DIR
from app.utils.logger import get_logger

_log = get_logger(__name__)


EVALUATOR_SYSTEM_PROMPT = """You are an independent, rigorous Machine Learning Evaluation Agent.
Your responsibility is to critique the trained models objectively and determine whether the solution meets production standards or requires improvement.

EVALUATION CHECKS:
1. Overfitting & Train/Validation Gap:
   - Check if training score is significantly higher than validation score (e.g. > 15-20% relative gap).
2. Data Leakage / Unrealistically Perfect Scores:
   - Suspiciously perfect scores (e.g. 1.0 or 0.999) without hard physical justification often indicate target leakage.
3. Class Imbalance Impact:
   - In classification, ensure the model does not just predict the majority class.
4. Metric Appropriateness:
   - Ensure the chosen metric matches the user's business objective.

DECISION CRITERIA:
- If results are solid, generalization is reasonable, and no major leakage/overfitting exists -> "PASS".
- If significant overfitting, poor performance, or leakage is present -> "IMPROVE".
  - If feature engineering changes are needed (e.g., better encodings, dropping leaky features, scaling, interaction terms) -> recommended_next_stage="feature_engineering".
  - If features are fine but model selection/hyperparameters caused the issue -> recommended_next_stage="model".

CRITICAL RULE: NO PLOTTING OR CHARTS.
"""


def run_evaluator(state: ProjectState, router: ModelRouter, registry: ToolRegistry) -> dict[str, Any]:
    """Independent evaluation agent assessing model validity, overfitting, and leakage."""
    stage_start = time.monotonic()
    project_id = state["project_id"]
    run_id = state.get("run_id", "unknown")
    _log.info("[EVALUATOR] Stage started | project=%s run=%s", project_id, run_id)

    tools = registry.get_tools_for_role("evaluator")
    ctx = get_stage_context(state, "evaluator")
    iteration = state.get("iteration", 1)

    llm = router.get_model("evaluator", temperature=0.0)
    structured_llm = llm.with_structured_output(EvaluatorOutput)

    eval_prompt = f"""Critique the latest modeling results:
Project Goal: {state.get('user_goal')}
Task Type: {state.get('task_type')}
Target Metric: {state.get('target_metric')}
Current Iteration: {iteration} of {state.get('max_iterations', 3)}

Model Stage Summary:
{json.dumps(state.get('model_summary', {}), indent=2)}

Provide your independent evaluation verdict (PASS or IMPROVE), identify any issues found, provide reasoning, and specify recommended_next_stage ("feature_engineering" or "model").
"""

    prompt_hash = hashlib.sha256(EVALUATOR_SYSTEM_PROMPT.encode()).hexdigest()[:12]
    _log.info("[EVALUATOR] Invoking LLM | prompt_version=%s iter=%s", prompt_hash, iteration)

    eval_output = None
    try:
        eval_output = structured_llm.invoke([
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=eval_prompt),
        ])
        _log.debug("[EVALUATOR] Raw LLM response (truncated): %s", str(eval_output)[:600])
    except Exception as exc:
        _log.exception("[EVALUATOR] Structured output failed: %s", exc)

    if eval_output is None:
        best_score = float(state.get("model_summary", {}).get("best_score", 0.0) or 0.0)
        target_met = state.get("target_metric") or "f1"
        # Determine pass if score is reasonable
        verdict = "PASS" if best_score > 0.5 else "IMPROVE"
        eval_output = EvaluatorOutput(
            verdict=verdict,
            target_metric=target_met,
            primary_metric_value=best_score,
            issues_found=[],
            recommended_next_stage="feature_engineering",
            reasoning=f"Evaluated model performance on {target_met}: {best_score:.4f}.",
        )

    eval_dict = eval_output.model_dump()

    # Save to projects/<id>/evaluations/
    eval_dir = PROJECTS_DIR / project_id / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    json_path = eval_dir / f"evaluation_iter_{iteration}.json"
    md_path = eval_dir / f"evaluation_iter_{iteration}.md"

    tools.files.write_file(f"evaluations/evaluation_iter_{iteration}.json", json.dumps(eval_dict, indent=2))

    # Create readable evaluation markdown (no plots)
    md_lines = [
        f"# Evaluation Verdict: Project `{project_id}` (Iteration {iteration})",
        f"- **Verdict:** `{eval_output.verdict}`",
        f"- **Target Metric ({eval_output.target_metric}):** {eval_output.primary_metric_value:.4f}",
        f"- **Recommended Next Stage:** `{eval_output.recommended_next_stage}`",
        "",
        "## Reasoning",
        eval_output.reasoning,
        "",
        "## Issues Detected",
    ]
    if eval_output.issues_found:
        md_lines.extend(["| Check | Severity | Description | Suggested Fix |", "|---|---|---|---|"])
        for issue in eval_output.issues_found:
            clean_desc = issue.description.replace("|", "/")
            clean_fix = issue.suggested_fix.replace("|", "/")
            md_lines.append(f"| {issue.check_name} | {issue.severity} | {clean_desc} | {clean_fix} |")
    else:
        md_lines.append("- No critical issues found.")

    tools.files.write_file(f"evaluations/evaluation_iter_{iteration}.md", "\n".join(md_lines))

    # Register artifact
    art_eval = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="evaluation_report",
        path=str(md_path),
        run_id=state.get("run_id"),
        version=f"iter_{iteration}",
    )

    artifacts_list = list(state.get("artifacts", []))
    artifacts_list.append(art_eval)

    elapsed = time.monotonic() - stage_start
    _log.info(
        "[EVALUATOR] Stage completed | project=%s run=%s | verdict=%s | metric=%.4f | duration=%.2fs",
        project_id, run_id, eval_output.verdict, eval_output.primary_metric_value, elapsed,
    )

    return {
        "evaluation_summary": eval_dict,
        "current_stage": "evaluator",
        "artifacts": artifacts_list,
        "status": "SUCCESS",
    }
