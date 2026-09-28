import json
import hashlib
import time
from pathlib import Path
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError

from app.core.state import ProjectState, EDAOutput, to_evidence_str
from app.core.model_router import ModelRouter
from app.core.memory import get_stage_context
from app.agents.coder import CoderSubAgent
from app.tools.registry import ToolRegistry
from app.config import PROJECTS_DIR
from app.utils.logger import get_logger

_log = get_logger(__name__)


EDA_SYSTEM_PROMPT = """You are an expert Tabular Exploratory Data Analysis (EDA) Agent.
Your responsibility is to analyze tabular data characteristics and produce actionable machine learning findings.

CRITICAL RULES:
1. NO charts or plots of any kind. Never suggest, output, or generate images.
2. Focus on:
   - Skewness and distribution of numerical columns
   - Outliers and high-leverage points
   - High cardinality or rare categories in categorical columns
   - Feature correlations and multicollinearity
   - Target correlation and relationship
   - Data leakage risks (e.g. ID columns, target proxies, post-event features)
   - Class imbalance (if classification)
3. Return clear, typed, structured findings with numerical evidence.
   EVIDENCE MUST ALWAYS BE A PLAIN HUMAN-READABLE STRING. Never return a dict, list, or
   object for the evidence field. Summarise the numbers inline, for example:
     Good: "std=1.39M, IQR=[7.99M, 9.29M], skewness=2.4"
     Good: "missing_pct=12.3%, unique_count=487"
     Bad:  {"std": 1390000, "iqr": [7990000, 9290000]}   <-- NEVER DO THIS
"""


def run_eda(state: ProjectState, router: ModelRouter, registry: ToolRegistry) -> dict[str, Any]:
    """Runs adaptive, LLM-driven EDA using Coder for computation and producing structured findings."""
    stage_start = time.monotonic()
    project_id = state["project_id"]
    run_id = state.get("run_id", "unknown")
    _log.info("[EDA] Stage started | project=%s run=%s", project_id, run_id)

    tools = registry.get_tools_for_role("eda")
    coder_tools = registry.get_tools_for_role("coder")
    coder = CoderSubAgent(project_id, router, coder_tools.files, coder_tools.execution)

    ctx = get_stage_context(state, "eda")
    dataset_version = state.get("dataset_version", "dataset_v1")
    dataset_path = str(tools.dataset.get_dataset_path(dataset_version).resolve()).replace("\\", "/")
    target_col = state.get("target_column")
    task_type = state.get("task_type")

    # 1. Ask Coder to compute statistical summaries and save to JSON
    eda_script_task = f"""Write a Python script to compute statistical EDA metrics for a tabular dataset.
Dataset file path: '{dataset_path}'
Target column: '{target_col}'
Task type: '{task_type}'

Requirements:
1. Load dataset using pandas.
2. Compute:
   - Numerical feature statistics: skewness, min, max, median, 25%, 75%
   - Categorical feature statistics: cardinality, top 3 values with frequency percentages
   - Correlations: Pearson correlation with target (if numeric target) or between numerical features
   - Missingness rates per feature
   - Leakage check: features with near 1.0 correlation with target or identical unique ID counts
   - Class balance ratio (if classification)
3. Print the computed summary as a valid JSON object wrapped in <JSON_OUTPUT> and </JSON_OUTPUT> tags. Use `json.dumps(summary, default=str)` so all numeric and dictionary types serialize cleanly.
4. NO plotting libraries (do NOT import matplotlib/seaborn).
"""

    coder_res = coder.run_task(task_description=eda_script_task, context=ctx)
    stdout = coder_res.get("full_stdout", "")

    # Extract JSON or use stdout
    extracted_stats = {}
    if "<JSON_OUTPUT>" in stdout and "</JSON_OUTPUT>" in stdout:
        try:
            raw_json = stdout.split("<JSON_OUTPUT>")[1].split("</JSON_OUTPUT>")[0].strip()
            extracted_stats = json.loads(raw_json)
        except Exception:
            extracted_stats = {"raw_output": stdout[-1500:]}
    else:
        extracted_stats = {"raw_output": stdout[-1500:]}

    # 2. Use LLM with structured output to synthesize findings
    llm = router.get_model("eda", temperature=0.0)
    structured_llm = llm.with_structured_output(EDAOutput)

    prompt = f"""Review the statistical analysis of the tabular dataset:
Task Type: {task_type}
Target Column: {target_col}
User Goal: {state.get('user_goal')}
Profile Summary: {json.dumps(state.get('profile_summary', {}), indent=2)[:2000]}
Computed Statistics from Data:
{json.dumps(extracted_stats, indent=2)[:3000]}

Produce a comprehensive EDA report with:
- Executive summary of the data dynamics
- Structured findings (category, finding, evidence formatted as string, implication, recommendation)
- Leakage risks detected
- Suggested feature engineering ideas
Remember: NO PLOTS, NO IMAGES. The 'evidence' field in every finding MUST be a plain string summary.
"""

    # Log prompt hash at INFO (not the full prompt, which may contain column names but not values)
    prompt_hash = hashlib.sha256(EDA_SYSTEM_PROMPT.encode()).hexdigest()[:12]
    _log.info("[EDA] Invoking structured LLM | prompt_version=%s | target=%s | task=%s", prompt_hash, target_col, task_type)

    eda_output: EDAOutput | None = None
    try:
        raw_response = structured_llm.invoke([
            SystemMessage(content=EDA_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        _log.debug("[EDA] Raw LLM response (truncated): %s", str(raw_response)[:800])
        eda_output = raw_response
    except ValidationError as ve:
        _log.exception(
            "[EDA] Pydantic ValidationError | fields=%s | payload_preview=%s",
            [e['loc'] for e in ve.errors()],
            str(ve.errors())[:500],
        )
        raise
    except Exception as exc:
        _log.exception("[EDA] LLM invocation failed: %s", exc)
        raise

    # 3. Save findings to disk
    eda_dir = PROJECTS_DIR / project_id / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    findings_json_path = eda_dir / "findings.json"
    summary_md_path = eda_dir / "summary.md"

    output_dict = eda_output.model_dump()
    tools.files.write_file("eda/findings.json", json.dumps(output_dict, indent=2))

    # Create summary.md (text-only)
    md_lines = [
        f"# EDA Findings: Project `{project_id}`",
        f"**Target Column:** `{target_col}` | **Task Type:** `{task_type}`",
        "",
        "## Executive Summary",
        eda_output.executive_summary,
        "",
        "## Leakage Risks",
    ]
    if eda_output.leakage_risks:
        for risk in eda_output.leakage_risks:
            md_lines.append(f"- [WARNING] {risk}")
    else:
        md_lines.append("- None detected.")

    md_lines.extend(["", "## Detailed Findings", "| Category | Finding | Evidence | Recommendation |", "|---|---|---|---|"])
    for f in eda_output.findings:
        clean_finding = str(f.finding).replace("|", "/")
        clean_ev = to_evidence_str(f.evidence).replace("|", "/")
        clean_rec = str(f.recommendation).replace("|", "/")
        md_lines.append(f"| {f.category} | {clean_finding} | {clean_ev} | {clean_rec} |")

    md_lines.extend(["", "## Suggested Feature Ideas"])
    for idea in eda_output.suggested_feature_ideas:
        md_lines.append(f"- {idea}")

    tools.files.write_file("eda/summary.md", "\n".join(md_lines))

    # Register artifacts
    art_json = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="eda_findings_json",
        path=str(findings_json_path),
        run_id=state.get("run_id"),
        version=dataset_version,
    )
    art_md = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="eda_summary_md",
        path=str(summary_md_path),
        run_id=state.get("run_id"),
        version=dataset_version,
    )

    artifacts_list = list(state.get("artifacts", []))
    artifacts_list.extend([art_json, art_md])

    elapsed = time.monotonic() - stage_start
    _log.info(
        "[EDA] Stage completed | project=%s run=%s | findings=%d | duration=%.2fs",
        project_id, run_id, len(eda_output.findings), elapsed,
    )

    return {
        "eda_findings": output_dict,
        "current_stage": "eda",
        "artifacts": artifacts_list,
        "status": "SUCCESS",
    }
