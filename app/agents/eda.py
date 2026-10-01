"""Exploratory Data Analysis (EDA) agent generating actionable tabular findings."""

import hashlib
import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.coder import CoderSubAgent
from app.agents.prompts import EDA_SYSTEM_PROMPT
from app.config import PROJECTS_DIR
from app.core.briefs import WORKER_BRIEF_LIMIT, build_worker_brief, check_and_log_budget
from app.core.llm_json import invoke_json
from app.core.model_router import ModelRouter
from app.core.run_memory import RunMemory
from app.core.schemas import WorkerConcern, WorkerReport
from app.core.state import EDAOutput, ProjectState, to_evidence_str
from app.tools.registry import ToolRegistry


def run_eda(
    state: ProjectState,
    router: ModelRouter,
    registry: ToolRegistry,
    brief: str | None = None,
) -> dict[str, Any]:
    """Runs adaptive, LLM-driven EDA using Coder for computation and producing structured findings."""
    stage_start = time.monotonic()
    project_id = state["project_id"]
    run_id = state.get("run_id", "unknown")
    _log.info("[EDA] Stage started | project=%s run=%s", project_id, run_id)

    tools = registry.get_tools_for_role("eda")
    coder_tools = registry.get_tools_for_role("coder")
    coder = CoderSubAgent(project_id, router, coder_tools.files, coder_tools.execution)

    dataset_version = state.get("dataset_version", "dataset_v1")
    dataset_path = state.get("split_train_path") or str(tools.dataset.get_dataset_path(dataset_version).resolve()).replace("\\", "/")
    target_col = state.get("target_column")
    task_type = state.get("task_type")
    task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)

    mem = RunMemory(project_id, run_id)
    if not brief:
        decision = state.get("supervisor_decision") or {
            "brief": {
                "objective": "Perform exploratory data analysis and detect leakage or skew",
                "focus_points": ["Target distribution", "Leakage detection", "Missing values and outliers"],
                "constraints": ["No plotting libraries", "Compute all statistics with code"],
                "mode": "standard",
            }
        }
        brief = build_worker_brief("eda", decision, mem, state)

    check_and_log_budget("EDA Brief", brief, WORKER_BRIEF_LIMIT)

    # 1. Ask Coder to compute statistical summaries and save to JSON
    eda_script_task = f"""Write a Python script to compute statistical EDA metrics for a tabular dataset.
Dataset file path: '{dataset_path}'
Target column: '{target_col}'
Task type: '{task_type_str}'

Requirements:
1. Load dataset using pandas (use pd.read_parquet if path ends with .parquet, else pd.read_csv).
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

    coder_res = coder.run_task(task_description=eda_script_task, context=brief)
    stdout = coder_res.get("full_stdout", "")

    extracted_stats = {}
    if "<JSON_OUTPUT>" in stdout and "</JSON_OUTPUT>" in stdout:
        try:
            raw_json = stdout.split("<JSON_OUTPUT>")[1].split("</JSON_OUTPUT>")[0].strip()
            extracted_stats = json.loads(raw_json)
        except Exception:
            extracted_stats = {"raw_output": stdout[-1500:]}
    else:
        extracted_stats = {"raw_output": stdout[-1500:]}

    # 2. Use LLM with invoke_json to synthesize findings
    llm = router.get_model("eda", temperature=0.0)

    stats_str = json.dumps(extracted_stats, indent=2)
    prompt = f"""Synthesize the EDA statistics into structured findings.

Worker Brief:
{brief}

Computed Statistical Summary:
{stats_str}

Ensure each finding provides human-readable string evidence (e.g. 'skewness=2.4, IQR=[10, 20]') and a specific action for feature engineering."""

    prompt_hash = hashlib.sha256(EDA_SYSTEM_PROMPT.encode()).hexdigest()[:12]
    _log.info("[EDA] Invoking LLM via invoke_json | prompt_version=%s | target=%s | task=%s", prompt_hash, target_col, task_type)

    eda_output, last_err_text, raw_text = invoke_json(
        llm,
        [
            SystemMessage(content=EDA_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ],
        EDAOutput,
        agent_name="eda",
    )

    if eda_output is None:
        err_msg = f"EDA structured output generation failed: {last_err_text or 'unknown error'}"
        _log.error("[EDA] %s", err_msg)
        return {
            "status": "FAILED",
            "error": err_msg,
            "current_stage": "eda",
            "report": {
                "status": "failed",
                "result_summary": err_msg[:300],
                "evidence": {},
                "concern": None,
                "suggestion": "Check dataset formatting or retry EDA with a revised brief.",
                "notebook": {
                    "step": state.get("step", 0),
                    "tried": "EDA structured analysis",
                    "outcome": "failed",
                    "lesson": "Failed to obtain valid EDAOutput from LLM",
                    "score_impact": None,
                    "errors": [err_msg[:200]],
                },
                "artifacts": [],
            },
        }

    # 3. Handle targeted second run merge into existing findings if applicable
    eda_dir = PROJECTS_DIR / project_id / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    findings_json_path = eda_dir / "findings.json"
    summary_md_path = eda_dir / "summary.md"

    output_dict = eda_output.model_dump()
    if findings_json_path.exists() and state.get("worker_runs", {}).get("eda", 0) > 0:
        try:
            with open(findings_json_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                old_findings = old_data.get("findings", [])
                existing_keys = {
                    (str(item.get("category", "")).lower(), str(item.get("finding", "")).strip().lower()[:50])
                    for item in old_findings
                }
                merged = list(old_findings)
                for new_f in output_dict.get("findings", []):
                    key = (str(new_f.get("category", "")).lower(), str(new_f.get("finding", "")).strip().lower()[:50])
                    if key not in existing_keys:
                        merged.append(new_f)
                        existing_keys.add(key)
                output_dict["findings"] = merged
        except Exception as e:
            _log.debug("[EDA] Failed merging with existing findings: %s", e)

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
    findings_count = len(output_dict.get("findings", []))
    _log.info(
        "[EDA] Stage completed | project=%s run=%s | findings=%d | duration=%.2fs",
        project_id, run_id, findings_count, elapsed,
    )

    # Check for potential evidence-backed worker concern
    concern_obj = None
    if eda_output.leakage_risks:
        concern_obj = {
            "claim": f"High risk of data leakage detected in {len(eda_output.leakage_risks)} columns",
            "evidence": str(eda_output.leakage_risks[0])[:150],
            "suggestion": "Drop suspected leakage features before model training",
        }

    worker_report = WorkerReport(
        status="ok",
        result_summary=f"EDA produced {findings_count} findings and {len(eda_output.leakage_risks)} leakage alerts. {eda_output.executive_summary[:120]}",
        evidence={"findings_count": findings_count, "leakage_risks_count": len(eda_output.leakage_risks)},
        concern=WorkerConcern(**concern_obj) if concern_obj else None,
        suggestion="Proceed to feature engineering addressing top high-priority findings." if not eda_output.leakage_risks else "Prune leakage columns in FE.",
        notebook={
            "step": state.get("step", 0),
            "tried": f"EDA analysis for {task_type_str}",
            "outcome": f"{findings_count} findings discovered",
            "lesson": f"Top leakage: {eda_output.leakage_risks[0] if eda_output.leakage_risks else 'None'}",
            "score_impact": None,
            "errors": [],
        },
        artifacts=[str(findings_json_path), str(summary_md_path)],
    )

    return {
        "eda_findings": output_dict,
        "current_stage": "eda",
        "artifacts": artifacts_list,
        "status": "SUCCESS",
        "report": worker_report.model_dump(),
    }
