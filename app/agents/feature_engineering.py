"""Tabular Feature Engineering agent creating leak-safe, reproducible pipelines (Phase 4.3)."""

import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.agents.coder import CoderSubAgent
from app.config import PROJECTS_DIR
from app.core.briefs import WORKER_BRIEF_LIMIT, build_worker_brief, check_and_log_budget
from app.core.model_router import ModelRouter
from app.core.run_memory import RunMemory
from app.core.schemas import WorkerReport
from app.core.state import ProjectState
from app.ml_harness.contract import (
    load_pipeline_from_script,
    postprocess_features,
)
from app.ml_harness.paths import RunPaths
from app.ml_harness.preprocess import make_basic_preprocessor
from app.tools.registry import ToolRegistry
from app.utils.logger import get_logger

_log = get_logger(__name__)


def _build_basic_script(profile: dict[str, Any]) -> str:
    """Generates the source code for features/feature_pipeline.py in basic mode."""
    prof_json = json.dumps(profile, default=str)
    return f'''"""Deterministic baseline feature engineering pipeline."""

import json
from app.ml_harness.preprocess import make_basic_preprocessor

_PROFILE_DATA = {prof_json}

def make_feature_pipeline():
    """Returns an unfitted ColumnTransformer conforming to column roles."""
    return make_basic_preprocessor(_PROFILE_DATA)
'''


def run_feature_engineering(
    state: ProjectState,
    router: ModelRouter,
    registry: ToolRegistry,
    brief: str | None = None,
) -> dict[str, Any]:
    """Designs and verifies an unfitted feature engineering pipeline without data leakage."""
    stage_start = time.monotonic()
    project_id = state["project_id"]
    run_id = state.get("run_id", "unknown")
    step = state.get("step", 0)
    iteration = state.get("iteration", 1)
    _log.info("[FE] Stage started | project=%s run=%s step=%d", project_id, run_id, step)

    tools = registry.get_tools_for_role("feature_engineering")
    coder_tools = registry.get_tools_for_role("coder")

    dataset_version = state.get("dataset_version", "dataset_v1")
    dataset_path_str = state.get("split_train_path")
    if not dataset_path_str:
        try:
            dataset_path_str = str(tools.dataset.get_dataset_path(dataset_version).resolve()).replace("\\", "/")
        except Exception:
            dataset_path_str = str((PROJECTS_DIR / project_id / "datasets" / f"{dataset_version}.csv").resolve()).replace("\\", "/")

    target_col = state.get("target_column")
    task_type = state.get("task_type")
    task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)

    mem = RunMemory(project_id, run_id)
    decision = state.get("supervisor_decision") or {}
    brief_dict = decision.get("brief", {}) if isinstance(decision.get("brief"), dict) else {}
    mode = brief_dict.get("mode") or ("basic" if step <= 1 else "new")

    if not brief:
        brief = build_worker_brief("fe", decision, mem, state)

    check_and_log_budget("FE Brief", brief, WORKER_BRIEF_LIMIT)

    run_paths = RunPaths(project_id, run_id)
    features_dir = PROJECTS_DIR / project_id / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load train sample (up to 200 rows) for contract verification
    dataset_path = Path(dataset_path_str)
    if not dataset_path.exists():
        err_msg = f"Dataset path does not exist: {dataset_path}"
        _log.error("[FE] %s", err_msg)
        return {
            "status": "FAILED",
            "current_stage": "feature_engineering",
            "error": err_msg,
            "report": WorkerReport(
                status="failed",
                result_summary=err_msg[:300],
                evidence={"error": err_msg},
                notebook={"step": step, "tried": "Load dataset sample", "outcome": "failed", "lesson": err_msg},
            ).model_dump(),
        }

    try:
        if str(dataset_path).endswith(".parquet"):
            df_full = pd.read_parquet(dataset_path)
            sample_df = df_full.head(200)
        else:
            sample_df = pd.read_csv(dataset_path, nrows=200)
    except Exception as exc:
        err_msg = f"Failed to read dataset sample: {exc}"
        _log.error("[FE] %s", err_msg)
        return {
            "status": "FAILED",
            "current_stage": "feature_engineering",
            "error": err_msg,
            "report": WorkerReport(
                status="failed",
                result_summary=err_msg[:300],
                evidence={"error": err_msg},
                notebook={"step": step, "tried": "Read CSV/Parquet", "outcome": "failed", "lesson": str(exc)},
            ).model_dump(),
        }

    sample_X = sample_df.drop(columns=[target_col], errors="ignore") if target_col else sample_df.copy()

    # Retrieve profile
    profile = state.get("profile_summary") or {}
    if not profile or not profile.get("columns"):
        prof_file = PROJECTS_DIR / project_id / "profile" / "profile.json"
        if prof_file.exists():
            try:
                with open(prof_file, "r", encoding="utf-8") as f:
                    profile = json.load(f)
            except Exception:
                pass

    pipeline_script_content = ""
    unfitted_pipeline = None
    kept_features: list[str] = []
    removed_features: list[str] = []

    # 2. Execution based on brief mode
    if mode == "basic":
        _log.info("[FE] Mode=basic: building deterministic preprocessor without LLM Coder call.")
        unfitted_pipeline = make_basic_preprocessor(profile)
        pipeline_script_content = _build_basic_script(profile)

        # Verify on sample
        try:
            sample_out, kept_features, removed_features = postprocess_features(
                unfitted_pipeline, sample_X, target_col=target_col
            )
            # Recreate unfitted instance for serialization
            unfitted_pipeline = make_basic_preprocessor(profile)
        except Exception as exc:
            err_msg = f"Basic preprocessor dry-run failed: {exc}"
            _log.error("[FE] %s", err_msg)
            return {
                "status": "FAILED",
                "current_stage": "feature_engineering",
                "error": err_msg,
                "report": WorkerReport(
                    status="failed",
                    result_summary=err_msg[:300],
                    evidence={"error": err_msg},
                    notebook={"step": step, "tried": "Basic preprocessor", "outcome": "failed", "lesson": str(exc)},
                ).model_dump(),
            }

    else:
        # Mode == "new" or "patch": use Coder SubAgent with contract verification
        _log.info("[FE] Mode=%s: Invoking Coder to write feature_pipeline.py", mode)
        coder = CoderSubAgent(project_id, router, coder_tools.files, coder_tools.execution)

        coder_prompt = f"""Write a COMPLETE, PRODUCTION-GRADE feature engineering script defining make_feature_pipeline().

Source dataset: '{dataset_path_str}'
Target column: '{target_col}'
Task type: '{task_type_str}'
Mode: '{mode}'

Worker Brief:
{brief}

MANDATORY CONTRACT RULES (The harness will verify these):
1. Define a function: def make_feature_pipeline() -> sklearn Pipeline / ColumnTransformer
2. Return a FRESH, UNFITTED sklearn-compatible transformer.
3. NEVER call train_test_split.
4. NEVER transform the full dataset.
5. Use safe transformers from app.ml_harness.transformers:
   (DropColumns, DateParts, CyclicEncoder, PairOps, LogPower, ClipQuantiles, FrequencyEncoder, TargetEncoderCV, RollingLag)
   or subclass SafeTransformer from app.ml_harness.base.
6. The pipeline must NEVER output the target column '{target_col}'.
7. Ensure all features produce no NaNs and are picklable with joblib.
"""
        max_attempts = 4
        last_error = ""

        for attempt in range(max_attempts):
            task_desc = coder_prompt
            if last_error:
                task_desc += f"\n\nPREVIOUS ATTEMPT FAILED THE HARNESS CONTRACT CHECK:\n{last_error}\nFix the pipeline to resolve this error."

            coder_res = coder.run_task(task_description=task_desc, context=brief)
            if coder_res["status"] == "FAILED":
                last_error = coder_res.get("error", "Script generation failed")
                continue

            candidate_code = coder_res.get("executed_code", "")
            # Temporarily write to features_dir to test loading
            temp_script = features_dir / "temp_feature_pipeline.py"
            temp_script.write_text(candidate_code, encoding="utf-8")

            try:
                candidate_pipeline = load_pipeline_from_script(temp_script)
                sample_out, kept_features, removed_features = postprocess_features(
                    candidate_pipeline, sample_X, target_col=target_col
                )
                if sample_out.isna().any().any():
                    nan_cols = sample_out.columns[sample_out.isna().any()].tolist()
                    raise ValueError(f"Transformed output contains NaNs in columns: {nan_cols}")

                # Test pickle safety
                test_pkl = features_dir / "test_pipe.pkl"
                fresh_pipe = load_pipeline_from_script(temp_script)
                joblib.dump(fresh_pipe, test_pkl)
                joblib.load(test_pkl)
                test_pkl.unlink(missing_ok=True)

                # Verification succeeded
                pipeline_script_content = candidate_code
                unfitted_pipeline = load_pipeline_from_script(temp_script)
                temp_script.unlink(missing_ok=True)
                break
            except Exception as verify_exc:
                last_error = f"Contract verification error: {verify_exc}"
                _log.warning("[FE] Attempt %d failed contract check: %s", attempt + 1, last_error)
                temp_script.unlink(missing_ok=True)

        if unfitted_pipeline is None:
            err_msg = f"Feature pipeline failed contract checks after {max_attempts} attempts: {last_error}"
            _log.error("[FE] %s", err_msg)
            return {
                "status": "FAILED",
                "current_stage": "feature_engineering",
                "error": err_msg,
                "report": WorkerReport(
                    status="failed",
                    result_summary=err_msg[:300],
                    evidence={"error": err_msg},
                    notebook={"step": step, "tried": f"FE {mode} mode", "outcome": "failed", "lesson": last_error[:200]},
                ).model_dump(),
            }

    # 3. Save artifacts (both run-scoped and project-scoped)
    # Save script
    run_paths.feature_pipeline_py.write_text(pipeline_script_content, encoding="utf-8")
    (features_dir / "feature_pipeline.py").write_text(pipeline_script_content, encoding="utf-8")

    # Save unfitted pipeline
    joblib.dump(unfitted_pipeline, run_paths.feature_pipeline_pkl)
    joblib.dump(unfitted_pipeline, features_dir / "feature_pipeline.pkl")

    # Generate and save schema
    schema_dict = {str(col): "float64" for col in kept_features}
    run_paths.feature_schema_json.write_text(json.dumps(schema_dict, indent=2), encoding="utf-8")
    (features_dir / "feature_schema.json").write_text(json.dumps(schema_dict, indent=2), encoding="utf-8")

    # Save Markdown report
    feature_cnt = len(kept_features)
    report_lines = [
        f"# Feature Engineering Report: Project `{project_id}` (Version v{iteration})",
        "- **Pipeline Script:** `features/feature_pipeline.py`",
        f"- **Target Column:** `{target_col}`",
        f"- **Total Engineered Features:** {feature_cnt}",
        "",
        "## Kept Features Sample",
        ", ".join(kept_features[:30]),
    ]
    if removed_features:
        report_lines.extend(["", "## Pruned / Leakage-Safe Dropped Features", ", ".join(removed_features)])

    report_content = "\n".join(report_lines)
    (features_dir / "feature_report.md").write_text(report_content, encoding="utf-8")
    (run_paths.reports_dir / "feature_report.md").write_text(report_content, encoding="utf-8")

    # Register artifacts
    art_pipe = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="feature_pipeline_py",
        path=str(run_paths.feature_pipeline_py),
        run_id=run_id,
        version=f"feat_v{iteration}",
    )
    art_pkl = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="feature_pipeline_pkl",
        path=str(run_paths.feature_pipeline_pkl),
        run_id=run_id,
        version=f"feat_v{iteration}",
    )
    artifacts_list = list(state.get("artifacts", []))
    artifacts_list.extend([art_pipe, art_pkl])

    output_summary = {
        "version": f"feat_v{iteration}",
        "feature_count": feature_cnt,
        "kept_features": kept_features[:20],
        "removed_features": removed_features,
        "pipeline_file": str(run_paths.feature_pipeline_py),
        "schema_file": str(run_paths.feature_schema_json),
    }

    worker_report = WorkerReport(
        status="ok",
        result_summary=f"FE generated {feature_cnt} clean features in v{iteration} ({mode} mode). Unfitted pipeline serialized.",
        evidence={
            "feature_count": feature_cnt,
            "version": f"v{iteration}",
            "mode": mode,
            "pruned_count": len(removed_features),
            "features_sample": kept_features[:8],
        },
        concern=None,
        suggestion="Evaluate candidate models with cross-validation harness.",
        notebook={
            "step": step,
            "tried": f"Feature pipeline ({mode} mode, v{iteration})",
            "outcome": f"{feature_cnt} features produced (contract verified)",
            "lesson": "Unfitted pipeline serialized without full data transformation.",
            "score_impact": None,
            "errors": [],
        },
        artifacts=[str(run_paths.feature_pipeline_py), str(run_paths.feature_pipeline_pkl)],
    )

    elapsed = time.monotonic() - stage_start
    _log.info("[FE] Stage completed | project=%s run=%s | features=%d | duration=%.2fs", project_id, run_id, feature_cnt, elapsed)

    return {
        "feature_summary": output_summary,
        "current_stage": "feature_engineering",
        "artifacts": artifacts_list,
        "status": "SUCCESS",
        "report": worker_report.model_dump(),
    }
