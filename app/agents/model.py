"""Scikit-Learn and Gradient Boosting Modeling Agent (Phase 2A).

LLM plans candidate models and hyperparameters dynamically using KnowledgeBoard digest.
Coder executes the candidates in one script via harness cv_evaluate using the current FE pipeline.
"""

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from app.agents.coder import CoderSubAgent
from app.agents.prompts import TASK_PLAYBOOKS
from app.config import PROJECTS_DIR
from app.core.briefs import WORKER_BRIEF_LIMIT, build_worker_brief, check_and_log_budget
from app.core.knowledge import KnowledgeBoard
from app.core.model_router import ModelRouter
from app.core.run_memory import RunMemory
from app.core.schemas import WorkerReport
from app.core.state import ProjectState
from app.core.versions import update_best
from app.ml_harness.cv import get_metric_direction
from app.ml_harness.libs import get_available_libs
from app.ml_harness.noise import is_significant_gain
from app.ml_harness.paths import RunPaths
from app.tools.registry import ToolRegistry
from app.utils.logger import get_logger

_log = get_logger(__name__)


class ModelExperimentItem(BaseModel):
    """A proposed model candidate experiment."""
    family: str = Field(description="Estimator class name, e.g. RandomForestClassifier, HistGradientBoostingClassifier, LogisticRegression, LGBMClassifier, XGBClassifier")
    params_or_search_space: dict[str, Any] = Field(default_factory=dict, description="Hyperparameters or search space for the estimator")
    reason: str = Field(default="", description="Why this candidate and hyperparameters were chosen based on EDA findings and prior results")


class ModelExperimentPlan(BaseModel):
    """Experiment plan containing candidate models to evaluate."""
    plan: list[ModelExperimentItem] = Field(default_factory=list, description="List of proposed candidate models")


def normalize_params(p: Any) -> str:
    """Normalizes estimator parameter representation for consistent duplicate matching."""
    if isinstance(p, dict):
        return json.dumps(p, sort_keys=True)
    if not p:
        return "{}"
    if isinstance(p, str):
        try:
            parsed = json.loads(p)
            if isinstance(parsed, dict):
                return json.dumps(parsed, sort_keys=True)
        except Exception:
            pass
        return p.strip()
    return str(p)


def filter_duplicate_candidates(
    plan: list[ModelExperimentItem],
    models_tried: list[Any],
) -> tuple[list[ModelExperimentItem], list[str]]:
    """Rejects any plan item whose family + params already exist in models_tried (code check)."""
    seen: set[tuple[str, str]] = set()
    for m in models_tried:
        family = (m.family if hasattr(m, "family") else m.get("family", "")).lower().strip()
        params = m.params_summary if hasattr(m, "params_summary") else m.get("params_summary", "")
        norm_params = normalize_params(params)
        seen.add((family, norm_params))

    kept: list[ModelExperimentItem] = []
    rejected: list[str] = []
    for item in plan:
        f = item.family.lower().strip()
        p = normalize_params(item.params_or_search_space)
        if (f, p) in seen:
            rejected.append(f"{item.family} with params {p}")
        else:
            kept.append(item)
            seen.add((f, p))

    return kept, rejected


def parse_and_validate_plan(raw_text: str) -> list[ModelExperimentItem]:
    """Parses and validates LLM experiment plan JSON with support for both list and wrapped dict formats."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    cleaned = cleaned.removesuffix("```").strip()

    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        # Search for first '[' or '{'
        idx_obj = cleaned.find("{")
        idx_arr = cleaned.find("[")
        if idx_obj == -1 and idx_arr == -1:
            raise ValueError("No JSON object or array found in LLM response.")

        if idx_arr != -1 and (idx_obj == -1 or idx_arr < idx_obj):
            depth, in_str, escape = 0, False, False
            for i in range(idx_arr, len(cleaned)):
                ch = cleaned[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    if in_str:
                        escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            data = json.loads(cleaned[idx_arr : i + 1])
                            break
        else:
            depth, in_str, escape = 0, False, False
            for i in range(idx_obj, len(cleaned)):
                ch = cleaned[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    if in_str:
                        escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            data = json.loads(cleaned[idx_obj : i + 1])
                            break

    if data is None:
        raise ValueError("Could not parse balanced JSON from LLM response.")

    items: list[ModelExperimentItem] = []
    if isinstance(data, list):
        items = [ModelExperimentItem.model_validate(x) for x in data]
    elif isinstance(data, dict):
        if "plan" in data and isinstance(data["plan"], list):
            items = [ModelExperimentItem.model_validate(x) for x in data["plan"]]
        elif "candidates" in data and isinstance(data["candidates"], list):
            items = [ModelExperimentItem.model_validate(x) for x in data["candidates"]]
        elif "family" in data:
            items = [ModelExperimentItem.model_validate(data)]
        else:
            raise ValueError(f"JSON object missing 'plan' key. Keys found: {list(data.keys())}")
    else:
        raise ValueError(f"Expected JSON array or object, got: {type(data)}")

    if not items:
        raise ValueError("Plan contains 0 candidates.")
    return items


def generate_experiment_plan(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
) -> tuple[list[ModelExperimentItem] | None, str]:
    """Generates and validates model experiment plan with exactly 1 repair retry on error."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Attempt 1
    try:
        response = llm.invoke(messages)
        raw_text = str(getattr(response, "content", ""))
        items = parse_and_validate_plan(raw_text)
        return items, ""
    except Exception as err1:
        from app.core.model_router import LLMRateLimitError
        if isinstance(err1, LLMRateLimitError):
            raise
        first_err = str(err1)
        _log.warning("[MODEL] Plan validation failed on attempt 1: %s. Retrying once with repair prompt.", first_err)

    # Attempt 2: Repair retry
    repair_prompt = (
        f"Your previous response failed validation with error:\n{first_err}\n"
        f"Fix the error and output ONLY valid JSON conforming to the schema:\n"
        f'{{"plan": [{{"family": "EstimatorName", "params_or_search_space": {{...}}, "reason": "..."}}]}}'
    )
    messages.append(HumanMessage(content=repair_prompt))
    try:
        response = llm.invoke(messages)
        raw_text = str(getattr(response, "content", ""))
        items = parse_and_validate_plan(raw_text)
        return items, ""
    except Exception as err2:
        from app.core.model_router import LLMRateLimitError
        if isinstance(err2, LLMRateLimitError):
            raise
        final_err = f"Plan validation failed after repair retry: {err2}"
        _log.error("[MODEL] %s", final_err)
        return None, final_err


def run_modeling(
    state: ProjectState,
    router: ModelRouter,
    registry: ToolRegistry,
    brief: str | None = None,
) -> dict[str, Any]:
    """Evaluates candidate models inside cross-validation folds against noise floor."""
    stage_start = time.monotonic()
    project_id = state["project_id"]
    run_id = state.get("run_id", "unknown")
    step = state.get("step", 0)
    iteration = state.get("iteration", 1)
    _log.info("[MODEL] Stage started | project=%s run=%s step=%d iter=%s", project_id, run_id, step, iteration)

    tools = registry.get_tools_for_role("model")
    coder_tools = registry.get_tools_for_role("coder")

    target_col = state.get("target_column")
    task_type = state.get("task_type")
    task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type or "binary_classification")
    from app.core.state import get_default_metric
    target_metric = (state.get("target_metric") or get_default_metric(task_type)).lower()
    direction = get_metric_direction(target_metric)

    available_libs = get_available_libs()
    available_libs_list = sorted([k for k, v in available_libs.items() if v] if isinstance(available_libs, dict) else list(available_libs))

    mem = RunMemory(project_id, run_id)
    kb = KnowledgeBoard(run_dir=mem.run_dir)
    run_paths = RunPaths(project_id, run_id)
    models_dir = PROJECTS_DIR / project_id / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    decision = state.get("supervisor_decision") or {}
    brief_dict = decision.get("brief", {}) if isinstance(decision.get("brief"), dict) else {}
    mode = brief_dict.get("mode") or ("baseline" if (step <= 2 and not mem.get_best()) else "new")

    if not brief:
        brief = build_worker_brief("model", decision, mem, state)

    check_and_log_budget("Model Brief", brief, WORKER_BRIEF_LIMIT)

    # 1. Resolve dataset path
    dataset_version = state.get("dataset_version", "dataset_v1")
    dataset_path_str = state.get("split_train_path")
    if not dataset_path_str or not Path(dataset_path_str).exists():
        if tools.dataset is not None:
            try:
                p = tools.dataset.get_dataset_path(dataset_version)
                if p.exists():
                    dataset_path_str = str(p.resolve()).replace("\\", "/")
            except Exception:
                pass

        if not dataset_path_str or not Path(dataset_path_str).exists():
            candidates = [
                PROJECTS_DIR / project_id / "datasets" / dataset_version / "data.csv",
                PROJECTS_DIR / project_id / "datasets" / dataset_version / "data.parquet",
                PROJECTS_DIR / project_id / "datasets" / f"{dataset_version}.csv",
                PROJECTS_DIR / project_id / "datasets" / f"{dataset_version}.parquet",
                PROJECTS_DIR / project_id / "datasets" / "dataset_v1" / "data.csv",
            ]
            for cand in candidates:
                if cand.exists():
                    dataset_path_str = str(cand.resolve()).replace("\\", "/")
                    break

            if not dataset_path_str or not Path(dataset_path_str).exists():
                d_dir = PROJECTS_DIR / project_id / "datasets"
                if d_dir.exists():
                    all_data_files = list(d_dir.glob("**/*.csv")) + list(d_dir.glob("**/*.parquet"))
                    if all_data_files:
                        dataset_path_str = str(all_data_files[0].resolve()).replace("\\", "/")

    dataset_path = Path(dataset_path_str or "")
    if not dataset_path.exists():
        err_msg = f"Dataset path not found: {dataset_path}"
        _log.error("[MODEL] %s", err_msg)
        return {
            "status": "FAILED",
            "current_stage": "model",
            "error": err_msg,
            "report": WorkerReport(
                status="failed",
                result_summary=err_msg[:300],
                evidence={"error": err_msg},
                notebook={"step": step, "tried": "Load dataset", "outcome": "failed", "lesson": err_msg},
            ).model_dump(),
        }

    # Validate target column presence in headers
    try:
        if str(dataset_path).endswith(".parquet"):
            import pyarrow.parquet as pq
            headers = pq.read_schema(dataset_path).names
        else:
            headers = pd.read_csv(dataset_path, nrows=0).columns.tolist()
    except Exception as exc:
        err_msg = f"Failed to inspect dataset headers: {exc}"
        _log.error("[MODEL] %s", err_msg)
        return {
            "status": "FAILED",
            "current_stage": "model",
            "error": err_msg,
            "report": WorkerReport(
                status="failed",
                result_summary=err_msg[:300],
                evidence={"error": err_msg},
                notebook={"step": step, "tried": "Read headers", "outcome": "failed", "lesson": str(exc)},
            ).model_dump(),
        }

    if target_col not in headers:
        err_msg = f"Target column '{target_col}' not found in dataset headers: {headers}"
        _log.error("[MODEL] %s", err_msg)
        return {
            "status": "FAILED",
            "current_stage": "model",
            "error": err_msg,
            "report": WorkerReport(
                status="failed",
                result_summary=err_msg[:300],
                evidence={"error": err_msg},
                notebook={"step": step, "tried": "Target column check", "outcome": "failed", "lesson": err_msg},
            ).model_dump(),
        }

    # 2. Determine split strategy
    split_strategy = state.get("split_strategy")
    if not split_strategy:
        profile_dict = state.get("profile_summary", {})
        if profile_dict.get("time_signal") or profile_dict.get("row_order_meaningful"):
            split_strategy = "time"
        elif "classification" in task_type_str:
            split_strategy = "stratified"
        else:
            split_strategy = "kfold"

    # 3. STEP 1: LLM writes experiment plan as JSON [{family, params_or_search_space, reason}]
    llm = router.get_model("model", temperature=0.2)
    model_digest = kb.digest(role="model")

    system_prompt = (
        "You are an expert Tabular Machine Learning Modeler.\n"
        "Your task is to propose an experiment plan with 2 to 3 candidate models to evaluate using cross-validation.\n"
        f"Available libraries installed in the environment: {available_libs_list}.\n"
        "RULES:\n"
        f"1. You must ONLY choose model families from installed libraries: {available_libs_list}.\n"
        "   Allowed examples: RandomForestClassifier, HistGradientBoostingClassifier, LogisticRegression, LGBMClassifier, XGBClassifier, CatBoostClassifier.\n"
        "2. Do NOT propose estimators from packages that are not installed.\n"
        "3. Incorporate relevant EDA findings, previous models tried, and judge advice from the context.\n"
        "4. Output MUST be valid JSON with key 'plan':\n"
        '{\n  "plan": [\n    {"family": "EstimatorClassName", "params_or_search_space": {"param1": val1}, "reason": "why chosen"}\n  ]\n}'
    )

    user_prompt = f"""Problem Context:
- Project: {project_id} | Run: {run_id} | Step: {step}
- Target column: '{target_col}'
- Task type: {task_type_str}
- Target metric: {target_metric} (direction: {direction})
- Split strategy: {split_strategy}
- Mode: {mode}

Knowledge Board Digest:
{model_digest}

Supervisor Brief:
{brief}

Propose 2-3 candidate models as a JSON object with 'plan': [{{"family": "...", "params_or_search_space": {{...}}, "reason": "..."}}].
"""

    plan_items, plan_err = generate_experiment_plan(llm, system_prompt, user_prompt)
    if plan_err or not plan_items:
        fail_err = f"Failed to generate valid model experiment plan: {plan_err}"
        _log.error("[MODEL] %s", fail_err)
        return {
            "status": "FAILED",
            "current_stage": "model",
            "error": fail_err,
            "report": WorkerReport(
                status="failed",
                result_summary=fail_err[:300],
                evidence={"error": fail_err},
                notebook={"step": step, "tried": "Plan generation", "outcome": "failed", "lesson": fail_err},
            ).model_dump(),
        }

    _log.info("[MODEL] Generated experiment plan with %d items: %s", len(plan_items), [p.family for p in plan_items])

    # 4. Reject duplicates against models_tried (code check)
    valid_items, rejected = filter_duplicate_candidates(plan_items, kb.models_tried)
    if rejected:
        _log.info("[MODEL] Rejected %d duplicate candidates: %s", len(rejected), rejected)

    if not valid_items:
        err_msg = f"All planned candidates were duplicates of already evaluated models: {rejected}"
        _log.warning("[MODEL] %s", err_msg)
        return {
            "status": "FAILED",
            "current_stage": "model",
            "error": err_msg,
            "report": WorkerReport(
                status="failed",
                result_summary=err_msg[:300],
                evidence={"error": err_msg, "rejected": rejected},
                notebook={"step": step, "tried": "Duplicate check", "outcome": "failed", "lesson": err_msg},
            ).model_dump(),
        }

    # 5. STEP 2: Coder runs the plan in ONE script using harness cv_evaluate with the FE pipeline
    pipeline_script = run_paths.feature_pipeline_py if run_paths.feature_pipeline_py.exists() else (PROJECTS_DIR / project_id / "features" / "feature_pipeline.py")
    pipeline_script_str = str(pipeline_script.resolve()).replace("\\", "/") if pipeline_script.exists() else ""
    profile_data = state.get("profile_summary", {})

    current_ver = state.get("current_version") or f"v{iteration}"
    ver_dir = run_paths.version_dir(current_ver)
    ver_dir.mkdir(parents=True, exist_ok=True)
    eval_results_json = models_dir / f"eval_results_step_{step}.json"

    candidates_spec = [item.model_dump() for item in valid_items]

    coder_task = f"""Write and run a complete, self-contained Python script to train and evaluate {len(valid_items)} candidate model(s) using cv_evaluate.

CANDIDATES TO EVALUATE:
{json.dumps(candidates_spec, indent=2)}

DATASET & TARGET:
- Dataset path: r"{dataset_path_str}"
- Target column: '{target_col}'
- Task type: '{task_type_str}'
- Target metric: '{target_metric}' (direction: '{direction}')
- Split strategy: '{split_strategy}'

FEATURE PIPELINE:
{f'Pipeline script: r"{pipeline_script_str}"' if pipeline_script_str else 'No custom script; use make_basic_preprocessor'}

EVALUATION HARNESS & IMPORTS:
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from app.ml_harness.cv import cv_evaluate

If pipeline script exists:
    from app.ml_harness.contract import load_pipeline_from_script
    pipeline_factory = lambda: load_pipeline_from_script(Path(r"{pipeline_script_str}"))
Else:
    from app.ml_harness.preprocess import make_basic_preprocessor
    pipeline_factory = lambda: make_basic_preprocessor({json.dumps(profile_data)})

CRITICAL SCRIPT STEPS:
1. Load dataset from r"{dataset_path_str}" (handle .parquet or .csv).
2. X = df.drop(columns=['{target_col}'], errors='ignore')
   y = df['{target_col}']
   If '{task_type_str}'.startswith('binary') and (y.dtype == object or y.dtype == bool or set(y.dropna().unique()) <= {{'0', '1', 'yes', 'no', 'true', 'false', 'True', 'False'}}):
       unique_vals = list(y.dropna().unique())
       if len(unique_vals) == 2:
           val_map = {{unique_vals[0]: 0, unique_vals[1]: 1}}
           if str(unique_vals[0]).lower() in ('yes', 'true', '1', 'positive'):
               val_map = {{unique_vals[0]: 1, unique_vals[1]: 0}}
           y = y.map(val_map).astype(int)

3. For each candidate in CANDIDATES:
   - Instantiate the estimator based on 'family' and 'params_or_search_space'. Always set random_state=42 (or random_seed=42 for CatBoost) if accepted.
   - Run cv_evaluate:
       res = cv_evaluate(
           pipeline_factory=pipeline_factory,
           estimator=est,
           X=X,
           y=y,
           task_type='{task_type_str}',
           metric='{target_metric}',
           split_strategy='{split_strategy}',
           seed=42,
       )
   - Store candidate evaluation metrics.

4. Select the best candidate based on cv_mean ({'maximum' if direction == 'higher' else 'minimum'}).
5. Fit the full pipeline on all data (X, y):
   full_pipe = Pipeline([('features', pipeline_factory()), ('model', clone(best_est))])
   full_pipe.fit(X, y)

6. Save artifacts:
   import joblib
   joblib.dump(full_pipe, r"{run_paths.best_model_pkl}")
   joblib.dump(full_pipe, r"{models_dir / 'best_model.pkl'}")
   ver_p = Path(r"{ver_dir}")
   ver_p.mkdir(parents=True, exist_ok=True)
   joblib.dump(full_pipe, ver_p / "pipeline.pkl")
   np.save(ver_p / "oof.npy", best_res["oof_predictions"])

7. Save evaluation summary to JSON at r"{eval_results_json}":
   with open(r"{eval_results_json}", "w", encoding="utf-8") as f:
       json.dump({{
           "candidates": evaluated_candidates,
           "best_name": best_name,
           "best_score": float(best_res["cv_mean"]),
           "best_std": float(best_res["cv_std"]),
           "best_fold_scores": [float(s) for s in best_res["fold_scores"]],
           "best_train_score": float(best_res["train_score"]),
           "best_train_val_gap": float(best_res["train_val_gap"]),
           "best_params": best_params,
       }}, f, indent=2)
print("EVALUATION_COMPLETE")
"""

    coder = CoderSubAgent(project_id, router, coder_tools.files, coder_tools.execution)
    coder_res = coder.run_task(
        task_description=coder_task,
        context={
            "stage": "model",
            "run_id": run_id,
            "step": step,
            "candidates_count": len(valid_items),
        },
    )

    if not eval_results_json.exists():
        err_msg = f"Coder failed to execute model experiment plan: {coder_res.get('error', 'eval_results.json was not created')}"
        _log.error("[MODEL] %s", err_msg)
        return {
            "status": "FAILED",
            "current_stage": "model",
            "error": err_msg,
            "report": WorkerReport(
                status="failed",
                result_summary=err_msg[:300],
                evidence={"error": err_msg, "attempts": coder_res.get("attempts", 0)},
                notebook={"step": step, "tried": "Model evaluation via Coder", "outcome": "failed", "lesson": err_msg},
            ).model_dump(),
        }

    try:
        with open(eval_results_json, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    except Exception as exc:
        err_msg = f"Failed to parse evaluation results JSON: {exc}"
        _log.error("[MODEL] %s", err_msg)
        return {
            "status": "FAILED",
            "current_stage": "model",
            "error": err_msg,
            "report": WorkerReport(
                status="failed",
                result_summary=err_msg[:300],
                evidence={"error": err_msg},
                notebook={"step": step, "tried": "Parse results", "outcome": "failed", "lesson": str(exc)},
            ).model_dump(),
        }

    evaluated_candidates = eval_data.get("candidates", [])
    if not evaluated_candidates:
        err_msg = "Model evaluation returned 0 evaluated candidates."
        _log.error("[MODEL] %s", err_msg)
        return {
            "status": "FAILED",
            "current_stage": "model",
            "error": err_msg,
            "report": WorkerReport(
                status="failed",
                result_summary=err_msg,
                evidence={"error": err_msg},
                notebook={"step": step, "tried": "Evaluate candidates", "outcome": "failed", "lesson": err_msg},
            ).model_dump(),
        }

    best_name = eval_data["best_name"]
    best_score = float(eval_data["best_score"])
    best_std = float(eval_data["best_std"])
    best_folds = [float(s) for s in eval_data.get("best_fold_scores", [])]
    best_train_score = float(eval_data.get("best_train_score", 0.0))
    best_train_val_gap = float(eval_data.get("best_train_val_gap", 0.0))
    best_params = eval_data.get("best_params", {})

    # 6. Log every evaluated candidate to KnowledgeBoard
    for cand in evaluated_candidates:
        c_name = cand.get("name", "model")
        kb.add_model_tried({
            "id": f"{c_name}_{step}_{int(time.time()*1000)%10000}",
            "family": c_name,
            "params_summary": json.dumps(cand.get("params", {}), sort_keys=True) if isinstance(cand.get("params"), dict) else str(cand.get("params", "")),
            "cv_mean": float(cand.get("cv_mean", 0.0)),
            "cv_std": float(cand.get("cv_std", 0.0)),
            "train_score": float(cand.get("train_score", 0.0)) if cand.get("train_score") is not None else None,
            "gap": float(cand.get("train_val_gap", 0.0)) if cand.get("train_val_gap") is not None else None,
            "fit_time": None,
            "outcome": "best" if c_name == best_name else "tested",
        })

    # 7. Significance comparison vs previous best
    prev_best = mem.get_best() or {}
    prev_best_scores = prev_best.get("fold_scores", [])
    prev_best_score = prev_best.get("score")
    prev_best_name = prev_best.get("model_name", "baseline")

    if mode == "baseline" or prev_best_score is None or not prev_best_scores:
        # Baseline run: fix C8 by NOT marking significant=True
        is_sig = None
        mean_diff = 0.0
        nf = 0.0
    elif len(prev_best_scores) == len(best_folds):
        is_sig, mean_diff, nf = is_significant_gain(best_folds, prev_best_scores, direction=direction)
    else:
        mean_diff = best_score - float(prev_best_score)
        is_sig = (mean_diff > 0) if direction == "higher" else (mean_diff < 0)
        nf = 0.0

    # 8. Update best version in versions.py (Fix C9: single writer)
    if is_sig is not False:  # None (baseline) or True (significant improvement)
        update_best(
            project_id=project_id,
            run_id=run_id,
            version_id=current_ver,
            score=best_score,
            metric=target_metric,
            step=step,
            state=state,
        )

    summary_lines = [f"{c.get('name', 'cand')}: {float(c.get('cv_mean', 0.0)):.4f}±{float(c.get('cv_std', 0.0)):.4f}" for c in evaluated_candidates]
    result_summary = f"{mode.capitalize()} complete. Best: {best_name} ({target_metric}={best_score:.4f}±{best_std:.4f}). [{', '.join(summary_lines)}]"

    worker_report = WorkerReport(
        status="ok" if (is_sig is not False) else "no_gain",
        result_summary=result_summary[:300],
        evidence={
            "best_model_name": best_name,
            "score": best_score,
            "mean": best_score,
            "cv_mean": best_score,
            "std": best_std,
            "cv_std": best_std,
            "train_score": best_train_score,
            "train_val_gap": best_train_val_gap,
            "noise_floor": nf,
            "significant": is_sig,
            "delta": mean_diff,
            "target_metric": target_metric,
            "candidates_count": len(evaluated_candidates),
        },
        concern=None,
        suggestion="Evaluate pipeline generalization with Judge." if (is_sig is not False) else "Consider domain-specific features in FE.",
        notebook={
            "step": step,
            "tried": f"Candidates ({best_name})",
            "outcome": f"Best {best_name} {target_metric}={best_score:.4f}",
            "lesson": f"{'Baseline established' if mode == 'baseline' else ('Exceeded noise floor' if is_sig else 'Within noise floor')}",
            "score_impact": best_score,
            "errors": [],
        },
        artifacts=[str(run_paths.best_model_pkl)],
    )

    return {
        "model_summary": {
            "validation_strategy": f"5-Fold CV ({split_strategy})",
            "target_metric": target_metric,
            "best_model_name": best_name,
            "best_score": best_score,
            "summary": result_summary,
        },
        "target_metric": target_metric,
        "best_metric_value": best_score if (is_sig is not False) else prev_best_score,
        "best_version": current_ver if (is_sig is not False) else state.get("best_version"),
        "current_stage": "model",
        "status": "SUCCESS",
        "report": worker_report.model_dump(),
    }
