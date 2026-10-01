"""Scikit-Learn Modeling Agent with Paired CV and Noise Floor Evaluation (Phase 4.4)."""

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from app.agents.prompts import TASK_PLAYBOOKS
from app.config import PROJECTS_DIR
from app.core.briefs import WORKER_BRIEF_LIMIT, build_worker_brief, check_and_log_budget
from app.core.model_router import ModelRouter
from app.core.run_memory import RunMemory
from app.core.schemas import WorkerReport
from app.core.state import ProjectState
from app.ml_harness.contract import load_pipeline_from_script
from app.ml_harness.cv import cv_evaluate, get_metric_direction
from app.ml_harness.libs import get_available_libs
from app.ml_harness.noise import is_significant_gain
from app.ml_harness.paths import RunPaths
from app.ml_harness.preprocess import make_basic_preprocessor
from app.tools.registry import ToolRegistry
from app.utils.logger import get_logger

_log = get_logger(__name__)


def _build_default_candidates(
    task_type: str,
    available_libs: list[str] | set[str] | dict[str, Any],
    seed: int = 42,
) -> list[tuple[str, BaseEstimator]]:
    """Builds a diverse set of competitive model estimators based on available libraries."""
    is_classification = "classification" in task_type
    candidates: list[tuple[str, BaseEstimator]] = []

    if isinstance(available_libs, dict):
        libs_set = {k for k, v in available_libs.items() if v}
    else:
        libs_set = set(available_libs) if available_libs else set()

    if is_classification:
        candidates.append(("HistGradientBoosting_tuned", HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=31, l2_regularization=0.1, random_state=seed
        )))
        candidates.append(("RandomForest_100", RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_split=5, random_state=seed, n_jobs=-1
        )))
        candidates.append(("ExtraTrees_100", ExtraTreesClassifier(
            n_estimators=100, max_depth=12, min_samples_split=5, random_state=seed, n_jobs=-1
        )))

        # LightGBM if installed
        if "lightgbm" in libs_set:
            try:
                import lightgbm as lgb
                candidates.append(("LightGBM", lgb.LGBMClassifier(
                    n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1
                )))
            except Exception:
                pass

        # XGBoost if installed
        if "xgboost" in libs_set:
            try:
                import xgboost as xgb
                candidates.append(("XGBoost", xgb.XGBClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.05, random_state=seed, eval_metric="logloss", verbosity=0
                )))
            except Exception:
                pass

        # CatBoost if installed
        if "catboost" in libs_set:
            try:
                import catboost as cb
                candidates.append(("CatBoost", cb.CatBoostClassifier(
                    iterations=150, learning_rate=0.05, depth=5, random_seed=seed, verbose=0
                )))
            except Exception:
                pass

    else:
        # Regression
        candidates.append(("HistGradientBoosting_tuned", HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=31, l2_regularization=0.1, random_state=seed
        )))
        candidates.append(("RandomForest_100", RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_split=5, random_state=seed, n_jobs=-1
        )))
        candidates.append(("ExtraTrees_100", ExtraTreesRegressor(
            n_estimators=100, max_depth=12, min_samples_split=5, random_state=seed, n_jobs=-1
        )))

        if "lightgbm" in libs_set:
            try:
                import lightgbm as lgb
                candidates.append(("LightGBM", lgb.LGBMRegressor(
                    n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1
                )))
            except Exception:
                pass

        if "xgboost" in libs_set:
            try:
                import xgboost as xgb
                candidates.append(("XGBoost", xgb.XGBRegressor(
                    n_estimators=150, max_depth=5, learning_rate=0.05, random_state=seed, verbosity=0
                )))
            except Exception:
                pass

        if "catboost" in libs_set:
            try:
                import catboost as cb
                candidates.append(("CatBoost", cb.CatBoostRegressor(
                    iterations=150, learning_rate=0.05, depth=5, random_seed=seed, verbose=0
                )))
            except Exception:
                pass

    return candidates


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
    target_metric = (state.get("target_metric") or ("roc_auc" if "binary" in task_type_str else ("f1" if "classification" in task_type_str else "rmse"))).lower()
    direction = get_metric_direction(target_metric)

    available_libs = get_available_libs()
    playbook = TASK_PLAYBOOKS.get(task_type_str, "")

    mem = RunMemory(project_id, run_id)
    run_paths = RunPaths(project_id, run_id)
    models_dir = PROJECTS_DIR / project_id / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    decision = state.get("supervisor_decision") or {}
    brief_dict = decision.get("brief", {}) if isinstance(decision.get("brief"), dict) else {}
    mode = brief_dict.get("mode") or ("baseline" if (step <= 2 and not mem.get_best()) else "new")

    if not brief:
        brief = build_worker_brief("model", decision, mem, state)

    check_and_log_budget("Model Brief", brief, WORKER_BRIEF_LIMIT)

    # 1. Load dataset
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

    try:
        if str(dataset_path).endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path)
    except Exception as exc:
        err_msg = f"Failed to load dataset: {exc}"
        _log.error("[MODEL] %s", err_msg)
        return {
            "status": "FAILED",
            "current_stage": "model",
            "error": err_msg,
            "report": WorkerReport(
                status="failed",
                result_summary=err_msg[:300],
                evidence={"error": err_msg},
                notebook={"step": step, "tried": "Read data", "outcome": "failed", "lesson": str(exc)},
            ).model_dump(),
        }

    if target_col not in df.columns:
        err_msg = f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}"
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

    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]

    # Convert binary classification string target to int if needed
    if "binary" in task_type_str and y.dtype == object:
        unique_vals = list(y.dropna().unique())
        if len(unique_vals) == 2:
            val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
            # Prefer common positive labels
            if str(unique_vals[0]).lower() in ("yes", "true", "1", "positive"):
                val_map = {unique_vals[0]: 1, unique_vals[1]: 0}
            y = y.map(val_map)

    # 2. Setup pipeline factory
    pipeline_script = run_paths.feature_pipeline_py if run_paths.feature_pipeline_py.exists() else (PROJECTS_DIR / project_id / "features" / "feature_pipeline.py")
    if pipeline_script.exists():
        pipeline_factory = lambda: load_pipeline_from_script(pipeline_script)
    else:
        profile_data = state.get("profile_summary", {})
        pipeline_factory = lambda: make_basic_preprocessor(profile_data)

    # 3. Determine split strategy
    split_strategy = state.get("split_strategy")
    if not split_strategy:
        profile_dict = state.get("profile_summary", {})
        if profile_dict.get("time_signal") or profile_dict.get("row_order_meaningful"):
            split_strategy = "time"
        elif "classification" in task_type_str:
            split_strategy = "stratified"
        else:
            split_strategy = "kfold"

    # 4. Mode execution: Baseline vs New/Tune
    evaluated_results: list[dict[str, Any]] = []

    if mode == "baseline":
        _log.info("[MODEL] Mode=baseline: deterministic evaluation of baseline model families.")
        is_classification = "classification" in task_type_str

        if is_classification:
            baseline_candidates = [
                ("DummyClassifier", DummyClassifier(strategy="prior")),
                ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
                ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=42)),
            ]
        else:
            baseline_candidates = [
                ("DummyRegressor", DummyRegressor(strategy="mean")),
                ("Ridge", Ridge(random_state=42)),
                ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=42)),
            ]

        for cand_name, cand_est in baseline_candidates:
            try:
                cv_res = cv_evaluate(
                    pipeline_factory=pipeline_factory,
                    estimator=cand_est,
                    X=X,
                    y=y,
                    task_type=task_type_str,
                    metric=target_metric,
                    split_strategy=split_strategy,
                    seed=42,
                )
                evaluated_results.append({
                    "name": cand_name,
                    "estimator": cand_est,
                    "cv_res": cv_res,
                })
                _log.info("[MODEL] Baseline %s | mean=%.4f std=%.4f gap=%.4f", cand_name, cv_res["cv_mean"], cv_res["cv_std"], cv_res["train_val_gap"])
            except Exception as cv_exc:
                _log.warning("[MODEL] Baseline candidate %s failed: %s", cand_name, cv_exc)

        if not evaluated_results:
            err_msg = "All baseline model evaluations failed."
            return {
                "status": "FAILED",
                "current_stage": "model",
                "error": err_msg,
                "report": WorkerReport(
                    status="failed",
                    result_summary=err_msg,
                    evidence={"error": err_msg},
                    notebook={"step": step, "tried": "Baselines", "outcome": "failed", "lesson": "All failed"},
                ).model_dump(),
            }

        # Select best baseline
        if direction == "lower":
            best_cand = min(evaluated_results, key=lambda x: x["cv_res"]["cv_mean"])
        else:
            best_cand = max(evaluated_results, key=lambda x: x["cv_res"]["cv_mean"])

        best_name = best_cand["name"]
        best_est = best_cand["estimator"]
        best_res = best_cand["cv_res"]
        best_score = best_res["cv_mean"]

        # Fit full pipeline on all data
        full_pipe = Pipeline([
            ("features", pipeline_factory()),
            ("model", clone(best_est)),
        ])
        full_pipe.fit(X, y)

        # Save artifacts
        joblib.dump(full_pipe, run_paths.best_model_pkl)
        joblib.dump(full_pipe, models_dir / "best_model.pkl")

        v1_dir = run_paths.version_dir("v1")
        np.save(v1_dir / "oof.npy", best_res["oof_predictions"])
        joblib.dump(full_pipe, v1_dir / "pipeline.pkl")

        # Update run memory best
        mem.set_best({
            "version": "v1",
            "score": best_score,
            "mean": best_score,
            "std": best_res["cv_std"],
            "fold_scores": best_res["fold_scores"],
            "metric": target_metric,
            "model_name": best_name,
            "train_score": best_res["train_score"],
            "train_val_gap": best_res["train_val_gap"],
        })

        summary_lines = [f"{c['name']}: {c['cv_res']['cv_mean']:.4f}±{c['cv_res']['cv_std']:.4f}" for c in evaluated_results]
        result_summary = f"Baseline complete. Best: {best_name} ({target_metric}={best_score:.4f}±{best_res['cv_std']:.4f}). [{', '.join(summary_lines)}]"

        worker_report = WorkerReport(
            status="ok",
            result_summary=result_summary[:300],
            evidence={
                "best_model_name": best_name,
                "score": best_score,
                "mean": best_score,
                "cv_mean": best_score,
                "std": best_res["cv_std"],
                "cv_std": best_res["cv_std"],
                "noise_floor": None,
                "significant": True,
                "target_metric": target_metric,
                "candidates_count": len(evaluated_results),
            },
            concern=None,
            suggestion="Train advanced model candidates against baseline noise floor.",
            notebook={
                "step": step,
                "tried": f"Baseline models ({len(evaluated_results)} candidates)",
                "outcome": f"Best {best_name} {target_metric}={best_score:.4f}",
                "lesson": "Baseline established. Folds cached for paired comparison.",
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
            "best_metric_value": best_score,
            "baseline_score": best_score,
            "current_version": "v1",
            "best_version": "v1",
            "current_stage": "model",
            "status": "SUCCESS",
            "report": worker_report.model_dump(),
        }

    else:
        # Mode == "new" or "tune" or "add_candidates"
        _log.info("[MODEL] Mode=%s: Evaluating competitive candidate models.", mode)
        prev_best = mem.get_best() or {}
        prev_best_scores = prev_best.get("fold_scores", [])
        prev_best_score = prev_best.get("score") or prev_best.get("mean")
        prev_best_name = prev_best.get("model_name", "baseline")

        # Propose competitive candidates
        candidates = _build_default_candidates(task_type_str, available_libs, seed=42)

        for cand_name, cand_est in candidates:
            try:
                cv_res = cv_evaluate(
                    pipeline_factory=pipeline_factory,
                    estimator=cand_est,
                    X=X,
                    y=y,
                    task_type=task_type_str,
                    metric=target_metric,
                    split_strategy=split_strategy,
                    seed=42,
                )
                evaluated_results.append({
                    "name": cand_name,
                    "estimator": cand_est,
                    "cv_res": cv_res,
                })
                _log.info("[MODEL] Candidate %s | mean=%.4f std=%.4f gap=%.4f", cand_name, cv_res["cv_mean"], cv_res["cv_std"], cv_res["train_val_gap"])
            except Exception as cv_exc:
                _log.warning("[MODEL] Candidate %s failed: %s", cand_name, cv_exc)

        if not evaluated_results:
            err_msg = "Candidate model evaluations failed."
            return {
                "status": "FAILED",
                "current_stage": "model",
                "error": err_msg,
                "report": WorkerReport(
                    status="failed",
                    result_summary=err_msg,
                    evidence={"error": err_msg},
                    notebook={"step": step, "tried": "Candidates", "outcome": "failed", "lesson": "All failed"},
                ).model_dump(),
            }

        # Find best candidate in this run
        if direction == "lower":
            top_cand = min(evaluated_results, key=lambda x: x["cv_res"]["cv_mean"])
        else:
            top_cand = max(evaluated_results, key=lambda x: x["cv_res"]["cv_mean"])

        top_name = top_cand["name"]
        top_est = top_cand["estimator"]
        top_res = top_cand["cv_res"]
        top_scores = top_res["fold_scores"]
        top_mean = top_res["cv_mean"]
        top_std = top_res["cv_std"]

        # Check significance vs previous best
        if prev_best_scores and len(prev_best_scores) == len(top_scores):
            is_sig, mean_diff, nf = is_significant_gain(top_scores, prev_best_scores, direction=direction)
        else:
            # No paired scores available: fall back to direct comparison
            if prev_best_score is not None:
                mean_diff = top_mean - prev_best_score
                is_sig = (mean_diff > 0) if direction == "higher" else (mean_diff < 0)
                nf = 0.0
            else:
                is_sig = True
                mean_diff = 0.0
                nf = 0.0

        current_ver = state.get("current_version") or f"v{iteration}"

        if is_sig:
            _log.info("[MODEL] Significant improvement by %s: mean=%.4f vs prev=%.4f (nf=%.4f diff=%.4f)", top_name, top_mean, prev_best_score or 0.0, nf, mean_diff)
            # Train full pipeline on all data
            full_pipe = Pipeline([
                ("features", pipeline_factory()),
                ("model", clone(top_est)),
            ])
            full_pipe.fit(X, y)

            joblib.dump(full_pipe, run_paths.best_model_pkl)
            joblib.dump(full_pipe, models_dir / "best_model.pkl")

            ver_dir = run_paths.version_dir(current_ver)
            np.save(ver_dir / "oof.npy", top_res["oof_predictions"])
            joblib.dump(full_pipe, ver_dir / "pipeline.pkl")

            mem.set_best({
                "version": current_ver,
                "score": top_mean,
                "mean": top_mean,
                "std": top_std,
                "fold_scores": top_scores,
                "metric": target_metric,
                "model_name": top_name,
                "train_score": top_res["train_score"],
                "train_val_gap": top_res["train_val_gap"],
            })

            res_summary = f"Significant improvement: {top_name} {target_metric}={top_mean:.4f}±{top_std:.4f} (diff {mean_diff:+.4f}, noise floor {nf:.4f} vs {prev_best_name})."
            worker_report = WorkerReport(
                status="ok",
                result_summary=res_summary[:300],
                evidence={
                    "best_model_name": top_name,
                    "score": top_mean,
                    "mean": top_mean,
                    "cv_mean": top_mean,
                    "std": top_std,
                    "cv_std": top_std,
                    "noise_floor": nf,
                    "significant": True,
                    "delta": mean_diff,
                    "target_metric": target_metric,
                },
                concern=None,
                suggestion="Evaluate pipeline generalization with Judge.",
                notebook={
                    "step": step,
                    "tried": f"Candidate models ({top_name})",
                    "outcome": f"Gain verified: {top_mean:.4f} (delta {mean_diff:+.4f})",
                    "lesson": f"Exceeded noise floor ({nf:.4f}). Promoted to best.",
                    "score_impact": top_mean,
                    "errors": [],
                },
                artifacts=[str(run_paths.best_model_pkl)],
            )

            return {
                "model_summary": {
                    "validation_strategy": f"5-Fold CV ({split_strategy})",
                    "target_metric": target_metric,
                    "best_model_name": top_name,
                    "best_score": top_mean,
                    "summary": res_summary,
                },
                "best_metric_value": top_mean,
                "best_version": current_ver,
                "current_stage": "model",
                "status": "SUCCESS",
                "report": worker_report.model_dump(),
            }

        else:
            _log.info("[MODEL] No significant gain by %s (score=%.4f vs best=%.4f, nf=%.4f). Keeping %s.", top_name, top_mean, prev_best_score or 0.0, nf, prev_best_name)
            res_summary = f"No significant gain: {top_name} scored {top_mean:.4f}±{top_std:.4f} (noise floor {nf:.4f}, diff {mean_diff:+.4f} vs {prev_best_name} {prev_best_score:.4f}). Retaining current best."
            worker_report = WorkerReport(
                status="no_gain",
                result_summary=res_summary[:300],
                evidence={
                    "best_model_name": prev_best_name,
                    "score": top_mean,
                    "mean": top_mean,
                    "cv_mean": top_mean,
                    "std": top_std,
                    "cv_std": top_std,
                    "noise_floor": nf,
                    "significant": False,
                    "delta": mean_diff,
                    "target_metric": target_metric,
                },
                concern=None,
                suggestion="Consider domain-specific features in FE or transition to report if headroom is exhausted.",
                notebook={
                    "step": step,
                    "tried": f"Candidate models ({top_name})",
                    "outcome": f"No significant gain ({top_mean:.4f} within noise floor {nf:.4f})",
                    "lesson": f"Candidate did not beat noise floor. Keeping {prev_best_name}.",
                    "score_impact": top_mean,
                    "errors": [],
                },
                artifacts=[],
            )

            return {
                "model_summary": {
                    "validation_strategy": f"5-Fold CV ({split_strategy})",
                    "target_metric": target_metric,
                    "best_model_name": prev_best_name,
                    "best_score": prev_best_score,
                    "summary": res_summary,
                },
                "best_metric_value": prev_best_score,
                "current_stage": "model",
                "status": "SUCCESS",
                "report": worker_report.model_dump(),
            }
