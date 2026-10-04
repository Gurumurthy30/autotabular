"""Version snapshots, best-model tracking, and best-version restoration for tabular ML runs."""

import json
import shutil
from pathlib import Path
from typing import Any

from app.config import PROJECTS_DIR
from app.core.run_memory import RunMemory
from app.tools.mlflow_tools import is_higher_better


def snapshot_version(
    project_id: str,
    run_id: str,
    version_id: str,
    stage: str = "fe",
    meta: dict[str, Any] | None = None,
) -> Path:
    """Snapshots a pipeline stage (fe or model) into runs/<run_id>/versions/<version_id>/."""
    run_dir = PROJECTS_DIR / project_id / "runs" / run_id
    version_dir = run_dir / "versions" / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    if stage in ("fe", "features", "feature_engineering"):
        fe_src_dir = PROJECTS_DIR / project_id / "features"
        fe_dst_dir = version_dir / "features"
        fe_dst_dir.mkdir(parents=True, exist_ok=True)

        if fe_src_dir.exists():
            for f in fe_src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, fe_dst_dir / f.name)

    elif stage in ("model", "modeling"):
        model_src_dir = PROJECTS_DIR / project_id / "models"
        model_dst_dir = version_dir / "models"
        model_dst_dir.mkdir(parents=True, exist_ok=True)

        if model_src_dir.exists():
            for f in model_src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, model_dst_dir / f.name)

        # Write or update meta.json
        meta_file = version_dir / "meta.json"
        existing_meta = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
            except Exception:
                existing_meta = {}

        if meta:
            existing_meta.update(meta)

        # Check feature schema if not present
        schema_file = version_dir / "features" / "feature_schema.json"
        if schema_file.exists() and "feature_count" not in existing_meta:
            try:
                with open(schema_file, "r", encoding="utf-8") as f:
                    schema_data = json.load(f)
                    features_list = schema_data.get("features", [])
                    existing_meta["feature_count"] = len(features_list)
                    existing_meta["feature_names"] = [
                        item.get("name") if isinstance(item, dict) else str(item)
                        for item in features_list
                    ]
            except Exception:
                pass

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(existing_meta, f, indent=2, default=str)

    return version_dir


def update_best(
    project_id: str,
    run_id: str,
    version_id: str,
    score: float,
    metric: str | None = None,
    step: int = 0,
    state: dict[str, Any] | None = None,
    experiment_id: str | None = None,
) -> bool:
    """Updates best.json if the given score improves upon the previous best."""
    mem = RunMemory(project_id, run_id)
    current_best = mem.get_best()

    from app.core.state import get_default_metric
    task_type = state.get("task_type") if state else None
    metric_name = metric or (state.get("target_metric") if state else None) or get_default_metric(task_type)
    higher_is_better = is_higher_better(metric_name)

    is_improved = False
    if current_best is None or "score" not in current_best or current_best["score"] is None:
        is_improved = True
    else:
        old_score = float(current_best["score"])
        if higher_is_better:
            is_improved = score > old_score
        else:
            is_improved = score < old_score

    if is_improved:
        best_data = {
            "version_id": version_id,
            "score": float(score),
            "metric": metric,
            "step": step,
            "experiment_id": experiment_id,
        }
        mem.set_best(best_data)

        if state is not None:
            state["best_metric_value"] = float(score)
            state["best_version"] = version_id
            if experiment_id:
                state["best_experiment_id"] = experiment_id

        return True

    return False


def restore_version(project_id: str, run_id: str, version_id: str) -> bool:
    """Restores the artifacts of a specific version back to active features/ and models/ directories."""
    version_dir = PROJECTS_DIR / project_id / "runs" / run_id / "versions" / version_id
    if not version_dir.exists():
        return False

    features_src = version_dir / "features"
    if features_src.exists():
        features_dst = PROJECTS_DIR / project_id / "features"
        features_dst.mkdir(parents=True, exist_ok=True)
        for f in features_src.iterdir():
            if f.is_file():
                shutil.copy2(f, features_dst / f.name)

    models_src = version_dir / "models"
    if models_src.exists():
        models_dst = PROJECTS_DIR / project_id / "models"
        models_dst.mkdir(parents=True, exist_ok=True)
        for f in models_src.iterdir():
            if f.is_file():
                shutil.copy2(f, models_dst / f.name)

    return True


def get_best_version(project_id: str, run_id: str) -> dict[str, Any] | None:
    """Reads best version record from RunMemory."""
    mem = RunMemory(project_id, run_id)
    return mem.get_best()
