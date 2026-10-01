"""Unit tests for RunMemory and versions module."""

import hashlib
from pathlib import Path

from app.core.run_memory import RunMemory
from app.core.schemas import LedgerRow, NotebookRow
from app.core.versions import (
    restore_version,
    snapshot_version,
    update_best,
)


def file_sha256(path: Path) -> str:
    """Computes sha256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def test_run_memory_mission_and_plan(tmp_path, monkeypatch):
    """Tests mission and plan get/set in RunMemory."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")

    mission = {
        "user_goal": "Predict churn",
        "target_column": "churn",
        "target_metric": "roc_auc",
        "task_type": "binary_classification",
        "metric_direction": "max",
    }
    mem.init_mission(mission)
    assert mem.get_mission() == mission

    plan = [
        {"id": "p1", "text": "Profile data", "status": "done"},
        {"id": "p2", "text": "Run EDA", "status": "doing"},
    ]
    mem.set_plan(plan)
    loaded_plan = mem.get_plan()
    assert len(loaded_plan) == 2
    assert loaded_plan[0]["id"] == "p1"
    assert loaded_plan[1]["status"] == "doing"


def test_ledger_append_read_and_render_under_budget(tmp_path, monkeypatch):
    """Tests ledger append, read, and rendering table staying under budget after 30 rows."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")

    # Append 30 rows
    for i in range(30):
        row = LedgerRow(
            step=i,
            ts="2026-09-29T12:00:00Z",
            agent="model" if i % 2 == 0 else "fe",
            action="model" if i % 2 == 0 else "fe",
            brief_summary=f"Brief for step {i}",
            status="ok",
            result_summary=f"Result for step {i} with some metrics and information",
            version_id=f"v{i // 2 + 1}",
            score=0.80 + i * 0.001,
            delta_vs_best=0.001 if i > 0 else 0.0,
            judge_overall=None,
            issues=[],
            concern=None,
            decision_reason=f"Supervisor decided to run step {i}",
            duration_s=5.0,
        )
        mem.append_ledger(row)

    rows = mem.read_ledger()
    assert len(rows) == 30

    last_5 = mem.read_ledger(last_n=5)
    assert len(last_5) == 5
    assert last_5[-1].step == 29

    # Render ledger
    rendered = mem.render_ledger(max_chars=3000)
    assert len(rendered) <= 3000
    assert "collapsed" in rendered
    assert "| Step | Agent | Action |" in rendered
    assert "| 29 |" in rendered


def test_notebook_and_top_lessons(tmp_path, monkeypatch):
    """Tests notebook append and top_lessons deduplication."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")

    agent = "fe"
    mem.append_notebook(agent, NotebookRow(step=1, tried="StandardScaler", outcome="ok", lesson="StandardScaler works well for bell-shaped features"))
    mem.append_notebook(agent, NotebookRow(step=2, tried="StandardScaler", outcome="ok", lesson="StandardScaler works well for bell-shaped features"))  # Duplicate
    mem.append_notebook(agent, NotebookRow(step=3, tried="Log1p", outcome="ok", lesson="Log1p reduces positive skew on price"))
    mem.append_notebook(agent, NotebookRow(step=4, tried="OneHot", outcome="ok", lesson="OneHot is safe for low cardinality"))

    lessons = mem.top_lessons(agent, n=5)
    assert len(lessons) == 3
    assert "StandardScaler works well for bell-shaped features" in lessons
    assert "Log1p reduces positive skew on price" in lessons
    assert "OneHot is safe for low cardinality" in lessons


def test_project_lessons_dedupe_and_cap(tmp_path, monkeypatch):
    """Tests add_project_lesson deduplication and 10-item cap."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")

    # Add 15 lessons
    for i in range(15):
        mem.add_project_lesson(f"Lesson number {i}")

    # Add duplicate
    mem.add_project_lesson("Lesson number 14")

    lessons = mem.get_project_lessons()
    assert len(lessons) == 10
    assert lessons[-1] == "Lesson number 14"
    assert lessons[0] == "Lesson number 5"


def test_versions_snapshot_and_restore_identical_hashes(tmp_path, monkeypatch):
    """Tests version snapshot and restore returns identical file hashes."""
    monkeypatch.setattr("app.core.versions.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)

    project_id = "proj_snap"
    run_id = "run_1"

    # Setup original features and models directories
    features_dir = tmp_path / project_id / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    models_dir = tmp_path / project_id / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = features_dir / "feature_data.parquet"
    parquet_file.write_bytes(b"PARQUET_DUMMY_DATA_V1")
    pipe_file = features_dir / "feature_pipeline.pkl"
    pipe_file.write_bytes(b"PICKLE_DUMMY_PIPE_V1")
    model_file = models_dir / "best_model.pkl"
    model_file.write_bytes(b"BEST_MODEL_V1")

    orig_parquet_hash = file_sha256(parquet_file)
    orig_pipe_hash = file_sha256(pipe_file)
    orig_model_hash = file_sha256(model_file)

    # 1. Snapshot FE
    snapshot_version(project_id, run_id, "v1", stage="fe")
    # 2. Snapshot Model
    snapshot_version(project_id, run_id, "v1", stage="model", meta={"best_model": "RandomForest", "score": 0.88})

    # Overwrite active features and models with v2 data
    parquet_file.write_bytes(b"PARQUET_MODIFIED_DATA_V2")
    pipe_file.write_bytes(b"PICKLE_MODIFIED_PIPE_V2")
    model_file.write_bytes(b"BEST_MODEL_V2")
    assert file_sha256(parquet_file) != orig_parquet_hash

    # Restore v1
    success = restore_version(project_id, run_id, "v1")
    assert success is True

    # Assert restored hashes are identical to original
    assert file_sha256(parquet_file) == orig_parquet_hash
    assert file_sha256(pipe_file) == orig_pipe_hash
    assert file_sha256(model_file) == orig_model_hash


def test_update_best_higher_and_lower_better(tmp_path, monkeypatch):
    """Tests update_best with both higher-is-better and lower-is-better metrics."""
    monkeypatch.setattr("app.core.versions.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)

    project_id = "proj_best"
    run_id = "run_best"
    state = {}

    # 1. Higher is better (roc_auc)
    # First candidate
    up1 = update_best(project_id, run_id, "v1", 0.85, "roc_auc", 3, state)
    assert up1 is True
    assert state["best_metric_value"] == 0.85
    assert state["best_version"] == "v1"

    # Worse candidate
    up2 = update_best(project_id, run_id, "v2", 0.82, "roc_auc", 5, state)
    assert up2 is False
    assert state["best_metric_value"] == 0.85

    # Better candidate
    up3 = update_best(project_id, run_id, "v3", 0.89, "roc_auc", 7, state)
    assert up3 is True
    assert state["best_metric_value"] == 0.89
    assert state["best_version"] == "v3"

    # 2. Lower is better (rmse)
    run_id_rmse = "run_rmse"
    state_rmse = {}
    up_r1 = update_best(project_id, run_id_rmse, "v1", 10.5, "rmse", 3, state_rmse)
    assert up_r1 is True
    assert state_rmse["best_metric_value"] == 10.5

    # Worse candidate (higher rmse)
    up_r2 = update_best(project_id, run_id_rmse, "v2", 12.0, "rmse", 5, state_rmse)
    assert up_r2 is False
    assert state_rmse["best_metric_value"] == 10.5

    # Better candidate (lower rmse)
    up_r3 = update_best(project_id, run_id_rmse, "v3", 8.2, "rmse", 7, state_rmse)
    assert up_r3 is True
    assert state_rmse["best_metric_value"] == 8.2
    assert state_rmse["best_version"] == "v3"
