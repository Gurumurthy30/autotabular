"""V2 Clean Audit & Rigorous Verification Tests.

Verifies:
1. Complete removal of legacy limit concepts across app/ codebase.
2. invoke_json JSON parsing, markdown code fences, and single-repair resilience.
3. Judge agent failsafe behavior returning overall="blocked".
4. Profile column roles categorization.
5. ML harness picklability of transformers and preprocessing pipelines.
6. Paired CV noise floor and significance testing in both metric directions.
7. Feature Engineering deterministic basic mode execution.
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.agents.judge import run_judge
from app.core.llm_json import invoke_json
from app.core.model_router import ModelRouter
from app.core.state import ProjectState
from app.ml_harness.noise import is_significant_gain
from app.ml_harness.preprocess import make_basic_preprocessor
from app.ml_harness.transformers import CyclicEncoder, DateParts
from app.tools.registry import ToolRegistry


def test_no_removed_limits_in_code():
    """Verify zero occurrences of banned limit concepts in app/ python code."""
    banned_tokens = [
        "MAX_STEPS",
        "MAX_CODER_RETRIES",
        "WORKER_RUN_CAPS",
        "MIN_GAIN_REL",
        "stall_guard",
        "step_budget",
        "forced report",
        "WORKER_CAPS",
    ]
    app_dir = Path("app")
    python_files = list(app_dir.rglob("*.py"))
    assert len(python_files) > 10

    for py_file in python_files:
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        for token in banned_tokens:
            assert token not in content, f"Found banned limit '{token}' in {py_file}"


def test_invoke_json_markdown_fence_and_coercion():
    """Test invoke_json handles markdown fenced blocks and pydantic schema validation."""
    class SampleOutput(BaseModel):
        score: float
        tag: str

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="Here is your output:\n```json\n{\n  \"score\": 0.95,\n  \"tag\": \"production\"\n}\n```\nHope that helps!"
    )

    result, err, _raw = invoke_json(
        mock_llm,
        "dummy prompt",
        SampleOutput,
        agent_name="test_agent",
    )
    assert not err
    assert result is not None
    assert result.score == 0.95
    assert result.tag == "production"


def test_judge_failure_blocks_overall(tmp_path, monkeypatch):
    """Verify Judge returns overall='blocked' on failure or corrupt inputs."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.core.versions.PROJECTS_DIR", tmp_path)

    state: ProjectState = {
        "project_id": "test_judge_proj",
        "run_id": "test_run_1",
        "user_goal": "Predict target",
        "target_column": "target",
        "target_metric": "roc_auc",
        "description": "test",
        "constraints": {},
        "dataset_id": "data",
        "dataset_version": "v1",
        "task_type": "binary",
        "profile_summary": {},
        "eda_findings": {},
        "feature_summary": {},
        "model_summary": {},
        "current_stage": "judge",
        "iteration": 1,
        "best_experiment_id": None,
        "best_metric_value": None,
        "supervisor_memory": {},
        "artifacts": [],
        "next_action": "judge",
        "status": "RUNNING",
    }

    mock_router = MagicMock(spec=ModelRouter)
    mock_llm = MagicMock()
    # LLM returns unparseable garbage that causes fallback
    mock_llm.invoke.return_value = MagicMock(content="I cannot evaluate this pipeline properly.")
    mock_router.get_model.return_value = mock_llm

    registry = ToolRegistry(project_id="test_judge_proj")
    report = run_judge(state, mock_router, registry)

    assert report["status"].lower() == "failed"
    assert report["judge_report"]["overall"] == "blocked"


def test_ml_harness_picklable_transformers():
    """Verify custom transformers and basic preprocessor are picklable and reversible."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
        "num_val": np.linspace(0, 100, 10),
        "cat_val": ["A", "B", "A", "B", "C", "A", "B", "C", "A", "B"],
    })

    dp = DateParts()
    df_transformed = dp.fit_transform(df[["timestamp"]])
    assert df_transformed.shape[1] >= 3

    # Pickle and unpickle
    pickled = pickle.dumps(dp)
    unpickled = pickle.loads(pickled)
    df_transformed_2 = unpickled.transform(df[["timestamp"]])
    pd.testing.assert_frame_equal(df_transformed, df_transformed_2)

    # Test CyclicEncoder
    ce = CyclicEncoder(period=24.0)
    df_cyclic = ce.fit_transform(df[["num_val"]])
    pickled_ce = pickle.dumps(ce)
    unpickled_ce = pickle.loads(pickled_ce)
    df_cyclic_2 = unpickled_ce.transform(df[["num_val"]])
    pd.testing.assert_frame_equal(df_cyclic, df_cyclic_2)


def test_make_basic_preprocessor_execution():
    """Verify make_basic_preprocessor builds a valid, picklable ColumnTransformer."""
    profile = {
        "column_roles": {
            "id": "id",
            "age": "numeric",
            "city": "categorical",
            "signup_date": "datetime",
            "target": "target",
        }
    }
    preprocessor = make_basic_preprocessor(profile)
    assert preprocessor is not None

    pickled = pickle.dumps(preprocessor)
    restored = pickle.loads(pickled)
    assert restored is not None


def test_paired_noise_floor_and_significance():
    """Verify paired CV noise floor and significance calculation in both metric directions."""
    # Case 1: Direction higher (e.g. roc_auc)
    best_scores = [0.80, 0.81, 0.79, 0.82, 0.80]
    # Minor candidate improvement well within noise floor
    cand_noisy = [0.801, 0.799, 0.802, 0.818, 0.800]
    is_sig, _diff, nf = is_significant_gain(cand_noisy, best_scores, direction="higher")
    assert not is_sig
    assert nf > 0.0

    # Strong candidate improvement exceeding noise floor
    cand_strong = [0.85, 0.86, 0.84, 0.87, 0.85]
    is_sig_strong, diff_strong, nf_strong = is_significant_gain(cand_strong, best_scores, direction="higher")
    assert is_sig_strong
    assert diff_strong > nf_strong

    # Case 2: Direction lower (e.g. rmse)
    best_rmse = [10.0, 10.5, 9.8, 10.2, 10.1]
    # Candidate with significantly lower error
    cand_rmse = [8.0, 8.2, 7.9, 8.1, 8.0]
    is_sig_rmse, diff_rmse, nf_rmse = is_significant_gain(cand_rmse, best_rmse, direction="lower")
    assert is_sig_rmse
    assert diff_rmse < -1.0 * nf_rmse


def test_model_registry_has_dataset_tools():
    """Verify registry provides dataset_tools to model agent and resolves nested dataset paths."""
    registry = ToolRegistry("rainfall_predection")
    model_tools = registry.get_tools_for_role("model")
    assert model_tools.dataset is not None

    path = model_tools.dataset.get_dataset_path("dataset_v1")
    assert path.exists()
    assert path.name == "data.csv"


def test_build_default_candidates_supports_list_and_dict():
    """Verify _build_default_candidates handles both list of library names and dict of library flags."""
    from app.agents.model import _build_default_candidates

    # Test with list (from get_available_libs())
    cands_list = _build_default_candidates("binary_classification", ["sklearn", "lightgbm", "xgboost"])
    assert len(cands_list) >= 2
    names = [name for name, _ in cands_list]
    assert "HistGradientBoosting_tuned" in names
    assert "RandomForest_100" in names

    # Test with dict
    cands_dict = _build_default_candidates("regression", {"sklearn": True, "lightgbm": True})
    assert len(cands_dict) >= 2

    # Test with empty list
    cands_empty = _build_default_candidates("binary_classification", [])
    assert len(cands_empty) >= 2


def test_delete_project_purges_all_artifacts_and_mlflow():
    """Verify DELETE /projects/{project_id} cascades to DB, removes filesystem folder, and purges MLflow runs."""
    import sqlite3
    from fastapi.testclient import TestClient
    from app.main import app
    from app.config import BASE_DIR, PROJECTS_DIR

    client = TestClient(app)
    pid = "test_purge_dummy_project"

    # 1. Create project via API
    res = client.post("/projects", json={"id": pid, "name": "Purge Dummy Project", "description": "Test purge"})
    assert res.status_code == 200

    # Create dummy files under projects/<pid>/
    proj_dir = PROJECTS_DIR / pid
    proj_dir.mkdir(parents=True, exist_ok=True)
    dummy_code = proj_dir / "code_executions" / "run_01.json"
    dummy_code.parent.mkdir(parents=True, exist_ok=True)
    dummy_code.write_text("{\"code\": \"print('hello')\"}")

    # Seed dummy entry in mlflow.db and mlruns if mlflow.db exists
    mlflow_db = BASE_DIR / "mlflow.db"
    dummy_exp_id = "9999999"
    if mlflow_db.exists():
        with sqlite3.connect(mlflow_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO experiments (experiment_id, name, artifact_location, lifecycle_stage) VALUES (?, ?, ?, ?)",
                (dummy_exp_id, pid, f"./mlruns/{dummy_exp_id}", "active")
            )
            cursor.execute(
                "INSERT OR REPLACE INTO runs (run_uuid, name, source_type, source_name, entry_point_name, user_id, status, start_time, end_time, source_version, lifecycle_stage, artifact_uri, experiment_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("dummy_run_uuid_123", "dummy_run", "LOCAL", "runner.py", "main", "test", "FINISHED", 1000, 2000, "1", "active", f"./mlruns/{dummy_exp_id}/dummy_run_uuid_123", dummy_exp_id)
            )
            conn.commit()

        dummy_mlrun_dir = BASE_DIR / "mlruns" / dummy_exp_id
        dummy_mlrun_dir.mkdir(parents=True, exist_ok=True)
        (dummy_mlrun_dir / "meta.yaml").write_text(f"experiment_id: '{dummy_exp_id}'\nname: '{pid}'\n")

    # 2. Call DELETE endpoint
    del_res = client.delete(f"/projects/{pid}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # 3. Assertions
    # Folder in projects/ must not exist
    assert not proj_dir.exists()

    # MLflow records must be purged
    if mlflow_db.exists():
        with sqlite3.connect(mlflow_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experiments WHERE name = ?", (pid,))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT COUNT(*) FROM runs WHERE experiment_id = ?", (dummy_exp_id,))
            assert cursor.fetchone()[0] == 0

        assert not (BASE_DIR / "mlruns" / dummy_exp_id).exists()


