"""Unit tests for KnowledgeBoard: persistence, CRUD, role digests, and brief integration."""

import json
from pathlib import Path

import pytest

from app.core.briefs import build_worker_brief, render_supervisor_context
from app.core.knowledge import (
    Decision,
    EDAFinding,
    FeatureTried,
    Hypothesis,
    JudgeVerdict,
    KnowledgeBoard,
    ModelTried,
)
from app.core.run_memory import RunMemory
from app.core.schemas import BriefContent, SupervisorDecision


def test_knowledge_board_crud_and_persistence(tmp_path: Path):
    run_dir = tmp_path / "runs" / "test_run"
    kb = KnowledgeBoard(run_dir=run_dir)

    # 1. Add EDA finding
    kb.add_eda_finding({
        "id": "eda_001",
        "category": "skew",
        "finding": "Fare feature is heavily right-skewed",
        "recommendation": "Apply log1p transform",
        "priority": "high",
        "status": "open",
    })

    # 2. Add Feature tried
    kb.add_feature_tried({
        "id": "fare_log1p",
        "version": "v1",
        "description": "Log1p transformed fare",
        "source_cols": ["Fare"],
        "reason": "Reduce positive skew",
        "outcome": "kept",
    })

    # 3. Add Model tried
    kb.add_model_tried({
        "id": "lgbm_v1",
        "family": "LightGBM",
        "params_summary": "n_estimators=100, lr=0.05",
        "cv_mean": 0.835,
        "cv_std": 0.015,
        "train_score": 0.890,
        "gap": 0.055,
        "fit_time": 1.25,
        "outcome": "best",
    })

    # 4. Add Hypothesis
    kb.add_hypothesis({
        "id": "hypo_01",
        "text": "Family size interaction improves predictive signal",
        "owner_agent": "fe",
        "status": "open",
    })

    # 5. Add Judge verdict
    kb.add_judge_verdict({
        "step": 2,
        "blame_stage": "fe",
        "overall": "improve",
        "advice": ["Prune low-importance features to avoid overfitting"],
        "addressed": False,
    })

    # 6. Add Decision
    kb.add_decision({
        "step": 1,
        "action": "eda",
        "reason": "Initial data exploration and leakage checks",
    })

    # Verify knowledge.json exists and is structured
    json_path = run_dir / "memory" / "knowledge.json"
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["eda_findings"]) == 1
    assert data["eda_findings"][0]["id"] == "eda_001"
    assert len(data["features_tried"]) == 1
    assert len(data["models_tried"]) == 1
    assert len(data["hypotheses"]) == 1
    assert len(data["judge_verdicts"]) == 1
    assert len(data["decisions"]) == 1

    # Reload from disk into fresh KnowledgeBoard
    kb_loaded = KnowledgeBoard(run_dir=run_dir)
    assert len(kb_loaded.eda_findings) == 1
    assert kb_loaded.eda_findings[0].finding == "Fare feature is heavily right-skewed"
    assert len(kb_loaded.features_tried) == 1
    assert kb_loaded.features_tried[0].outcome == "kept"
    assert len(kb_loaded.models_tried) == 1
    assert kb_loaded.models_tried[0].cv_mean == 0.835
    assert len(kb_loaded.judge_verdicts) == 1
    assert kb_loaded.judge_verdicts[0].addressed is False


def test_knowledge_board_update_and_addressed(tmp_path: Path):
    run_dir = tmp_path / "runs" / "test_run"
    kb = KnowledgeBoard(run_dir=run_dir)

    kb.add_eda_finding({"id": "eda_001", "finding": "Test finding", "status": "open"})
    kb.add_hypothesis({"id": "hypo_01", "text": "Hypo", "owner_agent": "fe", "status": "open"})
    kb.add_judge_verdict({
        "step": 3,
        "blame_stage": "fe",
        "overall": "improve",
        "advice": ["Add missing indicators"],
        "addressed": False,
    })
    kb.add_judge_verdict({
        "step": 4,
        "blame_stage": "model",
        "overall": "improve",
        "advice": ["Tune regularization"],
        "addressed": False,
    })

    # Update status
    assert kb.update_status("eda_findings", "eda_001", "used") is True
    assert kb.eda_findings[0].status == "used"

    assert kb.update_status("hypotheses", "hypo_01", "confirmed") is True
    assert kb.hypotheses[0].status == "confirmed"

    # Mark judge advice addressed targeting "fe"
    addressed_count = kb.mark_judge_advice_addressed("fe")
    assert addressed_count == 1
    assert kb.judge_verdicts[0].addressed is True
    assert kb.judge_verdicts[1].addressed is False

    # Second call should not double-address
    assert kb.mark_judge_advice_addressed("fe") == 0

    # Address model advice
    assert kb.mark_judge_advice_addressed("model") == 1
    assert kb.judge_verdicts[1].addressed is True


def test_knowledge_board_role_digests(tmp_path: Path):
    run_dir = tmp_path / "runs" / "test_run"
    kb = KnowledgeBoard(run_dir=run_dir)

    kb.add_eda_finding({"id": "eda_01", "category": "leakage", "finding": "ID column is unique", "status": "open", "priority": "high"})
    kb.add_eda_finding({"id": "eda_02", "category": "missingness", "finding": "Cabin is 80% null", "status": "used", "priority": "low"})
    kb.add_hypothesis({"id": "h1", "text": "Deck from cabin correlates with survival", "owner_agent": "fe", "status": "open"})
    kb.add_hypothesis({"id": "h2", "text": "Age imputation by title", "owner_agent": "fe", "status": "confirmed", "evidence": "+0.02 CV"})
    kb.add_feature_tried({"id": "deck_code", "version": "v1", "source_cols": ["Cabin"], "outcome": "kept", "cv_effect": 0.01})
    kb.add_model_tried({"id": "rf_1", "family": "RandomForest", "params_summary": "n=100", "cv_mean": 0.81, "cv_std": 0.02, "outcome": "baseline"})
    kb.add_judge_verdict({"step": 2, "blame_stage": "fe", "overall": "improve", "advice": ["Drop Cabin raw column"], "addressed": False})
    kb.add_judge_verdict({"step": 3, "blame_stage": "model", "overall": "improve", "advice": ["Try gradient boosting"], "addressed": False})

    # 1. Supervisor digest
    sup_digest = kb.digest("supervisor")
    assert "eda_01" in sup_digest
    assert "eda_02" not in sup_digest  # only open findings
    assert "h1" in sup_digest
    assert "rf_1" in sup_digest
    assert "deck_code" in sup_digest
    assert "Drop Cabin raw column" in sup_digest

    # 2. EDA digest
    eda_digest = kb.digest("eda")
    assert "DO NOT DUPLICATE" in eda_digest
    assert "eda_01" in eda_digest
    assert "eda_02" in eda_digest
    assert "h2" in eda_digest  # tested hypotheses

    # 3. FE digest
    fe_digest = kb.digest("fe")
    assert "eda_01" in fe_digest
    assert "deck_code" in fe_digest
    assert "Drop Cabin raw column" in fe_digest
    assert "Try gradient boosting" not in fe_digest  # model advice not for FE

    # 4. Model digest
    model_digest = kb.digest("model")
    assert "DO NOT REPEAT FAMILY + PARAMS" in model_digest
    assert "RandomForest" in model_digest
    assert "Try gradient boosting" in model_digest
    assert "Drop Cabin raw column" not in model_digest

    # 5. Judge digest
    judge_digest = kb.digest("judge")
    assert "KNOWLEDGE BOARD SUMMARY (JUDGE)" in judge_digest

    # 6. Report digest
    report_digest = kb.digest("report")
    assert "KNOWLEDGE BOARD LINEAGE (REPORT)" in report_digest


def test_brief_builder_knowledge_injection(tmp_path: Path, monkeypatch):
    from app.config import PROJECTS_DIR
    proj_dir = tmp_path / "projects"
    monkeypatch.setattr("app.core.briefs.PROJECTS_DIR", proj_dir)
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", proj_dir)
    monkeypatch.setattr("app.core.knowledge.PROJECTS_DIR", proj_dir)

    project_id = "test_p"
    run_id = "test_r"
    mem = RunMemory(project_id, run_id)
    mem.init_mission({"user_goal": "Predict survival", "target_column": "Survived", "task_type": "binary_classification"})

    kb = KnowledgeBoard(run_dir=mem.run_dir)
    kb.add_eda_finding({"id": "eda_inj", "finding": "Injected finding", "status": "open"})

    state = {
        "project_id": project_id,
        "run_id": run_id,
        "user_goal": "Predict survival",
        "target_column": "Survived",
        "step": 1,
    }

    # Verify Supervisor context injection
    sup_ctx = render_supervisor_context(mem=mem, state=state)
    assert "=== KNOWLEDGE BOARD ===" in sup_ctx
    assert "eda_inj" in sup_ctx

    # Verify Worker brief injection
    decision = SupervisorDecision(
        action="fe",
        brief=BriefContent(objective="Engineer features from findings"),
        reason="Test",
    )
    fe_brief = build_worker_brief("fe", decision, mem, state)
    assert "=== KNOWLEDGE BOARD (FE) ===" in fe_brief
    assert "eda_inj" in fe_brief


def test_worker_failure_leaves_judge_advice_unaddressed(tmp_path: Path, monkeypatch):
    """Verifies advice appears in target worker's digest/brief and stays unaddressed if worker fails."""
    from unittest.mock import MagicMock
    from app.config import PROJECTS_DIR
    from app.graph.build_graph import make_worker_node
    from app.tools.registry import ToolRegistry

    proj_dir = tmp_path / "projects"
    monkeypatch.setattr("app.core.briefs.PROJECTS_DIR", proj_dir)
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", proj_dir)
    monkeypatch.setattr("app.core.knowledge.PROJECTS_DIR", proj_dir)

    project_id = "test_p_fail"
    run_id = "test_r_fail"
    mem = RunMemory(project_id, run_id)
    kb = KnowledgeBoard(run_dir=mem.run_dir)

    # Add unaddressed judge advice targeting FE
    kb.add_judge_verdict({
        "step": 2,
        "blame_stage": "fe",
        "overall": "improve",
        "advice": ["Targeted advice for FE: fix skew"],
        "addressed": False,
    })

    # Assert advice appears in FE digest and brief
    fe_digest = kb.digest("fe")
    assert "Targeted advice for FE: fix skew" in fe_digest

    state = {
        "project_id": project_id,
        "run_id": run_id,
        "step": 3,
        "supervisor_decision": {"action": "fe", "brief": {"objective": "Fix skew"}},
    }

    # Worker function that fails
    def failing_worker(state, router, registry, brief=None):
        assert "Targeted advice for FE: fix skew" in brief
        return {
            "status": "FAILED",
            "error": "Execution exception",
            "report": {
                "status": "failed",
                "result_summary": "Worker failed",
                "evidence": {},
            },
        }

    mock_router = MagicMock()
    mock_registry = MagicMock()
    node = make_worker_node("features", failing_worker, mock_router, mock_registry)
    res = node(state)

    # Verify verdict remains unaddressed because worker failed
    kb_after = KnowledgeBoard(run_dir=mem.run_dir)
    assert len(kb_after.judge_verdicts) == 1
    assert kb_after.judge_verdicts[0].addressed is False

    # Now run a successful worker
    def succeeding_worker(state, router, registry, brief=None):
        return {
            "status": "SUCCESS",
            "report": {
                "status": "ok",
                "result_summary": "Worker succeeded",
                "evidence": {},
            },
        }

    node_success = make_worker_node("features", succeeding_worker, mock_router, mock_registry)
    res_success = node_success(state)

    # Verify verdict is now addressed
    kb_after_success = KnowledgeBoard(run_dir=mem.run_dir)
    assert kb_after_success.judge_verdicts[0].addressed is True
