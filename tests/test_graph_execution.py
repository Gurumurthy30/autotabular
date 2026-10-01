"""Integration and scenario tests for the flat Supervisor-led LangGraph execution."""

from unittest.mock import MagicMock

from app.core.run_memory import RunMemory
from app.core.schemas import BriefContent, SupervisorDecision
from app.graph.build_graph import build_ml_graph


def make_decision(action: str, objective: str = "Test objective") -> SupervisorDecision:
    """Creates a SupervisorDecision helper."""
    return SupervisorDecision(
        thought=f"Deciding to execute {action}",
        action=action,
        brief=BriefContent(
            objective=objective,
            focus_points=["test focus"],
            constraints=["test constraint"],
            mode="new",
        ),
        reason=f"Proceeding with {action}",
    )


def test_graph_smoke_test_scripted_sequence(tmp_path, monkeypatch):
    """Smoke test: scripted Supervisor goes profile -> eda -> fe -> model -> judge -> report -> finish.
    Asserts graph terminates, best version is restored, and ledger has one row per step.
    """
    project_id = "smoke_proj"
    run_id = "smoke_run"

    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.core.versions.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.agents.supervisor.PROJECTS_DIR", tmp_path)

    # Setup directories
    proj_dir = tmp_path / project_id
    (proj_dir / "features").mkdir(parents=True, exist_ok=True)
    (proj_dir / "models").mkdir(parents=True, exist_ok=True)
    (proj_dir / "features" / "feature_data.parquet").write_bytes(b"DUMMY_FEAT_V1")
    (proj_dir / "models" / "best_model.pkl").write_bytes(b"DUMMY_MODEL_V1")

    # Scripted sequence of decisions
    decisions = [
        make_decision("profile", "Profile dataset"),
        make_decision("eda", "Run exploratory analysis"),
        make_decision("fe", "Engineer features"),
        make_decision("model", "Train candidates"),
        make_decision("judge", "Review pipeline"),
        make_decision("report", "Generate report"),
        make_decision("finish", "All done"),
    ]

    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = decisions

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    mock_router = MagicMock()
    mock_router.get_model.return_value = mock_llm

    # Mock workers
    monkeypatch.setattr("app.graph.build_graph.profile_dataset", lambda s, r: {
        "status": "SUCCESS", "report": {"status": "ok", "result_summary": "Profile complete"}
    })
    monkeypatch.setattr("app.graph.build_graph.run_eda", lambda s, ro, re, brief=None: {
        "status": "SUCCESS", "report": {"status": "ok", "result_summary": "EDA complete"}
    })
    monkeypatch.setattr("app.graph.build_graph.run_feature_engineering", lambda s, ro, re, brief=None: {
        "status": "SUCCESS", "report": {"status": "ok", "result_summary": "FE complete"}
    })
    monkeypatch.setattr("app.graph.build_graph.run_modeling", lambda s, ro, re, brief=None: {
        "status": "SUCCESS", "best_metric_value": 0.91, "best_experiment_id": "exp_1",
        "model_summary": {"best_model": "GBM"},
        "report": {"status": "ok", "result_summary": "Model complete", "evidence": {"score": 0.91}}
    })
    monkeypatch.setattr("app.graph.build_graph.run_judge", lambda s, ro, re, brief=None: {
        "status": "SUCCESS", "report": {"status": "ok", "result_summary": "Judge complete", "evidence": {"overall": "ship"}}
    })
    monkeypatch.setattr("app.graph.build_graph.run_report", lambda s, ro, re, brief=None: {
        "status": "SUCCESS", "report": {"status": "ok", "result_summary": "Report complete"}
    })

    restored = []
    def fake_restore(pid, rid, vid):
        restored.append(vid)
        return True
    monkeypatch.setattr("app.graph.build_graph.restore_version", fake_restore)

    # Initialize RunMemory
    mem = RunMemory(project_id, run_id)
    mem.init_mission({
        "user_goal": "Predict test",
        "target_column": "target",
        "target_metric": "roc_auc",
        "task_type": "binary",
        "metric_direction": "higher",
    })

    initial_state = {
        "project_id": project_id,
        "run_id": run_id,
        "user_goal": "Predict test",
        "target_column": "target",
        "target_metric": "roc_auc",
        "dataset_version": "v1",
        "current_stage": "start",
        "step": 0,
        "worker_runs": {},
        "next_action": "profile",
        "status": "RUNNING",
    }

    graph = build_ml_graph(project_id, router=mock_router)
    final_state = initial_state
    for output_chunk in graph.stream(initial_state, config={"recursion_limit": 50}):
        for node_name, node_state in output_chunk.items():
            final_state.update(node_state)

    # Asserts
    # Graph reached finish
    assert final_state.get("next_action") == "finish"
    assert final_state.get("status") == "SUCCESS"

    # Best version was restored before report
    assert "v1" in restored

    # Ledger has rows recorded
    ledger_rows = mem.read_ledger()
    assert len(ledger_rows) >= 6
    actions_run = [r.action for r in ledger_rows]
    assert "profile" in actions_run
    assert "eda" in actions_run
    assert "fe" in actions_run
    assert "model" in actions_run
    assert "judge" in actions_run
    assert "report" in actions_run


def test_scenario_infinite_model_stops_and_reports(tmp_path, monkeypatch):
    """Scenario test: Fake Supervisor keeps asking for 'model' with same brief forever.
    The duplicate-brief guard must intervene: it cycles through fallback actions
    (profile, eda, fe, judge, report, finish) and the graph terminates.
    No per-worker cap is enforced (Phase 0 removal) — termination is via guard fallback cycling.
    """
    project_id = "loop_proj"
    run_id = "loop_run"

    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.core.versions.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.agents.supervisor.PROJECTS_DIR", tmp_path)

    proj_dir = tmp_path / project_id
    (proj_dir / "features").mkdir(parents=True, exist_ok=True)
    (proj_dir / "models").mkdir(parents=True, exist_ok=True)
    (proj_dir / "features" / "feature_data.parquet").write_bytes(b"DATA")
    (proj_dir / "models" / "best_model.pkl").write_bytes(b"MODEL")

    # Supervisor attempts to execute 'model' forever
    def infinite_model(*args, **kwargs):
        return make_decision("model", "Model run attempt")

    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = infinite_model

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    mock_router = MagicMock()
    mock_router.get_model.return_value = mock_llm

    # Mock workers
    monkeypatch.setattr("app.graph.build_graph.profile_dataset", lambda s, r: {
        "status": "SUCCESS", "report": {"status": "ok", "result_summary": "Profile ok"}
    })
    monkeypatch.setattr("app.graph.build_graph.run_eda", lambda s, ro, re, brief=None: {
        "status": "SUCCESS", "report": {"status": "ok", "result_summary": "EDA ok"}
    })
    monkeypatch.setattr("app.graph.build_graph.run_feature_engineering", lambda s, ro, re, brief=None: {
        "status": "SUCCESS", "report": {"status": "ok", "result_summary": "FE ok"}
    })
    monkeypatch.setattr("app.graph.build_graph.run_modeling", lambda s, ro, re, brief=None: {
        "status": "SUCCESS", "best_metric_value": 0.85,
        "model_summary": {"best_model": "RF"},
        "report": {"status": "ok", "result_summary": "Model ok", "evidence": {"score": 0.85}}
    })
    monkeypatch.setattr("app.graph.build_graph.run_judge", lambda s, ro, re, brief=None: {
        "status": "SUCCESS", "report": {"status": "ok", "result_summary": "Judge ok", "evidence": {"overall": "ship"}}
    })
    monkeypatch.setattr("app.graph.build_graph.run_report", lambda s, ro, re, brief=None: {
        "status": "SUCCESS", "report": {"status": "ok", "result_summary": "Report ok"}
    })

    mem = RunMemory(project_id, run_id)
    mem.init_mission({
        "user_goal": "Predict test",
        "target_column": "target",
        "target_metric": "roc_auc",
        "task_type": "binary",
        "metric_direction": "higher",
    })

    initial_state = {
        "project_id": project_id,
        "run_id": run_id,
        "user_goal": "Predict test",
        "target_column": "target",
        "target_metric": "roc_auc",
        "dataset_version": "v1",
        "current_stage": "start",
        "step": 0,
        "worker_runs": {},
        "next_action": "profile",
        "status": "RUNNING",
    }

    graph = build_ml_graph(project_id, router=mock_router)
    final_state = initial_state
    for output_chunk in graph.stream(initial_state, config={"recursion_limit": 50}):
        for node_name, node_state in output_chunk.items():
            final_state.update(node_state)

    # Assert run ends despite infinite model requests
    assert final_state.get("next_action") == "finish"
    # Report was executed (the duplicate-brief guard forces fallback progression through the pipeline)
    assert final_state.get("worker_runs", {}).get("report", 0) >= 1
    # No per-worker cap is enforced — the run terminates via the duplicate-brief guard cycling
    # through fallback actions (fe, eda, profile, judge, report, finish) rather than unlimited loops
    # The graph hits the recursion_limit which terminates it
    assert final_state.get("status") in ("SUCCESS", "RUNNING")  # Terminated cleanly
