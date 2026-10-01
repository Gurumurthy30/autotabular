"""Unit tests for the deterministic supervisor guard rules.

The following guards are KEPT:
  - Prerequisite ordering (eda needs profile, fe needs eda, model needs fe, etc.)
  - Duplicate-brief guard (identical action + brief hash blocked)
  - Judge-consecutive guard (judge cannot run twice in a row)
"""

from app.agents.guards import (
    compute_brief_hash,
    get_deterministic_fallback,
    guard_action,
)
from app.core.run_memory import RunMemory
from app.core.schemas import BriefContent, LedgerRow, SupervisorDecision


def make_decision(action: str, objective: str = "test", focus: list[str] = None, mode: str = "new", reason: str = "test reason") -> SupervisorDecision:
    """Helper to create a SupervisorDecision object."""
    return SupervisorDecision(
        thought="Testing guard rule",
        action=action,
        brief=BriefContent(
            objective=objective,
            focus_points=focus or ["focus1"],
            constraints=["c1"],
            mode=mode,
        ),
        reason=reason,
    )


def test_guard_prerequisites_profile_before_eda(tmp_path, monkeypatch):
    """Prerequisite: eda requires profile to be completed first."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")
    state = {"step": 0, "worker_runs": {}}

    action, note = guard_action(state, mem, make_decision("eda"))
    assert action == "profile"
    assert note is not None
    assert "profile" in note.lower()


def test_guard_prerequisites_eda_before_fe(tmp_path, monkeypatch):
    """Prerequisite: fe requires eda findings first."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")
    state = {"step": 0, "worker_runs": {"profile": 1}}

    action, note = guard_action(state, mem, make_decision("fe"))
    assert action == "eda"
    assert "eda" in note.lower()


def test_guard_prerequisites_fe_before_model(tmp_path, monkeypatch):
    """Prerequisite: model requires a successful feature engineering version."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")
    state = {"step": 0, "worker_runs": {"profile": 1, "eda": 1}}

    action, note = guard_action(state, mem, make_decision("model"))
    assert action == "fe"
    assert "feature engineering" in note.lower()


def test_guard_prerequisites_model_before_judge(tmp_path, monkeypatch):
    """Prerequisite: judge requires at least 1 completed model run."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")
    state = {
        "step": 0,
        "current_version": "v1",
        "worker_runs": {"profile": 1, "eda": 1, "fe": 1},
        "best_metric_value": None,
    }

    action, note = guard_action(state, mem, make_decision("judge"))
    assert action == "model"
    assert "model" in note.lower()


def test_guard_prerequisites_model_before_report(tmp_path, monkeypatch):
    """Prerequisite: report requires at least 1 completed model run."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")
    state = {
        "step": 0,
        "current_version": "v1",
        "worker_runs": {"profile": 1, "eda": 1, "fe": 1},
        "best_metric_value": None,
    }

    action, note = guard_action(state, mem, make_decision("report"))
    assert action == "model"
    assert "model" in note.lower()


def test_guard_prerequisites_report_before_finish(tmp_path, monkeypatch):
    """Prerequisite: cannot finish before generating final report."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")
    state = {
        "step": 0,
        "worker_runs": {"profile": 1, "eda": 1, "fe": 1, "model": 1},
        "best_metric_value": 0.8,
    }

    action, note = guard_action(state, mem, make_decision("finish"))
    assert action == "report"
    assert "report" in note.lower()


def test_guard_duplicate_brief_blocked(tmp_path, monkeypatch):
    """Duplicate-brief guard: same action + identical brief hash is blocked."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")

    dec1 = make_decision("fe", objective="Scale numerical features", focus=["Age", "Fare"], mode="patch")
    h = compute_brief_hash(dec1.brief)
    row = LedgerRow(
        step=3,
        ts="2026-09-29T12:00:00Z",
        agent="fe",
        action="fe",
        brief_summary="fe: Scale numerical features",
        status="ok",
        result_summary="Completed",
        version_id="v1",
        score=None,
        delta_vs_best=None,
        judge_overall=None,
        issues=[],
        concern=None,
        decision_reason="reason",
        brief_hash=h,
        duration_s=2.0,
    )
    mem.append_ledger(row)

    state = {
        "step": 4,
        "current_version": "v1",
        "worker_runs": {"profile": 1, "eda": 1, "fe": 1, "model": 1},
    }

    action, note = guard_action(state, mem, dec1)
    assert note is not None
    assert "identical brief" in note


def test_guard_judge_consecutive_blocked(tmp_path, monkeypatch):
    """Judge-consecutive guard: judge cannot run twice in a row."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")

    mem.append_ledger(LedgerRow(
        step=4, ts="t", agent="judge", action="judge", brief_summary="j1", status="ok",
        result_summary="done", version_id="v1", score=None, delta_vs_best=None,
        judge_overall="improve", issues=[], concern=None, decision_reason="r", duration_s=1.0,
    ))

    state = {
        "step": 5,
        "current_version": "v1",
        "worker_runs": {"profile": 1, "eda": 1, "fe": 1, "model": 1, "judge": 1},
    }

    action, note = guard_action(state, mem, make_decision("judge"))
    assert action != "judge"
    assert "Judge cannot run twice" in note


def test_guard_allows_valid_actions(tmp_path, monkeypatch):
    """Guards do not block valid actions when prerequisites are met."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")

    state = {
        "step": 3,
        "current_version": "v1",
        "worker_runs": {"profile": 1, "eda": 1, "fe": 1},
        "best_metric_value": None,
    }

    # model is valid here: profile+eda+fe done and current_version set
    dec = make_decision("model", objective="Train RF and LogReg with 5-fold CV")
    action, note = guard_action(state, mem, dec)
    assert action == "model"
    assert note is None


def test_guard_allows_unrestricted_model_runs(tmp_path, monkeypatch):
    """Phase 0: No per-worker cap — model can run many times as supervisor judges necessary."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")

    # Add previous model ledger rows but with different brief hashes
    for i in range(5):
        mem.append_ledger(LedgerRow(
            step=i + 2, ts="t", agent="model", action="model",
            brief_summary=f"model run {i}", status="ok",
            result_summary="done", version_id=f"v{i+1}", score=0.7 + i * 0.01,
            delta_vs_best=0.01, judge_overall=None,
            issues=[], concern=None, decision_reason="r",
            brief_hash=f"hash_{i}", duration_s=5.0,
        ))

    state = {
        "step": 8,
        "current_version": "v5",
        "worker_runs": {"profile": 1, "eda": 1, "fe": 3, "model": 5},
        "best_metric_value": 0.74,
    }

    # Model run 6 with a NEW objective should pass — no caps enforced
    dec = make_decision("model", objective="Try GradientBoosting with deeper trees", mode="tune")
    action, note = guard_action(state, mem, dec)
    assert action == "model"
    assert note is None


def test_get_deterministic_fallback_ordering(tmp_path, monkeypatch):
    """get_deterministic_fallback returns the next unfinished pipeline step in order."""
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)
    mem = RunMemory("p_test", "r_test")

    # Empty state: should start at profile
    assert get_deterministic_fallback({"worker_runs": {}}, mem) == "profile"

    # After profile: eda
    assert get_deterministic_fallback({"worker_runs": {"profile": 1}}, mem) == "eda"

    # After eda: fe
    assert get_deterministic_fallback({"worker_runs": {"profile": 1, "eda": 1}}, mem) == "fe"

    # After fe (but no version): still fe
    assert get_deterministic_fallback({"worker_runs": {"profile": 1, "eda": 1, "fe": 1}, "current_version": None}, mem) == "fe"

    # After fe with version: model
    assert get_deterministic_fallback({"worker_runs": {"profile": 1, "eda": 1, "fe": 1}, "current_version": "v1"}, mem) == "model"

    # After model: judge (if no report yet)
    state = {"worker_runs": {"profile": 1, "eda": 1, "fe": 1, "model": 1}, "current_version": "v1", "best_metric_value": 0.8}
    assert get_deterministic_fallback(state, mem) == "judge"

    # After judge: report
    state["worker_runs"]["judge"] = 1
    assert get_deterministic_fallback(state, mem) == "report"

    # After report: finish
    state["worker_runs"]["report"] = 1
    assert get_deterministic_fallback(state, mem) == "finish"
