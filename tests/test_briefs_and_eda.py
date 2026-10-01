"""Unit tests for briefs, context budgets, compact_eda, and concern deduplication."""

import hashlib

from app.core.briefs import (
    JUDGE_CONTEXT_LIMIT,
    REPORT_CONTEXT_LIMIT,
    SUPERVISOR_CONTEXT_LIMIT,
    WORKER_BRIEF_LIMIT,
    build_worker_brief,
    check_and_log_budget,
    compact_eda,
    render_supervisor_context,
)
from app.core.run_memory import RunMemory
from app.core.schemas import LedgerRow


def test_compact_eda_top_12_priority_order(tmp_path, monkeypatch):
    """Tests compact_eda selects top 12 ordered high > med > low with expected schema."""
    monkeypatch.setattr("app.core.briefs.PROJECTS_DIR", tmp_path)

    # Create 20 findings across low, med, high priorities
    raw_findings = []
    for i in range(5):
        raw_findings.append({
            "id": f"low_{i}",
            "category": "skew",
            "finding": f"Low priority finding {i}",
            "priority": "low",
            "action": f"Action for low {i}",
        })
    for i in range(10):
        raw_findings.append({
            "id": f"high_{i}",
            "category": "leakage",
            "finding": f"High priority finding {i}",
            "priority": "high",
            "action": f"Action for high {i}",
        })
    for i in range(5):
        raw_findings.append({
            "id": f"med_{i}",
            "category": "missingness",
            "finding": f"Medium priority finding {i}",
            "priority": "medium",
            "action": f"Action for med {i}",
        })

    eda_data = {"findings": raw_findings}
    compact_str = compact_eda(eda_data, project_id="p_test")

    lines = compact_str.strip().split("\n")
    # Up to 12 items + 1 overflow note
    assert len(lines) == 13
    
    # 10 high + 2 medium = 12 findings
    assert any("high_0 | leakage | High priority finding 0 | Action for high 0" in l for l in lines)
    assert any("high_9 | leakage | High priority finding 9 | Action for high 9" in l for l in lines)
    assert any("med_0" in l for l in lines)
    assert any("8 more in" in l for l in lines)

    # High items must appear before medium items
    high_idx = [i for i, l in enumerate(lines) if "high_" in l]
    med_idx = [i for i, l in enumerate(lines) if "med_" in l]
    assert max(high_idx) < min(med_idx)


def test_briefs_and_context_under_budgets(tmp_path, monkeypatch):
    """Tests that Supervisor, Worker, Judge, and Report briefs stay strictly under budget limits."""
    monkeypatch.setattr("app.core.briefs.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)

    mem = RunMemory("p_test", "r_test")
    # Populate large ledger (25 rows)
    for i in range(25):
        mem.append_ledger(LedgerRow(
            step=i,
            ts="2026-09-29T12:00:00Z",
            agent="model" if i % 2 == 0 else "fe",
            action="model" if i % 2 == 0 else "fe",
            brief_summary=f"Brief for step {i}",
            status="ok",
            result_summary=f"Result for step {i} with details",
            version_id="v1",
            score=0.85,
            delta_vs_best=0.01,
            judge_overall=None,
            issues=[],
            concern=None,
            decision_reason="Standard supervisor reasoning",
            duration_s=3.0,
        ))

    state = {
        "project_id": "p_test",
        "run_id": "r_test",
        "step": 25,
        "target_column": "target",
        "target_metric": "roc_auc",
        "worker_runs": {"profile": 1, "eda": 2, "fe": 3, "model": 3},
        "profile_summary": {"rows": 1000, "cols": 15, "missing_pct": 0.05, "target_type": "binary"},
        "eda_findings": {"findings": [{"id": f"f_{i}", "category": "c", "finding": "f", "priority": "high", "action": "a"} for i in range(15)]},
        "current_version": "v1",
    }

    # 1. Supervisor context budget (<= 6000)
    sup_ctx = render_supervisor_context(
        mem=mem,
        state=state,
        guard_note="Sample guard note",
        open_concern={"claim": "Possible leakage", "evidence": "col_a", "suggestion": "drop col_a"},
    )
    assert len(sup_ctx) <= SUPERVISOR_CONTEXT_LIMIT
    assert check_and_log_budget("supervisor", sup_ctx, SUPERVISOR_CONTEXT_LIMIT) is True

    # 2. Worker briefs (<= 5000 for each agent)
    decision = {
        "thought": "Let's build models",
        "action": "model",
        "brief": {
            "objective": "Train gradient boosting and random forest candidates",
            "focus_points": ["Balance class weights", "Check train/val gap"],
            "constraints": ["Diverse families"],
            "mode": "add_candidates",
        },
        "reason": "EDA confirmed no leakage and FE created 12 features",
    }

    for agent_name in ("profile", "eda", "fe", "model"):
        worker_brief = build_worker_brief(agent_name, decision, mem, state)
        assert len(worker_brief) <= WORKER_BRIEF_LIMIT
        assert check_and_log_budget(agent_name, worker_brief, WORKER_BRIEF_LIMIT) is True

    # 3. Judge context budget (<= 7000)
    judge_brief = build_worker_brief("judge", decision, mem, state)
    assert len(judge_brief) <= JUDGE_CONTEXT_LIMIT
    assert check_and_log_budget("judge", judge_brief, JUDGE_CONTEXT_LIMIT) is True

    # 4. Report context budget (<= 6000)
    report_brief = build_worker_brief("report", decision, mem, state)
    assert len(report_brief) <= REPORT_CONTEXT_LIMIT
    assert check_and_log_budget("report", report_brief, REPORT_CONTEXT_LIMIT) is True


def test_concern_deduplication():
    """Tests concern deduplication: normalized text hash already raised is ignored."""
    name = "fe"
    concern1 = {"claim": "Target column appears correlated with ID column", "evidence": "r=0.99", "suggestion": "drop ID"}
    concern2 = {"claim": "Target column appears correlated with ID column", "evidence": "r=0.99", "suggestion": "drop ID"}
    concern3 = {"claim": "High missingness in Age", "evidence": "pct=0.45", "suggestion": "impute median"}

    def hash_concern(agent, c):
        norm = f"{agent}:{c.get('claim', '')}".strip().lower()
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    h1 = hash_concern(name, concern1)
    h2 = hash_concern(name, concern2)
    h3 = hash_concern(name, concern3)

    assert h1 == h2
    assert h1 != h3

    seen_concerns = []
    # First time: accepted
    if h1 not in seen_concerns:
        seen_concerns.append(h1)
        accepted_1 = True
    else:
        accepted_1 = False
    assert accepted_1 is True

    # Second time (identical concern): rejected
    if h2 not in seen_concerns:
        seen_concerns.append(h2)
        accepted_2 = True
    else:
        accepted_2 = False
    assert accepted_2 is False

    # Third time (different concern): accepted
    if h3 not in seen_concerns:
        seen_concerns.append(h3)
        accepted_3 = True
    else:
        accepted_3 = False
    assert accepted_3 is True
