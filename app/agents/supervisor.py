"""The Lead ML Supervisor Agent: real-time LLM decision-maker orchestrating the worker team."""

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.guards import get_deterministic_fallback, guard_action
from app.agents.prompts import SUPERVISOR_SYSTEM_PROMPT
from app.config import PROJECTS_DIR
from app.core.briefs import (
    compute_advisory_notes,
    render_supervisor_context,
)
from app.core.llm_json import invoke_json
from app.core.model_router import ModelRouter
from app.core.run_memory import RunMemory
from app.core.schemas import BriefContent, LedgerRow, SupervisorDecision
from app.core.state import ProjectState
from app.tools.registry import ToolRegistry
from app.utils.logger import get_logger

_log = get_logger(__name__)

__all__ = ["PROJECTS_DIR", "extract_cross_run_lessons", "run_supervisor"]


def extract_cross_run_lessons(
    mem: RunMemory,
    router: ModelRouter,
    state: ProjectState,
) -> None:
    """Invokes LLM once to extract at most 2 cross-run lessons and stores them in project memory."""
    try:
        from pydantic import BaseModel, Field

        class ProjectLessonsExtraction(BaseModel):
            lessons: list[str] = Field(default_factory=list)

        llm = router.get_model("supervisor", temperature=0.2)

        prompt = f"""Review the completed run for Project '{state.get('project_id')}'.
Target: {state.get('target_column')} | Metric: {state.get('target_metric')} | Best Score: {state.get('best_metric_value')}

Ledger Summary:
{mem.render_ledger(max_chars=2000)}

Extract at most 2 short, highly actionable data science lessons (max 1 sentence each) to remember for future runs on similar datasets."""

        parsed, err, raw = invoke_json(
            llm,
            [
                SystemMessage(content="You are a data science team lead summarizing key learnings. Return JSON with 'lessons': [str, str]."),
                HumanMessage(content=prompt),
            ],
            ProjectLessonsExtraction,
            agent_name="supervisor_lessons",
        )
        if parsed and parsed.lessons:
            for l in parsed.lessons[:2]:
                clean_l = str(l).strip()
                if clean_l:
                    mem.add_project_lesson(clean_l)
                    _log.info("[SUPERVISOR] Saved cross-run lesson: %s", clean_l)
    except Exception as exc:
        _log.warning("[SUPERVISOR] Failed to extract cross-run lessons: %s", exc)


def supervisor_node(
    state: ProjectState,
    router: ModelRouter,
    registry: ToolRegistry,
) -> dict[str, Any]:
    """The real LLM Supervisor agent: evaluates state, issues brief, and determines next action."""
    start_time = time.monotonic()
    project_id = state["project_id"]
    run_id = state.get("run_id", "unknown")
    step = state.get("step", 0)

    mem = RunMemory(project_id, run_id)
    degraded = state.get("degraded", False)

    # 1. Cooperative cancellation check
    if mem.is_cancelled() or state.get("cancelled", False):
        _log.info("[SUPERVISOR] Run has been cancelled cooperatively at step %d.", step)
        mem.append_ledger(
            LedgerRow(
                step=step,
                ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                agent="supervisor",
                action="finish",
                brief_summary="Run cancelled cooperatively by user",
                status="ok",
                result_summary="Run CANCELLED via cooperative cancellation",
                decision_reason="Stop flag detected",
                duration_s=time.monotonic() - start_time,
            )
        )
        # If any model exists, allow report from best version if not yet done
        has_best = mem.get_best() is not None
        report_done = state.get("worker_runs", {}).get("report", 0) > 0
        if has_best and not report_done:
            return {
                "next_action": "report",
                "step": step + 1,
                "status": "CANCELLED",
                "cancelled": True,
            }
        return {
            "next_action": "finish",
            "step": step + 1,
            "status": "CANCELLED",
            "cancelled": True,
        }

    # 2. Record advisory notes in ledger if new
    notes = compute_advisory_notes(mem, state)
    for note_text in notes:
        # Check if identical note logged in last 3 rows to avoid flooding
        recent_ledger = mem.read_ledger()[-3:]
        already_logged = any(r.status == "note" and r.brief_summary == note_text for r in recent_ledger)
        if not already_logged:
            mem.append_ledger(
                LedgerRow(
                    step=step,
                    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    agent="system",
                    action="note",
                    brief_summary=note_text[:200],
                    status="note",
                    result_summary=note_text[:300],
                    decision_reason="Advisory warning from system context",
                    duration_s=0.0,
                )
            )

    llm = router.get_model("supervisor", temperature=0.1)

    open_concern = state.get("open_concern")
    guard_note = state.get("guard_note")

    decision: SupervisorDecision | None = None
    final_action = "profile"
    guard_overrides_list: list[str] = []

    # Ask the Supervisor LLM (with up to 2 retries if blocked by guards)
    for attempt in range(3):
        context = render_supervisor_context(
            mem=mem,
            state=state,
            guard_note=guard_note,
            open_concern=open_concern,
        )

        parsed_decision, err_msg, raw_text = invoke_json(
            llm,
            [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ],
            SupervisorDecision,
            agent_name="supervisor",
        )
        decision = parsed_decision

        if decision is None:
            # LLM parse failure after repair: use deterministic fallback
            degraded = True
            fallback_action = get_deterministic_fallback(state, mem)
            _log.warning(
                "[SUPERVISOR] LLM decision failed to parse after repair (err: %s); using fallback '%s'. Run marked degraded.",
                err_msg,
                fallback_action,
            )
            unique_objective = f"Step {step} fallback {fallback_action}: follow standard quality checks (recovering from parse failure: {err_msg[:60]})"
            decision = SupervisorDecision(
                thought=f"Deterministic fallback at step {step} due to LLM decision parse error.",
                action=fallback_action,
                brief=BriefContent(
                    objective=unique_objective,
                    focus_points=["Standard execution", f"Step {step} recovery"],
                    constraints=["Adhere to standard quality checks"],
                    mode="standard",
                ),
                reason=f"Step {step} automatic progression following parse error: {err_msg[:100]}",
            )

        # Run code guards on proposed action
        validated_action, guard_override_note = guard_action(state, mem, decision)

        if guard_override_note is None:
            # Action approved by all guards
            final_action = validated_action
            break
        else:
            guard_overrides_list.append(guard_override_note)
            _log.warning(
                "[SUPERVISOR] Guard override on attempt %d: proposed '%s' blocked: %s",
                attempt + 1, decision.action, guard_override_note,
            )
            # Record guard override in ledger
            mem.append_ledger(
                LedgerRow(
                    step=step,
                    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    agent="supervisor",
                    action=decision.action,
                    brief_summary=f"Proposed {decision.action} ({decision.brief.objective[:60]})"[:200],
                    status="guard_override",
                    result_summary=f"Blocked: {guard_override_note}"[:300],
                    decision_reason=decision.reason[:300],
                    duration_s=time.monotonic() - start_time,
                )
            )

            if attempt < 2:
                # Re-ask Supervisor with updated GUARD NOTE
                guard_note = guard_override_note
            else:
                # Retries exhausted: enforce deterministic fallback
                final_action = validated_action
                _log.warning("[SUPERVISOR] Guard retry limit reached. Enforcing action: %s", final_action)
                decision.action = final_action
                break

    # Save plan updates if proposed
    split_strategy = state.get("split_strategy")
    if decision and decision.plan_update:
        mem.set_plan(decision.plan_update)
        if isinstance(decision.plan_update, dict) and "split_strategy" in decision.plan_update:
            split_strategy = decision.plan_update["split_strategy"]

    # If action is finish, run cross-run learning
    if final_action == "finish":
        extract_cross_run_lessons(mem, router, state)

    # Maintain version iteration for DB/UI backward compatibility
    current_ver = state.get("current_version") or "v1"
    try:
        iter_num = int(current_ver.replace("v", ""))
    except Exception:
        iter_num = 1

    elapsed = time.monotonic() - start_time
    _log.info(
        "[SUPERVISOR] Turn complete | step=%d | action=%s | reason=%s | degraded=%s | elapsed=%.2fs",
        step, final_action, decision.reason[:80] if decision else "N/A", degraded, elapsed,
    )

    plan_val = mem.get_plan()

    return {
        "next_action": final_action,
        "step": step + 1,
        "iteration": iter_num,
        "plan": plan_val if isinstance(plan_val, list) else [plan_val],
        "split_strategy": split_strategy,
        "supervisor_decision": decision.model_dump() if decision else {},
        "guard_overrides": guard_overrides_list,
        "guard_note": None,          # Clear guard note for next turn
        "open_concern": None,        # Clear open concern as Supervisor has addressed it
        "degraded": degraded,
        "status": "RUNNING" if final_action != "finish" else "SUCCESS",
    }
