"""Code guards enforcing structural prerequisite rules for the Supervisor-led pipeline.

KEPT:
  - Prerequisite order guard (eda needs profile, fe needs eda, model needs fe, etc.)
  - Duplicate-brief guard (blocks identical re-submission of same action+brief)
  - Judge-consecutive guard (judge cannot run twice in a row without a work step)
"""

import hashlib
import json
from typing import Any

from app.core.run_memory import RunMemory
from app.core.schemas import BriefContent, SupervisorDecision
from app.core.state import ProjectState
from app.utils.logger import get_logger

_log = get_logger(__name__)


def compute_brief_hash(brief: BriefContent | dict[str, Any]) -> str:
    """Computes a deterministic hash of a worker brief."""
    data = brief.model_dump() if hasattr(brief, "model_dump") else brief
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_deterministic_fallback(state: ProjectState, mem: RunMemory) -> str:
    """Determines the next unfinished step in pipeline sequence: profile -> eda -> fe -> model -> judge -> report -> finish."""
    worker_runs = state.get("worker_runs", {})

    if worker_runs.get("profile", 0) == 0:
        return "profile"
    if worker_runs.get("eda", 0) == 0:
        return "eda"
    if worker_runs.get("fe", 0) == 0 or state.get("current_version") is None:
        return "fe"
    if worker_runs.get("model", 0) == 0 or state.get("best_metric_value") is None:
        return "model"
    if worker_runs.get("judge", 0) == 0 and worker_runs.get("report", 0) == 0:
        return "judge"
    if worker_runs.get("report", 0) == 0:
        return "report"
    return "finish"


def guard_action(
    state: ProjectState,
    mem: RunMemory,
    decision: SupervisorDecision,
) -> tuple[str, str | None]:
    """Applies structural prerequisite guards on the supervisor's chosen action.

    Returns (final_action, guard_note).
    If guard_note is None, the chosen action passed all guards.
    If guard_note is not None, the action was overridden by guard rules.

    Guards (in priority order):
      1. Prerequisites: pipeline ordering must be respected.
      2. Duplicate-brief: block identical action+brief re-submission.
      3. Judge-consecutive: judge cannot run twice in a row.
    """
    worker_runs = state.get("worker_runs", {})
    action = decision.action
    report_done = worker_runs.get("report", 0) > 0

    # 1. Prerequisites: pipeline ordering must be respected.
    if action == "eda" and worker_runs.get("profile", 0) == 0:
        return "profile", "Prerequisite missing: eda requires profile to be completed first."

    if action == "fe":
        if worker_runs.get("profile", 0) == 0:
            return "profile", "Prerequisite missing: fe requires profile to be completed first."
        mode = getattr(decision.brief, "mode", None)
        if mode != "basic" and worker_runs.get("eda", 0) == 0:
            return "eda", "Prerequisite missing: non-basic fe requires eda findings first."

    if action == "model":
        if worker_runs.get("fe", 0) == 0 or state.get("current_version") is None:
            return "fe", "Prerequisite missing: model requires a successful feature engineering version."

    if action == "judge":
        if worker_runs.get("model", 0) == 0 and state.get("best_metric_value") is None:
            return "model", "Prerequisite missing: judge requires at least 1 completed model run."

    if action == "report":
        if worker_runs.get("model", 0) == 0 and state.get("best_metric_value") is None:
            return "model", "Prerequisite missing: report requires at least 1 completed model run."

    if action == "finish":
        if not report_done:
            return "report", "Prerequisite missing: cannot finish before generating final report."

    # 2. Duplicate-brief guard: same action + same brief hash as the previous step -> blocked.
    ledger = mem.read_ledger()
    if ledger:
        last_row = ledger[-1]
        if last_row.action == action:
            curr_hash = compute_brief_hash(decision.brief)
            prev_brief = getattr(last_row, "brief_summary", "")
            prev_hash = getattr(last_row, "brief_hash", None)
            if (prev_hash and prev_hash == curr_hash) or (curr_hash in prev_brief) or (decision.brief.objective and decision.brief.objective in prev_brief):
                fallback = get_deterministic_fallback(state, mem)
                return fallback, f"Action '{action}' repeated with identical brief hash. Supervisor cannot repeat identical brief."

    # 3. Judge-consecutive guard: judge cannot run twice in a row without a work step between.
    if action == "judge" and ledger:
        if ledger[-1].action in ("judge", "evaluator"):
            fallback = get_deterministic_fallback(state, mem)
            return fallback, "Judge cannot run twice consecutively without an intervening work step."

    return action, None
