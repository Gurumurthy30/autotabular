"""Deterministic brief construction and context engineering for supervisor and workers."""

import json
import time
from typing import Any

from app.config import PROJECTS_DIR
from app.core.run_memory import RunMemory
from app.core.schemas import SupervisorDecision
from app.core.state import ProjectState
from app.utils.logger import get_logger

_log = get_logger(__name__)

SUPERVISOR_CONTEXT_LIMIT = 8000
WORKER_BRIEF_LIMIT = 6000
JUDGE_CONTEXT_LIMIT = 8000
REPORT_CONTEXT_LIMIT = 7000


def check_and_log_budget(name: str, text: str, budget: int) -> bool:
    """Logs the size of the generated context and warns if the character budget is exceeded."""
    size = len(text)
    if size > budget:
        _log.warning(
            "[%s] Context length %d exceeded character budget %d by %d chars!",
            name, size, budget, size - budget
        )
        return False
    else:
        _log.info("[%s] Context length: %d / %d chars", name, size, budget)
        return True


def compact_profile(profile_data: dict[str, Any]) -> str:
    """Generates a concise summary of the dataset profile."""
    if not profile_data:
        return "No profile data available."

    rows = profile_data.get("row_count", "unknown")
    cols = profile_data.get("column_count", "unknown")
    missing_pct = profile_data.get("missing_total_pct", 0.0)
    task_guess = profile_data.get("task_type_guess", "unknown")
    target_dist = profile_data.get("target_distribution", {})

    columns = profile_data.get("columns", [])
    dtypes_count: dict[str, int] = {}
    high_missing = []
    for c in columns:
        dt = str(c.get("dtype", "unknown"))
        dtypes_count[dt] = dtypes_count.get(dt, 0) + 1
        if c.get("missing_pct", 0) > 10.0:
            high_missing.append(f"{c.get('name')}({c.get('missing_pct'):.1f}%)")

    dtypes_str = ", ".join(f"{k}:{v}" for k, v in dtypes_count.items())
    missing_str = ", ".join(high_missing[:5]) if high_missing else "None (>10%)"
    if len(high_missing) > 5:
        missing_str += f" (+{len(high_missing) - 5} more)"

    return (
        f"Shape: ({rows}, {cols}) | Task: {task_guess} | Total Missing: {missing_pct:.1f}%\n"
        f"Dtypes: [{dtypes_str}]\n"
        f"High Missing: [{missing_str}]\n"
        f"Target Stats: {json.dumps(target_dist, default=str)[:200]}"
    )


def compact_eda(findings_data: dict[str, Any], project_id: str | None = None) -> str:
    """Selects top 12 findings ordered by priority (high>med>low) and formats them deterministically."""
    if not findings_data:
        return "No EDA findings available."

    raw_findings = findings_data.get("findings", [])
    if not raw_findings:
        return "No EDA findings listed."

    priority_weight = {"high": 3, "med": 2, "medium": 2, "low": 1}

    def sort_key(item: dict[str, Any]) -> int:
        p = str(item.get("priority", "low")).lower()
        return priority_weight.get(p, 0)

    sorted_findings = sorted(raw_findings, key=sort_key, reverse=True)
    top_12 = sorted_findings[:12]

    lines = []
    for idx, f in enumerate(top_12, 1):
        f_id = f.get("id", f"F{idx}")
        cat = f.get("category", "General")
        finding_text = str(f.get("finding", "")).replace("\n", " ").strip()[:200]
        action = str(f.get("action") or f.get("recommendation") or "").replace("\n", " ").strip()[:150]
        lines.append(f"{f_id} | {cat} | {finding_text} | {action}")

    if len(sorted_findings) > 12:
        rem = len(sorted_findings) - 12
        path_hint = f"projects/{project_id}/eda/findings.json" if project_id else "eda/findings.json"
        lines.append(f"{rem} more in {path_hint}")

    result = "\n".join(lines)

    # Cache compact findings if project_id is given
    if project_id:
        try:
            cache_path = PROJECTS_DIR / project_id / "eda" / "findings_compact.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fp:
                json.dump({"compact": lines, "count": len(sorted_findings)}, fp, indent=2)
        except Exception as e:
            _log.debug("Failed to cache compact EDA: %s", e)

    return result


def compact_feature_schema(schema_data: dict[str, Any]) -> str:
    """Generates a compact summary of created features and data schema."""
    if not schema_data:
        return "No feature schema available."
    features = schema_data.get("features", [])
    count = len(features)
    names = [f.get("name") if isinstance(f, dict) else str(f) for f in features[:20]]
    names_str = ", ".join(names)
    if count > 20:
        names_str += f" (+{count - 20} more)"
    return f"Total Features: {count}\nFeatures Sample: [{names_str}]"


def compact_model_summary(model_summary: dict[str, Any]) -> str:
    """Generates a concise summary of models evaluated and train/val scores."""
    if not model_summary:
        return "No model summary available."
    best = model_summary.get("best_model_name", "unknown")
    best_score = model_summary.get("best_metric_value")
    models_list = model_summary.get("models", [])
    details = []
    for m in models_list:
        name = m.get("model_name", "unknown")
        val_score = m.get("validation_score") or m.get("val_score")
        train_score = m.get("train_score")
        gap = m.get("train_val_gap")
        if gap is None and train_score is not None and val_score is not None:
            gap = abs(train_score - val_score)
        val_str = f"{val_score:.4f}" if isinstance(val_score, (int, float)) else str(val_score)
        gap_str = f"gap={gap:.4f}" if isinstance(gap, (int, float)) else ""
        details.append(f"{name}: score={val_str} {gap_str}".strip())

    return f"Best: {best} (score={best_score})\nCandidates: " + "; ".join(details)


def compute_score_history(mem: RunMemory) -> list[dict[str, Any]]:
    """Deterministically extracts model score progression from the ledger."""
    rows = mem.read_ledger()
    history: list[dict[str, Any]] = []
    prev_score: float | None = None
    best_score: float | None = None

    for r in rows:
        if r.action in ("model", "modeling") and r.score is not None:
            score = float(r.score)
            delta_prev = (score - prev_score) if prev_score is not None else 0.0
            delta_best = (score - best_score) if best_score is not None else 0.0
            history.append({
                "version": r.version_id or f"step_{r.step}",
                "step": r.step,
                "score": score,
                "delta_vs_prev": round(delta_prev, 5),
                "delta_vs_best": round(delta_best, 5),
            })
            prev_score = score
            if best_score is None or score > best_score:
                best_score = score

    return history


def compute_advisory_notes(mem: RunMemory, state: ProjectState) -> list[str]:
    """Generates advisory notes that the Supervisor sees in context and decides on. Never blocks."""
    notes: list[str] = []
    rows = mem.read_ledger()
    step_cur = state.get("step", 0)

    # 1. M steps and T minutes elapsed
    if rows:
        try:
            t0 = time.strptime(rows[0].ts, "%Y-%m-%dT%H:%M:%SZ")
            t_now = time.gmtime()
            elapsed_m = max(0.0, round((time.mktime(t_now) - time.mktime(t0)) / 60.0, 1))
        except Exception:
            elapsed_m = 0.0
        notes.append(f"{step_cur} steps and {elapsed_m:.1f} minutes elapsed")
    elif step_cur > 0:
        notes.append(f"{step_cur} steps elapsed")

    # 2. Worker X failed N times with the same error
    failed_rows = [r for r in rows if r.status == "failed"]
    if len(failed_rows) >= 2:
        last_f = failed_rows[-1]
        prev_f = failed_rows[-2]
        if last_f.agent == prev_f.agent:
            notes.append(f"worker {last_f.agent} failed 2+ times recently")

    # 3. Last K passes gained less than noise_floor
    model_rows = [r for r in rows if r.action in ("model", "modeling") and r.score is not None]
    if len(model_rows) >= 2:
        insignificant_count = sum(1 for r in model_rows[-2:] if r.significant is False)
        if insignificant_count >= 2:
            notes.append("last 2 passes gained less than noise_floor")

    # 4. Same action and same brief as step N
    if len(rows) >= 2:
        last_r = rows[-1]
        for prev in reversed(rows[:-1]):
            if prev.action == last_r.action and prev.brief_hash and prev.brief_hash == last_r.brief_hash:
                notes.append(f"same action and same brief as step {prev.step}")
                break

    return notes


def render_supervisor_context(
    mem: RunMemory,
    state: ProjectState,
    guard_note: str | None = None,
    open_concern: dict[str, Any] | None = None,
) -> str:
    """Renders the strictly formatted supervisor context.

    Sections: MISSION | PLAN | BEST | PROGRESS | LEDGER | NOTES | OPEN CONCERN | PROJECT LESSONS
    """
    mission = mem.get_mission()
    user_goal = mission.get("user_goal") or state.get("user_goal", "Train tabular ML model")
    target_col = mission.get("target_column") or state.get("target_column")
    target_metric = mission.get("target_metric") or state.get("target_metric")
    task_type = mission.get("task_type") or state.get("task_type")

    sec_mission = (
        "=== MISSION ===\n"
        f"Goal: {user_goal}\n"
        f"Target Column: {target_col} | Target Metric: {target_metric} | Task Type: {task_type}"
    )

    plan_items = mem.get_plan()
    if plan_items:
        plan_str = "\n".join(f"- [{p.get('status', 'todo')}] ({p.get('id')}): {p.get('text')}" for p in plan_items)
    else:
        plan_str = "No active plan yet."
    sec_plan = f"=== PLAN ===\n{plan_str}"

    best = mem.get_best()
    if best:
        best_str = f"Version: {best.get('version_id')} | Score: {best.get('score')} | Metric: {best.get('metric')} | Step: {best.get('step')}"
    else:
        best_str = "None yet."
    sec_best = f"=== BEST ===\n{best_str}"

    # Show actual pipeline progress (what has been done)
    step_cur = state.get("step", 0)
    worker_runs = state.get("worker_runs", {})
    worker_counts_str = ", ".join(f"{w}:{cnt}" for w, cnt in worker_runs.items()) if worker_runs else "none yet"
    sec_progress = (
        f"=== PROGRESS ===\n"
        f"Supervisor Steps Taken: {step_cur}\n"
        f"Worker Runs: {worker_counts_str}"
    )

    ledger_str = mem.render_ledger(max_chars=3000)
    sec_ledger = f"=== LEDGER ===\n{ledger_str}"

    notes_list = compute_advisory_notes(mem, state)
    if guard_note:
        notes_list.append(f"Guard notification: {guard_note}")
    if notes_list:
        notes_str = "\n".join(f"- {n}" for n in notes_list)
    else:
        notes_str = "None"
    sec_notes = f"=== NOTES ===\n{notes_str}"

    if open_concern:
        claim = open_concern.get("claim", "")
        ev = open_concern.get("evidence", "")
        sug = open_concern.get("suggestion", "")
        sec_concern = f"=== OPEN CONCERN ===\nClaim: {claim}\nEvidence: {ev}\nSuggestion: {sug}"
    else:
        sec_concern = "=== OPEN CONCERN ===\nNone"

    lessons = mem.get_project_lessons()
    if lessons:
        lessons_str = "\n".join(f"- {l}" for l in lessons)
    else:
        lessons_str = "None recorded."
    sec_lessons = f"=== PROJECT LESSONS ===\n{lessons_str}"

    sections = [
        sec_mission,
        sec_plan,
        sec_best,
        sec_progress,
        sec_ledger,
        sec_notes,
        sec_concern,
        sec_lessons,
    ]

    context = "\n\n".join(sections)
    check_and_log_budget("Supervisor Context", context, SUPERVISOR_CONTEXT_LIMIT)
    return context


def build_worker_brief(
    agent: str,
    decision: SupervisorDecision | dict[str, Any],
    mem: RunMemory,
    state: ProjectState,
) -> str:
    """Builds a targeted, bounded brief for an individual worker node."""
    project_id = state["project_id"]
    run_id = state.get("run_id", "unknown")

    if isinstance(decision, dict):
        brief_data = decision.get("brief", {})
        objective = brief_data.get("objective", "")
        focus_points = brief_data.get("focus_points", [])
        constraints = brief_data.get("constraints", [])
        mode = brief_data.get("mode")
    else:
        objective = decision.brief.objective
        focus_points = decision.brief.focus_points
        constraints = decision.brief.constraints
        mode = decision.brief.mode

    mission = mem.get_mission()
    target_col = mission.get("target_column") or state.get("target_column")
    target_metric = mission.get("target_metric") or state.get("target_metric")
    task_type = mission.get("task_type") or state.get("task_type")
    user_goal = mission.get("user_goal") or state.get("user_goal")

    # Section A: Supervisor Brief
    sec_a = (
        f"=== SUPERVISOR BRIEF ===\n"
        f"Objective: {objective}\n"
        f"Focus Points: {', '.join(focus_points) if focus_points else 'None'}\n"
        f"Constraints: {', '.join(constraints) if constraints else 'None'}\n"
        f"Mode: {mode or 'default'}"
    )

    # Section B: Mission
    sec_b = (
        f"=== MISSION ===\n"
        f"Goal: {user_goal}\n"
        f"Target: {target_col} | Metric: {target_metric} | Task Type: {task_type}"
    )

    # Section C: Agent-Specific Inputs
    sec_c_parts = []
    if agent == "eda":
        profile_text = compact_profile(state.get("profile_summary", {}))
        sec_c_parts.append(f"Profile Compact:\n{profile_text}")
        eda_data = state.get("eda_findings", {})
        existing_ids = [f.get("id") for f in eda_data.get("findings", []) if f.get("id")]
        if existing_ids:
            sec_c_parts.append(f"Existing Finding IDs: {', '.join(existing_ids[:15])}")

    elif agent in ("fe", "features", "feature_engineering"):
        eda_compact = compact_eda(state.get("eda_findings", {}), project_id=project_id)
        profile_text = compact_profile(state.get("profile_summary", {}))
        sec_c_parts.append(f"Top EDA Findings:\n{eda_compact}")
        sec_c_parts.append(f"Profile Compact:\n{profile_text}")
        schema_path = f"projects/{project_id}/features/feature_schema.json"
        prev_features = state.get("feature_summary", {}).get("created_features", [])
        prev_names = [f.get("feature_name") for f in prev_features[:10]]
        sec_c_parts.append(f"Previous Features ({len(prev_features)}): {', '.join(prev_names)}")
        sec_c_parts.append(f"Schema Path: {schema_path}")
        judge_rep = state.get("judge_report", {})
        advice = judge_rep.get("advice", [])
        if advice:
            sec_c_parts.append(f"Judge Advice: {'; '.join(advice[:3])}")
        sec_c_parts.append(f"Execution Mode: {mode or 'new'}")

    elif agent in ("model", "modeling"):
        schema_compact = compact_feature_schema(state.get("feature_summary", {}))
        sec_c_parts.append(f"Feature Schema:\n{schema_compact}")
        curr_ver = state.get("current_version") or "v1"
        sec_c_parts.append(f"Current Feature Version: {curr_ver}")
        prev_models = compact_model_summary(state.get("model_summary", {}))
        sec_c_parts.append(f"Previous Model Results:\n{prev_models}")
        sec_c_parts.append(f"Execution Mode: {mode or 'new'}")

    elif agent in ("judge", "evaluator"):
        ledger_text = mem.render_ledger(max_chars=2500)
        eda_compact = compact_eda(state.get("eda_findings", {}), project_id=project_id)
        schema_compact = compact_feature_schema(state.get("feature_summary", {}))
        model_compact = compact_model_summary(state.get("model_summary", {}))
        score_history = compute_score_history(mem)
        sec_c_parts.append(f"Ledger:\n{ledger_text}")
        sec_c_parts.append(f"Compact EDA:\n{eda_compact}")
        sec_c_parts.append(f"Feature Schema:\n{schema_compact}")
        sec_c_parts.append(f"Model Summary:\n{model_compact}")
        sec_c_parts.append(f"Score History: {json.dumps(score_history, default=str)}")

    elif agent in ("report", "reporting"):
        best_meta = mem.get_best() or {}
        ledger_text = mem.render_ledger(max_chars=2500)
        judge_report = state.get("judge_report", {})
        sec_c_parts.append(f"Best Version Meta:\n{json.dumps(best_meta, default=str)}")
        sec_c_parts.append(f"Ledger:\n{ledger_text}")
        sec_c_parts.append(f"Last Judge Report:\n{json.dumps(judge_report, default=str)[:800]}")
        # Extract decision reasons from ledger
        rows = mem.read_ledger()
        reasons = [f"Step {r.step} ({r.action}): {r.decision_reason}" for r in rows if r.decision_reason]
        sec_c_parts.append("Decision Log:\n" + "\n".join(reasons[-10:]))

    sec_c = "=== AGENT SPECIFIC INPUTS ===\n" + "\n\n".join(sec_c_parts)

    # Section D: Agent Top Lessons
    lessons = mem.top_lessons(agent, n=5)
    lessons_str = "\n".join(f"- {l}" for l in lessons) if lessons else "None"
    sec_d = f"=== YOUR NOTEBOOK LESSONS ===\n{lessons_str}"

    # Section E: File Pointers
    file_pointers = [
        f"Profile JSON: projects/{project_id}/profile/profile.json",
        f"EDA Findings: projects/{project_id}/eda/findings.json",
        f"Feature Schema: projects/{project_id}/features/feature_schema.json",
        f"Best Version Meta: projects/{project_id}/runs/{run_id}/memory/best.json",
    ]
    sec_e = "=== FILE POINTERS (Inspect on demand, never dump raw) ===\n" + "\n".join(file_pointers)

    brief_text = "\n\n".join([sec_a, sec_b, sec_c, sec_d, sec_e])

    budget = (
        JUDGE_CONTEXT_LIMIT if agent in ("judge", "evaluator")
        else (REPORT_CONTEXT_LIMIT if agent in ("report", "reporting") else WORKER_BRIEF_LIMIT)
    )
    check_and_log_budget(f"Worker Brief: {agent}", brief_text, budget)
    return brief_text
