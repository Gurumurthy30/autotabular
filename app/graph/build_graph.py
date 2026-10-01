"""LangGraph workflow definition for the Supervisor-led autonomous ML pipeline."""

import hashlib
import time
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.agents.eda import run_eda
from app.agents.feature_engineering import run_feature_engineering
from app.agents.guards import compute_brief_hash
from app.agents.judge import run_judge
from app.agents.model import run_modeling
from app.agents.profile import profile_dataset
from app.agents.report import run_report
from app.agents.supervisor import supervisor_node
from app.core.briefs import build_worker_brief
from app.core.model_router import ModelRouter
from app.core.run_memory import RunMemory
from app.core.schemas import LedgerRow
from app.core.state import ProjectState
from app.core.versions import restore_version, snapshot_version, update_best
from app.tools.registry import ToolRegistry
from app.utils.logger import get_logger

_log = get_logger(__name__)


def make_worker_node(name: str, fn: Any, router: ModelRouter, registry: ToolRegistry):
    """Wraps every specialist worker to enforce standardized timing, ledger/notebook logging, version snapshots, best tracking, and concern deduplication."""
    def worker_wrapper(state: ProjectState) -> dict[str, Any]:
        project_id = state["project_id"]
        run_id = state.get("run_id", "unknown")
        mem = RunMemory(project_id, run_id)
        decision_dict = state.get("supervisor_decision") or {}

        # 1. Build targeted brief
        brief = build_worker_brief(name, decision_dict, mem, state)

        # 2. Time worker execution
        t0 = time.monotonic()
        if name == "profile":
            res = fn(state, registry)
        else:
            res = fn(state, router, registry, brief=brief)
        duration_s = time.monotonic() - t0

        # 3. Update worker execution counts
        worker_runs = dict(state.get("worker_runs", {}))
        canonical_key = "fe" if name == "features" else ("judge" if name == "evaluator" else name)
        worker_runs[canonical_key] = worker_runs.get(canonical_key, 0) + 1

        # 4. Extract WorkerReport
        raw_report = res.get("report") or {}
        if isinstance(raw_report, dict):
            status = raw_report.get("status", "ok" if res.get("status") == "SUCCESS" else "failed")
            result_summary = str(raw_report.get("result_summary") or res.get("error") or f"{name} completed")[:300]
            evidence = raw_report.get("evidence", {})
            concern = raw_report.get("concern")
            notebook_data = raw_report.get("notebook") or {
                "step": state.get("step", 0),
                "tried": f"Executed {name}",
                "outcome": status,
                "lesson": result_summary[:100],
            }
        else:
            status = "ok"
            result_summary = f"{name} completed"
            evidence = {}
            concern = None
            notebook_data = {"step": state.get("step", 0), "tried": name, "outcome": "ok", "lesson": "done"}

        # 5. Concern deduplication: ignore concern whose normalized text hash was already raised
        open_concern = None
        if concern:
            claim = str(concern.get("claim", ""))
            norm = f"{name}:{claim}".strip().lower()
            h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            seen_concerns = list(state.get("seen_concerns", []))
            if h not in seen_concerns:
                seen_concerns.append(h)
                open_concern = concern
                res["seen_concerns"] = seen_concerns
            else:
                _log.info("[GRAPH] Suppressing duplicate concern from '%s': %s", name, claim[:60])

        # 6. Extract score and delta
        score = None
        delta_vs_best = None
        curr_best = mem.get_best()
        best_score = float(curr_best["score"]) if curr_best and curr_best.get("score") is not None else None

        if name == "model":
            score_val = res.get("best_metric_value") or evidence.get("best_metric_value") or evidence.get("score")
            if score_val is not None:
                score = float(score_val)
                if best_score is not None:
                    delta_vs_best = score - best_score

        judge_overall = None
        if name in ("judge", "evaluator"):
            judge_overall = evidence.get("overall")

        mean_val = evidence.get("mean") or evidence.get("cv_mean")
        std_val = evidence.get("std") or evidence.get("cv_std")
        noise_floor_val = evidence.get("noise_floor")
        significant_val = evidence.get("significant")

        error_text = None
        if status != "ok":
            err_raw = evidence.get("error") or res.get("error") or result_summary
            error_text = str(err_raw)[-300:]

        decision_reason = str(decision_dict.get("reason", ""))[:300]
        brief_summary = str(decision_dict.get("brief", {}).get("objective", f"Execute {name}"))[:200]
        brief_hash = compute_brief_hash(decision_dict.get("brief", {}))

        # 7. Record ledger and notebook in ONE place
        ledger_row = LedgerRow(
            step=state.get("step", 0),
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            agent=name,
            action=canonical_key,
            brief_summary=brief_summary,
            status="ok" if status == "ok" else ("no_gain" if status == "no_gain" else "failed"),
            result_summary=result_summary,
            version_id=state.get("current_version"),
            score=score,
            mean=float(mean_val) if mean_val is not None else None,
            std=float(std_val) if std_val is not None else None,
            noise_floor=float(noise_floor_val) if noise_floor_val is not None else None,
            significant=bool(significant_val) if significant_val is not None else None,
            delta_vs_best=delta_vs_best,
            judge_overall=judge_overall,
            issues=[],
            concern=open_concern,
            decision_reason=decision_reason,
            brief_hash=brief_hash,
            duration_s=duration_s,
            error_text=error_text,
        )
        mem.append_ledger(ledger_row)
        mem.append_notebook(canonical_key, notebook_data)

        # 8. Versioning: snapshot and update_best
        current_ver = state.get("current_version")
        if name in ("features", "fe") and status == "ok":
            fe_count = worker_runs.get("fe", 1)
            current_ver = f"v{fe_count}"
            snapshot_version(project_id, run_id, current_ver, stage="fe")
            res["current_version"] = current_ver
            res["iteration"] = fe_count

        if name == "model" and status == "ok" and score is not None:
            ver = current_ver or "v1"
            snapshot_version(project_id, run_id, ver, stage="model", meta=res.get("model_summary"))
            metric = state.get("target_metric", "f1")
            exp_id = res.get("best_experiment_id")
            improved = update_best(project_id, run_id, ver, score, metric, state.get("step", 0), state, experiment_id=exp_id)
            if improved:
                res["best_metric_value"] = score
                res["best_version"] = ver

        res["worker_runs"] = worker_runs
        if open_concern is not None:
            res["open_concern"] = open_concern

        # Worker FAILED status does not end run, except profile
        if name == "profile" and status == "failed":
            res["status"] = "FAILED"
            res["error"] = "Profile step failed: dataset could not be read or profiled."
        else:
            res["status"] = "RUNNING"

        return res

    return worker_wrapper


def build_ml_graph(project_id: str, router: ModelRouter | None = None) -> StateGraph:
    """Constructs the flat LangGraph supervisor-led workflow: START -> supervisor -> worker -> supervisor -> ... -> END."""
    if router is None:
        router = ModelRouter()
    registry = ToolRegistry(project_id)

    builder = StateGraph(ProjectState)

    # 1. Define nodes
    def _supervisor(state: ProjectState):
        _log.info("[SUPERVISOR] Turn step=%d current_stage='%s'", state.get("step", 0), state.get("current_stage"))
        result = supervisor_node(state, router, registry)
        _log.info("[SUPERVISOR] Chosen next action: '%s'", result.get("next_action"))
        return result

    _profile = make_worker_node("profile", profile_dataset, router, registry)
    _eda = make_worker_node("eda", run_eda, router, registry)
    _features = make_worker_node("features", run_feature_engineering, router, registry)
    _model = make_worker_node("model", run_modeling, router, registry)
    _judge = make_worker_node("judge", run_judge, router, registry)
    _report = make_worker_node("report", run_report, router, registry)

    builder.add_node("supervisor", _supervisor)
    builder.add_node("profile", _profile)
    builder.add_node("eda", _eda)
    builder.add_node("features", _features)
    builder.add_node("model", _model)
    builder.add_node("judge", _judge)
    builder.add_node("report", _report)

    # 2. Add edges: workers always return control to the supervisor
    builder.add_edge(START, "supervisor")
    builder.add_edge("profile", "supervisor")
    builder.add_edge("eda", "supervisor")
    builder.add_edge("features", "supervisor")
    builder.add_edge("model", "supervisor")
    builder.add_edge("judge", "supervisor")
    builder.add_edge("report", "supervisor")

    # 3. Conditional routing from supervisor
    def route_supervisor(state: ProjectState) -> Literal["profile", "eda", "features", "model", "judge", "report", "__end__"]:
        action = state.get("next_action")
        if action == "profile":
            return "profile"
        elif action == "eda":
            return "eda"
        elif action in ("features", "fe"):
            return "features"
        elif action == "model":
            return "model"
        elif action in ("judge", "evaluator"):
            return "judge"
        elif action == "report":
            # Restore best version before report runs
            best_v = state.get("best_version")
            run_id = state.get("run_id", "unknown")
            if best_v and project_id:
                _log.info("[GRAPH] Restoring best version '%s' before report generation", best_v)
                restore_version(project_id, run_id, best_v)
            return "report"
        elif action == "finish":
            return END
        return END

    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "profile": "profile",
            "eda": "eda",
            "features": "features",
            "model": "model",
            "judge": "judge",
            "report": "report",
            END: END,
        },
    )

    return builder.compile()
