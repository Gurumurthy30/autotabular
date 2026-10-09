"""LangGraph pipeline graph assembly and orchestrator.

Purpose:
  Coordinates the execution flow of the mlagent system as a LangGraph state graph.
  Enforces deterministic routing, budget bounds, crash recovery, and state persistence.
  The graph IS the controller: plain code decides every edge; LLMs only reason inside agent nodes.

What it reads:
  - Ledger state (ledger.json) for problems, profiles, strategies, queue items, runs, and budgets.
  - Routing state (control.json) for retry counters, tuner stage, and pending validations.

What it writes:
  - Updates routing state in control.json after every node step.
  - Updates ledger status / stop reason when stopping or completing.

Who calls it:
  - CLI runner (mlagent run, mlagent resume) via run_pipeline(ws).
  - Diagnostic and inspection tools via mermaid().
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import leaderboard_check, ui
from .agents import (
    api_docs_lookup,
    data_profiler,
    experiment_planner,
    experiment_runner,
    final_submission,
    hyperparameter_tuner,
    results_analyzer,
    run_validator,
)
from .ledger import (
    Workspace,
    experiments_used,
    hard_stop,
    is_api_error,
    is_oom_error,
    load,
    load_control,
    next_pending,
    save_control,
    session,
    set_item_status,
    stop_reason,
    unvalidated_ids,
    verdicts,
)
from .llm import RateLimitError

# Maximum number of API-error retry attempts per experiment with library cheat-sheets.
MAX_API_RETRIES: int = 2

# Maximum number of experiments allowed after a hyperparameter tuner failure.
REENTRY_RUNS: int = 2

# Recursion limit for graph traversal to support extensive exploration loops.
GRAPH_RECURSION_LIMIT: int = 5000

# Error tail length captured for diagnostic crash routing.
ERROR_TAIL_CHARS: int = 1500


class GraphState(TypedDict, total=False):
    """Execution state passed along the LangGraph pipeline."""
    workspace: str
    next: str                        # controller decision for next node
    current_batch: list[str]         # queue ids executing currently (len <= parallel.workers)
    retries: dict[str, int]          # exp_id -> Docs-assisted retries used
    oom_retries: dict[str, int]      # exp_id -> downgrade retries used
    crashes: list[dict[str, str]]    # [{exp, kind: api|oom|logic, error}] awaiting routing
    pending_validation: list[str]
    runs_since_analysis: int
    analysis_trigger: str            # periodic | empty_queue | tuner_failure | crash
    analyzer_empty: bool             # Results Analyzer already ran and queue is still empty
    tuned: bool                      # Hyperparameter tuner has run (loop re-entry is bounded)
    tuner_ids: list[str]
    tuner_retry_used: bool
    reentry_left: int
    resume_tuner: bool               # resume found unvalidated / untuned tuner work
    abort: bool


# ---- node plumbing -----------------------------------------------------------------------------

def stop_pipeline(ws: Workspace, reason: str, status: str = "stopped") -> None:
    """Updates ledger status and stop reason atomically when terminating the pipeline."""
    with session(ws) as L:
        L.status = status           # persisted in ledger.json: do not rename
        L.stop_reason = reason      # persisted in ledger.json: do not rename


def node_wrapper(name: str, fatal: bool = False) -> Callable:
    """Wraps an agent execution step with error logging, rate-limit handling, and control persistence."""
    def decorator(fn: Callable[[Workspace, GraphState], dict[str, Any] | None]) -> Callable[[GraphState], dict[str, Any]]:
        def wrapped(state: GraphState) -> dict[str, Any]:
            ws = Workspace(state["workspace"])
            try:
                output = fn(ws, state) or {}
                # Persist state after each completed node execution
                save_control(ws, {**state, **output})
                return output
            except RateLimitError as exc:
                # Rate limit stops gracefully so user can resume when quota resets
                ui.error(f"rate limit exhausted in {name}: {exc}")
                stop_pipeline(ws, "rate_limit")
                save_control(ws, state)
                return {"abort": True}
            except Exception as exc:
                ws.logs.mkdir(parents=True, exist_ok=True)
                error_log_path = ws.logs / "errors.log"
                with error_log_path.open("a", encoding="utf-8") as f:
                    f.write(f"[{name}] {time.ctime()}\n{traceback.format_exc()}\n")
                ui.error(f"{name} failed: {type(exc).__name__}: {exc} (details in logs/errors.log)")
                if fatal:
                    stop_pipeline(ws, f"error:{name}")
                    return {"abort": True}
                # Non-fatal error: clear pending validation and prevent re-dispatching same batch
                return {"crashes": list(state.get("crashes") or []), "pending_validation": []}
        wrapped.__name__ = name
        return wrapped
    return decorator


def classify_crash(err: str, exp: str, state: GraphState, ws: Workspace) -> str:
    """Classifies an experiment failure as out-of-memory, missing library API, or generic logic bug."""
    hardware_cfg = ws.config().hardware
    oom_count = state.get("oom_retries", {}).get(exp, 0)
    if hardware_cfg.oom_downgrade and is_oom_error(err) and oom_count < hardware_cfg.max_oom_retries:
        return "oom"

    api_count = state.get("retries", {}).get(exp, 0)
    if is_api_error(err) and api_count < MAX_API_RETRIES:
        return "api"

    return "logic"


# ---- agent nodes ---------------------------------------------------------------------------------

@node_wrapper("data_profiler", fatal=True)
def data_profiler_node(ws: Workspace, s: GraphState) -> None:
    """Data Profiler node: inspects datasets, types, flags, and creates the baseline Profile."""
    data_profiler.run(ws)


@node_wrapper("experiment_planner", fatal=True)
def experiment_planner_node(ws: Workspace, s: GraphState) -> None:
    """Experiment Planner node: selects validation split scheme and populates initial hypotheses."""
    experiment_planner.run(ws)


@node_wrapper("pipeline_controller")
def pipeline_controller_node(ws: Workspace, s: GraphState) -> dict[str, Any]:
    """Pure-code controller node: evaluates stopping criteria, batches queue items, and directs routing."""
    L = load(ws)
    is_tuned = s.get("tuned", False)
    reentry_left = s.get("reentry_left", 0)

    # Calculate stop reason based on budget or re-entry constraints
    if is_tuned:
        stop_cond = hard_stop(L) or ("reentry_done" if reentry_left <= 0 else None)
    else:
        stop_cond = stop_reason(L)

    pending_items = [q for q in L.queue if q.status == "pending"]

    # Handle queue exhaustion
    if stop_cond is None and not pending_items:
        if s.get("analyzer_empty"):
            stop_cond = "empty_queue"  # safe exit to prevent infinite empty analyzer looping
        else:
            ui.say("Pipeline Controller", "queue empty → Results Analyzer")
            return {
                "next": "results_analyzer",
                "analysis_trigger": "empty_queue",
                "current_batch": [],
            }

    # Handle loop termination
    if stop_cond:
        with session(ws) as L_curr:
            L_curr.stop_reason = L_curr.stop_reason or stop_cond
        next_target = "final_submission" if is_tuned else "hyperparameter_tuner"
        ui.say("Pipeline Controller", f"loop stopped: {stop_cond} → {next_target}")
        return {"next": next_target, "current_batch": []}

    # Dispatch next batch of experiments
    max_parallel_workers = ws.config().parallel.workers
    experiments_remaining = L.env.budget.max_experiments - experiments_used(L)
    extra_runs_allowed = reentry_left if is_tuned else 10**6

    batch_size = max(1, min(max_parallel_workers, len(pending_items), experiments_remaining, extra_runs_allowed))
    batch_ids = [q.id for q in pending_items[:batch_size]]

    if batch_size > 1:
        ui.say("Pipeline Controller", f"running {batch_size} experiments in parallel: {', '.join(batch_ids)}")

    output: dict[str, Any] = {"next": "experiment_runner", "current_batch": batch_ids}
    if is_tuned:
        output["reentry_left"] = reentry_left - batch_size
    return output


@node_wrapper("experiment_runner")
def experiment_runner_node(ws: Workspace, s: GraphState) -> dict[str, Any]:
    """Experiment Runner node: executes batch experiments in parallel subprocesses."""
    batch = s.get("current_batch", [])
    workers = ws.config().parallel.workers
    results = experiment_runner.run_batch(ws, batch, workers)

    crashes = [
        {"exp": exp_id, "kind": classify_crash(err, exp_id, s, ws), "error": err[-ERROR_TAIL_CHARS:]}
        for exp_id, err in results.items() if err
    ]
    successful_runs = [exp_id for exp_id, err in results.items() if not err]

    return {
        "crashes": crashes,
        "pending_validation": successful_runs,
        "analyzer_empty": False,
        "runs_since_analysis": s.get("runs_since_analysis", 0) + len(successful_runs),
    }


@node_wrapper("api_docs_lookup")
def api_docs_lookup_node(ws: Workspace, s: GraphState) -> dict[str, Any]:
    """API Docs Lookup node: inspects library signatures and scrapes official docs on API failure."""
    retries = dict(s.get("retries", {}))
    api_crashes = [c for c in s.get("crashes", []) if c["kind"] == "api"]

    for crash in api_crashes:
        api_docs_lookup.lookup(ws, crash["error"], crash["exp"])
        set_item_status(ws, crash["exp"], "pending")  # retry with cheat sheet in context
        retries[crash["exp"]] = retries.get(crash["exp"], 0) + 1

    remaining_crashes = [c for c in s.get("crashes", []) if c["kind"] != "api"]
    return {"retries": retries, "crashes": remaining_crashes}


@node_wrapper("downgrade")
def downgrade_node(ws: Workspace, s: GraphState) -> dict[str, Any]:
    """Downgrade node: handles OOM errors by retrying lighter variants (smaller batch/epochs, then CPU)."""
    oom_counts = dict(s.get("oom_retries", {}))
    oom_crashes = [c for c in s.get("crashes", []) if c["kind"] == "oom"]

    for crash in oom_crashes:
        next_level = oom_counts.get(crash["exp"], 0) + 1
        with session(ws) as L:
            for item in L.queue:
                if item.id == crash["exp"]:
                    item.params = {**item.params, "downgrade": next_level}
                    item.status = "pending"
        oom_counts[crash["exp"]] = next_level
        ui.say("Pipeline Controller", f"{crash['exp']} ran out of memory → retry with downgrade level {next_level}")

    remaining_crashes = [c for c in s.get("crashes", []) if c["kind"] != "oom"]
    return {"oom_retries": oom_counts, "crashes": remaining_crashes}


@node_wrapper("run_validator")
def run_validator_node(ws: Workspace, s: GraphState) -> dict[str, Any]:
    """Run Validator node: audits completed runs for leakage, sanity, and valid predictions."""
    run_validator.run(ws, s.get("pending_validation", []))
    return {"pending_validation": []}


@node_wrapper("results_analyzer")
def results_analyzer_node(ws: Workspace, s: GraphState) -> dict[str, Any]:
    """Results Analyzer node: diagnoses bottlenecks and adds new hypothesis queue items."""
    crashes = s.get("crashes", [])
    trigger = "crash" if crashes else s.get("analysis_trigger", "periodic")
    error_summary = "\n---\n".join(f"{c['exp']}: {c['error']}" for c in crashes[:3]) if crashes else None

    results_analyzer.run(ws, trigger, error_summary)

    has_pending = next_pending(load(ws)) is not None
    return {
        "runs_since_analysis": 0,
        "analyzer_empty": not has_pending,
        "analysis_trigger": "periodic",
        "crashes": [],
    }


@node_wrapper("hyperparameter_tuner")
def hyperparameter_tuner_node(ws: Workspace, s: GraphState) -> dict[str, Any]:
    """Hyperparameter Tuner node: conducts Optuna search over top approved models."""
    tuned_ids = hyperparameter_tuner.run(ws)
    all_tuner_ids = list(dict.fromkeys(tuned_ids + [i for i in unvalidated_ids(load(ws)) if i.startswith("t_")]))
    return {
        "tuned": True,
        "tuner_ids": all_tuner_ids,
        "pending_validation": all_tuner_ids,
        "resume_tuner": False,
    }


@node_wrapper("tuner_validator")
def tuner_validator_node(ws: Workspace, s: GraphState) -> dict[str, Any]:
    """Tuner Validator node: validates tuned models and gates entry into final ensembling."""
    tuner_ids = s.get("tuner_ids", [])
    run_validator.run(ws, tuner_ids)

    L = load(ws)
    v_dict = verdicts(L)
    rejected_ids = [t_id for t_id in tuner_ids if v_dict.get(t_id) == "reject"]

    output: dict[str, Any] = {"pending_validation": []}
    if rejected_ids and not s.get("tuner_retry_used") and hard_stop(L) is None:
        ui.say("Pipeline Controller", f"tuner results rejected ({', '.join(rejected_ids)}) → Results Analyzer, bounded loop re-entry")
        output.update(analysis_trigger="tuner_failure", reentry_left=REENTRY_RUNS, tuner_retry_used=True)
    elif rejected_ids:
        ui.say("Pipeline Controller", "tuner rejected; accepting best-effort (no re-entry budget)")

    return output


@node_wrapper("final_submission")
def final_submission_node(ws: Workspace, s: GraphState) -> None:
    """Final Submission node: computes ensemble, generates submission CSV, and writes final report."""
    final_submission.run(ws)


@node_wrapper("lb_check")
def leaderboard_check_node(ws: Workspace, s: GraphState) -> None:
    """Leaderboard Check node: checks local validation against public leaderboard if enabled."""
    if not s.get("abort"):
        leaderboard_check.run(ws)


# ---- routers (pure functions of state / config) ---------------------------------------------------

def crash_route(s: GraphState) -> str | None:
    """Routes an active crash to the appropriate recovery node (API docs lookup, downgrade, or analyzer)."""
    crash_kinds = {c["kind"] for c in s.get("crashes", [])}
    route_priority = (("api", "api_docs_lookup"), ("oom", "downgrade"), ("logic", "results_analyzer"))
    for kind, target in route_priority:
        if kind in crash_kinds:
            return target
    return None


# Where the graph goes next: starts at Data Profiler, Experiment Planner, Tuner, Validator, or Controller based on state.
def route_start(s: GraphState) -> str:
    """Entry point router determining where the pipeline should resume or begin."""
    L = load(Workspace(s["workspace"]))
    if L.profile is None:
        return "data_profiler"
    if L.strategy is None:
        return "experiment_planner"
    if s.get("resume_tuner"):
        return "hyperparameter_tuner"
    if s.get("pending_validation"):
        return "run_validator"
    return "pipeline_controller"


# Where the graph goes next: advances to the next setup step or aborts to Final Submission.
def after_setup(nxt: str) -> Callable[[GraphState], str]:
    """Routes setup step to the next configured node or terminates early if aborted."""
    return lambda s: "final_submission" if s.get("abort") else nxt


# Where the graph goes next: advances to the node selected by the Controller or aborts.
def after_controller(s: GraphState) -> str:
    """Routes after controller decision: dispatches to experiment runner, analyzer, tuner, or finisher."""
    return "final_submission" if s.get("abort") else s["next"]


# Where the graph goes next: routes to Run Validator, crash recovery, or back to Pipeline Controller.
def after_experimenter(s: GraphState) -> str:
    """Routes after experiment batch: validates successful runs or triages crash handling."""
    if s.get("abort"):
        return "final_submission"
    if s.get("pending_validation"):
        return "run_validator"
    recovery_node = crash_route(s)
    return recovery_node if recovery_node else "pipeline_controller"


# Where the graph goes next: routes to crash recovery, periodic Results Analyzer, or Pipeline Controller.
def after_validator(s: GraphState) -> str:
    """Routes after validation audit: checks for crashes or triggers periodic progress analysis."""
    if s.get("abort"):
        return "final_submission"
    recovery_node = crash_route(s)
    if recovery_node:
        return recovery_node

    analyze_interval = load(Workspace(s["workspace"])).env.budget.analyze_every
    if s.get("runs_since_analysis", 0) >= analyze_interval:
        return "results_analyzer"
    return "pipeline_controller"


# Where the graph goes next: resumes Pipeline Controller after crash recovery, or aborts.
def after_crash_node(s: GraphState) -> str:
    """Routes after docs lookup or downgrade retry preparation back into the loop."""
    if s.get("abort"):
        return "final_submission"
    recovery_node = crash_route(s)
    return recovery_node if recovery_node else "pipeline_controller"


# Where the graph goes next: returns to Pipeline Controller after Results Analyzer completes.
def after_analyzer(s: GraphState) -> str:
    """Routes after analysis back to controller to schedule newly queued hypotheses."""
    return "final_submission" if s.get("abort") else "pipeline_controller"


# Where the graph goes next: validates tuned models or skips directly to Final Submission.
def after_tuner(s: GraphState) -> str:
    """Routes after tuning: validates tuner outputs or advances directly to final submission."""
    if s.get("tuner_ids") and not s.get("abort"):
        return "tuner_validator"
    return "final_submission"


# Where the graph goes next: triggers re-entry analysis on tuner failure or advances to Final Submission.
def after_tuner_validator(s: GraphState) -> str:
    """Routes after tuner validation: triggers re-entry analysis or completes ensembling."""
    if s.get("abort"):
        return "final_submission"
    if s.get("analysis_trigger") == "tuner_failure":
        return "results_analyzer"
    return "final_submission"


# ---- graph assembly ------------------------------------------------------------------------------

def build_graph():
    """Assembles and compiles the full LangGraph state machine with all nodes and conditional edges."""
    graph = StateGraph(GraphState)

    node_registry = [
        ("data_profiler", data_profiler_node),
        ("experiment_planner", experiment_planner_node),
        ("pipeline_controller", pipeline_controller_node),
        ("experiment_runner", experiment_runner_node),
        ("api_docs_lookup", api_docs_lookup_node),
        ("downgrade", downgrade_node),
        ("run_validator", run_validator_node),
        ("results_analyzer", results_analyzer_node),
        ("hyperparameter_tuner", hyperparameter_tuner_node),
        ("tuner_validator", tuner_validator_node),
        ("final_submission", final_submission_node),
        ("lb_check", leaderboard_check_node),
    ]

    for node_name, node_fn in node_registry:
        graph.add_node(node_name, node_fn)

    crash_destinations = ["api_docs_lookup", "downgrade", "results_analyzer"]

    graph.add_conditional_edges(
        START,
        route_start,
        ["data_profiler", "experiment_planner", "pipeline_controller", "run_validator", "hyperparameter_tuner"]
    )
    graph.add_conditional_edges("data_profiler", after_setup("experiment_planner"), ["experiment_planner", "final_submission"])
    graph.add_conditional_edges("experiment_planner", after_setup("pipeline_controller"), ["pipeline_controller", "final_submission"])
    graph.add_conditional_edges(
        "pipeline_controller",
        after_controller,
        ["experiment_runner", "results_analyzer", "hyperparameter_tuner", "final_submission"]
    )
    graph.add_conditional_edges(
        "experiment_runner",
        after_experimenter,
        ["run_validator", "pipeline_controller", "final_submission", *crash_destinations]
    )
    graph.add_conditional_edges(
        "run_validator",
        after_validator,
        ["pipeline_controller", "final_submission", *crash_destinations]
    )
    graph.add_conditional_edges(
        "api_docs_lookup",
        after_crash_node,
        ["pipeline_controller", "final_submission", *crash_destinations]
    )
    graph.add_conditional_edges(
        "downgrade",
        after_crash_node,
        ["pipeline_controller", "final_submission", *crash_destinations]
    )
    graph.add_conditional_edges("results_analyzer", after_analyzer, ["pipeline_controller", "final_submission"])
    graph.add_conditional_edges("hyperparameter_tuner", after_tuner, ["tuner_validator", "final_submission"])
    graph.add_conditional_edges("tuner_validator", after_tuner_validator, ["results_analyzer", "final_submission"])
    graph.add_edge("final_submission", "lb_check")
    graph.add_edge("lb_check", END)

    return graph.compile()


def initial_state(ws: Workspace) -> GraphState:
    """Prepares fresh or resumed routing state from control.json and unfinished ledger items."""
    with session(ws) as L:
        for item in L.queue:
            if item.status == "running":  # interrupted mid-experiment
                item.status = "pending"

    L = load(ws)
    persisted_control = load_control(ws)
    state: GraphState = {
        "workspace": str(ws.root),
        "retries": {},
        "oom_retries": {},
        "crashes": [],
        "runs_since_analysis": 0,
        **persisted_control,  # type: ignore[typeddict-item]
    }

    pending_ids = list(dict.fromkeys(list(state.get("pending_validation") or []) + unvalidated_ids(L)))
    state["pending_validation"] = [i for i in pending_ids if not i.startswith("t_")]
    state["resume_tuner"] = any(i.startswith("t_") for i in pending_ids)
    return state


def run_pipeline(ws: Workspace) -> None:
    """Executes or resumes the entire multi-agent ML engineering pipeline."""
    graph = build_graph()
    try:
        graph.invoke(initial_state(ws), config={"recursion_limit": GRAPH_RECURSION_LIMIT})
    except KeyboardInterrupt:
        stop_pipeline(ws, "user")
        ui.warn(f"interrupted. Everything is saved in {ws.root} — resume with: mlagent resume {ws.root}")


def mermaid() -> str:
    """Generates mermaid markdown representation of the compiled pipeline graph."""
    return build_graph().get_graph().draw_mermaid()
