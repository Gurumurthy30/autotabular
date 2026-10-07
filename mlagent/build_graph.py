"""LangGraph assembly. The graph IS the controller: plain code decides every edge, LLMs only reason
inside agent nodes. The ledger (ledger.json in the data's workspace) is the source of truth; the graph
state carries routing facts and is mirrored to control.json after every node, so a run resumes exactly
where it stopped (retry counters, tuner stage, pending validations).

   START ─► profiler ─► strategist ─► controller ◄────────────────────────────────────────┐
                                         │ picks a BATCH of ≤ parallel.workers queue items │
          ┌──────────────┬───────────────┼───────────────┐                                  │
          ▼              ▼               ▼               ▼                                  │
     experimenter     analyzer         tuner         finisher ─► lb_check ─► END            │
       │  │                └────────────────────────────────────────────────────────────────┤
       │  └ successes ─► validator ─► (crashes? ─► docs/downgrade/analyzer) ─► every K runs ─► analyzer
       └ crashes:  api ─► docs ─► retry      oom ─► downgrade (smaller, then CPU) ─► retry
                   logic ─► analyzer(crash)                                                │
        tuner ─► tuner_validator ─► (rejected? analyzer ─► controller [bounded re-entry]) / finisher
"""

from __future__ import annotations

import time
import traceback
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import ui
from . import lb
from .agents import analyzer, docs, experimenter, finisher, profiler, strategist, tuner, validator
from .ledger import (Workspace, experiments_used, hard_stop, is_api_error, is_oom_error, load, load_control,
                     next_pending, save_control, session, set_item_status, stop_reason, unvalidated_ids, verdicts)
from .llm import RateLimitError

MAX_API_RETRIES = 2      # Docs-assisted retries per experiment
REENTRY_RUNS = 2         # experiments allowed after a tuner failure


class GraphState(TypedDict, total=False):
    workspace: str
    next: str                        # controller decision
    current_batch: list[str]         # queue ids running now (len ≤ parallel.workers)
    retries: dict[str, int]          # exp_id -> Docs-assisted retries used
    oom_retries: dict[str, int]      # exp_id -> downgrade retries used
    crashes: list[dict]              # [{exp, kind: api|oom|logic, error}] awaiting routing
    pending_validation: list[str]
    runs_since_analysis: int
    analysis_trigger: str            # periodic | empty_queue | tuner_failure   (crash is inferred from crashes)
    analyzer_empty: bool             # Analyzer already tried and the queue is still empty
    tuned: bool                      # Tuner has run (loop re-entry is bounded from here on)
    tuner_ids: list[str]
    tuner_retry_used: bool
    reentry_left: int
    resume_tuner: bool               # resume found unvalidated / untuned tuner work
    abort: bool


# ---- node plumbing -----------------------------------------------------------------------------

def _stop(ws: Workspace, reason: str, status: str = "stopped") -> None:
    with session(ws) as L:
        L.status, L.stop_reason = status, reason      # type: ignore[assignment]


def node(name: str, fatal: bool = False):
    """Wrap an agent: rate limits end the run gracefully, bugs are logged instead of killing the graph,
    and the routing state is persisted so `resume` is exact."""
    def deco(fn):
        def wrapped(state: GraphState) -> dict[str, Any]:
            ws = Workspace(state["workspace"])
            try:
                out = fn(ws, state) or {}
                save_control(ws, {**state, **out})
                return out
            except RateLimitError as e:
                ui.error(f"rate limit exhausted in {name}: {e}")
                _stop(ws, "rate_limit")
                save_control(ws, state)
                return {"abort": True}
            except Exception as e:                                     # noqa: BLE001
                ws.logs.mkdir(parents=True, exist_ok=True)
                with (ws.logs / "errors.log").open("a", encoding="utf-8") as _f:
                    _f.write(f"[{name}] {time.ctime()}\n{traceback.format_exc()}\n")
                ui.error(f"{name} failed: {type(e).__name__}: {e}  (details in logs/errors.log)")
                if fatal:
                    _stop(ws, f"error:{name}")
                    return {"abort": True}
                # Safe sentinel: prevent the controller from re-dispatching the same batch
                return {"crashes": list(state.get("crashes") or []), "pending_validation": []}
        wrapped.__name__ = name
        return wrapped
    return deco


def classify_crash(err: str, exp: str, s: GraphState, ws: Workspace) -> str:
    hw = ws.config().hardware
    if hw.oom_downgrade and is_oom_error(err) and s.get("oom_retries", {}).get(exp, 0) < hw.max_oom_retries:
        return "oom"
    if is_api_error(err) and s.get("retries", {}).get(exp, 0) < MAX_API_RETRIES:
        return "api"
    return "logic"


# ---- nodes ---------------------------------------------------------------------------------------

@node("profiler", fatal=True)
def profiler_node(ws, s):
    profiler.run(ws)


@node("strategist", fatal=True)
def strategist_node(ws, s):
    strategist.run(ws)


@node("controller")
def controller_node(ws, s):
    """Pure code: stop conditions, the next batch of queue items, or hand-off to Analyzer / Tuner / Finisher."""
    L = load(ws)
    tuned, left = s.get("tuned", False), s.get("reentry_left", 0)
    reason = (hard_stop(L) or ("reentry_done" if left <= 0 else None)) if tuned else stop_reason(L)
    pending = [q for q in L.queue if q.status == "pending"]
    if reason is None and not pending:
        if s.get("analyzer_empty"):
            reason = "empty_queue"                                     # safe exit: no infinite loop
        else:
            ui.say("Controller", "queue empty → Analyzer")
            return {"next": "analyzer", "analysis_trigger": "empty_queue", "current_batch": []}
    if reason:
        with session(ws) as L:
            L.stop_reason = L.stop_reason or reason                    # keep the loop's original reason
        ui.say("Controller", f"loop stopped: {reason} → {'finisher' if tuned else 'tuner'}")
        return {"next": "finisher" if tuned else "tuner", "current_batch": []}
    room = L.env.budget.max_experiments - experiments_used(L)
    n = max(1, min(ws.config().parallel.workers, len(pending), room, left if tuned else 10 ** 6))
    batch = [q.id for q in pending[:n]]
    if n > 1:
        ui.say("Controller", f"running {n} experiments in parallel: {', '.join(batch)}")
    out: dict[str, Any] = {"next": "experimenter", "current_batch": batch}
    if tuned:
        out["reentry_left"] = left - n
    return out


@node("experimenter")
def experimenter_node(ws, s):
    batch = s.get("current_batch", [])
    results = experimenter.run_batch(ws, batch, ws.config().parallel.workers)
    crashes = [{"exp": e, "kind": classify_crash(err, e, s, ws), "error": err[-1500:]}
               for e, err in results.items() if err]
    ok = [e for e, err in results.items() if not err]
    return {"crashes": crashes, "pending_validation": ok, "analyzer_empty": False,
            "runs_since_analysis": s.get("runs_since_analysis", 0) + len(ok)}


@node("docs")
def docs_node(ws, s):
    retries = dict(s.get("retries", {}))
    for c in [c for c in s.get("crashes", []) if c["kind"] == "api"]:
        docs.lookup(ws, c["error"], c["exp"])
        set_item_status(ws, c["exp"], "pending")                       # retry with the cheat sheet in context
        retries[c["exp"]] = retries.get(c["exp"], 0) + 1
    return {"retries": retries, "crashes": [c for c in s.get("crashes", []) if c["kind"] != "api"]}


@node("downgrade")
def downgrade_node(ws, s):
    """Out of memory (GPU/RAM): retry the SAME idea lighter (L1), then on CPU with a lighter model (L2)."""
    oom = dict(s.get("oom_retries", {}))
    for c in [c for c in s.get("crashes", []) if c["kind"] == "oom"]:
        level = oom.get(c["exp"], 0) + 1
        with session(ws) as L:
            for q in L.queue:
                if q.id == c["exp"]:
                    q.params = {**q.params, "downgrade": level}
                    q.status = "pending"
        oom[c["exp"]] = level
        ui.say("Controller", f"{c['exp']} ran out of memory → retry with downgrade level {level}")
    return {"oom_retries": oom, "crashes": [c for c in s.get("crashes", []) if c["kind"] != "oom"]}


@node("validator")
def validator_node(ws, s):
    validator.run(ws, s.get("pending_validation", []))
    return {"pending_validation": []}


@node("analyzer")
def analyzer_node(ws, s):
    crashes = s.get("crashes", [])
    trigger = "crash" if crashes else s.get("analysis_trigger", "periodic")
    err = "\n---\n".join(f"{c['exp']}: {c['error']}" for c in crashes[:3]) if crashes else None
    analyzer.run(ws, trigger, err)
    return {"runs_since_analysis": 0, "analyzer_empty": next_pending(load(ws)) is None,
            "analysis_trigger": "periodic", "crashes": []}


@node("tuner")
def tuner_node(ws, s):
    ids = tuner.run(ws)
    ids = list(dict.fromkeys(ids + [i for i in unvalidated_ids(load(ws)) if i.startswith("t_")]))
    return {"tuned": True, "tuner_ids": ids, "pending_validation": ids, "resume_tuner": False}


@node("tuner_validator")
def tuner_validator_node(ws, s):
    ids = s.get("tuner_ids", [])
    validator.run(ws, ids)
    L = load(ws)
    v = verdicts(L)
    rejected = [i for i in ids if v.get(i) == "reject"]
    out: dict[str, Any] = {"pending_validation": []}
    if rejected and not s.get("tuner_retry_used") and hard_stop(L) is None:
        ui.say("Controller", f"tuner results rejected ({', '.join(rejected)}) → Analyzer, bounded loop re-entry")
        out.update(analysis_trigger="tuner_failure", reentry_left=REENTRY_RUNS, tuner_retry_used=True)
    elif rejected:
        ui.say("Controller", "tuner rejected; accepting best-effort (no re-entry budget)")
    return out


@node("finisher")
def finisher_node(ws, s):
    finisher.run(ws)


@node("lb_check")
def lb_check_node(ws, s):
    if not s.get("abort"):
        lb.run(ws)


# ---- routers (pure functions of state / config) ---------------------------------------------------

def crash_route(s: GraphState) -> str | None:
    kinds = {c["kind"] for c in s.get("crashes", [])}
    for kind, target in (("api", "docs"), ("oom", "downgrade"), ("logic", "analyzer")):
        if kind in kinds:
            return target
    return None


def route_start(s: GraphState) -> str:
    L = load(Workspace(s["workspace"]))
    if L.profile is None:
        return "profiler"
    if L.strategy is None:
        return "strategist"
    if s.get("resume_tuner"):
        return "tuner"
    return "validator" if s.get("pending_validation") else "controller"


def after_setup(nxt: str):
    return lambda s: "finisher" if s.get("abort") else nxt


def after_controller(s: GraphState) -> str:
    return "finisher" if s.get("abort") else s["next"]


def after_experimenter(s: GraphState) -> str:
    if s.get("abort"):
        return "finisher"
    return "validator" if s.get("pending_validation") else (crash_route(s) or "controller")


def after_validator(s: GraphState) -> str:
    if s.get("abort"):
        return "finisher"
    if crash_route(s):
        return crash_route(s)
    k = load(Workspace(s["workspace"])).env.budget.analyze_every
    return "analyzer" if s.get("runs_since_analysis", 0) >= k else "controller"


def after_crash_node(s: GraphState) -> str:                              # docs / downgrade
    return "finisher" if s.get("abort") else (crash_route(s) or "controller")


def after_analyzer(s: GraphState) -> str:
    return "finisher" if s.get("abort") else "controller"


def after_tuner(s: GraphState) -> str:
    return "tuner_validator" if s.get("tuner_ids") and not s.get("abort") else "finisher"


def after_tuner_validator(s: GraphState) -> str:
    if s.get("abort"):
        return "finisher"
    return "analyzer" if s.get("analysis_trigger") == "tuner_failure" else "finisher"


# ---- graph ---------------------------------------------------------------------------------------

def build_graph():
    g = StateGraph(GraphState)
    for name, fn in [("profiler", profiler_node), ("strategist", strategist_node), ("controller", controller_node),
                     ("experimenter", experimenter_node), ("docs", docs_node), ("downgrade", downgrade_node),
                     ("validator", validator_node), ("analyzer", analyzer_node), ("tuner", tuner_node),
                     ("tuner_validator", tuner_validator_node), ("finisher", finisher_node),
                     ("lb_check", lb_check_node)]:
        g.add_node(name, fn)

    crash_targets = ["docs", "downgrade", "analyzer"]
    g.add_conditional_edges(START, route_start, ["profiler", "strategist", "controller", "validator", "tuner"])
    g.add_conditional_edges("profiler", after_setup("strategist"), ["strategist", "finisher"])
    g.add_conditional_edges("strategist", after_setup("controller"), ["controller", "finisher"])
    g.add_conditional_edges("controller", after_controller, ["experimenter", "analyzer", "tuner", "finisher"])
    g.add_conditional_edges("experimenter", after_experimenter, ["validator", "controller", "finisher", *crash_targets])
    g.add_conditional_edges("validator", after_validator, ["controller", "finisher", *crash_targets])
    g.add_conditional_edges("docs", after_crash_node, ["controller", "finisher", *crash_targets])
    g.add_conditional_edges("downgrade", after_crash_node, ["controller", "finisher", *crash_targets])
    g.add_conditional_edges("analyzer", after_analyzer, ["controller", "finisher"])
    g.add_conditional_edges("tuner", after_tuner, ["tuner_validator", "finisher"])
    g.add_conditional_edges("tuner_validator", after_tuner_validator, ["analyzer", "finisher"])
    g.add_edge("finisher", "lb_check")
    g.add_edge("lb_check", END)
    return g.compile()


def initial_state(ws: Workspace) -> GraphState:
    """Fresh-or-resumed routing state: persisted control.json + anything the ledger proves is unfinished."""
    with session(ws) as L:
        for q in L.queue:
            if q.status == "running":                                  # interrupted mid-experiment
                q.status = "pending"
    L = load(ws)
    st: GraphState = {"workspace": str(ws.root), "retries": {}, "oom_retries": {}, "crashes": [],
                      "runs_since_analysis": 0, **load_control(ws)}          # type: ignore[typeddict-item]
    ids = list(dict.fromkeys(list(st.get("pending_validation") or []) + unvalidated_ids(L)))
    st["pending_validation"] = [i for i in ids if not i.startswith("t_")]
    st["resume_tuner"] = any(i.startswith("t_") for i in ids)
    return st


def run_pipeline(ws: Workspace) -> None:
    """Run (or resume) the whole pipeline for a prepared workspace."""
    graph = build_graph()
    try:
        graph.invoke(initial_state(ws), config={"recursion_limit": 5000})
    except KeyboardInterrupt:
        _stop(ws, "user")
        ui.warn(f"interrupted. Everything is saved in {ws.root}  — resume with: mlagent resume {ws.root}")


def mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()
