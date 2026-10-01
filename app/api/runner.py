import asyncio
from datetime import UTC, datetime

from sqlmodel import Session, select

from app.config import PROJECTS_DIR
from app.core.events import event_manager
from app.core.model_router import ModelRouter
from app.core.run_memory import RunMemory
from app.core.state import ProjectState, TaskType, to_evidence_str
from app.core.versions import is_higher_better
from app.db.models import (
    ArtifactIndex,
    EDAFinding,
    FeatureVersion,
    WorkflowRun,
)
from app.db.session import engine
from app.graph.build_graph import build_ml_graph
from app.tools.dataset_tools import DatasetTools
from app.utils.logger import get_logger

_log = get_logger(__name__)

_CANCELLED_RUNS: set[str] = set()
_CANCELLED_PROJECTS: set[str] = set()


def cancel_run(run_id: str) -> None:
    _CANCELLED_RUNS.add(run_id)


def cancel_project_runs(project_id: str) -> None:
    _CANCELLED_PROJECTS.add(project_id)


def is_run_cancelled(run_id: str) -> bool:
    return run_id in _CANCELLED_RUNS


def is_project_cancelled(project_id: str) -> bool:
    return project_id in _CANCELLED_PROJECTS


def execute_workflow_sync(project_id: str, run_id: str, dataset_version: str, target_column: str | None, target_metric: str | None, constraints: dict | None = None) -> None:
    """Executes the workflow graph synchronously and emits structured events throughout."""
    start_time = datetime.now(UTC)
    _log.info(
        "[RUNNER] Workflow started | project=%s run=%s dataset=%s target=%s metric=%s",
        project_id, run_id, dataset_version, target_column, target_metric,
    )

    # 1. Update run to RUNNING
    with Session(engine) as session:
        run_record = session.get(WorkflowRun, run_id)
        if run_record:
            run_record.status = "RUNNING"
            session.add(run_record)
            session.commit()

    event_manager.emit_event(
        project_id=project_id,
        run_id=run_id,
        event_type="WORKFLOW_STARTED",
        stage="init",
        message=f"Starting autonomous workflow run '{run_id}' for project '{project_id}'.",
        data={"dataset_version": dataset_version, "target_column": target_column, "target_metric": target_metric},
    )

    try:
        # Check dataset file
        d_tools = DatasetTools(project_id)
        try:
            dataset_path = d_tools.get_dataset_path(dataset_version)
        except Exception as e:
            raise FileNotFoundError(f"Dataset for version '{dataset_version}' not found: {e}")

        # Check if target_column is missing initially
        if not target_column:
            with Session(engine) as session:
                run_record = session.get(WorkflowRun, run_id)
                if run_record:
                    run_record.status = "NEEDS_INPUT"
                    run_record.error = "Target column not specified."
                    session.add(run_record)
                    session.commit()

            event_manager.emit_event(
                project_id=project_id,
                run_id=run_id,
                event_type="SUPERVISOR_DECISION",
                stage="supervisor",
                message="Target column is missing. Run paused in NEEDS_INPUT status.",
                data={"status": "NEEDS_INPUT"},
            )
            event_manager.emit_event(
                project_id=project_id,
                run_id=run_id,
                event_type="WORKFLOW_COMPLETED",
                stage="supervisor",
                message="Workflow paused: Needs target column input.",
                data={"status": "NEEDS_INPUT"},
            )
            return

        # Initialize RunMemory and mission.json
        mem = RunMemory(project_id, run_id)
        metric_name = target_metric or "auto"
        mem.init_mission(
            user_goal=f"Train tabular ML model to predict '{target_column}'",
            target_column=target_column or "",
            target_metric=metric_name,
            task_type="",
            metric_direction="higher" if is_higher_better(metric_name) else "lower",
        )

        initial_state: ProjectState = {
            "project_id": project_id,
            "run_id": run_id,
            "user_goal": f"Train tabular ML model to predict '{target_column}'",
            "target_column": target_column,
            "target_metric": target_metric,
            "dataset_version": dataset_version,
            "task_type": None,
            "profile_summary": {},
            "eda_findings": {},
            "feature_summary": {},
            "model_summary": {},
            "current_stage": "start",
            "iteration": 1,
            "step": 0,
            "plan": [],
            "current_version": None,
            "best_version": None,
            "open_concern": None,
            "guard_note": None,
            "worker_runs": {},
            "report": None,
            "best_experiment_id": None,
            "best_metric_value": None,
            "artifacts": [],
            "next_action": "profile",
            "status": "RUNNING",
        }

        # Build graph and run
        router = ModelRouter()
        graph = build_ml_graph(project_id, router=router)

        # Stream graph step executions to emit fine-grained events
        final_state = initial_state
        last_stage = "start"

        for output_chunk in graph.stream(initial_state, config={"recursion_limit": 500}):
            if is_run_cancelled(run_id) or is_project_cancelled(project_id):
                _log.info("[RUNNER] Run %s cancelled or project deleted; terminating stream.", run_id)
                return
            for node_name, node_state in output_chunk.items():
                final_state.update(node_state)
                curr_stage = node_state.get("current_stage", node_name)
                iteration_idx = node_state.get("iteration", 1)

                if node_state.get("status") == "FAILED":
                    error_msg = node_state.get("error", f"Stage '{node_name}' failed.")
                    with Session(engine) as session:
                        run_record = session.get(WorkflowRun, run_id)
                        if run_record:
                            run_record.status = "FAILED"
                            run_record.error = error_msg
                            run_record.completed_at = datetime.now(UTC)
                            session.add(run_record)
                            session.commit()

                    event_manager.emit_event(
                        project_id=project_id,
                        run_id=run_id,
                        event_type="WORKFLOW_COMPLETED",
                        stage=node_name,
                        message=f"Workflow failed at stage '{node_name}': {error_msg}",
                        data={"status": "FAILED", "error": error_msg},
                    )
                    return

                if node_name == "supervisor":
                    dec = node_state.get("supervisor_decision") or {}
                    step_num = node_state.get("step", 1) - 1
                    next_act = node_state.get("next_action") or dec.get("action")
                    reason = dec.get("reason", "")
                    brief_summary = dec.get("brief", {}).get("objective", "") if isinstance(dec.get("brief"), dict) else ""

                    # Emit supervisor_decision (both lower and upper for client compatibility)
                    event_manager.emit_event(
                        project_id=project_id,
                        run_id=run_id,
                        event_type="supervisor_decision",
                        stage="supervisor",
                        message=f"Supervisor step {step_num}: chosen '{next_act}'. {reason}",
                        data={
                            "step": step_num,
                            "action": next_act,
                            "reason": reason,
                            "brief_summary": brief_summary,
                            "thought": dec.get("thought", ""),
                            "current_stage": curr_stage,
                            "next_action": next_act,
                            "iteration": iteration_idx,
                        },
                    )
                    event_manager.emit_event(
                        project_id=project_id,
                        run_id=run_id,
                        event_type="SUPERVISOR_DECISION",
                        stage="supervisor",
                        message=f"Supervisor reviewed '{curr_stage}'. Next route: '{next_act}' (Iteration {iteration_idx}).",
                        data={
                            "step": step_num,
                            "action": next_act,
                            "reason": reason,
                            "brief_summary": brief_summary,
                            "current_stage": curr_stage,
                            "next_action": next_act,
                            "iteration": iteration_idx,
                        },
                    )

                    # Emit guard_override if any occurred
                    for g_reason in node_state.get("guard_overrides", []):
                        event_manager.emit_event(
                            project_id=project_id,
                            run_id=run_id,
                            event_type="guard_override",
                            stage="supervisor",
                            message=f"Guard blocked action: {g_reason}",
                            data={"reason": g_reason},
                        )

                else:
                    # Worker completed: emit legacy stage event
                    event_manager.emit_event(
                        project_id=project_id,
                        run_id=run_id,
                        event_type="AGENT_COMPLETED",
                        stage=node_name,
                        message=f"Completed stage: '{node_name.upper()}' (Iteration {iteration_idx}).",
                        data={"stage": node_name, "iteration": iteration_idx},
                    )

                    report_data = node_state.get("report") or {}
                    score_val = None
                    if node_name == "model":
                        score_val = node_state.get("best_metric_value")
                    elif isinstance(report_data.get("evidence"), dict):
                        score_val = report_data.get("evidence", {}).get("score")

                    # Emit worker_report
                    event_manager.emit_event(
                        project_id=project_id,
                        run_id=run_id,
                        event_type="worker_report",
                        stage=node_name,
                        message=f"Worker '{node_name}' finished: {report_data.get('result_summary', 'completed')}",
                        data={
                            "agent": node_name,
                            "status": report_data.get("status", "ok"),
                            "result_summary": report_data.get("result_summary", ""),
                            "score": score_val,
                            "evidence": report_data.get("evidence", {}),
                        },
                    )

                    # Emit concern if raised by worker
                    if open_concern := node_state.get("open_concern"):
                        event_manager.emit_event(
                            project_id=project_id,
                            run_id=run_id,
                            event_type="concern",
                            stage=node_name,
                            message=f"Concern raised by {node_name}: {open_concern.get('claim')}",
                            data={
                                "agent": node_name,
                                "claim": open_concern.get("claim", ""),
                                "evidence": open_concern.get("evidence", ""),
                                "suggestion": open_concern.get("suggestion", ""),
                            },
                        )

                    if node_name == "profile":
                        task_guess = node_state.get("task_type")
                        if task_guess == TaskType.AMBIGUOUS:
                            with Session(engine) as session:
                                run_record = session.get(WorkflowRun, run_id)
                                if run_record:
                                    run_record.status = "NEEDS_INPUT"
                                    run_record.error = "Ambiguous target column."
                                    session.add(run_record)
                                    session.commit()
                            event_manager.emit_event(
                                project_id=project_id,
                                run_id=run_id,
                                event_type="SUPERVISOR_DECISION",
                                stage="supervisor",
                                message="Target column profile is ambiguous. Pausing for user input.",
                                data={"status": "NEEDS_INPUT"},
                            )
                            event_manager.emit_event(
                                project_id=project_id,
                                run_id=run_id,
                                event_type="WORKFLOW_COMPLETED",
                                stage="supervisor",
                                message="Workflow paused: Ambiguous target column needs user input.",
                                data={"status": "NEEDS_INPUT"},
                            )
                            return

                    elif node_name == "model":
                        event_manager.emit_event(
                            project_id=project_id,
                            run_id=run_id,
                            event_type="EXPERIMENT_COMPLETED",
                            stage="model",
                            message=f"Model training logged to MLflow. Best candidate score: {node_state.get('best_metric_value')}",
                            data={"best_experiment_id": node_state.get("best_experiment_id"), "best_metric_value": node_state.get("best_metric_value")},
                        )

                    elif node_name in ("judge", "evaluator"):
                        judge_rep = node_state.get("judge_report", {})
                        verdict = judge_rep.get("overall")
                        if verdict in ("improve", "blocked"):
                            event_manager.emit_event(
                                project_id=project_id,
                                run_id=run_id,
                                event_type="EVALUATION_FAILED",
                                stage="judge",
                                message=f"Judge verdict: {verdict.upper()}. Requesting refinement.",
                                data=judge_rep,
                            )

                # Log per-step metrics
                step_elapsed = (datetime.now(UTC) - start_time).total_seconds()
                _log.info(
                    "[STEP] step=%s action=%s duration=%.2fs best_score=%s",
                    final_state.get("step"),
                    node_name,
                    step_elapsed,
                    final_state.get("best_metric_value"),
                )

        # 3. Synchronize EDA findings to SQLite
        eda_data = final_state.get("eda_findings", {})
        findings_list = eda_data.get("findings", [])
        with Session(engine) as session:
            for item in findings_list:
                finding_rec = EDAFinding(
                    id=f"eda_{project_id}_{run_id}_{item.get('category')}_{hash(str(item.get('finding')))}",
                    project_id=project_id,
                    run_id=run_id,
                    category=item.get("category", "General"),
                    finding=str(item.get("finding", "")),
                    evidence=to_evidence_str(item.get("evidence", "")),
                    implication=str(item.get("implication", "")),
                    recommendation=str(item.get("recommendation", "")),
                    created_at=datetime.now(UTC),
                )
                session.merge(finding_rec)

            # 4. Synchronize FeatureVersion to SQLite
            fe_data = final_state.get("feature_summary", {})
            if fe_data:
                feat_rec = FeatureVersion(
                    id=f"fv_{project_id}_{run_id}_{fe_data.get('version', 'v1')}",
                    project_id=project_id,
                    run_id=run_id,
                    dataset_version=dataset_version,
                    version_tag=fe_data.get("version", "feat_v1"),
                    pipeline_path=fe_data.get("pipeline_file", ""),
                    data_path=fe_data.get("data_file", ""),
                    schema_path=fe_data.get("schema_file", ""),
                    feature_count=len(fe_data.get("created_features", [])),
                    created_at=datetime.now(UTC),
                )
                session.merge(feat_rec)

            # 5. Synchronize ArtifactIndex for created files (deduplicating by project_id and path)
            artifact_dirs = [
                ("profile", PROJECTS_DIR / project_id / "profile"),
                ("eda", PROJECTS_DIR / project_id / "eda"),
                ("features", PROJECTS_DIR / project_id / "features"),
                ("reports", PROJECTS_DIR / project_id / "reports"),
            ]
            for stage_name, stage_dir in artifact_dirs:
                if stage_dir.exists():
                    for f in stage_dir.rglob("*"):
                        if f.is_file():
                            path_str = str(f)
                            norm_path = str(f.resolve()).replace("\\", "/")
                            existing = session.exec(
                                select(ArtifactIndex).where(
                                    ArtifactIndex.project_id == project_id,
                                    (ArtifactIndex.path == path_str) | (ArtifactIndex.path == norm_path),
                                )
                            ).first()
                            if not existing:
                                art_id = f"art_{project_id}_{stage_name}_{f.name}"
                                art_rec = ArtifactIndex(
                                    id=art_id,
                                    project_id=project_id,
                                    run_id=run_id,
                                    artifact_type=stage_name,
                                    path=path_str,
                                    version=dataset_version,
                                    created_at=datetime.now(UTC).isoformat(),
                                )
                                session.add(art_rec)

            # 6. Update WorkflowRun record
            run_record = session.get(WorkflowRun, run_id)
            if run_record:
                run_record.status = "SUCCESS"
                run_record.current_stage = "report"
                run_record.iteration = final_state.get("iteration", 1)
                run_record.best_experiment_id = final_state.get("best_experiment_id")
                run_record.best_metric_value = final_state.get("best_metric_value")
                run_record.completed_at = datetime.now(UTC)
                session.add(run_record)
                session.commit()

        elapsed = (datetime.now(UTC) - start_time).total_seconds()
        _log.info(
            "[RUNNER] Workflow completed | project=%s run=%s | status=SUCCESS | elapsed=%.1fs",
            project_id, run_id, elapsed,
        )
        event_manager.emit_event(
            project_id=project_id,
            run_id=run_id,
            event_type="WORKFLOW_COMPLETED",
            stage="report",
            message=f"Workflow run '{run_id}' completed successfully in {elapsed:.1f}s.",
            data={
                "status": "SUCCESS",
                "elapsed_seconds": elapsed,
                "best_experiment_id": final_state.get("best_experiment_id"),
                "best_metric_value": final_state.get("best_metric_value"),
            },
        )

    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
        is_interrupted = isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
        err_msg = "Workflow execution cancelled or interrupted" if is_interrupted else str(exc)
        _log.exception(
            "[RUNNER] Workflow terminated | project=%s run=%s | error=%s",
            project_id, run_id, err_msg,
        )
        try:
            with Session(engine) as session:
                run_record = session.get(WorkflowRun, run_id)
                if run_record:
                    run_record.status = "FAILED"
                    run_record.error = err_msg
                    run_record.completed_at = datetime.now(UTC)
                    session.add(run_record)
                    session.commit()

            event_manager.emit_event(
                project_id=project_id,
                run_id=run_id,
                event_type="WORKFLOW_COMPLETED",
                stage="error",
                message=f"Workflow run terminated: {err_msg}",
                data={"status": "FAILED", "error": err_msg},
            )
        except Exception as inner_exc:
            _log.exception("[RUNNER] Failed to update DB on workflow failure: %s", inner_exc)

        if is_interrupted:
            raise exc


async def execute_workflow_async(project_id: str, run_id: str, dataset_version: str, target_column: str | None, target_metric: str | None, constraints: dict | None = None) -> None:
    """Async background task adapter that runs workflow synchronously in thread pool."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        execute_workflow_sync,
        project_id,
        run_id,
        dataset_version,
        target_column,
        target_metric,
        constraints,
    )
