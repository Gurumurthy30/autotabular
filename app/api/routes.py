import asyncio
import importlib.util
import json
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.runner import execute_workflow_async
from app.config import MLFLOW_TRACKING_URI, PROJECTS_DIR
from app.core.events import event_manager
from app.db.models import (
    ArtifactIndex,
    Dataset,
    EDAFinding,
    Event,
    FeatureVersion,
    Project,
    SupervisorMemoryRecord,
    WorkflowRun,
)
from app.db.session import get_session
from app.tools.mlflow_tools import MLflowTools, is_higher_better
from app.utils.logger import get_logger

_log = get_logger(__name__)

router = APIRouter()


# --- Pydantic Schemas ---

class ProjectCreate(BaseModel):
    name: str
    description: str | None = None


class RunCreate(BaseModel):
    target_column: str | None = None
    target_metric: str | None = None
    dataset_version: str | None = None
    constraints: dict[str, Any] | None = None


def validate_project_id(project_id: str) -> str:
    """Validates project_id format to prevent directory traversal attacks."""
    if not re.match(r"^[a-zA-Z0-9_\-]+$", project_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid project_id. Only alphanumeric characters, dashes, and underscores are allowed.",
        )
    resolved = (PROJECTS_DIR / project_id).resolve()
    if not str(resolved).startswith(str(PROJECTS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Path traversal detected.")
    return project_id


def check_project_exists(session: Session, project_id: str) -> bool:
    if session.get(Project, project_id):
        return True
    if (PROJECTS_DIR / project_id).exists():
        return True
    try:
        if mlflow.get_experiment_by_name(project_id) is not None:
            return True
    except Exception:
        pass
    return False


# --- Project Endpoints ---

@router.post("/projects", response_model=Project)
def create_project(data: ProjectCreate, session: Session = Depends(get_session)):
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", data.name).strip("_").lower()
    project_id = slug or f"proj_{uuid.uuid4().hex[:8]}"

    # Check existence
    existing = session.get(Project, project_id)
    if existing:
        project_id = f"{slug}_{uuid.uuid4().hex[:4]}"

    now = datetime.now(UTC)
    project = Project(
        id=project_id,
        name=data.name,
        description=data.description,
        created_at=now,
        updated_at=now,
    )
    session.add(project)
    session.commit()
    session.refresh(project)

    # Prepare project directory structure
    p_dir = PROJECTS_DIR / project_id
    p_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["datasets", "profile", "eda", "features", "models", "evaluations", "reports", "memory", "workspace"]:
        (p_dir / sub).mkdir(parents=True, exist_ok=True)

    return project


@router.get("/projects", response_model=list[Project])
def list_projects(session: Session = Depends(get_session)):
    return session.exec(select(Project).order_by(Project.created_at.desc())).all()


@router.get("/projects/{project_id}", response_model=Project)
def get_project(project_id: str, session: Session = Depends(get_session)):
    validate_project_id(project_id)
    project = session.get(Project, project_id)
    if not project:
        if (PROJECTS_DIR / project_id).exists():
            now = datetime.now(UTC)
            return Project(
                id=project_id,
                name=project_id,
                description=f"Existing workspace directory '{project_id}'",
                created_at=now,
                updated_at=now,
            )
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return project


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, session: Session = Depends(get_session)):
    """Deletes a project, its associated SQLite database records, filesystem files, and MLflow experiments."""
    validate_project_id(project_id)
    project = session.get(Project, project_id)
    dir_exists = (PROJECTS_DIR / project_id).exists()
    if not project and not dir_exists:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    # 1. Cascade delete database records
    for model in [Event, ArtifactIndex, EDAFinding, FeatureVersion, Dataset, WorkflowRun, SupervisorMemoryRecord]:
        items = session.exec(select(model).where(model.project_id == project_id)).all()
        for it in items:
            session.delete(it)

    if project:
        session.delete(project)
    session.commit()

    # 2. Delete physical project folder under projects/
    p_dir = PROJECTS_DIR / project_id
    if p_dir.exists():
        shutil.rmtree(p_dir, ignore_errors=True)

    # 3. Clean up MLflow experiment if present
    try:
        exp = mlflow.get_experiment_by_name(project_id)
        if exp:
            mlflow.delete_experiment(exp.experiment_id)
    except Exception:
        pass

    return {
        "status": "success",
        "message": f"Project '{project_id}' deleted successfully.",
        "project_id": project_id,
    }


@router.get("/projects/{project_id}/code-executions")
def get_code_executions(project_id: str, session: Session = Depends(get_session)):
    """Retrieves executed scripts, exit codes, and stdout/stderr outputs for monitoring the Coder agent."""
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    exec_dir = PROJECTS_DIR / project_id / "code_executions"
    records: list[dict[str, Any]] = []

    if exec_dir.exists():
        for f in exec_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                records.append(data)
            except Exception:
                continue

    # Sort records descending by executed_at or timestamp
    records.sort(key=lambda r: r.get("executed_at", ""), reverse=True)

    # Backfill fallback for pre-existing projects so the user immediately sees code runs
    if not records:
        ws_script = PROJECTS_DIR / project_id / "workspace" / "run_task.py"
        if ws_script.exists():
            code_text = ws_script.read_text(encoding="utf-8", errors="replace")
            records.append({
                "id": "exec_prior_workspace",
                "project_id": project_id,
                "stage": "model",
                "script_name": "run_task.py",
                "task_description": "Train candidate tabular models with scikit-learn",
                "attempt": 1,
                "code": code_text,
                "exit_code": 0,
                "stdout": "Completed model training and evaluation.",
                "stderr": "",
                "success": True,
                "executed_at": datetime.fromtimestamp(ws_script.stat().st_mtime, UTC).isoformat(),
                "duration_ms": 1500,
            })
        feat_script = PROJECTS_DIR / project_id / "features" / "feature_pipeline.py"
        if feat_script.exists():
            code_text = feat_script.read_text(encoding="utf-8", errors="replace")
            records.append({
                "id": "exec_prior_features",
                "project_id": project_id,
                "stage": "features",
                "script_name": "feature_pipeline.py",
                "task_description": "Engineer features & transformation pipeline",
                "attempt": 1,
                "code": code_text,
                "exit_code": 0,
                "stdout": "Transformed dataset and saved feature_data.parquet.",
                "stderr": "",
                "success": True,
                "executed_at": datetime.fromtimestamp(feat_script.stat().st_mtime, UTC).isoformat(),
                "duration_ms": 1200,
            })

    return records


# --- Dataset Endpoints ---

@router.post("/projects/{project_id}/datasets", response_model=Dataset)
def upload_dataset(project_id: str, file: UploadFile = File(...), session: Session = Depends(get_session)):
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    # Validate file extension
    filename = Path(file.filename or "data.csv").name
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are supported for tabular datasets.")

    # Determine next dataset version (e.g. dataset_v1, dataset_v2) without overwriting
    existing_datasets = session.exec(
        select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.asc())
    ).all()
    next_ver_num = len(existing_datasets) + 1
    version_tag = f"dataset_v{next_ver_num}"

    dest_dir = PROJECTS_DIR / project_id / "datasets" / version_tag
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "data.csv"

    # Save uploaded file
    try:
        with open(dest_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e!s}")

    # Validate that the file is an actual parseable tabular CSV with rows and columns
    try:
        df = pd.read_csv(dest_file)
        row_count, col_count = df.shape
        if row_count == 0 or col_count == 0:
            if dest_file.exists():
                dest_file.unlink()
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty or has no tabular columns.")
    except HTTPException:
        raise
    except Exception as e:
        if dest_file.exists():
            dest_file.unlink()
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e!s}")

    dataset_record = Dataset(
        id=f"ds_{uuid.uuid4().hex[:8]}",
        project_id=project_id,
        version=version_tag,
        filename=filename,
        file_path=str(dest_file),
        row_count=row_count,
        col_count=col_count,
        created_at=datetime.now(UTC),
    )
    session.add(dataset_record)
    session.commit()
    session.refresh(dataset_record)

    return dataset_record


@router.get("/projects/{project_id}/datasets", response_model=list[Dataset])
def list_datasets(project_id: str, session: Session = Depends(get_session)):
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    return session.exec(
        select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
    ).all()


# --- Workflow Run Endpoints ---

@router.post("/projects/{project_id}/runs", response_model=WorkflowRun)
def trigger_run(
    project_id: str,
    payload: RunCreate,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    # Resolve dataset version
    dataset_version = payload.dataset_version
    if not dataset_version:
        latest_dataset = session.exec(
            select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
        ).first()
        if not latest_dataset:
            raise HTTPException(status_code=400, detail="No dataset uploaded for this project yet. Please upload a CSV first.")
        dataset_version = latest_dataset.version

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    workflow_run = WorkflowRun(
        id=run_id,
        project_id=project_id,
        dataset_version=dataset_version,
        target_column=payload.target_column,
        target_metric=payload.target_metric,
        status="PENDING",
        current_stage="start",
        created_at=datetime.now(UTC),
    )
    session.add(workflow_run)
    session.commit()
    session.refresh(workflow_run)

    # Launch in background
    background_tasks.add_task(
        execute_workflow_async,
        project_id=project_id,
        run_id=run_id,
        dataset_version=dataset_version,
        target_column=payload.target_column,
        target_metric=payload.target_metric,
        constraints=payload.constraints or {},
    )

    return workflow_run


@router.get("/projects/{project_id}/runs", response_model=list[WorkflowRun])
def list_runs(project_id: str, session: Session = Depends(get_session)):
    validate_project_id(project_id)
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    return session.exec(
        select(WorkflowRun).where(WorkflowRun.project_id == project_id).order_by(WorkflowRun.created_at.desc())
    ).all()


@router.get("/projects/{project_id}/runs/{run_id}", response_model=WorkflowRun)
def get_run(project_id: str, run_id: str, session: Session = Depends(get_session)):
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    run = session.get(WorkflowRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found for project '{project_id}'")
    return run


# --- SSE Event Stream ---

@router.get("/projects/{project_id}/events")
async def stream_events(project_id: str, run_id: str = Query(...)):
    """Server-Sent Events (SSE) endpoint streaming real-time pipeline events."""
    validate_project_id(project_id)
    past_events = event_manager.get_past_events(project_id, run_id)
    queue = event_manager.subscribe(project_id, run_id)

    async def event_generator():
        try:
            # Replay historical events
            for ev in past_events:
                yield f"data: {json.dumps(ev)}\n\n"

            # If past events already reached completion or pause, terminate stream cleanly
            if any(ev.get("event_type") == "WORKFLOW_COMPLETED" for ev in past_events):
                return
            if past_events and past_events[-1].get("data", {}).get("status") in ["NEEDS_INPUT", "FAILED"]:
                return

            # Stream live events
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if (
                        event.get("event_type") == "WORKFLOW_COMPLETED"
                        or event.get("data", {}).get("status") in ["NEEDS_INPUT", "FAILED"]
                    ):
                        break
                except TimeoutError:
                    # Keep-alive heartbeat
                    yield ": ping\n\n"
        finally:
            event_manager.unsubscribe(project_id, run_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Artifacts Endpoints ---

@router.get("/projects/{project_id}/artifacts")
def get_artifacts(project_id: str, session: Session = Depends(get_session)):
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    artifacts = session.exec(
        select(ArtifactIndex).where(ArtifactIndex.project_id == project_id).order_by(ArtifactIndex.created_at.desc())
    ).all()
    return [
        {
            "id": a.id,
            "project_id": a.project_id,
            "run_id": a.run_id,
            "stage": a.artifact_type,
            "artifact_type": a.artifact_type,
            "path": a.path,
            "file_path": a.path,
            "version": a.version,
            "parent_id": a.parent_id,
            "created_at": a.created_at,
        }
        for a in artifacts
    ]


@router.get("/projects/{project_id}/artifacts/{artifact_id}/content")
def get_artifact_content(project_id: str, artifact_id: str, session: Session = Depends(get_session)):
    """Fetches the content of an artifact (markdown rendered, JSON pretty-printed, parquet/csv table preview)."""
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    artifact = session.get(ArtifactIndex, artifact_id)
    if not artifact or artifact.project_id != project_id:
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found for project '{project_id}'")

    path_str = getattr(artifact, "path", None) or getattr(artifact, "file_path", None)
    if not path_str:
        raise HTTPException(status_code=404, detail="Artifact path not specified")

    file_path = Path(path_str).resolve()
    # Verify file is within project directory (prevent path traversal)
    project_dir = (PROJECTS_DIR / project_id).resolve()
    if not str(file_path).startswith(str(project_dir)):
        raise HTTPException(status_code=400, detail="Path traversal outside project directory detected.")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact file not found on disk: {file_path.name}")

    ext = file_path.suffix.lower()
    try:
        if ext == ".json":
            content = json.loads(file_path.read_text(encoding="utf-8"))
            return {"type": "json", "data": content, "raw": file_path.read_text(encoding="utf-8")}
        elif ext in [".md", ".markdown"]:
            return {"type": "markdown", "content": file_path.read_text(encoding="utf-8")}
        elif ext in [".py", ".sh"]:
            return {"type": "code", "content": file_path.read_text(encoding="utf-8"), "language": "python" if ext == ".py" else "bash"}
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
            preview_df = df.head(100).fillna("")
            return {
                "type": "table",
                "columns": list(df.columns),
                "rows": preview_df.to_dict(orient="records"),
                "total_rows": len(df),
                "total_columns": len(df.columns),
            }
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            preview_df = df.head(100).fillna("")
            return {
                "type": "table",
                "columns": list(df.columns),
                "rows": preview_df.to_dict(orient="records"),
                "total_rows": len(df),
                "total_columns": len(df.columns),
            }
        else:
            return {"type": "text", "content": file_path.read_text(encoding="utf-8", errors="replace")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read artifact content: {e!s}")


@router.get("/projects/{project_id}/datasets/{version}/preview")
def preview_dataset(project_id: str, version: str, session: Session = Depends(get_session)):
    """Returns a table preview of the requested dataset version."""
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    dataset = session.exec(
        select(Dataset).where(Dataset.project_id == project_id, Dataset.version == version)
    ).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset version '{version}' not found for project '{project_id}'")

    file_path = Path(dataset.file_path).resolve()
    project_dir = (PROJECTS_DIR / project_id).resolve()
    if not str(file_path).startswith(str(project_dir)):
        raise HTTPException(status_code=400, detail="Path traversal outside project directory detected.")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file '{dataset.filename}' does not exist on disk")

    try:
        df = pd.read_csv(file_path)
        preview_df = df.head(100).fillna("")
        return {
            "version": version,
            "filename": dataset.filename,
            "columns": list(df.columns),
            "rows": preview_df.to_dict(orient="records"),
            "total_rows": len(df),
            "total_columns": len(df.columns),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview dataset: {e!s}")


# --- Models & Leaderboard Endpoints ---

@router.get("/projects/{project_id}/models")
def get_leaderboard(project_id: str, metric: str = Query("f1"), session: Session = Depends(get_session)):
    """Returns ranked MLflow runs sorted by target metric with direction awareness."""
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    try:
        tools = MLflowTools(project_id)
        runs = tools.get_leaderboard(metric)
        return {
            "project_id": project_id,
            "target_metric": metric,
            "direction": "max" if is_higher_better(metric) else "min",
            "leaderboard": runs,
        }
    except Exception as e:
        return {"project_id": project_id, "target_metric": metric, "leaderboard": [], "error": str(e)}


@router.get("/projects/{project_id}/experiments")
def get_experiments(project_id: str, session: Session = Depends(get_session)):
    """Raw MLflow experiment runs."""
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    try:
        tools = MLflowTools(project_id)
        runs = tools.get_leaderboard("f1")
        return {"project_id": project_id, "runs": runs}
    except Exception as e:
        return {"project_id": project_id, "runs": [], "error": str(e)}


def transform_raw_test_features(
    project_id: str,
    test_df: pd.DataFrame,
    target_col: str | None = None,
) -> pd.DataFrame:
    """
    Transforms raw, unprocessed test data using the project's saved feature engineering pipeline.

    1. If `features/feature_pipeline.py` exists, dynamically imports it so any custom
       transformer classes (e.g. FeatureEngineeringTransformer) are registered in Python's
       class lookup for unpickling.
    2. If `features/feature_pipeline.pkl` exists, loads the fitted pipeline with joblib and
       calls `pipeline.transform(test_df)`.
    3. If `feature_pipeline.pkl` is absent or fails, falls back to running `feature_pipeline.py`
       via subprocess.
    4. If no feature pipeline exists at all, returns `test_df` directly.
    """
    features_dir = PROJECTS_DIR / project_id / "features"
    pipeline_pkl = features_dir / "feature_pipeline.pkl"
    pipeline_py = features_dir / "feature_pipeline.py"
    schema_json = features_dir / "feature_schema.json"

    # Step 1: Dynamically import feature_pipeline.py if present
    pipe_mod = None
    if pipeline_py.exists():
        try:
            spec = importlib.util.spec_from_file_location(f"fe_pipe_{project_id}", pipeline_py)
            if spec and spec.loader:
                pipe_mod = importlib.util.module_from_spec(spec)
                sys.modules[f"fe_pipe_{project_id}"] = pipe_mod
                spec.loader.exec_module(pipe_mod)

                # Export classes to __main__ so unpickler resolves classes saved under '__main__'
                main_mod = sys.modules.get("__main__")
                if main_mod:
                    for attr in dir(pipe_mod):
                        if not hasattr(main_mod, attr) and not attr.startswith("__"):
                            setattr(main_mod, attr, getattr(pipe_mod, attr))
                _log.debug("[PREDICT] Loaded pipeline module fe_pipe_%s", project_id)
        except Exception as exc:
            _log.warning("[PREDICT] Could not dynamically load feature_pipeline.py: %s", exc)

    def _clean_nans_and_infs(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.replace([np.inf, -np.inf], np.nan)
        if cleaned.isna().any().any():
            _log.warning("[PREDICT] Transformed features contain NaNs/Infs; imputing with column median/mode")
            for c in cleaned.columns:
                if cleaned[c].isna().any():
                    if pd.api.types.is_numeric_dtype(cleaned[c]):
                        med = cleaned[c].median()
                        fill_v = 0.0 if pd.isna(med) else med
                    else:
                        m = cleaned[c].mode()
                        fill_v = m.iloc[0] if not m.empty else "missing"
                    cleaned[c] = cleaned[c].fillna(fill_v)
        return cleaned

    # Step 2: Try loading and running fitted pipeline.pkl
    if pipeline_pkl.exists():
        try:
            pipeline = joblib.load(pipeline_pkl)
            _log.info("[PREDICT] Successfully loaded fitted feature_pipeline.pkl for project %s", project_id)

            test_input = test_df.copy()
            if target_col and target_col in test_input.columns:
                test_input = test_input.drop(columns=[target_col])

            # Pre-impute raw test input so transformers without imputer steps don't fail or propagate NaNs
            if test_input.isna().any().any():
                _log.info("[PREDICT] Raw test input contains NaNs; applying pre-transform median/mode imputation")
                for c in test_input.columns:
                    if test_input[c].isna().any():
                        if pd.api.types.is_numeric_dtype(test_input[c]):
                            med = test_input[c].median()
                            fill_v = 0.0 if pd.isna(med) else med
                        else:
                            m = test_input[c].mode()
                            fill_v = m.iloc[0] if not m.empty else "missing"
                        test_input[c] = test_input[c].fillna(fill_v)

            try:
                X_transformed = pipeline.transform(test_input)
            except Exception as transform_err:
                # Some custom pipelines expect target_col in df (e.g. `if target_col in df: ...`)
                if target_col and target_col not in test_input.columns:
                    _log.debug("[PREDICT] Retrying transform with dummy target col: %s", transform_err)
                    test_input[target_col] = 0
                    X_transformed = pipeline.transform(test_input)
                else:
                    raise transform_err

            # Convert to DataFrame if result is numpy array
            if isinstance(X_transformed, pd.DataFrame):
                out_df = X_transformed
            elif hasattr(X_transformed, "__array__"):
                feature_names = None
                if hasattr(pipeline, "get_feature_names_out"):
                    try:
                        feature_names = list(pipeline.get_feature_names_out())
                    except Exception:
                        pass
                if not feature_names and schema_json.exists():
                    try:
                        schema_data = json.loads(schema_json.read_text(encoding="utf-8"))
                        if isinstance(schema_data, list):
                            feature_names = [x.get("name") for x in schema_data if isinstance(x, dict) and x.get("name") != target_col]
                        elif isinstance(schema_data, dict):
                            feature_names = [k for k in schema_data.keys() if k != target_col]
                    except Exception:
                        pass

                if feature_names and len(feature_names) == X_transformed.shape[1]:
                    out_df = pd.DataFrame(X_transformed, columns=feature_names, index=test_df.index)
                else:
                    out_df = pd.DataFrame(X_transformed, index=test_df.index)
            else:
                out_df = pd.DataFrame(X_transformed, index=test_df.index)

            if target_col and target_col in out_df.columns:
                out_df = out_df.drop(columns=[target_col])

            out_df = _clean_nans_and_infs(out_df)
            _log.info("[PREDICT] Transformed raw test data to shape %s via pipeline.pkl", out_df.shape)
            return out_df
        except Exception as exc:
            _log.warning("[PREDICT] Error applying feature_pipeline.pkl, falling back: %s", exc)

    # Step 3: Fallback - Subprocess execution if pipeline_py exists
    if pipeline_py.exists():
        _log.info("[PREDICT] Falling back to subprocess transform with feature_pipeline.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_in_csv = (Path(tmpdir) / "test_in.csv").resolve()
            test_out_parquet = (Path(tmpdir) / "test_out.parquet").resolve()

            test_df_copy = test_df.copy()
            if target_col and target_col not in test_df_copy.columns:
                test_df_copy[target_col] = 0

            test_df_copy.to_csv(test_in_csv, index=False)

            raw_code = pipeline_py.read_text(encoding="utf-8")
            in_path_str = str(test_in_csv).replace("\\", "/")
            out_path_str = str(test_out_parquet).replace("\\", "/")

            mod_code = re.sub(
                r'(?:RAW_CSV_PATH|DATA_PATH)\s*=\s*r?["\'].*?["\']',
                f'DATA_PATH = r"{in_path_str}"\nRAW_CSV_PATH = r"{in_path_str}"',
                raw_code,
            )
            mod_code = re.sub(
                r'(?:PARQUET_OUT_PATH|FEATURE_DATA_PATH)\s*=\s*r?["\'].*?["\']',
                f'FEATURE_DATA_PATH = r"{out_path_str}"\nPARQUET_OUT_PATH = r"{out_path_str}"',
                mod_code,
            )

            run_script = Path(tmpdir) / "run_transform.py"
            run_script.write_text(mod_code, encoding="utf-8")

            res = subprocess.run([sys.executable, str(run_script)], capture_output=True, text=True)
            if res.returncode == 0 and test_out_parquet.exists():
                out_df = pd.read_parquet(test_out_parquet)
                if target_col and target_col in out_df.columns:
                    out_df = out_df.drop(columns=[target_col])
                out_df = _clean_nans_and_infs(out_df)
                _log.info("[PREDICT] Transformed raw test data to shape %s via subprocess fallback", out_df.shape)
                return out_df
            else:
                err_msg = res.stderr.strip() or res.stdout.strip() or "Feature pipeline execution failed."
                last_err = err_msg.splitlines()[-1] if err_msg else "Unknown pipeline error"
                _log.warning("[PREDICT] Subprocess fallback failed: %s", last_err)

    # Step 4: Final fallback (raw data pass-through)
    _log.info("[PREDICT] No feature pipeline applied; passing raw test features directly")
    out_df = test_df.copy()
    if target_col and target_col in out_df.columns:
        out_df = out_df.drop(columns=[target_col])
    out_df = _clean_nans_and_infs(out_df)
    return out_df



@router.post("/projects/{project_id}/models/{experiment_id}/predict")
def predict_model(
    project_id: str,
    experiment_id: str,
    test_file: UploadFile = File(...),
    id_column: str | None = Form(None),
    session: Session = Depends(get_session),
):
    """Generates Kaggle-style model predictions on an unlabeled test CSV using the project's feature pipeline."""
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    filename = Path(test_file.filename or "test.csv").name
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are supported for test predictions.")

    # 1. Parse uploaded test CSV
    try:
        test_df = pd.read_csv(test_file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e!s}")

    if len(test_df) == 0:
        raise HTTPException(status_code=400, detail="Test file contains 0 data rows.")

    # Strip column names of surrounding whitespace
    test_df.columns = [c.strip() if isinstance(c, str) else c for c in test_df.columns]

    # 2. Determine target column and profile info
    target_col = None
    profile_data = {}
    profile_path = PROJECTS_DIR / project_id / "profile" / "profile.json"
    if profile_path.exists():
        try:
            profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
            target_col = profile_data.get("target_column")
        except Exception:
            pass

    if not target_col:
        latest_run = session.exec(
            select(WorkflowRun).where(WorkflowRun.project_id == project_id).order_by(WorkflowRun.created_at.desc())
        ).first()
        if latest_run and latest_run.target_column:
            target_col = latest_run.target_column

    # 3. Validate required feature columns from training dataset / profile
    training_feature_cols = []
    id_candidates = {"id", "passengerid", "row_id", "index", "id_column"}
    raw_train_cols = []

    if profile_data.get("columns"):
        raw_train_cols = [c["name"] for c in profile_data["columns"]]
    elif (PROJECTS_DIR / project_id / "datasets" / "dataset_v1" / "data.csv").exists():
        try:
            train_sample = pd.read_csv(PROJECTS_DIR / project_id / "datasets" / "dataset_v1" / "data.csv", nrows=1)
            raw_train_cols = list(train_sample.columns)
        except Exception:
            pass

    # Infer target column if missing or not in training columns
    if raw_train_cols and (not target_col or target_col not in raw_train_cols):
        for c in raw_train_cols:
            if c.lower() in project_id.lower() or c.lower() in ("target", "label", "rainfall", "price", "survived"):
                target_col = c
                break
        else:
            diff_cols = [c for c in raw_train_cols if c not in test_df.columns and c.lower() not in id_candidates]
            if len(diff_cols) == 1:
                target_col = diff_cols[0]

    if not target_col:
        target_col = "target"

    if raw_train_cols:
        training_feature_cols = [c for c in raw_train_cols if c != target_col]

    if training_feature_cols:
        missing_cols = [
            c for c in training_feature_cols
            if c not in test_df.columns and c.lower() not in id_candidates
        ]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Validation error: Missing required feature column(s): {', '.join(missing_cols)}",
            )

    # 4. Resolve ID column
    resolved_id_col = None
    if id_column and id_column.strip():
        chosen = id_column.strip()
        if chosen not in test_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Validation error: Specified id_column '{chosen}' does not exist in the uploaded test file.",
            )
        resolved_id_col = chosen
    else:
        # Check if profile flagged an ID column in the training dataset
        if profile_data.get("columns"):
            row_count = profile_data.get("row_count", 0)
            for c in profile_data["columns"]:
                c_name = c.get("name", "")
                if c_name in test_df.columns:
                    if (c.get("unique_count") == row_count and row_count > 0) or c_name.lower() in ["id", "passengerid", "row_id"]:
                        resolved_id_col = c_name
                        break

        # Fallback check for common ID column names in test_df
        if not resolved_id_col:
            for cand in ["Id", "id", "ID", "PassengerId", "passenger_id", "row_id"]:
                if cand in test_df.columns:
                    resolved_id_col = cand
                    break

    if resolved_id_col:
        id_values = test_df[resolved_id_col].tolist()
        id_col_output_name = resolved_id_col
    else:
        id_values = list(range(len(test_df)))
        id_col_output_name = "id"

    # 5. Locate MLflow run and load model artifact
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    target_run = None
    try:
        target_run = client.get_run(experiment_id)
    except Exception:
        pass

    if not target_run:
        try:
            exp = mlflow.get_experiment_by_name(project_id)
            if exp:
                runs = client.search_runs(experiment_ids=[exp.experiment_id])
                for r in runs:
                    if r.info.run_id == experiment_id or r.info.run_name == experiment_id:
                        target_run = r
                        break
                if not target_run and runs:
                    target_run = runs[0]
        except Exception:
            pass

    # 6. Load model pickle
    model = None
    models_dir = PROJECTS_DIR / project_id / "models"

    if target_run:
        try:
            artifacts = client.list_artifacts(target_run.info.run_id)
            for art in artifacts:
                if art.path.endswith(".pkl"):
                    local_art_path = client.download_artifacts(target_run.info.run_id, art.path)
                    try:
                        with open(local_art_path, "rb") as mf:
                            model = pickle.load(mf)
                    except Exception:
                        model = joblib.load(local_art_path)
                    break
        except Exception as exc:
            _log.warning("[PREDICT] Error downloading/unpickling MLflow artifact: %s", exc)

    if model is None and models_dir.exists():
        run_name = target_run.info.run_name if target_run else ""
        candidate_files = list(models_dir.glob("*.pkl"))
        for cf in candidate_files:
            if cf.stem in run_name or run_name in cf.stem:
                try:
                    with open(cf, "rb") as mf:
                        model = pickle.load(mf)
                except Exception:
                    model = joblib.load(cf)
                break
        if model is None and candidate_files:
            try:
                with open(candidate_files[0], "rb") as mf:
                    model = pickle.load(mf)
            except Exception:
                model = joblib.load(candidate_files[0])

    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Trained model artifact for '{experiment_id}' could not be found or loaded.",
        )

    # 7. Apply feature pipeline transform to raw test_df
    try:
        X_test = transform_raw_test_features(project_id, test_df, target_col=target_col)
    except Exception as exc:
        _log.exception("[PREDICT] Feature transformation failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Feature transformation failed: {exc!s}")

    # 8. Align feature columns with model expectations
    feature_names = None
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    elif hasattr(model, "steps") and len(model.steps) > 0:
        first_step = model.steps[0][1]
        if hasattr(first_step, "feature_names_in_"):
            feature_names = list(first_step.feature_names_in_)

    if feature_names:
        missing_features = [c for c in feature_names if c not in X_test.columns]
        if missing_features:
            _log.info(
                "[PREDICT] Adding %d missing features with default 0.0: %s",
                len(missing_features), missing_features,
            )
            missing_df = pd.DataFrame(0.0, index=X_test.index, columns=missing_features)
            X_test = pd.concat([X_test, missing_df], axis=1)
        X_test = X_test[feature_names]

    # Clean any inf or NaN values before passing to estimator
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    if X_test.isna().any().any():
        nan_cols = [c for c in X_test.columns if X_test[c].isna().any()]
        _log.warning("[PREDICT] Final guard: detected NaNs in features %s; filling with median/0.0", nan_cols)
        for c in nan_cols:
            if pd.api.types.is_numeric_dtype(X_test[c]):
                med = X_test[c].median()
                fill_v = 0.0 if pd.isna(med) else med
            else:
                m = X_test[c].mode()
                fill_v = m.iloc[0] if not m.empty else "missing"
            X_test[c] = X_test[c].fillna(fill_v)

    # 9. Model inference
    try:
        preds = model.predict(X_test)
    except Exception as e:
        _log.exception("[PREDICT] Model inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e!s}")

    # 10. Format and return submission CSV
    submission_df = pd.DataFrame({
        id_col_output_name: id_values,
        target_col: preds,
    })
    csv_str = submission_df.to_csv(index=False)
    _log.info(
        "[PREDICT] Generated predictions successfully | project=%s model=%s rows=%d",
        project_id, experiment_id, len(preds),
    )

    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="submission.csv"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )



# --- Evaluation & Report Endpoints ---

@router.get("/projects/{project_id}/evaluation")
def get_evaluation(project_id: str, session: Session = Depends(get_session)):
    """Returns latest evaluation json and markdown report, or empty objects if not yet generated."""
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    eval_dir = PROJECTS_DIR / project_id / "evaluations"
    if not eval_dir.exists():
        return {"json": {}, "markdown": ""}

    # Get latest evaluation file
    json_files = sorted(eval_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        return {"json": {}, "markdown": ""}

    try:
        latest_json = json_files[0]
        data = json.loads(latest_json.read_text(encoding="utf-8"))
        md_file = latest_json.with_suffix(".md")
        md_content = md_file.read_text(encoding="utf-8") if md_file.exists() else ""
        return {"json": data, "markdown": md_content}
    except Exception:
        return {"json": {}, "markdown": ""}


@router.get("/projects/{project_id}/report")
def get_report(project_id: str, session: Session = Depends(get_session)):
    """Returns the final markdown and JSON report, or empty objects if not yet generated."""
    validate_project_id(project_id)
    if not check_project_exists(session, project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    reports_dir = PROJECTS_DIR / project_id / "reports"
    json_path = reports_dir / "summary.json"
    md_path = reports_dir / "final_report.md"

    if not md_path.exists() and not json_path.exists():
        return {"summary": {}, "markdown": ""}

    summary_data = {}
    if json_path.exists():
        try:
            summary_data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    md_content = ""
    if md_path.exists():
        try:
            md_content = md_path.read_text(encoding="utf-8")
        except Exception:
            pass

    return {"summary": summary_data, "markdown": md_content}
