from datetime import UTC, datetime
from typing import Any

from pydantic import field_validator
from sqlmodel import Field, SQLModel

from app.core.state import to_evidence_str


def utc_now() -> datetime:
    return datetime.now(UTC)


class Project(SQLModel, table=True):
    __tablename__ = "projects"

    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    description: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Dataset(SQLModel, table=True):
    __tablename__ = "datasets"

    id: str = Field(primary_key=True)
    project_id: str = Field(index=True)
    version: str = Field(index=True)  # e.g. "dataset_v1"
    filename: str
    file_path: str
    row_count: int | None = None
    col_count: int | None = None
    created_at: datetime = Field(default_factory=utc_now)


class WorkflowRun(SQLModel, table=True):
    __tablename__ = "workflow_runs"

    id: str = Field(primary_key=True)
    project_id: str = Field(index=True)
    dataset_id: str | None = None
    dataset_version: str = "dataset_v1"
    target_column: str | None = None
    target_metric: str | None = None
    status: str = Field(default="PENDING", index=True)  # PENDING, RUNNING, SUCCESS, FAILED, NEEDS_INPUT
    current_stage: str = "start"
    iteration: int = 1
    max_iterations: int = 0  # No longer enforced; kept for DB/API compatibility
    best_experiment_id: str | None = None
    best_metric_value: float | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class EDAFinding(SQLModel, table=True):
    __tablename__ = "eda_findings"

    id: str = Field(primary_key=True)
    project_id: str = Field(index=True)
    run_id: str = Field(index=True)
    category: str
    finding: str
    evidence: str
    implication: str
    recommendation: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("evidence", mode="before")
    @classmethod
    def coerce_evidence_to_str(cls, v: Any) -> str:
        return to_evidence_str(v)

    def __init__(self, **data: Any):
        if "evidence" in data:
            data["evidence"] = to_evidence_str(data["evidence"])
        super().__init__(**data)


class FeatureVersion(SQLModel, table=True):
    __tablename__ = "feature_versions"

    id: str = Field(primary_key=True)
    project_id: str = Field(index=True)
    run_id: str = Field(index=True)
    dataset_version: str = "dataset_v1"
    version_tag: str  # e.g. "feat_v1"
    pipeline_path: str
    data_path: str
    schema_path: str
    feature_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class ArtifactIndex(SQLModel, table=True):
    __tablename__ = "artifact_index"

    id: str = Field(primary_key=True)
    project_id: str = Field(index=True)
    run_id: str | None = Field(default=None, index=True)
    artifact_type: str = Field(index=True)
    path: str
    version: str | None = None
    parent_id: str | None = None
    created_at: str


class Event(SQLModel, table=True):
    __tablename__ = "events"

    id: str = Field(primary_key=True)
    project_id: str = Field(index=True)
    run_id: str = Field(index=True)
    event_type: str = Field(index=True)
    stage: str
    message: str
    data_json: str = "{}"
    timestamp: datetime = Field(default_factory=utc_now)


class SupervisorMemoryRecord(SQLModel, table=True):
    __tablename__ = "supervisor_memory"

    id: str = Field(primary_key=True)
    project_id: str = Field(index=True, unique=True)
    memory_json: str = "{}"
    updated_at: datetime = Field(default_factory=utc_now)
