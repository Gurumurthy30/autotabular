import json
from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from app.utils.logger import get_logger

_log = get_logger(__name__)


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    AMBIGUOUS = "ambiguous"


def get_default_metric(task_type: TaskType | str | None) -> str:
    """Returns the default evaluation metric for a given task type:
    binary -> roc_auc, multiclass -> f1_macro, regression -> rmse.
    """
    if task_type is None:
        return "roc_auc"
    t_str = task_type.value if hasattr(task_type, "value") else str(task_type).lower()
    if "binary" in t_str:
        return "roc_auc"
    if "multi" in t_str:
        return "f1_macro"
    if "regress" in t_str:
        return "rmse"
    return "roc_auc"


class ProjectState(TypedDict, total=False):
    project_id: str
    run_id: str
    user_goal: str
    target_column: str | None
    target_metric: str | None          # e.g. "f1", "roc_auc", "rmse", "r2"

    dataset_version: str
    task_type: TaskType | None

    profile_summary: dict
    eda_findings: dict
    feature_summary: dict
    model_summary: dict

    current_stage: str
    iteration: int                     # for UI/DB compatibility (reflects version count)
    step: int                          # current supervisor step count (0, 1, 2, ...)
    plan: list[dict]                   # [{id, text, status: todo|doing|done|dropped}]
    current_version: str | None        # e.g. "v1", "v2"
    best_version: str | None           # e.g. "v1"
    open_concern: dict | None          # {claim, evidence, suggestion}
    guard_note: str | None
    worker_runs: dict[str, int]        # counts per worker e.g. {"profile": 1, ...}
    report: dict | None                # last worker report dict

    best_experiment_id: str | None     # MLflow run_id
    best_metric_value: float | None

    artifacts: list[dict]              # references only, never raw content

    next_action: str | None
    status: Literal["SUCCESS", "FAILED", "NEEDS_INPUT", "RETRY", "RUNNING"]
    error: str | None
    split_train_path: str | None
    split_val_path: str | None



# --- Structured Output Models for Agents ---

from pydantic import field_validator, model_validator


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    is_constant: bool
    sample_values: list[Any] = Field(default_factory=list)


class ProfileSummary(BaseModel):
    row_count: int
    column_count: int
    columns: list[ColumnProfile] = Field(default_factory=list)
    target_column: str | None = None
    task_type_guess: TaskType
    duplicates_count: int
    missing_total_pct: float
    target_distribution: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    column_roles: dict[str, str] = Field(default_factory=dict)
    id_suspects: list[dict[str, Any]] = Field(default_factory=list)
    row_order_meaningful: bool = False
    possible_time_column: str | None = None
    possible_group_columns: list[str] = Field(default_factory=list)
    near_duplicate_pairs: list[list[str]] = Field(default_factory=list)
    compact: str = ""


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively converts numpy and pandas types to standard Python primitives."""
    if obj is None:
        return None
    try:
        import pandas as pd
        if pd.isna(obj):
            if not isinstance(obj, (list, dict)):
                return None
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return [_convert_numpy_types(x) for x in obj.tolist()]
    except Exception:
        pass

    try:
        import pandas as pd
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        if isinstance(obj, pd.Series):
            return {str(k): _convert_numpy_types(v) for k, v in obj.to_dict().items()}
        if isinstance(obj, pd.DataFrame):
            return {str(k): _convert_numpy_types(v) for k, v in obj.to_dict().items()}
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_convert_numpy_types(x) for x in obj]
    return obj


def to_evidence_str(data: Any, _field: str = "evidence") -> str:
    """Converts raw statistics, dicts, numpy/pandas scalars, and lists into a deterministic,
    human-readable evidence string.

    When a non-string value is coerced, a WARNING is logged with tag EVIDENCE_COERCION so it
    can be grepped in logs/app.log.
    """
    if data is None:
        return ""
    if isinstance(data, str):
        return data

    # --- coercion path: the LLM returned a non-string value ---
    _repr = str(data)[:200]  # truncated for safe logging, no raw dataset rows
    _log.warning(
        "EVIDENCE_COERCION | field=%s | input_type=%s | preview=%s",
        _field, type(data).__name__, _repr,
    )

    cleaned = _convert_numpy_types(data)

    if isinstance(cleaned, (int, float, bool)):
        return str(cleaned)

    if isinstance(cleaned, dict):
        # Nested dictionaries (e.g. correlation matrices) are serialized deterministically via json.dumps
        if any(isinstance(v, dict) for v in cleaned.values()):
            return json.dumps(cleaned, default=str, sort_keys=True)

        # Flat dictionaries are formatted as clean, deterministic key=value pairs
        parts = []
        for k in sorted(cleaned.keys()):
            v = cleaned[k]
            if isinstance(v, float):
                val_str = str(round(v, 4)) if abs(v) >= 1e-4 else str(v)
            elif isinstance(v, list):
                formatted_items = [
                    str(round(x, 4)) if isinstance(x, float) and abs(x) >= 1e-4 else str(x)
                    for x in v
                ]
                val_str = "[" + ", ".join(formatted_items) + "]"
            else:
                val_str = str(v)
            parts.append(f"{k}={val_str}")
        return ", ".join(parts)

    if isinstance(cleaned, (list, tuple)):
        return json.dumps(cleaned, default=str, sort_keys=True)

    try:
        return json.dumps(cleaned, default=str, sort_keys=True)
    except Exception:
        return str(cleaned)


class EDAFindingItem(BaseModel):
    id: str = ""
    category: str = Field(default="general", description="e.g. correlation, skewness, outliers, leakage, class_imbalance")
    finding: str = Field(default="", description="Clear statement of what was discovered")
    evidence: str = Field(default="", description="Numerical statistics or evidence supporting the finding (formatted as string)")
    implication: str = Field(default="", description="Impact on modeling or feature engineering")
    recommendation: str = Field(default="", description="Actionable suggestion for feature engineering")
    priority: Literal["high", "med", "low"] = "med"

    @field_validator("evidence", mode="before")
    @classmethod
    def coerce_evidence_to_str(cls, v: Any) -> str:
        return to_evidence_str(v, _field="EDAFindingItem.evidence")

    @model_validator(mode="before")
    @classmethod
    def normalize_finding_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            p = str(data.get("priority", "med")).lower()
            if p not in ("high", "med", "low"):
                p = "med"
            data["priority"] = p
            if data.get("implication") is None:
                data["implication"] = ""
        return data


class EDAOutput(BaseModel):
    executive_summary: str = ""
    findings: list[EDAFindingItem] = Field(default_factory=list)
    leakage_risks: list[str] = Field(default_factory=list)
    suggested_feature_ideas: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def autofill_finding_ids(cls, data: Any) -> Any:
        if isinstance(data, dict) and "findings" in data and isinstance(data["findings"], list):
            for i, f in enumerate(data["findings"], 1):
                if isinstance(f, dict) and not f.get("id"):
                    f["id"] = f"E{i}"
        return data

    @field_validator("findings", mode="before")
    @classmethod
    def coerce_findings_list(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return [v]
        return v

    @field_validator("leakage_risks", "suggested_feature_ideas", mode="before")
    @classmethod
    def coerce_to_strings(cls, v):
        if isinstance(v, list):
            coerced = []
            for item in v:
                if isinstance(item, dict):
                    coerced.append(" | ".join(f"{k}: {val}" for k, val in item.items()))
                else:
                    coerced.append(str(item))
            return coerced
        return v



class FeatureMetadataItem(BaseModel):
    feature_name: str
    source_columns: list[str]
    transformation: str
    reason: str
    eda_evidence: str
    leakage_check: str
    inference_available: bool = True

    @field_validator("eda_evidence", mode="before")
    @classmethod
    def coerce_eda_evidence_to_str(cls, v: Any) -> str:
        return to_evidence_str(v, _field="FeatureMetadataItem.eda_evidence")


class FeatureEngineeringOutput(BaseModel):
    version: str
    created_features: list[FeatureMetadataItem]
    pipeline_file: str
    data_file: str
    schema_file: str
    notes: str


class TrainedModelResult(BaseModel):
    model_name: str
    model_family: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    train_score: float
    val_score: float
    metrics: dict[str, float] = Field(default_factory=dict)
    mlflow_run_id: str
    model_path: str


class ModelStageOutput(BaseModel):
    validation_strategy: str
    target_metric: str
    trained_models: list[TrainedModelResult]
    best_model_name: str
    best_mlflow_run_id: str
    best_score: float
    summary: str


class ReportSummaryOutput(BaseModel):
    project_id: str
    objective: str
    dataset_summary: dict[str, Any] = Field(default_factory=dict)
    best_model_name: str = "best_model"
    best_score: float | None = 0.0
    metric_name: str = "metric"
    iterations_run: int | None = 1
    key_findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def clean_report_summary(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("best_model_name") is None:
                data["best_model_name"] = "N/A"
            if data.get("best_score") is None:
                data["best_score"] = 0.0
            if data.get("iterations_run") is None:
                data["iterations_run"] = 1
        return data

    @field_validator("key_findings", "recommendations", "limitations", mode="before")
    @classmethod
    def coerce_to_strings(cls, v):
        if isinstance(v, list):
            coerced = []
            for item in v:
                if isinstance(item, dict):
                    coerced.append(" | ".join(f"{k}: {val}" for k, val in item.items()))
                else:
                    coerced.append(str(item))
            return coerced
        return v
