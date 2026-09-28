import json
from enum import Enum
from typing import Literal, TypedDict, Any
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

_log = get_logger(__name__)


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    AMBIGUOUS = "ambiguous"


class ProjectState(TypedDict):
    project_id: str
    run_id: str
    user_goal: str
    target_column: str | None
    target_metric: str | None          # e.g. "f1", "roc_auc", "rmse", "r2"
    description: str | None
    constraints: dict

    dataset_id: str
    dataset_version: str

    task_type: TaskType | None

    profile_summary: dict
    eda_findings: dict
    feature_summary: dict
    model_summary: dict
    evaluation_summary: dict

    current_stage: str
    iteration: int
    max_iterations: int

    best_experiment_id: str | None     # MLflow run_id
    best_metric_value: float | None

    supervisor_memory: dict
    artifacts: list[dict]              # references only, never raw content

    next_action: str | None
    status: Literal["SUCCESS", "FAILED", "NEEDS_INPUT", "RETRY", "RUNNING"]


# --- Structured Output Models for Agents ---

from pydantic import BaseModel, Field, field_validator


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
    columns: list[ColumnProfile]
    target_column: str | None = None
    task_type_guess: TaskType
    duplicates_count: int
    missing_total_pct: float
    target_distribution: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


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
    category: str = Field(description="e.g. correlation, skewness, outliers, leakage, class_imbalance")
    finding: str = Field(description="Clear statement of what was discovered")
    evidence: str = Field(description="Numerical statistics or evidence supporting the finding (formatted as string)")
    implication: str = Field(description="Impact on modeling or feature engineering")
    recommendation: str = Field(description="Actionable suggestion for feature engineering")

    @field_validator("evidence", mode="before")
    @classmethod
    def coerce_evidence_to_str(cls, v: Any) -> str:
        return to_evidence_str(v, _field="EDAFindingItem.evidence")


class EDAOutput(BaseModel):
    executive_summary: str
    findings: list[EDAFindingItem]
    leakage_risks: list[str] = Field(default_factory=list)
    suggested_feature_ideas: list[str] = Field(default_factory=list)

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


from pydantic import BaseModel, Field, field_validator, model_validator


class EvaluatorIssue(BaseModel):
    check_name: str = "general_check"
    severity: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    description: str = ""
    suggested_fix: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_issue(cls, data: Any) -> Any:
        if isinstance(data, dict):
            c_name = data.get("check_name") or data.get("issue") or data.get("name") or "model_check"
            sev = str(data.get("severity", "MEDIUM")).upper()
            if sev not in ("HIGH", "MEDIUM", "LOW"):
                sev = "MEDIUM"
            desc = data.get("description") or data.get("issue") or data.get("details") or str(data)
            fix = data.get("suggested_fix") or data.get("fix") or data.get("recommendation") or "Apply regularization or review features."
            return {
                "check_name": str(c_name),
                "severity": sev,
                "description": str(desc),
                "suggested_fix": str(fix),
            }
        elif isinstance(data, str):
            return {
                "check_name": "check",
                "severity": "MEDIUM",
                "description": data,
                "suggested_fix": "Review model",
            }
        return data


class EvaluatorOutput(BaseModel):
    verdict: Literal["PASS", "IMPROVE"] = "PASS"
    target_metric: str = "metric"
    primary_metric_value: float = 0.0
    issues_found: list[EvaluatorIssue] = Field(default_factory=list)
    recommended_next_stage: Literal["feature_engineering", "model"] = "feature_engineering"
    reasoning: str = ""


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



class SupervisorReview(BaseModel):
    action: Literal["PROCEED", "RETRY", "IMPROVE", "COMPLETE", "FAIL"]
    next_stage: str
    reasoning: str
    instructions_for_next_stage: str
