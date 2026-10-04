"""Pydantic schemas for the Supervisor-led team multi-agent architecture."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class BriefContent(BaseModel):
    """Specific task brief provided by Supervisor to a worker."""
    objective: str = "Standard execution"
    focus_points: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    mode: str | None = None

    @field_validator("constraints", "focus_points", mode="before")
    @classmethod
    def normalize_list_strings(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, dict):
            return [f"{k}: {val}" for k, val in v.items()]
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple)):
            res = []
            for item in v:
                if isinstance(item, dict):
                    res.extend([f"{k}: {val}" for k, val in item.items()])
                else:
                    res.append(str(item))
            return res
        return [str(v)]


class PlanItem(BaseModel):
    """Individual item in the supervisor's dynamic plan."""
    id: str
    text: str
    status: Literal["todo", "doing", "done", "dropped"] = "todo"


class SupervisorDecision(BaseModel):
    """The real-time decision emitted by the Supervisor LLM on each turn."""
    thought: str = Field(default="", description="Internal reasoning <= 300 chars")
    action: Literal["profile", "eda", "fe", "model", "judge", "report", "finish"]
    brief: BriefContent = Field(default_factory=lambda: BriefContent(objective="Standard execution"))
    reason: str = Field(default="", description="Justification <= 300 chars; must answer open concern if any")
    plan_update: list[PlanItem] | list[dict[str, Any]] | dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def clean_decision(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Ensure brief is always a valid dict with an objective
            b = data.get("brief")
            if b is None or not isinstance(b, dict):
                data["brief"] = {"objective": f"Execute {data.get('action', 'task')}"}
            elif not b.get("objective") or not str(b.get("objective")).strip():
                data["brief"]["objective"] = f"Execute {data.get('action', 'task')}"
        return data

    @field_validator("thought", "reason", mode="before")
    @classmethod
    def truncate_strings(cls, v: Any) -> str:
        s = str(v) if v is not None else ""
        return s[:300]


class WorkerConcern(BaseModel):
    """Evidence-backed concern raised by a worker."""
    claim: str
    evidence: str
    suggestion: str

    @field_validator("evidence", mode="before")
    @classmethod
    def validate_evidence(cls, v: Any) -> str:
        s = str(v).strip() if v is not None else ""
        if not s:
            raise ValueError("Worker concern requires non-empty evidence string")
        return s


class NotebookRow(BaseModel):
    """Entry into an agent's persistent notebook."""
    step: int
    tried: str
    outcome: str
    lesson: str
    score_impact: float | None = None
    errors: list[str] = Field(default_factory=list)


class WorkerReport(BaseModel):
    """Structured report returned by every worker after completing a brief."""
    status: Literal["ok", "failed", "no_gain"]
    result_summary: str = Field(..., max_length=300)
    evidence: dict[str, Any] = Field(default_factory=dict)
    concern: WorkerConcern | None = None
    suggestion: str | None = None
    notebook: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    error_text: str | None = None

    @field_validator("result_summary", mode="before")
    @classmethod
    def truncate_summary(cls, v: Any) -> str:
        s = str(v) if v is not None else ""
        return s[:300]


class StageVerdict(BaseModel):
    """Judge verdict on an individual pipeline stage."""
    status: Literal["ok", "weak", "bad"]
    issue: str = ""
    evidence: str = ""


class JudgeReport(BaseModel):
    """Judge structured assessment extending WorkerReport evidence."""
    stage_verdicts: dict[str, StageVerdict] = Field(default_factory=dict)
    blame_stage: Literal["eda", "fe", "model", "none"] = "none"
    overall: Literal["ship", "improve", "blocked"] = "ship"
    advice: list[str] = Field(default_factory=list)


class LedgerRow(BaseModel):
    """Append-only row in the run ledger recording a supervisor step, worker execution, or note."""
    step: int
    ts: str
    agent: str
    action: str
    brief_summary: str = Field(..., max_length=200)
    status: Literal["ok", "failed", "guard_override", "note", "no_gain"]
    result_summary: str = Field(..., max_length=300)
    version_id: str | None = None
    score: float | None = None
    mean: float | None = None
    std: float | None = None
    noise_floor: float | None = None
    significant: bool | None = None
    delta_vs_best: float | None = None
    judge_overall: str | None = None
    issues: list[str] = Field(default_factory=list)
    concern: dict[str, Any] | None = None
    decision_reason: str = Field(..., max_length=300)
    brief_hash: str | None = None
    duration_s: float = 0.0
    error_text: str | None = None

    @field_validator("brief_summary", mode="before")
    @classmethod
    def truncate_brief(cls, v: Any) -> str:
        s = str(v) if v is not None else ""
        return s[:200]

    @field_validator("result_summary", "decision_reason", mode="before")
    @classmethod
    def truncate_result_reason(cls, v: Any) -> str:
        s = str(v) if v is not None else ""
        return s[:300]

    @field_validator("error_text", mode="before")
    @classmethod
    def truncate_error_text(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v)
        return s[-300:] if len(s) > 300 else s
