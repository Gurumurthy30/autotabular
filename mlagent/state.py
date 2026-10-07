"""Ledger state: every shape the agents read and write. Data only, no logic.

Rules (dedupe, gain > std, plateau, budget) live in ledger.py / controller.py.
Models used to validate LLM output ignore extra keys, so a weak model's
stray fields do not fail the run.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TaskType = Literal["binary", "multiclass", "regression"]
Direction = Literal["maximize", "minimize"]
CVKind = Literal["kfold", "stratified", "group", "time"]
ExpKind = Literal["feature", "model", "tune"]
ExpStatus = Literal["pending", "running", "done", "failed"]
QueueSource = Literal["strategist", "analyzer"]
RunSource = Literal["experimenter", "tuner"]
VerdictKind = Literal["approve", "reject"]
RunStatus = Literal["running", "done", "stopped"]


class Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ---- problem / env (written by CLI) ----------------------------------------

class Problem(Base):
    goal: str
    target: str
    metric: str
    direction: Direction
    task_type: TaskType
    train_path: str
    test_path: str | None = None
    id_column: str | None = None
    sample_submission_path: str | None = None


class Budget(Base):
    max_experiments: int = 15
    max_minutes: int = 120
    plateau_n: int = 3        # approved runs without a counted gain
    analyze_every: int = 3    # K: run the Analyzer every K runs
    tune_top_n: int = 3
    tune_trials: int = 20     # search trials per tuned model
    max_consecutive_crashes: int = 3
    loop_fraction: float = 0.85   # loop stops at this share of max_minutes; rest is for Tuner + Finisher


class Env(Base):
    seed: int = 42
    started_at: float | None = None          # epoch seconds, set by CLI
    library_versions: dict[str, str] = Field(default_factory=dict)
    budget: Budget = Field(default_factory=Budget)


# ---- profile (Profiler) ----------------------------------------------------

class Profile(Base):
    questions: list[str] = Field(default_factory=list)
    column_roles: dict[str, str] = Field(default_factory=dict)  # col -> numeric|categorical|datetime|id|text|constant
    target_stats: dict[str, Any] = Field(default_factory=dict)
    missing: dict[str, float] = Field(default_factory=dict)     # col -> fraction missing
    leakage_flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---- strategy (Strategist) -------------------------------------------------

class CVScheme(Base):
    kind: CVKind
    n_splits: int = 5
    group_col: str | None = None
    time_col: str | None = None


class Strategy(Base):
    validation: CVScheme
    folds_path: str | None = None      # set by code after folds.json is saved
    holdout_path: str | None = None    # set by code after the holdout is saved
    target_score: float | None = None
    risks: list[str] = Field(default_factory=list)
    domain_notes: list[str] = Field(default_factory=list)  # recalled approaches, UNVERIFIED


# ---- queue / runs ----------------------------------------------------------

class QueueItem(Base):
    id: str = ""                       # assigned by ledger.py, not by the LLM
    hypothesis: str
    change: str                        # the ONE change this experiment makes
    kind: ExpKind
    params: dict[str, Any] = Field(default_factory=dict)
    base: str | None = None            # exp_id this builds on (hill-climbing)
    status: ExpStatus = "pending"
    source: QueueSource = "strategist"


class Run(Base):
    exp_id: str
    cv_mean: float | None = None
    cv_std: float | None = None
    holdout: float | None = None
    oof_path: str | None = None
    test_pred_path: str | None = None  # test predictions in submission format
    artifact: str | None = None
    script_path: str | None = None
    seconds: float = 0.0
    seed: int = 42
    error: str | None = None           # a crash is a Run with error, not a CV attempt
    source: RunSource = "experimenter"


# ---- validation / analysis (Validator, Analyzer) ---------------------------

class ValidationRecord(Base):
    exp_id: str
    verdict: VerdictKind
    reasons: list[str] = Field(default_factory=list)
    signals_checked: list[str] = Field(default_factory=list)


class Analysis(Base):
    after_exp: str
    bottleneck: str
    evidence: str
    new_ids: list[str] = Field(default_factory=list)


# ---- final (Finisher) ------------------------------------------------------

class FinalResult(Base):
    ensemble: dict[str, Any] | None = None     # {method, members, weights, cv}
    submission_path: str | None = None
    checks: dict[str, bool] = Field(default_factory=dict)
    report_path: str | None = None


# ---- the ledger ------------------------------------------------------------

class Ledger(Base):
    problem: Problem
    env: Env = Field(default_factory=Env)
    profile: Profile | None = None
    strategy: Strategy | None = None
    queue: list[QueueItem] = Field(default_factory=list)
    runs: list[Run] = Field(default_factory=list)
    validation: list[ValidationRecord] = Field(default_factory=list)
    analysis: list[Analysis] = Field(default_factory=list)
    docs: dict[str, str] = Field(default_factory=dict)  # library -> cheat sheet
    final: FinalResult | None = None
    status: RunStatus = "running"
    stop_reason: str | None = None     # target | plateau | budget | minutes | empty_queue | rate_limit
