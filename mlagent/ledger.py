"""Ledger I/O + every code-enforced rule (dedupe, gain > std, plateau, budget) + role digests.

The workspace lives NEXT TO THE DATA: <data_dir>/mlagent_<train_stem>/ — no central location.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Iterator

from .state import Ledger, QueueItem, Run

_LOCK = threading.RLock()      # parallel workers share one ledger file
MINIMIZE_METRICS = {"rmse", "mae", "mse", "logloss", "log_loss", "rmsle"}
RARE_LIBS = ("lightgbm", "xgboost", "catboost", "optuna")


# ---- workspace --------------------------------------------------------------

class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.scripts = self.root / "scripts"
        self.artifacts = self.root / "artifacts"
        self.logs = self.root / "logs"

    @classmethod
    def for_data(cls, train_path: str | Path) -> "Workspace":
        p = Path(train_path).expanduser().resolve()
        return cls(p.parent / f"mlagent_{p.stem}")

    @classmethod
    def resolve(cls, p: str | Path) -> "Workspace":
        """Accept a workspace dir, a dir that contains one, or the data file itself."""
        p = Path(p).expanduser().resolve()
        if p.is_dir():
            if (p / "ledger.json").exists():
                return cls(p)
            found = sorted(p.glob("mlagent_*/ledger.json"), key=lambda f: f.stat().st_mtime)
            if found:
                return cls(found[-1].parent)
            raise FileNotFoundError(f"no mlagent workspace found in {p}")
        return cls.for_data(p)

    def make(self) -> "Workspace":
        for d in (self.root, self.scripts, self.artifacts, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def ledger_path(self) -> Path: return self.root / "ledger.json"
    @property
    def folds_path(self) -> Path: return self.root / "folds.json"
    @property
    def holdout_path(self) -> Path: return self.root / "holdout.json"
    @property
    def submission_path(self) -> Path: return self.root / "submission.csv"
    @property
    def report_path(self) -> Path: return self.root / "report.md"
    @property
    def control_path(self) -> Path: return self.root / "control.json"
    @property
    def lb_path(self) -> Path: return self.root / "lb.json"

    def config(self):
        from .config import load_workspace
        return load_workspace(self.root)

    def load_mlkit(self):
        """Import the workspace's own mlkit.py (same helper the generated scripts use)."""
        spec = importlib.util.spec_from_file_location("mlkit_ws", self.root / "mlkit.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


# ---- load / save ------------------------------------------------------------

def _retry(fn, tries: int = 40, delay: float = 0.05):
    """Windows: a file replaced/read at the same instant by another process raises PermissionError."""
    for i in range(tries):
        try:
            return fn()
        except (PermissionError, FileNotFoundError):
            if i == tries - 1:
                raise
            time.sleep(delay)


def load(ws: Workspace) -> Ledger:
    return Ledger.model_validate_json(_retry(lambda: ws.ledger_path.read_text(encoding="utf-8")))


def save(ws: Workspace, L: Ledger) -> None:
    tmp = ws.ledger_path.with_suffix(".tmp")
    tmp.write_text(L.model_dump_json(indent=2), encoding="utf-8")
    _retry(lambda: os.replace(tmp, ws.ledger_path))          # atomic: scripts never read a half-written ledger


@contextlib.contextmanager
def session(ws: Workspace) -> Iterator[Ledger]:
    """Short read-modify-write. Never hold it across an LLM call."""
    with _LOCK:
        L = load(ws)
        yield L
        save(ws, L)


def set_item_status(ws: Workspace, exp_id: str, status: str) -> None:
    with session(ws) as L:
        for q in L.queue:
            if q.id == exp_id:
                q.status = status  # type: ignore[assignment]


# ---- queue rules ------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _fps(it: QueueItem) -> set[str]:
    p = json.dumps(it.params, sort_keys=True, default=str)
    return {hashlib.md5(f"{_norm(t)}|{p}".encode()).hexdigest() for t in (it.hypothesis, it.change)}


def add_queue_items(L: Ledger, items: list[QueueItem], source: str) -> list[str]:
    """Rule 1: reject an item whose hypothesis (or change) + params matches an earlier one."""
    seen: set[str] = set()
    for q in L.queue:
        seen |= _fps(q)
    new: list[str] = []
    for it in items:
        if not it.hypothesis.strip() or not it.change.strip():
            continue
        fps = _fps(it)
        if fps & seen:
            continue
        seen |= fps
        it.id = f"e{len(L.queue) + 1:03d}"
        it.status, it.source = "pending", source  # type: ignore[assignment]
        L.queue.append(it)
        new.append(it.id)
    return new


def next_pending(L: Ledger) -> QueueItem | None:
    return next((q for q in L.queue if q.status == "pending"), None)


# ---- run rules --------------------------------------------------------------

def verdicts(L: Ledger) -> dict[str, str]:
    d: dict[str, str] = {}
    for v in L.validation:
        d[v.exp_id] = v.verdict
    return d


def ok_runs(L: Ledger, source: str | None = None) -> list[Run]:
    return [r for r in L.runs if r.error is None and r.cv_mean is not None
            and (source is None or r.source == source)]


def approved_runs(L: Ledger, source: str | None = None) -> list[Run]:
    v = verdicts(L)
    return [r for r in ok_runs(L, source) if v.get(r.exp_id) == "approve"]


def latest_ok_run(L: Ledger, exp_id: str) -> Run | None:
    rs = [r for r in ok_runs(L) if r.exp_id == exp_id]
    return rs[-1] if rs else None


def _better(a: float, b: float, direction: str) -> bool:
    return a > b if direction == "maximize" else a < b


def beats(new_mean: float, best_mean: float, best_std: float, direction: str) -> bool:
    """Rule 2: a gain counts only if it clears the best run's fold std."""
    std = best_std or 0.0
    return new_mean > best_mean + std if direction == "maximize" else new_mean < best_mean - std


def best_run(L: Ledger, source: str | None = None) -> Run | None:
    rs = approved_runs(L, source)
    if not rs:
        return None
    d = L.problem.direction
    best = rs[0]
    for r in rs[1:]:
        if _better(r.cv_mean, best.cv_mean, d):
            best = r
    return best


def counted_gain_flags(L: Ledger) -> list[bool]:
    d, best, flags = L.problem.direction, None, []
    for r in approved_runs(L, "experimenter"):
        flags.append(best is None or beats(r.cv_mean, best.cv_mean, best.cv_std or 0.0, d))
        if best is None or _better(r.cv_mean, best.cv_mean, d):
            best = r
    return flags


def plateau(L: Ledger, n: int | None = None) -> bool:
    """Rule 3: last N approved runs show zero counted gains (the baseline never counts as 'last')."""
    n = n or L.env.budget.plateau_n
    flags = counted_gain_flags(L)
    return len(flags) > n and not any(flags[-n:])


def experiments_used(L: Ledger) -> int:
    return sum(1 for r in L.runs if r.source == "experimenter" and r.error is None)


def consecutive_crashes(L: Ledger) -> int:
    """Distinct experiments that crashed since the last success (Docs-assisted retries count once)."""
    ids: set[str] = set()
    for r in reversed([r for r in L.runs if r.source == "experimenter"]):
        if r.error is None:
            break
        ids.add(r.exp_id)
    return len(ids)


def minutes_elapsed(L: Ledger) -> float:
    return (time.time() - (L.env.started_at or time.time())) / 60.0


def target_hit(L: Ledger) -> bool:
    t = L.strategy.target_score if L.strategy else None
    b = best_run(L)
    if t is None or b is None:
        return False
    return b.cv_mean >= t if L.problem.direction == "maximize" else b.cv_mean <= t


def stop_reason(L: Ledger) -> str | None:
    """Rule 4 + friends: why the experiment loop must stop (None = keep going)."""
    b = L.env.budget
    if target_hit(L): return "target"
    if experiments_used(L) >= b.max_experiments: return "budget"
    if minutes_elapsed(L) >= b.max_minutes * b.loop_fraction: return "minutes"
    if consecutive_crashes(L) >= b.max_consecutive_crashes: return "crashes"
    if plateau(L): return "plateau"
    return None


def hard_stop(L: Ledger) -> str | None:
    """Limits that also bind during bounded re-entry (plateau/target are ignored there)."""
    b = L.env.budget
    if experiments_used(L) >= b.max_experiments: return "budget"
    if minutes_elapsed(L) >= b.max_minutes: return "minutes"
    return None


_OOM_ERR = re.compile(
    r"CUDA out of memory|OutOfMemoryError|out of memory|MemoryError|Unable to allocate|bad_alloc|"
    r"cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED|ResourceExhausted", re.I)


def is_oom_error(err: str) -> bool:
    """GPU/RAM exhaustion: handled by a downgrade retry, not by Docs."""
    return bool(_OOM_ERR.search(err or ""))


_API_ERR = re.compile(
    r"AttributeError|unexpected keyword|got an unexpected|positional argument|"
    r"cannot import name|TypeError|has no attribute|is not a valid parameter|Invalid parameter", re.I)


def is_api_error(err: str) -> bool:
    """Library-usage mistakes go to Docs; everything else is a logic problem for the Analyzer."""
    return bool(_API_ERR.search(err or ""))


# ---- digests (each agent sees a slice, never the whole file) ------------------

def compact(obj, limit: int = 6000) -> str:
    s = json.dumps(obj, default=str, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + " ...[truncated]"


def _r(x):
    return None if x is None else round(float(x), 4)


def unvalidated_ids(L: Ledger) -> list[str]:
    """Successful runs that never got a verdict (e.g. the process was interrupted before validation)."""
    v = verdicts(L)
    return [r.exp_id for r in ok_runs(L) if r.exp_id not in v]


# ---- persisted routing flags (so resume does not forget retries / the tuner stage) ----------------

CONTROL_KEYS = ("retries", "oom_retries", "tuned", "tuner_ids", "tuner_retry_used", "reentry_left",
                "runs_since_analysis", "analyzer_empty", "pending_validation", "crashes", "analysis_trigger")


def save_control(ws: Workspace, state: dict) -> None:
    ws.control_path.write_text(json.dumps({k: state[k] for k in CONTROL_KEYS if k in state}, default=str), encoding="utf-8")


def load_control(ws: Workspace) -> dict:
    try:
        return json.loads(ws.control_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run_rows(L: Ledger, last: int | None = None, only: str | None = None) -> list[dict]:
    v = verdicts(L)
    rows = [{"exp": r.exp_id, "src": r.source, "cv": _r(r.cv_mean), "std": _r(r.cv_std),
             "holdout": _r(r.holdout), "verdict": None if r.error else v.get(r.exp_id),
             "error": (r.error or "")[:160] or None}
            for r in L.runs if only is None or r.exp_id == only]
    return rows[-last:] if last else rows


def profile_digest(L: Ledger) -> dict:
    p = L.profile
    if p is None:
        return {}
    roles: dict[str, list[str]] = {}
    for c, r in p.column_roles.items():
        roles.setdefault(r, []).append(c)
    return {"roles": roles, "target": p.target_stats,
            "missing": {c: round(f, 3) for c, f in p.missing.items() if f > 0.01},
            "leakage_flags": p.leakage_flags, "notes": p.notes[:8]}


def digest(L: Ledger, role: str, exp_id: str | None = None) -> dict:
    pr = L.problem
    d: dict = {"problem": {k: getattr(pr, k) for k in ("goal", "target", "metric", "direction", "task_type")}}
    s = L.strategy
    strat = {"validation": s.validation.model_dump(), "risks": s.risks, "domain_notes": s.domain_notes} if s else {}
    if role == "profiler":
        return d
    if role == "strategist":
        return {**d, "profile": profile_digest(L), "libraries": L.env.library_versions}
    if role == "experimenter":
        b = best_run(L)
        return {**d, "strategy": strat, "profile": profile_digest(L),
                "best_so_far": {"exp": b.exp_id, "cv": _r(b.cv_mean)} if b else None,
                "libraries": L.env.library_versions}
    if role == "validator":
        return {**d, "validation": strat.get("validation"), "runs": run_rows(L, only=exp_id)}
    if role == "analyzer":
        return {**d, "profile": profile_digest(L), "strategy": strat, "runs": run_rows(L, last=12),
                "validation": [v.model_dump() for v in L.validation[-6:]],
                "queue": [{"id": q.id, "kind": q.kind, "change": q.change[:100], "status": q.status}
                          for q in L.queue[-25:]]}
    return d
