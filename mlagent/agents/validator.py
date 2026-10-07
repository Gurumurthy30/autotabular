"""Validator (independent). Sees the SCRIPT and the RESULT numbers only — never the hypothesis or the
Experimenter's reasoning. Never edits model artifacts; its own scripts are scripts/val_<exp>.py.

Checks: (code) OOF recomputed on the fixed folds must match the reported CV, no NaN OOF inside valid
folds, CV-vs-holdout gap; (LLM-written script) static leakage review of the experiment script.
"Refit" = recomputation from OOF; a full re-run is skipped to protect the free-tier budget."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from pydantic import Field

from .. import ui
from ..executor import parse_result, tail
from ..ledger import Workspace, compact, digest, latest_ok_run, load, session
from ..state import Base, ValidationRecord
from ..llm import LLMFormatError, get_llm
from .coder import write_and_run

SYSTEM = ("You are an independent ML validator. You have NOT seen the author's reasoning. Given the signals, "
          "reject if there is evidence of leakage, an invalid validation setup, or a suspicious CV/holdout gap; "
          "otherwise approve. Fields: verdict ('approve' or 'reject'), reasons (short list).")

SCRIPT_TASK = """You are verifying experiment {exp} that you did not write. Do NOT retrain models.
Experiment script under review:
```python
{code}
```
Write a verification script that emits a dict with EXACTLY these keys:
  features_used: best-effort list of feature column names the script feeds the model (from reading it)
  uses_id_as_feature: bool — is ID (or an id-like column) a model input?
  fit_before_split: bool — is any scaler/encoder/target-encoder/imputer fit on data that includes validation rows
  suspect_features: columns whose |spearman| with the target on load_train() exceeds 0.9 (compute it)
  notes: short string
Wrap anything fragile in try/except and always call emit(...)."""


class Verdict(Base):
    verdict: str = "approve"
    reasons: list[str] = Field(default_factory=list)


def core_signals(L, r, kit) -> tuple[dict, list[str]]:
    sig: dict = {}
    hard: list[str] = []
    if not (r.oof_path and Path(r.oof_path).exists()):
        return sig, ["no OOF file for this run"]
    oof, y, d = np.load(r.oof_path), kit.get_y(), L.problem.direction
    scores, nan_in_valid = [], False
    for f in kit.folds():
        va = np.asarray(f["valid"])
        if not np.isfinite(oof[va]).all():
            nan_in_valid = True
            continue
        scores.append(kit.score(y[va], oof[va]))
    std = r.cv_std or 0.0
    if nan_in_valid or not scores:
        hard.append("OOF predictions contain NaN inside validation folds")
    else:
        rec = float(np.mean(scores))
        sig["recomputed_cv"], sig["reported_cv"] = round(rec, 5), round(r.cv_mean, 5)
        if abs(rec - r.cv_mean) > 1e-3 + 0.1 * std:
            hard.append(f"reported CV {r.cv_mean:.4f} != CV recomputed from OOF {rec:.4f}")
    if r.holdout is not None:
        gap = (r.cv_mean - r.holdout) if d == "maximize" else (r.holdout - r.cv_mean)
        sig["cv_minus_holdout_gap"] = round(gap, 5)
        if gap > max(4 * std, 0.1 * abs(r.cv_mean)):
            hard.append(f"holdout much worse than CV (gap {gap:.4f}): overfit to CV or leakage")
    return sig, hard


def validate_one(ws: Workspace, exp_id: str) -> ValidationRecord | None:
    L = load(ws)
    r = latest_ok_run(L, exp_id)
    if r is None:
        return None
    sig, hard = core_signals(L, r, ws.load_mlkit())

    code = Path(r.script_path).read_text() if r.script_path and Path(r.script_path).exists() else ""
    _, res = write_and_run(ws, f"val_{exp_id}", SCRIPT_TASK.format(exp=exp_id, code=code[:6000]),
                           kind="analysis", timeout=300)
    extra = parse_result(res.stdout) if res.ok else None
    if extra:
        sig.update({k: extra.get(k) for k in ("uses_id_as_feature", "fit_before_split", "suspect_features", "notes")})
        if extra.get("uses_id_as_feature") is True:
            hard.append("an id column is used as a model input")
    else:
        sig["static_review"] = "unavailable: " + re.sub(r"\s+", " ", tail(res.stderr, 120))

    if hard:
        verdict, reasons = "reject", hard
    else:
        try:
            v = get_llm().json(SYSTEM, compact({**digest(L, "validator", exp_id), "script_excerpt": code[:2500],
                                                "signals": sig}), Verdict)
            verdict = "reject" if v.verdict.strip().lower().startswith("rej") else "approve"
            reasons = v.reasons[:5]
        except LLMFormatError:
            flagged = bool(sig.get("fit_before_split") or sig.get("suspect_features"))
            verdict, reasons = ("reject", ["flags present, verdict model unavailable"]) if flagged else ("approve", [])
    rec = ValidationRecord(exp_id=exp_id, verdict=verdict, reasons=reasons, signals_checked=sorted(sig))  # type: ignore[arg-type]
    with session(ws) as L:
        L.validation.append(rec)
    ui.say("Validator", f"{exp_id} → {verdict}" + (f"  ({'; '.join(reasons)[:110]})" if reasons else ""))
    return rec


def run(ws: Workspace, exp_ids: list[str]) -> list[ValidationRecord]:
    return [r for r in (validate_one(ws, e) for e in exp_ids) if r]
