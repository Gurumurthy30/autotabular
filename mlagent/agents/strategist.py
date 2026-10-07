"""Strategist: LLM picks the CV scheme + hypotheses; folds.json / holdout.json are built in plain code
(row indices = data, not code), so every experiment compares on identical splits."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold,
                                     TimeSeriesSplit, train_test_split)

from .. import ui
from ..ledger import Workspace, add_queue_items, compact, digest, load, session
from ..llm import LLMFormatError, get_llm
from ..state import Base, CVScheme, Problem, QueueItem, Strategy
from .coder import HypoOut

def goal_target(text: str) -> float | None:
    """Parse a numeric target score from free-form goal text.

    Handles: "reach 0.85", "at least 85%", "rmse below 0.4", "predict survival" (→ None).
    Percentages in (1, 100] are divided by 100. Values outside [0, 100] are ignored.
    """
    import re as _re
    for raw in _re.findall(r"\b(\d+(?:\.\d+)?)\s*%?", text):
        v = float(raw)
        if "%" in text[text.find(raw):text.find(raw) + len(raw) + 2]:
            v = v / 100.0
        elif v > 1.0:
            # bare integer like "85" without "%" — treat as percentage
            v = v / 100.0
        if 0.0 < v <= 1.0:
            return round(v, 6)
    return None


PROMPT_SYSTEM = """\
You are the strategist of a Kaggle-grandmaster-level ML team and you own the plan. You receive the problem, a data profile (column roles, target stats, missingness, leakage flags, notes) and the installed libraries. Your choices fix the validation scheme and the experiment queue that the team then executes mechanically. A wrong validation scheme poisons every later result, so decide that first and with care.

HOW TO DECIDE
1. Validation — kind is one of kfold | stratified | group | time. Use the evidence in the profile:
   - rows ordered in time, or a datetime column that separates train from test -> "time" (+ time_col)
   - an entity repeating across rows (user, patient, station, site) that is unseen in test -> "group" (+ group_col)
   - classification, especially with a rare class -> "stratified"; regression -> "kfold"
   - n_splits: 5 by default, 3 below ~1,000 rows, up to 10 above ~100k.
   If two schemes look plausible, choose the one that better mimics how the test set was split from train, and say why in "risks".
2. Risks: 2-5 concrete risks for THIS data, naming columns (leakage flags, imbalance, small data, shift, high-cardinality categoricals, target skew).
3. Domain notes: approaches you recall that work on similar data. They are unverified recollections; write them as such.
4. Hypotheses: 5-8 experiments, each ONE change testable by one script. Spread your bets:
   - at least 2 feature ideas tied to specific columns or roles in the profile (never a generic "add features"),
   - at least 2 model/algorithm ideas that fit the data size and metric (gradient boosting if installed; regularised linear models for tiny data),
   - at most 1 tuning idea (a Tuner runs the main search later).
   Order by expected gain per minute of compute: cheap, high-signal changes first. No baseline (added automatically). Never repeat an idea in other words.
   base: "best" = build on the current best run (feature changes and tweaks); null = fresh script (a different model family).
   Propose only libraries that are installed; nothing that needs the internet or a GPU.
5. target_score: only a number explicitly stated in the goal text, otherwise null.

YOU MAY choose any validation scheme, model family or feature idea that the evidence supports. YOU MUST NOT use test labels, invent columns that are not in the profile, or change budgets and routing (code owns those).

OUTPUT (JSON): "reasoning" FIRST (<= 120 words: what the profile tells you and why this validation scheme), then validation {kind, n_splits, group_col, time_col}, target_score, risks[], domain_notes[], hypotheses[{hypothesis, change, kind, params, base}].
"hypothesis" = the claim and why it should help; "change" = the single concrete modification, specific enough to implement without asking questions; "kind" is feature | model | tune.
Example hypothesis: {"hypothesis": "Rainfall depends on humidity relative to temperature, which trees find hard to build", "change": "add humidity/temp and dew-point-gap ratio columns", "kind": "feature", "params": {}, "base": "best"}

SELF-CHECK before answering: Is the CV scheme justified by the profile? Does every hypothesis name a column or an algorithm? Any duplicates? Is each exactly ONE change?\
"""

PROMPTS = {
    "system": PROMPT_SYSTEM,
}

SYSTEM = PROMPT_SYSTEM


class CVOut(Base):
    kind: str = "stratified"
    n_splits: int = 5
    group_col: str | None = None
    time_col: str | None = None


class StrategyOut(Base):
    validation: CVOut = Field(default_factory=CVOut)
    target_score: float | None = None
    risks: list[str] = Field(default_factory=list)
    domain_notes: list[str] = Field(default_factory=list)
    hypotheses: list[HypoOut] = Field(default_factory=list)


def _sanitize_cv(cv: CVOut, p: Problem, cols: list[str], n: int) -> CVScheme:
    """Code overrides the LLM where its pick is invalid for the data."""
    kind = cv.kind if cv.kind in ("kfold", "stratified", "group", "time") else "stratified"
    clf = p.task_type != "regression"
    if kind == "group" and cv.group_col not in cols:
        kind = "stratified" if clf else "kfold"
    if kind == "time" and cv.time_col not in cols:
        kind = "stratified" if clf else "kfold"
    if kind == "kfold" and clf:
        kind = "stratified"
    if kind == "stratified" and not clf:
        kind = "kfold"
    k = max(3, min(10, int(cv.n_splits or 5)))
    k = min(k, max(3, n // 30))
    return CVScheme(kind=kind, n_splits=k,  # type: ignore[arg-type]
                    group_col=cv.group_col if kind == "group" else None,
                    time_col=cv.time_col if kind == "time" else None)


def make_folds(df: pd.DataFrame, p: Problem, cv: CVScheme, seed: int, holdout_frac: float = 0.15):
    n, idx, y = len(df), np.arange(len(df)), df[p.target]
    hf = holdout_frac if n * holdout_frac >= 20 else 0.0
    stratify = cv.kind == "stratified"
    if hf == 0:
        pool, ho = idx, np.array([], dtype=int)
    elif cv.kind == "time":
        s = pd.to_datetime(df[cv.time_col], errors="coerce")
        s = df[cv.time_col] if s.isna().all() else s
        order = np.argsort(s.to_numpy(), kind="stable")
        cut = int(n * (1 - hf))
        pool, ho = order[:cut], order[cut:]
    elif cv.kind == "group":
        pool, ho = next(GroupShuffleSplit(1, test_size=hf, random_state=seed).split(idx, groups=df[cv.group_col]))
    else:
        pool, ho = train_test_split(idx, test_size=hf, random_state=seed, shuffle=True,
                                    stratify=y if stratify and y.value_counts().min() >= 2 else None)
    yp, k = y.iloc[pool], cv.n_splits
    if cv.kind == "stratified" and yp.value_counts().min() >= k:
        splits = StratifiedKFold(k, shuffle=True, random_state=seed).split(pool, yp)
    elif cv.kind == "group":
        splits = GroupKFold(k).split(pool, groups=df[cv.group_col].iloc[pool])
    elif cv.kind == "time":
        splits = TimeSeriesSplit(k).split(pool)
    else:
        splits = KFold(k, shuffle=True, random_state=seed).split(pool)
    folds = [{"train": pool[a].tolist(), "valid": pool[b].tolist()} for a, b in splits]
    return folds, sorted(int(i) for i in ho)


BASELINE = QueueItem(
    hypothesis="A simple, sensible baseline sets the reference score every later change is judged against.",
    change="Minimal preprocessing (simple_prep) + RandomForest (Classifier or Regressor by task), no feature engineering.",
    kind="model", params={"model": "RandomForest", "baseline": True})


def run(ws: Workspace) -> None:
    L = load(ws)
    p = L.problem
    ui.say("Strategist", "choosing validation scheme and hypotheses")
    df = pd.read_csv(p.train_path)
    try:
        out = get_llm().json(SYSTEM, compact(digest(L, "strategist"), 7000), StrategyOut)
    except LLMFormatError:
        ui.warn("strategist reply unusable; falling back to defaults")
        out = StrategyOut()
    cv = _sanitize_cv(out.validation, p, list(df.columns), len(df))
    folds, ho = make_folds(df, p, cv, L.env.seed)
    ws.folds_path.write_text(json.dumps({"kind": cv.kind, "n": len(df), "folds": folds}))
    ws.holdout_path.write_text(json.dumps({"indices": ho}))

    hyps = [h.to_item() for h in out.hypotheses if h.change.strip()]
    if not hyps:                                   # weak-model safety net: a few generic ideas
        hyps = [QueueItem(hypothesis=h, change=c, kind=k, base="best") for h, c, k in [
            ("Gradient boosting usually beats a random forest on tabular data.",
             "Switch to HistGradientBoosting (sklearn) with default params.", "model"),
            ("Domain features add signal trees cannot build themselves.",
             "Add ratio/interaction/count features from the strongest raw columns.", "feature"),
            ("Rare categories add noise.", "Group categories seen < 10 times into 'other'.", "feature")]]
    with session(ws) as L:
        L.strategy = Strategy(validation=cv, folds_path=str(ws.folds_path), holdout_path=str(ws.holdout_path),
                              target_score=out.target_score, risks=out.risks[:6], domain_notes=out.domain_notes[:6])
        add_queue_items(L, [BASELINE.model_copy(deep=True)], "strategist")     # always first
        added = add_queue_items(L, hyps[:8], "strategist")
    ui.say("Strategist", f"cv={cv.kind}/{cv.n_splits}, holdout={len(ho)} rows, queue: baseline + {len(added)} hypotheses")
