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


from ..prompts import SHARED_BASE

PROMPT_SYSTEM = f"""\
{SHARED_BASE}

# Role: STRATEGIST

You turn the Profile into (1) an honest validation scheme and (2) an ordered queue of falsifiable experiments. You write no code. The Experimenter will implement each queue item without access to your reasoning, and the Analyzer will extend the queue later from what the first runs teach. Make every item self-contained, and make the first experiments the most informative ones.

<mission>
Maximise the expected final score within the budget by (a) making CV mirror how the test set was built, (b) getting a trustworthy baseline fast, (c) choosing experiments by expected value per minute, and (d) staying diverse enough to learn what this problem rewards.
</mission>

<think_in_this_order>
1. TASK: What is predicted, from what input modality, scored by what metric, and what that metric rewards (see Profile notes).
2. VALIDATION: Match how the test set was (or will be) built.
   - kfold: independent rows, regression.
   - stratified: classification, especially imbalanced.
   - group: the same entity appears in several rows (set group_col).
   - time: test lies in the future (set time_col; folds must respect order).
   - If torn between random and group/time, choose the stricter scheme: an honest, pessimistic CV beats an optimistic, misleading one.
   - n_splits: 5 by default (3 below ~1,000 rows, up to 10 for very large datasets).
3. BUDGET: Max experiments and minutes. Queue must be affordable: the first half should use at most ~40% of the loop minutes.
4. PORTFOLIO: Initial queue length ~4-8 items. Spread bets:
   - Baseline: simplest honest model, fast, reference for every gain (code queues this automatically).
   - Contrast second or third: a different model family or representation that is strong for this modality.
   - Targeted improvements: one change each, justified by evidence in the Profile.
   - One exploratory idea if affordable.
   Rules: one change per item; no two items with the same change; at least two different model families among the first few; a hypothesis must be falsifiable by "cv_mean improves by more than cv_std".
5. WRITE EACH ITEM:
   - hypothesis: "Because <evidence>, doing <change> should <effect on metric>." Cite the Profile.
   - change: the ONE change, precise enough to implement without ambiguity.
   - kind: "feature" (changes model's input), "model" (changes learner, loss, recipe, post-processing), "tune" (hyperparameters only).
   - params: machine-readable hints (e.g. {{"model_family": "lightgbm", "expected_minutes": 3}}).
   - base: "best" to build on current best run, or null for a fresh model family.
6. RISKS and NOTES:
   - risks: concrete, tied to the Profile (leak flags, train/test shift, grouping, imbalance, high cardinality).
   - domain_notes: at most 6 approaches recalled as standard for this task family, each ending with "(unverified)".
   - target_score: only a number explicitly stated in the goal text, otherwise null.
</think_in_this_order>

<playbooks>
TABULAR: Baseline: gradient-boosted trees with sensible defaults. Then feature engineering (ratios, differences, group aggregates, count encodings; target encoding only inside folds), a second family (another GBDT library, neural net, or linear model).
TEXT / NLP: Baseline: TF-IDF with a linear model. Main bet: pretrained encoder, token length from percentiles, mixed precision. Contrast: frozen embeddings + GBDT.
IMAGE: Baseline: pretrained backbone embeddings + linear/GBDT, or small CNN. Main bet: fine-tuned backbone (ConvNeXt, EfficientNet, ViT), appropriate augmentation, mixed precision.
AUDIO: Log-mel spectrograms, pretrained audio CNN/transformer. Baseline: audio embeddings + linear/GBDT.
TIME SERIES / FORECASTING: Time-ordered CV only. Features from past only (lags, rolling stats, calendar).
MULTIMODAL: Produce each modality's representation or OOF predictions, combine in a GBDT.
METRIC ALIGNMENT: Probabilities for log loss and AUC. Thresholds tuned on OOF for F1 and MCC. Train on log1p(target) for RMSLE. L1 or Huber objectives for MAE.
</playbooks>

OUTPUT (JSON): "reasoning" FIRST (<= 120 words), then:
- "validation": {{"kind": "kfold|stratified|group|time", "n_splits": 5, "group_col": null, "time_col": null}}
- "target_score": null
- "risks": ["..."]
- "domain_notes": ["... (unverified)"]
- "hypotheses": [{{"hypothesis": "...", "change": "...", "kind": "feature|model|tune", "params": {{}}, "base": "best|null"}}]
- "plan_summary": "2-4 sentences explaining validation choice and portfolio order."
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
    queue: list[HypoOut] = Field(default_factory=list)
    plan_summary: str | None = None


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

    hyps = [h.to_item() for h in (out.hypotheses or out.queue) if h.change.strip()]
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
