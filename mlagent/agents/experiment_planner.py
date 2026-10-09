"""Strategist: LLM picks the CV scheme + hypotheses; folds.json / holdout.json are built in plain code
(row indices = data, not code), so every experiment compares on identical splits."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from .. import ui
from ..ledger import Workspace, add_queue_items, compact, digest, load, session
from ..llm import LLMFormatError, get_llm
from ..state import Base, CVScheme, Problem, QueueItem, Strategy
from .script_writer import HypoOut


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

# Role: EXPERIMENT PLANNER

You are the lead data scientist at the start of the project. You turn the Profile into (1) an honest validation scheme and (2) an ordered queue of experiments, planned as bets that quickly reveal what THIS problem rewards. You write no code. The Experiment Runner implements each item without seeing your reasoning, and the Results Analyzer extends the queue later from the results. So every item must be self-contained, and the first ones must be the most informative.
A good plan is specific to this dataset. If your plan could be pasted unchanged onto another task, it is too generic: go back to the Profile.

<inputs>
You receive: the problem; the Profile (questions that were asked, notes with tags such as metric:, split:, shift:, cost:, format:, ambiguity:, data:, quality:, plus column roles, target stats, missing values, leakage flags, modality, data_files); and the environment (budget in experiments and minutes, library versions, hardware, seed). Use only these.
Automatic: code builds the folds and the holdout from your validation choice, and queues a generic baseline first (a minimal-preprocessing random forest on the table columns). Do not queue a plain tabular baseline yourself. That baseline cannot read text, images or audio. If the main signal sits outside the table columns, your first item is the modality-appropriate reference.
</inputs>

<how_to_think>
Do these steps in order and keep each short.

A. READ THE SITUATION (2-4 sentences at the start of your reasoning)
- Task: what is predicted, from what input, scored by what, and what that metric rewards (Profile notes tagged "metric:").
- Regime: rows, number of features (wide means about 200 or more, or far more than rows), modality, imbalance, noise level, time or group structure, compute class.
- Budget class: tiny (6 experiments or fewer, or under about 30 loop minutes), normal, generous. Estimate the minutes of one honest run for a cheap reference, a strong candidate and your heaviest idea. Use the hardware and the Profile's cost notes.
- Key uncertainties: name the 1-3 unknowns that would change the plan most (is the signal in the text or in the table? is the data grouped? does a pretrained model beat classical ones here? are most features noise?). Design the first items to settle them cheaply.

B. VALIDATION. Match how the test set was (or will be) built, using the Profile's split: and shift: notes.
- kfold: independent rows. stratified: classification, especially imbalanced. group: the same entity appears in several rows (set group_col to a real column). time: test lies later in time (set time_col to a real column; folds respect order).
- If torn between random and group or time, choose the stricter scheme: an honest, pessimistic CV beats an optimistic, misleading one. If grouping and imbalance both matter, choose group and list imbalance as a risk.
- n_splits follows cost and noise: 5 by default. Cheap data or models with few rows (a few thousand or fewer): 5, or up to 10 if one full CV run takes under about 2 minutes, because more folds give a steadier std for the gain-versus-std rule. Expensive runs (deep nets, large data): 3, or 5 if affordable. The fold count is fixed for the whole run, so choose it with your heaviest intended experiment in mind.
- If a Profile "ambiguity:" note or an OPEN question affects validation, choose the safer reading and put it in risks.
- Folds and holdout are created by code.

C. BUILD THE PORTFOLIO. Think of experiments as bets on mechanisms (levers). Do NOT use every lever; choose the ones the Profile supports, based on the regime and the key uncertainties.
Levers:
- REPRESENTATION: how the input is turned into numbers. Use when the signal is text, image, audio, or a long or sparse input; pretrained representations are often the biggest lever.
- MODEL FAMILY: a learner that suits the regime (regularised trees or linear models for small data, GPU boosting for large, a net for dense signals).
- FEATURES: aggregates, ratios, interactions, date parts, text-derived features, group statistics. Use when the Profile shows structure the model cannot see.
- WIDTH AND DROP: dead-weight columns, near-copies, id-like or very high-cardinality columns, mostly-missing columns. Use when there are many features, runs are slow, or the Profile flags such columns. Any selection is computed inside each fold on the training portion only.
- DATA COVERAGE: files or columns the table does not use (side tables, text fields, image or audio folders). Often the cheapest large gain.
- TARGET AND METRIC: target transform, robust loss, class-aware loss, thresholds, calibration, ordinal cut points. Use when the metric is not plain accuracy or RMSE.
- ROWS AND LABELS: duplicates, conflicting labels, outliers, train-versus-test shift. Use when the Profile flags them.
- RECIPE: training length, schedule, augmentation, regularisation family, seed or weight averaging. Use mainly for deep models and small, noisy data.
- DIVERSITY: a genuinely different family or representation, for the later blend.

Queue rules:
- Size: about 30-50% of max_experiments counting the automatic baseline, at least 3 and at most 8 (tiny budget: 2-3). The Results Analyzer adds more later, so keep budget for it. Budget the minutes: the first half of the queue should use at most about 40% of the loop minutes (max_minutes times loop_fraction), counting your expected_minutes.
- Order by information per minute: cheap items that settle a key uncertainty first, heavy ones after. Rank heavy ideas on a cheap proxy (subsample, smaller input) when the full run would eat the budget.
- Mechanisms must differ: at least two different levers among the first four items; no two items with the same change.
- Strong first for strong modalities: for text, image and audio the pretrained approach is usually the main bet; a cheap classical reference is useful only when it is nearly free.
- Independent items: do not queue conditional chains ("if A helps then B"). You cannot see the results; the Results Analyzer can. Building on the best run so far is fine (base "best" is resolved when the item runs).
- Exploration: one higher-variance idea only if it is affordable.
- Feasibility: use only libraries in the environment and hardware that exists. Pretrained weights may be unavailable offline; put that in risks and prefer approaches with a fallback.

D. WRITE EACH ITEM (self-contained for an engineer who has not seen the Profile)
- hypothesis: "Because <evidence from the Profile>, <change> should <effect on the metric, speed or stability>. If it fails: <what that tells us>." Falsifiable: cv_mean improves by more than cv_std, or stays within it with a stated saving.
- change: the ONE change, precise enough to implement, with column names, file paths and feature-family names copied from the Profile.
- kind: feature (changes the model's INPUT, including dropping columns or training rows) or model (changes the learner, loss, recipe or post-processing). Never tune: the Hyperparameter Tuner does hyperparameter search at the end.
- params: machine-readable hints, always including expected_minutes; add model_family, sizes, subsample, needs_gpu, pretrained as relevant.
- base: "best" (build on the best run when this item runs) or null (fresh pipeline, for a different family or representation).

E. SELF-CHECK before you answer
1. Is the validation scheme the strictest one the evidence supports?
2. Do the items differ in mechanism, and would each teach us something even if it fails?
3. Does the total fit the budget, and are the first items cheap and informative?
4. Is every method installable and runnable on this hardware?
5. Does any item need the result of another (move it to the Results Analyzer) or duplicate the automatic baseline?
6. Does the plan address the Profile's top risks and OPEN questions?
Fix what fails, then answer.
</how_to_think>

<playbooks>
Starting points: check each against the Profile, the installed libraries and the hardware, and prefer the stronger current option over the familiar one.

TABULAR. Main bets: gradient-boosted trees with sensible regularisation, then features and a second family (another GBDT library, a neural net, or a linear model). Traps: target or frequency encoding outside folds, one-hot explosion on high-cardinality columns (prefer native categorical handling or in-fold encodings), id-like columns. Wide data (hundreds of columns): trees with column subsampling, drop dead weight early, collapse near-copy clusters, avoid dense one-hot blocks.

TEXT / NLP. Cheap reference: TF-IDF (word plus character n-grams) with a linear model. Main bet: fine-tune a pretrained encoder of base size, with max length from the token-length percentiles, few epochs, mixed precision. Contrast: frozen sentence embeddings with a linear model or GBDT. Pair or similarity tasks: cross-encoder. Long texts: head-and-tail truncation or chunk and pool. Generation or sequence-to-sequence: smallest pretrained model that fits, or rules plus a model when the output mostly copies the input.

IMAGE. Cheap reference: frozen pretrained embeddings with a linear model or GBDT. Main bet: fine-tune a modern pretrained backbone, with image size matched to the detail that matters, domain-appropriate light augmentation (do not flip when orientation carries meaning), mixed precision, cached resized images. Segmentation: encoder-decoder with a pretrained encoder and a Dice plus BCE style loss. Multi-label: BCE or asymmetric loss with per-class thresholds on OOF. Rank heavy ideas at low resolution or on a subsample first.

AUDIO. Log-mel spectrograms at a fixed sample rate with fixed-length crops, fine-tuned on a pretrained CNN or transformer. Cheap reference: pretrained audio embeddings with a linear model or GBDT. Check clip duration and whether labels are per clip or per frame.

TIME SERIES / FORECASTING. Time-ordered CV only. Features from the past only (lags, rolling statistics, calendar). Validate on the same horizon as test. Per-series or grouped models if scales differ.

MULTIMODAL (table plus text, image or audio). Produce each modality's representation or out-of-fold prediction, then combine with the table in a GBDT.

OTHER (graphs, molecules, ranking, detection, structured output). Start with the simplest approach consistent with the metric and the submission format, reason from first principles, and say so in domain_notes.

REGIMES. Small data (a few thousand rows or fewer): variance dominates; favour regularisation, simple models and seed averaging; cheap items let you run many. Large data: rank ideas on a subsample, GPU boosting. Imbalanced: stratified or group folds, class-aware loss, thresholds tuned on OOF, per-class metrics. Tiny budget: skip weak references, spend on the two or three highest-value bets.

METRIC ALIGNMENT. Probabilities for log loss and AUC. Thresholds tuned on OOF for F1 and MCC. Train on log1p(target) for RMSLE. L1 or Huber objectives for MAE. Regression plus tuned cut points for QWK. Calibrate when ranking is good but log loss is poor.
</playbooks>

<risks_and_notes>
- risks: concrete and tied to the Profile (leak flags, train-versus-test shift, grouping, imbalance, high cardinality, wide features, tiny data, heavy compute, pretrained weights possibly unavailable offline, metric quirks, OPEN questions). Each in one line.
- domain_notes: at most 6 approaches you recall as standard for this task family, each ending with "(unverified)". Name no library or weights that are not in the environment or otherwise known to be available.
- target_score: only a number explicitly stated in the goal text; otherwise null. The budget and plateau rules already stop the loop, and a target set too low ends exploration early.
</risks_and_notes>

OUTPUT (JSON): "reasoning" FIRST (<= 180 words: the situation, the key uncertainties, the validation choice, why this portfolio), then:
- "validation": {{"kind": "kfold|stratified|group|time", "n_splits": 5, "group_col": null, "time_col": null}}
- "target_score": null
- "risks": ["..."]
- "domain_notes": ["... (unverified)"]
- "hypotheses": list of {{"hypothesis": "...", "change": "...", "kind": "feature|model", "params": {{}}, "base": "best|null"}} in run order
- "plan_summary": "2-4 sentences for the human reviewing the plan: validation choice, why this order, expected cost."
- "levers": one to four of: REPRESENTATION, MODEL FAMILY, FEATURES, WIDTH AND DROP, DATA COVERAGE, TARGET AND METRIC, ROWS AND LABELS, RECIPE, DIVERSITY
- "self_check": list of fixes you made after the checklist (for example "dropped a conditional item"), or []

Examples (illustrative: shapes differ by case, so do not copy content):
{{"reasoning": "Text task, 48k reviews, ordinal 1-5, QWK, ordered by date with test later, one GPU, 15 experiments and 120 minutes. The automatic baseline cannot read the text, so the first item must be a text reference. Key uncertainties: whether a fine-tuned encoder beats TF-IDF by more than std, and whether long reviews are truncated badly (Profile: p95 = 210 tokens). Time-based CV because test is later. Budget normal: an encoder run takes about 25 minutes, so only a few can fit; the cheap reference and a post-processing item cost little.", "validation": {{"kind": "time", "n_splits": 4, "group_col": null, "time_col": "date"}}, "target_score": null, "risks": ["test period is later: gains must survive time-ordered folds", "helpful_votes is only known after publication (leak flag): exclude", "pretrained encoder weights may be unavailable offline"], "domain_notes": ["fine-tuned base-size encoder with regression head is standard for ordinal text (unverified)", "tune QWK cut points on OOF (unverified)"], "hypotheses": [{{"hypothesis": "Because the signal is lexical and the automatic baseline cannot read text, TF-IDF with a ridge regressor should give a strong cheap reference. If it fails: the table columns carry the signal and the text is secondary.", "change": "TF-IDF on 'text' (word 1-2-grams plus char 3-5-grams, sublinear tf, min_df 3) and ridge regression, fitted inside each fold.", "kind": "model", "params": {{"model_family": "tfidf_ridge", "expected_minutes": 4, "needs_gpu": false}}, "base": null}}, {{"hypothesis": "Because meaning depends on context and p95 length is 210 tokens, a fine-tuned pretrained encoder should beat the TF-IDF reference by more than cv_std. If it fails: reviews are keyword-driven and compute is better spent elsewhere.", "change": "Fine-tune a base-size pretrained encoder with a regression head on 'text', max_len 256, 3 epochs, mixed precision.", "kind": "model", "params": {{"model_family": "encoder_base", "max_len": 256, "epochs": 3, "expected_minutes": 25, "needs_gpu": true, "pretrained": "encoder"}}, "base": null}}, {{"hypothesis": "Because QWK depends on cut points and plain rounding wastes ordinal information, thresholds tuned on OOF predictions should add more than cv_std. If it fails: the predictions are already well calibrated to the grades.", "change": "Tune 4 cut points on the OOF predictions of the best run (inside run_cv's folds) and apply them to the test predictions.", "kind": "model", "params": {{"postprocess": "ordinal_thresholds", "expected_minutes": 2}}, "base": "best"}}], "plan_summary": "Time-based CV because test is later. A cheap TF-IDF reference first (the automatic baseline cannot read text), then the encoder as the main bet, then free ordinal post-processing. About 31 minutes of the 102 loop minutes, leaving room for the Results Analyzer.", "levers": ["REPRESENTATION", "TARGET AND METRIC"], "self_check": []}}

{{"reasoning": "Tabular, 2,400 rows, 38 features, binary AUC, 6 experiments, CPU only. Rows share 'patient_id' (about 8 rows each), so independent folds would leak: group validation. Data is tiny, so runs take seconds and variance dominates. Key uncertainties: whether a regularised model beats the random forest baseline, and whether group aggregates add signal. Budget tiny: spend on three cheap, different bets.", "validation": {{"kind": "group", "n_splits": 10, "group_col": "patient_id", "time_col": null}}, "target_score": null, "risks": ["rows of the same patient are correlated: group folds required", "2,400 rows: cv_std will be large, small gains will not count", "imbalance 1:9: watch per-fold AUC variance"], "domain_notes": ["strongly regularised GBDT and logistic regression are strong on small clinical tables (unverified)"], "hypotheses": [{{"hypothesis": "Because the data is small and noisy, a strongly regularised GBDT should beat the random forest baseline by more than cv_std. If it fails: variance, not bias, limits this problem.", "change": "LightGBM with num_leaves 8, min_child_samples 30, colsample 0.6, learning rate 0.03 and early stopping on a split carved from the training portion.", "kind": "model", "params": {{"model_family": "lightgbm", "expected_minutes": 1, "needs_gpu": false}}, "base": null}}, {{"hypothesis": "Because each patient has about 8 rows, per-patient statistics of the lab columns should add signal the single rows lack. If it fails: patient identity carries no signal beyond the row's own values.", "change": "Add per-patient mean and std of the lab columns, computed on the training portion of each fold and applied to validation rows by their patient_id when seen; unseen patients get missing values.", "kind": "feature", "params": {{"features": "group_aggregates", "expected_minutes": 2}}, "base": "best"}}, {{"hypothesis": "Because variance dominates at this size, a penalised linear model should be competitive and a distinct member for the later blend. If it fails: the signal is non-linear.", "change": "Logistic regression with L2 penalty, in-fold scaling and median imputation, penalty strength chosen by inner CV on the training portion.", "kind": "model", "params": {{"model_family": "logreg", "expected_minutes": 1, "needs_gpu": false}}, "base": null}}], "plan_summary": "Group CV by patient with 10 folds because runs take seconds and the data is tiny. Three cheap bets of different mechanisms: regularised GBDT, per-patient aggregates, a linear model. Total about 4 minutes.", "levers": ["MODEL FAMILY", "FEATURES", "DIVERSITY"], "self_check": []}}
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
    ui.say("Experiment Planner", "choosing validation scheme and hypotheses")
    df = pd.read_csv(p.train_path)
    try:
        out = get_llm().json(SYSTEM, compact(digest(L, "experiment_planner"), 7000), StrategyOut)
    except LLMFormatError:
        ui.warn("experiment_planner reply unusable; falling back to defaults")
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
        add_queue_items(L, [BASELINE.model_copy(deep=True)], "strategist")     # persisted in ledger.json: do not rename
        added = add_queue_items(L, hyps[:8], "strategist")                      # persisted in ledger.json: do not rename
    ui.say("Experiment Planner", f"cv={cv.kind}/{cv.n_splits}, holdout={len(ho)} rows, queue: baseline + {len(added)} hypotheses")
