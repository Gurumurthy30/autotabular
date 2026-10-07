"""Analyzer: computes the signal list via Coder, finds the bottleneck, proposes NON-duplicate experiments.
The signal list lives in the prompt, not in code. Triggers: periodic | empty_queue | crash | tuner_failure."""

from __future__ import annotations

from pydantic import Field

from .. import ui
from ..executor import parse_result, tail
from ..ledger import Workspace, add_queue_items, best_run, compact, digest, load, session
from ..llm import LLMFormatError, get_llm
from ..state import Analysis, Base
from .coder import HypoOut, write_and_run

PROMPT_SIGNALS = """\
Compute these from the OOF predictions of run {best} (artifacts/{best}_oof.npy; rows with NaN OOF are holdout/unscored — ignore them). Skip any that do not apply to this task. Keep the output small (round to 4 dp) and wrap each block in try/except (store "error: ..." for that key):
1. per_class_error   (classification) per-class recall and confusion counts
2. residual_patterns (regression) mean/std of residual by target decile; the 3 features most correlated with |residual|
3. importance_shift  top-10 feature importances from a quick tree model on ONE fold, versus the correlation ranking
4. learning_curve    score of a quick model on 25/50/75/100% of fold-0 training rows (does more data still help?)
5. calibration       (binary probabilities) mean predicted vs observed rate in 5 bins
6. error_by_segment  error rate by the top 5 levels of the main categorical column
Finish with emit({{...signals...}}).\
"""

PROMPT_SYSTEM = """\
You are the analyst of an ML team. You decide what the team should try next, using only evidence: recent runs and validator verdicts, the experiment queue, the budget, and measured signals (or the failure context).

HOW TO THINK
1. Name the single biggest bottleneck and classify it: variance (CV std large, unstable), bias (all models similar and low), features (errors cluster in a segment or class), data/validation (CV and holdout disagree, leakage suspects), or engineering (crashes). Cite the specific numbers or signal keys as evidence.
2. Explore vs exploit: if the recent runs gained, push the same direction (exploit). If the last runs showed no counted gain (plateau), stop micro-tuning: change the model family, the representation of the strongest columns, or attack the weakest segment (explore).
3. Propose 1-3 NEW experiments, each ONE change, each aimed at that bottleneck and different from everything already in the queue or tried. Prefer cheap changes. Say what result would confirm the hypothesis.
4. If the evidence is thin, say so in "evidence" and propose the cheapest experiment that would produce evidence.

YOU MAY propose any feature, model or data-centric idea. YOU MUST NOT repeat queued ideas, use test labels, or assume columns that are not in the profile.

OUTPUT: Fields: reasoning FIRST (<= 100 words), then bottleneck (string), evidence (string, with numbers), hypotheses (list of {hypothesis, change, kind: feature|model|tune, params: object, base: "best" or null}).
Example: {"reasoning": "Recall for class 1 is 0.31 while class 0 is 0.92 and three runs gave no gain...", "bottleneck": "features: minority class under-served", "evidence": "per_class_error recall {0: 0.92, 1: 0.31}; last 3 runs +0.001", "hypotheses": [{"hypothesis": "Class weighting lets trees fit the minority class", "change": "set class_weight='balanced' in the model", "kind": "model", "params": {}, "base": "best"}]}\
"""

PROMPT_INSTR_CRASH = "The last experiment(s) crashed from a logic problem (see crash_error). Work out the root cause from the traceback, then propose a safer variant of the same idea (or a different idea if the idea itself is unsound for this data)."
PROMPT_INSTR_TUNER_FAILURE = "Tuned models were rejected by the validator (see validation reasons). Explain the most likely cause (e.g. tuning overfit the CV, leakage introduced by the search) and propose fixes or a more conservative alternative."
PROMPT_INSTR_CV_LB_GAP = "The public leaderboard score disagrees with our CV (see lb_gap). Treat this as a validation problem first: suspect leakage, folds that do not mimic the test split, target encoding, train/test shift. Propose experiments that test those causes (e.g. adversarial validation, dropping suspect features, group- or time-aware features)."

PROMPTS = {
    "system": PROMPT_SYSTEM,
    "signals": PROMPT_SIGNALS,
    "instr_crash": PROMPT_INSTR_CRASH,
    "instr_tuner_failure": PROMPT_INSTR_TUNER_FAILURE,
    "instr_cv_lb_gap": PROMPT_INSTR_CV_LB_GAP,
}

SIGNALS = PROMPT_SIGNALS
SYSTEM = PROMPT_SYSTEM


class AnalysisOut(Base):
    bottleneck: str = ""
    evidence: str = ""
    hypotheses: list[HypoOut] = Field(default_factory=list)


def run(ws: Workspace, trigger: str = "periodic", error: str | None = None) -> list[str]:
    L = load(ws)
    ui.say("Analyzer", f"trigger={trigger}")
    ctx: dict = {"trigger": trigger, **digest(L, "analyzer")}
    best = best_run(L)
    if trigger in ("periodic", "empty_queue") and best is not None:
        _, res = write_and_run(ws, f"signals_{len(L.analysis) + 1}", SIGNALS.format(best=best.exp_id),
                               kind="analysis", timeout=300)
        ctx["signals"] = parse_result(res.stdout) if res.ok else f"signal script failed: {tail(res.stderr, 300)}"
    elif trigger == "crash":
        ctx["crash_error"] = tail(error or "", 1200)
        ctx["instruction"] = "The last experiment crashed from a logic problem. Propose a safer variant."
    elif trigger == "tuner_failure":
        ctx["instruction"] = "Tuned models were rejected by the validator (see validation). Explain why and propose fixes."

    try:
        out = get_llm().json(SYSTEM, compact(ctx, 9000), AnalysisOut)
    except LLMFormatError:
        ui.warn("analyzer reply unusable; no new experiments")
        out = AnalysisOut(bottleneck="unknown", evidence="analyzer output invalid")
    items = [h.to_item() for h in out.hypotheses[:3] if h.change.strip()]
    with session(ws) as L:
        new_ids = add_queue_items(L, items, "analyzer")
        L.analysis.append(Analysis(after_exp=L.runs[-1].exp_id if L.runs else "-", bottleneck=out.bottleneck[:300],
                                   evidence=out.evidence[:600], new_ids=new_ids))
    ui.say("Analyzer", f"bottleneck: {out.bottleneck[:100]}  → +{len(new_ids)} queued ({len(items) - len(new_ids)} duplicates dropped)")
    return new_ids
