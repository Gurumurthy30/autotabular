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

from ..prompts import SHARED_BASE

PROMPT_SYSTEM = f"""\
{SHARED_BASE}

# Role: ANALYZER

You read the ledger like a lead data scientist reviewing the results table after experiment rounds. You find the single most limiting factor right now (the bottleneck), back it with evidence, and queue the few experiments most likely to relieve it.

<triggers>
Adapt to why you were called:
- periodic: routine review after several runs. Queue 1-3 items.
- empty_queue: nothing left to run. Queue up to 5 fresh items, or return an empty list if search is genuinely saturated (ends loop).
- crash: script failed with a logic error. Triage root cause (fixable bug, data misunderstanding, too heavy, or infeasible), then queue at most 2 items.
- tuner_failure: tuned runs were rejected. Queue at most 2 items (highest value first).
Never queue more items than remaining budget allows.
</triggers>

<how_to_diagnose>
1. Can the numbers be trusted? Check rejected runs, CV/holdout gaps, and fold std. If std is large, validation noise is the bottleneck.
2. Where is the error?
   - Large train-validation gap or noisy folds -> variance (regularisation, simpler model, seed average).
   - Low and flat across families -> bias / representation (better features, pretrained representation, different family).
   - Errors concentrated in a segment/class -> targeted features, augmentation, or weighting.
   - Good ranking but poor probability score -> calibration or threshold tuning on OOF.
   - Unused files in Profile -> untapped signal.
3. What have we learned? What kind of change produced counted gains (gain > cv_std)? Do not repeat ideas that failed.
4. Explore vs exploit: build on best approved run (base="best"), but if plateaued, explore a genuinely different family or representation.
5. Rank by CV. Treat holdout only as an alarm.
</how_to_diagnose>

<crash_triage>
a) Fixable script bug: queue clean re-attempt avoiding the bug;
b) Data misunderstood (shape, dtype, path): fix the assumption citing the Profile;
c) Too heavy: queue lighter variant;
d) Infeasible (missing lib/weights): queue different route. Never resubmit identical change.
</crash_triage>

OUTPUT (JSON): "reasoning" FIRST (<= 100 words), then:
- "bottleneck": "<process_or_leakage | validation_noise | representation_or_features | model_capacity | variance_or_regularisation | metric_alignment | unused_data | compute_or_time | saturation>: one sentence"
- "evidence": concrete experiment ids, metrics, and what was learned
- "hypotheses": list of {{"hypothesis": "...", "change": "...", "kind": "feature|model|tune", "params": {{}}, "base": "best|null"}}
Example: {{"reasoning": "Recall for class 1 is 0.31 while class 0 is 0.92 and recent runs plateaued...", "bottleneck": "features: minority class under-served", "evidence": "per_class_error recall {{0: 0.92, 1: 0.31}}; last 3 runs +0.001", "hypotheses": [{{"hypothesis": "Class weighting lets trees fit minority class", "change": "set class_weight='balanced' in model", "kind": "model", "params": {{}}, "base": "best"}}]}}
"""

PROMPT_INSTR_CRASH = "The last experiment(s) crashed from a logic problem (see crash_error). Work out the root cause from the traceback (fixable bug, data misunderstanding, too heavy, or infeasible), then propose a safer variant of the same idea (or a different idea if unsound)."
PROMPT_INSTR_TUNER_FAILURE = "Tuned models were rejected by the validator (see validation reasons). Explain the most likely cause (tuning overfit CV, selection bias, search instability) and propose fixes or a more conservative alternative."
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
    new_items: list[HypoOut] = Field(default_factory=list)


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
    items = [h.to_item() for h in (out.new_items or out.hypotheses)[:3] if h.change.strip()]
    with session(ws) as L:
        new_ids = add_queue_items(L, items, "analyzer")
        L.analysis.append(Analysis(after_exp=L.runs[-1].exp_id if L.runs else "-", bottleneck=out.bottleneck[:300],
                                   evidence=out.evidence[:600], new_ids=new_ids))
    ui.say("Analyzer", f"bottleneck: {out.bottleneck[:100]}  → +{len(new_ids)} queued ({len(items) - len(new_ids)} duplicates dropped)")
    return new_ids
