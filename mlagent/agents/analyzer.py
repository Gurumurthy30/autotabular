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

SIGNALS = """Compute these from the OOF predictions of run {best} (artifacts/{best}_oof.npy; rows with NaN OOF
are holdout/unscored — ignore them). Skip any that do not apply. Keep the output small (round to 4 dp):
1. per_class_error   (classification) per-class recall and confusion counts
2. residual_patterns (regression) mean/std of residual by target decile; 3 features most correlated with |residual|
3. importance_shift  top-10 feature importances from a quick tree model on ONE fold, vs correlation ranking
4. learning_curve    score of a quick model on 25/50/75/100% of fold-0 training rows
5. calibration       (binary probs) mean predicted vs observed rate in 5 bins
6. error_by_segment  error rate by the top 5 levels of the main categorical column
Finish with emit({{...signals...}})."""

SYSTEM = """You are the analyst of an ML team. Given recent runs, validator verdicts, the queue and the measured
signals (or the failure context), name the single biggest bottleneck, cite the evidence, and propose 1-3 NEW
experiments (ONE change each) that attack it. Do not repeat anything already in the queue.
Fields: bottleneck (string), evidence (string), hypotheses (list of {hypothesis, change, kind: feature|model|tune,
params: object, base: "best" or null})."""


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
