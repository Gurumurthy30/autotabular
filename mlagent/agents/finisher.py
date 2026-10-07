"""Finisher: ensemble (greedy weighted average on OOF, aligned via the fixed folds), submission.csv,
format checks, report. Deterministic; the LLM only writes an optional summary paragraph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import ensemble, ui
from ..ledger import Workspace, approved_runs, best_run, load, ok_runs, run_rows, session, verdicts
from ..llm import RateLimitError, get_llm
from ..state import FinalResult


def _mask(oofs: dict[str, np.ndarray]) -> np.ndarray:
    m = None
    for o in oofs.values():
        ok = np.isfinite(o).all(axis=tuple(range(1, o.ndim))) if o.ndim > 1 else np.isfinite(o)
        m = ok if m is None else (m & ok)
    return m


def greedy_blend(oofs: dict[str, np.ndarray], y, kit, direction: str, rounds: int = 10) -> dict[str, float]:
    sign, m = (1 if direction == "maximize" else -1), _mask(oofs)
    counts: dict[str, int] = {}
    cur, best_sc = None, None
    for r in range(rounds):
        cand = None
        for n, o in oofs.items():
            trial = o if cur is None else (cur * r + o) / (r + 1)
            sc = kit.score(y[m], trial[m])
            if cand is None or sign * sc > sign * cand[0]:
                cand = (sc, n, trial)
        if best_sc is not None and sign * cand[0] <= sign * best_sc:
            break
        best_sc, cur = cand[0], cand[2]
        counts[cand[1]] = counts.get(cand[1], 0) + 1
    tot = sum(counts.values())
    return {k: v / tot for k, v in counts.items()}


def _checks(ws: Workspace, L, sub: pd.DataFrame) -> dict[str, bool]:
    p = L.problem
    test = pd.read_csv(p.test_path)
    sample = pd.read_csv(p.sample_submission_path) if p.sample_submission_path and Path(p.sample_submission_path).exists() else None
    idc = sample.columns[0] if sample is not None else p.id_column
    grp = lambda k: "n" if k in "iuf" else k
    checks = {"row_count": len(sub) == len(test), "no_nans": not bool(sub.isna().any().any())}
    if sample is not None:
        checks["columns"] = list(sub.columns) == list(sample.columns)
        checks["dtypes"] = checks["columns"] and all(grp(sub[c].dtype.kind) == grp(sample[c].dtype.kind) for c in sample.columns)
    if idc and idc in sub.columns and idc in test.columns:
        checks["id_alignment"] = bool((sub[idc].to_numpy() == test[idc].to_numpy()).all())
    return checks


def _report(ws: Workspace, L, fin: FinalResult, summary: str) -> None:
    b, v = best_run(L), verdicts(L)
    lines = [f"# mlagent report — {L.problem.goal}", "",
             f"- metric: **{L.problem.metric}** ({L.problem.direction})   · stop reason: **{L.stop_reason}**",
             f"- best single run: **{b.exp_id}** cv={b.cv_mean:.4f} ± {(b.cv_std or 0):.4f}" if b else "- no approved runs",
             f"- ensemble: {fin.ensemble}", f"- submission: {fin.submission_path}", f"- checks: {fin.checks}", ""]
    if summary:
        lines += ["## Summary", summary, ""]
    lines += ["## Runs", "| exp | src | cv | std | holdout | verdict |", "|---|---|---|---|---|---|"]
    q = {x.id: x.change for x in L.queue}
    for r in L.runs:
        f = lambda x: "" if x is None else f"{x:.4f}"
        lines.append(f"| {r.exp_id} | {r.source} | {f(r.cv_mean)} | {f(r.cv_std)} | {f(r.holdout)} | {v.get(r.exp_id, '')} "
                     f"{'ERR' if r.error else ''} |")
    lines += ["", "## What was tried"] + [f"- **{k}**: {c}" for k, c in q.items()]
    lines += ["", "## Analyzer findings"] + [f"- after {a.after_exp}: {a.bottleneck} — {a.evidence}" for a in L.analysis]
    if L.strategy:
        lines += ["", "## Risks noted"] + [f"- {x}" for x in L.strategy.risks]
        lines += ["", "## Recalled domain approaches (UNVERIFIED)"] + [f"- {x}" for x in L.strategy.domain_notes]
    ws.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(ws: Workspace) -> None:
    L = load(ws)
    ui.say("Finisher", "building ensemble and submission")
    fin = FinalResult(report_path=str(ws.report_path))
    cands = approved_runs(L) or []
    if not cands:                                           # best effort: validator unavailable or all rejected
        cands = [r for r in ok_runs(L)]
        if cands:
            ui.warn("no approved runs; falling back to unvalidated runs")
    cands = [r for r in cands if r.oof_path and Path(r.oof_path).exists()]
    if cands:
        kit, d = ws.load_mlkit(), L.problem.direction
        y = kit.get_y()
        oofs = {r.exp_id: np.load(r.oof_path) for r in cands}
        cfg = ws.config()
        results, common = ensemble.evaluate(
            oofs, y, kit, d, kit.folds(), cfg.finisher.methods, cfg.finisher.voting_top_k
        )
        choice, table = ensemble.choose(results, y, kit, d, common)
        fin.ensemble = {
            "method": choice.name,
            "members": choice.members,
            "weights": {k: round(w, 3) for k, w in choice.weights.items()},
            "cv": round(float(choice.score), 5),
            "candidates": table,
            "best_single_cv_same_rows": round(float(table["single"]), 5),
        }
        ui.say("Finisher", f"{choice.name}: {fin.ensemble['weights']}  cv={choice.score:.4f}")
        raws = {k: ws.artifacts / f"{k}_test_raw.npy" for k in choice.members}
        if L.problem.test_path and all(p.exists() for p in raws.values()):
            test_arrs = {k: np.load(raws[k]) for k in choice.members}
            raw = choice.predict(test_arrs)
            sub = kit.to_submission(raw)
            sub.to_csv(ws.submission_path, index=False)
            fin.submission_path = str(ws.submission_path)
            fin.checks = _checks(ws, L, sub)
            (ui.ok if all(fin.checks.values()) else ui.warn)(f"submission checks: {fin.checks}")
        else:
            ui.warn("no test predictions available; submission.csv not written")
    else:
        ui.warn("no usable runs; writing a report only")

    summary = ""
    try:
        summary = get_llm().chat("You write a 5-sentence executive summary of an ML run for a data scientist. "
                                 "Use only the facts given.", f"stop={L.stop_reason} final={fin.model_dump()} runs={run_rows(L)}")
    except RateLimitError:
        pass
    except Exception:                                       # noqa: BLE001
        pass
    with session(ws) as L:
        L.final = fin
        if L.status == "running":
            L.status = "done"
        _report(ws, L, fin, summary.strip())
    ui.ok(f"report: {ws.report_path}")
