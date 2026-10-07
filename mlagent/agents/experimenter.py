"""Experimenter: ONE hypothesis, ONE change. Coder writes + runs the script on the fixed folds.
Sees only: strategy, the current queue item, docs for the libraries it needs (digest rule)."""

from __future__ import annotations

import math
import traceback
from pathlib import Path

from .. import ui
from ..executor import parse_result, tail
from ..ledger import (RARE_LIBS, Workspace, best_run, compact, digest, latest_ok_run, load,
                      minutes_elapsed, session)
from ..llm import RateLimitError
from ..state import Run
from . import docs
from .coder import write_and_run


PROMPT_TASK_FRESH = """\
Experiment <<exp_id>>.
Hypothesis (the claim being tested): <<hypothesis>>
The ONE change to implement: <<change>>
Hints (may be empty): <<params>>

Write a fresh script following the skeleton and implement exactly this change, nothing more.
Contract:
- Reply with exactly ONE ```python block and nothing else.
- The script begins with a plan docstring (at most 8 lines):
  HYPOTHESIS: ...
  CHANGE: the one change this script makes
  LEAKAGE CHECK: what is fit where (must be inside fit_predict on X_tr only)
  EXPECTED RUNTIME: estimate versus the time limit
  FALLBACKS / DOWNGRADE: anything reduced or subsampled
- Use helpers from `mlkit` (run_cv). Never touch test labels or split data yourself.
- Test predictions must match test data (probabilities for classification, values for regression).
If the change is ambiguous, take the most standard reading and record your interpretation in the plan docstring. If it cannot be done as written, implement the closest sound variant and note it there.\
"""

PROMPT_TASK_FROM_BASE = """\
Experiment <<exp_id>>.
Hypothesis (the claim being tested): <<hypothesis>>
The ONE change to implement: <<change>>
Hints (may be empty): <<params>>

Start from this working script (from <<base_id>>). Apply ONLY the change above, keep everything else identical, and set EXP_ID = "<<exp_id>>".
Contract:
- Reply with exactly ONE ```python block and nothing else.
- Update the plan docstring at the top of the script (HYPOTHESIS, CHANGE, LEAKAGE CHECK, EXPECTED RUNTIME, FALLBACKS).
- Ensure no leakage: transforms and encoders fit on X_tr only inside fit_predict.
- Keep output quiet and run within the time budget.
Script from <<base_id>> to build upon:
```python
<<base_code>>
```\
"""

PROMPT_CHEAT_HEADER = "LIBRARY CHEAT SHEETS (verified against the installed versions; they override your memory of these libraries):"

PROMPT_DOWNGRADE_1 = "DOWNGRADE LEVEL 1 — the previous attempt ran out of memory (GPU or RAM). Keep the same idea but make it lighter: halve batch size / n_estimators / max_bin / hidden sizes, use float32 and category dtypes, drop unneeded columns early, `del` big intermediates, and never build dense one-hot matrices of high-cardinality columns. Note what you reduced in the plan docstring."

PROMPT_DOWNGRADE_2 = "DOWNGRADE LEVEL 2 — the lighter attempt also ran out of memory. Run on CPU only (the GPU is disabled for this attempt: no device='cuda'/'gpu', no tree_method='gpu_hist'), subsample training rows inside fit_predict if needed (max 200k), and switch to a lighter model family for the same idea (e.g. HistGradientBoosting, or LightGBM with max_bin=63, instead of a deep net or a huge ensemble). Note what you changed in the plan docstring."

PROMPTS = {
    "task_fresh": PROMPT_TASK_FRESH,
    "task_from_base": PROMPT_TASK_FROM_BASE,
    "cheat_header": PROMPT_CHEAT_HEADER,
    "downgrade_1": PROMPT_DOWNGRADE_1,
    "downgrade_2": PROMPT_DOWNGRADE_2,
}


def _libs_in(text: str) -> list[str]:
    t = text.lower()
    return [l for l in RARE_LIBS if l in t] + [l for l in ("sklearn", "pandas") if l in t]


def _base_code(L, base: str | None) -> tuple[str | None, str | None]:
    run = best_run(L) if base == "best" else (latest_ok_run(L, base) if base else None)
    if run and run.script_path and Path(run.script_path).exists():
        return run.exp_id, Path(run.script_path).read_text()
    return None, None


def run(ws: Workspace, exp_id: str) -> str | None:
    """Returns None on success, or the error text (crash = Run with error, never a CV attempt)."""
    try:
        return _run(ws, exp_id)
    except RateLimitError:
        raise
    except Exception as e:                                           # noqa: BLE001 — agent bug != run loop death
        err = f"internal error: {type(e).__name__}: {e}"
        (ws.logs / "errors.log").open("a").write(f"[experimenter {exp_id}]\n{traceback.format_exc()}\n")
        with session(ws) as L:
            L.runs.append(Run(exp_id=exp_id, error=err, seed=L.env.seed))
            for q in L.queue:
                if q.id == exp_id:
                    q.status = "failed"
        ui.error(f"{exp_id}: {err}")
        return err


def _run(ws: Workspace, exp_id: str) -> str | None:
    with session(ws) as L:
        item = next(q for q in L.queue if q.id == exp_id)
        item.status = "running"
    L = load(ws)
    ui.say("Experimenter", f"{exp_id} [{item.kind}] {item.change}")

    base_id, base_code = _base_code(L, item.base)
    libs = _libs_in(item.change + item.hypothesis + str(item.params) + (base_code or ""))
    docs.ensure(ws, libs)
    L = load(ws)
    retry = any(r.exp_id == exp_id and r.error for r in L.runs)     # crashed before -> Docs may have written sheets
    sheets = {l: v for l, v in L.docs.items() if retry or l in libs}

    if base_code:
        task = (PROMPTS["task_from_base"]
                .replace("<<exp_id>>", exp_id)
                .replace("<<hypothesis>>", item.hypothesis)
                .replace("<<change>>", item.change)
                .replace("<<params>>", str(item.params))
                .replace("<<base_id>>", base_id or "base")
                .replace("<<base_code>>", base_code))
    else:
        task = (PROMPTS["task_fresh"]
                .replace("<<exp_id>>", exp_id)
                .replace("<<hypothesis>>", item.hypothesis)
                .replace("<<change>>", item.change)
                .replace("<<params>>", str(item.params)))

    if item.params and item.params.get("downgrade") == 1:
        task += "\n\n" + PROMPTS["downgrade_1"]
    elif item.params and item.params.get("downgrade") == 2:
        task += "\n\n" + PROMPTS["downgrade_2"]

    context = compact(digest(L, "experimenter"), 3500)
    if sheets:
        context += f"\n\n{PROMPTS['cheat_header']}\n" + "\n".join(
            f"## {k}\n{v}" for k, v in sheets.items())

    b = L.env.budget
    timeout = int(max(120, min(900, (b.max_minutes - minutes_elapsed(L)) * 60)))
    path, res = write_and_run(ws, exp_id, task, context, "experiment", timeout)

    result = parse_result(res.stdout) if res.ok else None
    err = None
    if not res.ok:
        err = tail(res.stderr or res.stdout, 1800)
    elif not result or not isinstance(result.get("cv_mean"), (int, float)) or not math.isfinite(result["cv_mean"]):
        err = "script finished but printed no valid RESULT_JSON line (did it call run_cv?)\n" + tail(res.stdout, 400)

    run_ = Run(exp_id=exp_id, seconds=round(res.seconds, 1), seed=L.env.seed, script_path=str(path), error=err)
    if not err:
        run_ = run_.model_copy(update={k: result.get(k) for k in
                                       ("cv_mean", "cv_std", "holdout", "oof_path", "test_pred_path", "artifact")})
    with session(ws) as L:
        L.runs.append(run_)
        for q in L.queue:
            if q.id == exp_id:
                q.status = "failed" if err else "done"
    if err:
        ui.warn(f"{exp_id} crashed: {err.strip().splitlines()[-1][:140]}")
    else:
        ho = f"  holdout={run_.holdout:.4f}" if run_.holdout is not None else ""
        ui.say("Experimenter", f"{exp_id} cv={run_.cv_mean:.4f} ± {run_.cv_std:.4f}{ho}  ({run_.seconds:.0f}s)")
    return err


def run_batch(ws: Workspace, batch: list[str], workers: int = 1) -> dict[str, str | None]:
    """Runs a batch of experiments, concurrently if workers > 1."""
    if not batch:
        return {}
    if workers <= 1 or len(batch) <= 1:
        return {exp_id: run(ws, exp_id) for exp_id in batch}
    import concurrent.futures
    results: dict[str, str | None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(batch))) as ex:
        futures = {ex.submit(run, ws, exp_id): exp_id for exp_id in batch}
        for fut in concurrent.futures.as_completed(futures):
            exp_id = futures[fut]
            try:
                results[exp_id] = fut.result()
            except Exception as e:
                results[exp_id] = str(e)
    return results

