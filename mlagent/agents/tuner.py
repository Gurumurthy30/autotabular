"""Tuner: bounded hyperparameter search (Optuna) for each of the top-N approved runs.
Writes runs tagged source='tuner'. Trials and wall-time are capped from the budget."""

from __future__ import annotations

import math
from pathlib import Path

from .. import ui
from ..executor import parse_result, tail
from ..ledger import Workspace, approved_runs, load, minutes_elapsed, session
from ..state import Run
from . import docs
from .coder import write_and_run

PROMPT_TASK = """\
Create a TUNED version of the experiment script below (exp <<base>>, cv=<<cv>>).
Keep the features and the model family identical; only search hyperparameters with Optuna:
- optuna.create_study(direction="<<direction>>", sampler=optuna.samplers.TPESampler(seed=SEED))
- study.optimize(objective, n_trials=<<trials>>, timeout=<<secs>>)
- objective(trial): build params from trial.suggest_*, return run_cv("<<eid>>", fit_predict_with(params), X, y, final=False)["cv_mean"]
  (final=False is fast and saves nothing; make fit_predict take the params and still guard X_te=None.)
- After the search: refit once with study.best_params and call run_cv("<<eid>>", ..., X, y, X_test, final=True).
Search-space guidance: learning rates log-uniform (0.01-0.3); tree depth/leaves and min-child/min-samples ranges that match the data size; always include a regularisation knob (L1/L2, subsample, colsample, min_child_weight); keep n_estimators moderate with early stopping carved from X_tr. Do not widen the space so far that trials exceed the time limit.
Set EXP_ID = "<<eid>>". Script to tune:
```python
<<code>>
```\
"""

PROMPTS = {
    "task": PROMPT_TASK,
}

TASK = """Create a TUNED version of the experiment script below (exp {base}, cv={cv:.4f}).
Keep features and model family identical; only search hyperparameters with Optuna:
- optuna.create_study(direction="{direction}", sampler=optuna.samplers.TPESampler(seed=SEED))
- study.optimize(objective, n_trials={trials}, timeout={secs})
- objective(trial): build params from trial.suggest_*, return run_cv("{eid}", fit_predict_with(params), X, y, final=False)["cv_mean"]
  (final=False is fast and saves nothing; make fit_predict take the params).
- After the search: refit once with study.best_params and call run_cv("{eid}", ..., X, y, X_test, final=True).
Set EXP_ID = "{eid}". Script to tune:
```python
{code}
```"""


def run(ws: Workspace) -> list[str]:
    L = load(ws)
    b, d = L.env.budget, L.problem.direction
    cands = sorted(approved_runs(L, "experimenter"), key=lambda r: r.cv_mean, reverse=(d == "maximize"))
    cands = [r for r in cands if r.script_path and Path(r.script_path).exists()][:b.tune_top_n]
    if not cands:
        ui.say("Tuner", "no approved runs to tune")
        return []
    docs.ensure(ws, ["optuna"])
    left_s = max(0.0, (b.max_minutes - minutes_elapsed(L)) * 60)
    per_model = int(max(60, min(900, left_s / (len(cands) + 1))))
    ui.say("Tuner", f"tuning {len(cands)} model(s): {b.tune_trials} trials / {per_model}s each")
    new_ids: list[str] = []
    for r in cands:
        eid = f"t_{r.exp_id}"
        if any(x.exp_id == eid and x.error is None for x in load(ws).runs):
            continue
        L = load(ws)
        task = TASK.format(base=r.exp_id, cv=r.cv_mean, direction=d, trials=b.tune_trials, secs=per_model,
                           eid=eid, code=Path(r.script_path).read_text()[:7000])
        ctx = ("OPTUNA CHEAT SHEET:\n" + L.docs["optuna"]) if "optuna" in L.docs else ""
        path, res = write_and_run(ws, eid, task, ctx, "experiment", timeout=per_model + 240)
        out = parse_result(res.stdout) if res.ok else None
        err = None
        if not res.ok:
            err = tail(res.stderr or res.stdout, 1500)
        elif not out or not isinstance(out.get("cv_mean"), (int, float)) or not math.isfinite(out["cv_mean"]):
            err = "tuned script printed no valid RESULT_JSON"
        run_ = Run(exp_id=eid, source="tuner", seconds=round(res.seconds, 1), seed=L.env.seed,
                   script_path=str(path), error=err)
        if not err:
            run_ = run_.model_copy(update={k: out.get(k) for k in
                                           ("cv_mean", "cv_std", "holdout", "oof_path", "test_pred_path", "artifact")})
        with session(ws) as L:
            L.runs.append(run_)
        if err:
            ui.warn(f"{eid} failed: {err.strip().splitlines()[-1][:120]}")
        else:
            ui.say("Tuner", f"{eid} cv={run_.cv_mean:.4f} (base {r.exp_id}: {r.cv_mean:.4f})")
            new_ids.append(eid)
    return new_ids
