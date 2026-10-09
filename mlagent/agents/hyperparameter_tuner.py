"""Hyperparameter Tuner: bounded hyperparameter search (Optuna) for each of the top-N approved runs.
Writes runs tagged source='tuner'. Trials and wall-time are capped from the budget."""

from __future__ import annotations

import math
from pathlib import Path

from .. import hw, ui
from ..executor import parse_result, tail
from ..ledger import Workspace, approved_runs, load, minutes_elapsed, session
from ..state import Run
from . import api_docs_lookup as docs
from .script_writer import write_and_run

MAX_SCRIPT_CHARS = 24_000

PROMPT_TASK = """\
<script experiment="{base}" cv_mean="{cv:.4f}" cv_std="{cv_std:.4f}" runtime_seconds="{base_seconds:.0f}">
{code}
</script>

<study>
New experiment id: {eid}. Direction: {direction}. Search budget: at most {trials} trials within {secs} seconds. Search at most {max_params} parameters. Hardware: {hardware}.
These numbers were computed from the base's runtime. Do not recompute them.
</study>

Create the TUNED version of the script above: the same pipeline, with ONLY its hyperparameters searched by Optuna. Tuning refines a model that already works. Its main risk is overfitting the CV folds by trying many settings, so search narrowly and cheaply, and change nothing else.

Plan
- Pick the {max_params} (or fewer) most sensitive parameters for the model family in the script:
  * gradient-boosted trees: learning_rate, num_leaves or max_depth, min_child_samples (or min_child_weight), subsample, colsample, reg_alpha, reg_lambda
  * linear and TF-IDF models: C or alpha, n-gram upper bound, min_df, max_features
  * random forests: max_features, min_samples_leaf, max_depth
  * fine-tuned networks: learning_rate, weight_decay, warmup ratio, dropout or label smoothing, augmentation strength, layer-wise learning-rate decay; epochs only by 1-2 and only if time allows
  * anything else: the parameters that govern regularisation and learning speed
- A gain smaller than cv_std cannot be detected. The larger cv_std is relative to the gain you can plausibly find, the narrower the ranges should be.
- Never tune: seeds, thread counts, fold definitions, features, preprocessing, model family, loss. If the script uses early stopping, keep it and do not tune n_estimators or the epoch cap. For deep models, never raise anything that increases memory or time (batch size, max length, image size, model size) beyond the script's value.

Ranges
- Centre every range on the value the script uses now. If the script relies on library defaults, write those defaults out explicitly (use the cheat sheet if given, else the documented default).
- Log scale for rates and strengths: learning rate about 2-3x each side, regularisation strengths about 10x each side. Integers about 40% each side. Clip to valid domains.

Script mechanics (all are required)
1. Keep the base's `fit_predict` body identical, but turn it into a factory: `make_fit_predict(params)` returns `fit_predict(X_tr, y_tr, X_va, X_te)` and reads the tuned values from `{{**CURRENT, **params}}`, so unsearched values stay as in the base. `CURRENT = {{...}}` holds the base's value for every searched parameter.
2. `SPACE = {{name: (low, high, log_flag)}}` at the top, right after the imports. The keys of SPACE and CURRENT are identical. Every CURRENT value lies inside its range, integers stay ints, and a categorical value is among its choices. Use only `trial.suggest_float(..., log=...)`, `suggest_int` and `suggest_categorical`.
3. `objective(trial)` builds params and returns `run_cv("{eid}", make_fit_predict(params), X, y, final=False)["cv_mean"]`. Never call final=True inside the objective. fit_predict must work with X_te=None.
4. `import optuna`; `optuna.logging.set_verbosity(optuna.logging.WARNING)`. `study = optuna.create_study(direction="{direction}", sampler=optuna.samplers.TPESampler(seed=SEED))`. No pruner (run_cv returns only the mean).
5. First `study.enqueue_trial(CURRENT)`, so the base is re-measured under exactly the same protocol as every other trial. Then `study.optimize(objective, n_trials={trials}, timeout={secs}, gc_after_trial=True, catch=(RuntimeError, MemoryError))`, so one oversized trial is recorded as failed instead of ending the study. A running trial is not interrupted, so keep every trial short.
6. After the study: `if not any(t.state.name == "COMPLETE" for t in study.trials): raise RuntimeError("tuning: no trial completed")`. Never hide a total failure.
7. Print ONE line: `TUNING trials=<number of COMPLETE trials> baseline=<value of the first trial> best=<study.best_value> edge=<names of searched numeric parameters whose best value lies within 5% of a range bound (in log space for log parameters), or none>`. All numbers come from the study object.
8. Then exactly ONE module-level call: `run_cv("{eid}", make_fit_predict(study.best_params), X, y, X_test, final=True)`. It saves the artifacts and prints the RESULT_JSON line. Set EXP_ID = "{eid}".
9. Plan docstring, at most 8 lines: HYPOTHESIS (tuning these parameters inside narrow ranges lifts cv_mean by more than cv_std), CHANGE (one `SEARCH name low-high [log]` entry per parameter, in one or two lines), LEAKAGE CHECK (unchanged from the base; the search uses the same fixed folds), EXPECTED RUNTIME (trials x runtime_seconds versus the timeout, plus the final refit), FALLBACKS / DOWNGRADE.

Check silently before you reply: only SPACE parameters differ from the base; run_cv(final=True) appears once, after the study; CURRENT and SPACE keys match; nothing else in the script changed; no progress bars or large prints.
Reply with exactly ONE ```python block (the full script) and nothing else.\
"""

PROMPTS = {
    "task": PROMPT_TASK,
}


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
    max_tune_s = b.tune_minutes_per_model * 60
    per_model = int(max(60, min(max_tune_s, left_s / (len(cands) + 1))))
    ui.say("Hyperparameter Tuner", f"tuning {len(cands)} model(s): {b.tune_trials} trials / {per_model}s each")
    new_ids: list[str] = []
    hardware_text = hw.describe()
    for r in cands:
        eid = f"t_{r.exp_id}"
        already_tuned = any(x.exp_id == eid and x.error is None for x in load(ws).runs)
        if already_tuned:
            continue                       # resume case: this run was tuned before
        L = load(ws)

        # How many trials fit? One trial costs about one full CV run of the base script.
        base_seconds = max(r.seconds or 30.0, 1.0)
        refit_reserve = base_seconds + 30          # time for the final run_cv(final=True) after the search
        search_seconds = int(max(30, per_model - refit_reserve))
        trials = int(min(b.tune_trials, search_seconds // base_seconds))
        if trials < 3:
            ui.say("Tuner", f"{r.exp_id}: one run takes {base_seconds:.0f}s, fewer than 3 trials fit; skipping")
            continue
        max_params = 6 if trials >= 8 else 3       # few trials -> search fewer parameters (less noise-fitting)

        # Never cut a script in the middle: a half script makes the model tune broken code.
        code = Path(r.script_path).read_text(encoding="utf-8")
        if len(code) > MAX_SCRIPT_CHARS:
            ui.warn(f"{r.exp_id}: script longer than {MAX_SCRIPT_CHARS} chars; skipping tuning")
            continue

        task = PROMPT_TASK.format(
            base=r.exp_id, cv=r.cv_mean, cv_std=r.cv_std or 0.0, base_seconds=base_seconds,
            direction=d, trials=trials, secs=search_seconds, max_params=max_params,
            hardware=hardware_text, eid=eid, code=code,
        )
        ctx = ("OPTUNA CHEAT SHEET:\n" + L.docs["optuna"]) if "optuna" in L.docs else ""
        timeout_seconds = int(search_seconds + 2 * base_seconds + 240)   # search + refit + safety margin
        path, res = write_and_run(ws, eid, task, ctx, "experiment", timeout=timeout_seconds)
        out = parse_result(res.stdout) if res.ok else None
        err = None
        if not res.ok:
            err = tail(res.stderr or res.stdout, 1500)
        elif not out or not isinstance(out.get("cv_mean"), (int, float)) or not math.isfinite(out["cv_mean"]):
            err = "tuned script printed no valid RESULT_JSON"
        run_ = Run(exp_id=eid, source="tuner", seconds=round(res.seconds, 1), seed=L.env.seed,  # persisted in ledger.json: do not rename
                   script_path=str(path), error=err)
        if not err:
            run_ = run_.model_copy(update={k: out.get(k) for k in
                                           ("cv_mean", "cv_std", "holdout", "oof_path", "test_pred_path", "artifact", "fold_scores", "train_score")})
        with session(ws) as L:
            L.runs.append(run_)
        if err:
            ui.warn(f"{eid} failed: {err.strip().splitlines()[-1][:120]}")
        else:
            ui.say("Hyperparameter Tuner", f"{eid} cv={run_.cv_mean:.4f} (base {r.exp_id}: {r.cv_mean:.4f})")
            new_ids.append(eid)
    return new_ids
