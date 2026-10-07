=== task ===
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
```
