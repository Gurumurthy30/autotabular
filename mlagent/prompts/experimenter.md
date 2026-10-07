=== task_fresh ===
Experiment <<exp_id>>.
Hypothesis (the claim being tested): <<hypothesis>>
The ONE change to implement: <<change>>
Hints (may be empty): <<params>>
Write a fresh script following the skeleton and implement exactly this change, nothing more. If the change is ambiguous, take the most standard reading and record your interpretation in the `# DECISIONS:` comment. If it cannot be done as written (a column does not exist, a library is missing), implement the closest sound variant and say so there.
=== task_from_base ===
Experiment <<exp_id>>.
Hypothesis (the claim being tested): <<hypothesis>>
The ONE change to implement: <<change>>
Hints (may be empty): <<params>>
Start from this working script (from <<base_id>>). Apply ONLY the change above, keep everything else identical, and set EXP_ID = "<<exp_id>>". Record any interpretation or deviation in the `# DECISIONS:` comment at the top. If the change cannot be done as written, implement the closest sound variant and say so there.
```python
<<base_code>>
```
=== cheat_header ===
LIBRARY CHEAT SHEETS (verified against the installed versions; they override your memory of these libraries):
=== downgrade_1 ===
DOWNGRADE LEVEL 1 — the previous attempt ran out of memory (GPU or RAM). Keep the same idea but make it lighter: halve batch size / n_estimators / max_bin / hidden sizes, use float32 and category dtypes, drop unneeded columns early, `del` big intermediates, and never build dense one-hot matrices of high-cardinality columns. Note what you reduced in `# DECISIONS:`.
=== downgrade_2 ===
DOWNGRADE LEVEL 2 — the lighter attempt also ran out of memory. Run on CPU only (the GPU is disabled for this attempt: no device='cuda'/'gpu', no tree_method='gpu_hist'), subsample training rows inside fit_predict if needed (max 200k), and switch to a lighter model family for the same idea (e.g. HistGradientBoosting, or LightGBM with max_bin=63, instead of a deep net or a huge ensemble). Note what you changed in `# DECISIONS:`.
