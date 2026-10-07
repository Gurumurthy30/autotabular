"""Coder: turns an instruction into a script, runs it through the Executor. Used by every agent.

Only the Experimenter/Tuner use it for model artifacts; the Validator uses it only for its own
verification scripts (written to scripts/val_*.py, never touching model artifacts).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import Field

from ..executor import ExecResult, run_script, tail, write_script
from ..ledger import Workspace, load
from ..llm import get_llm
from ..state import Base, QueueItem

KIT_API = """\
KIT API — module `mlkit` is in the working directory. ALWAYS start with `from mlkit import *`.
  Names: pd, np, json, SEED, TARGET, ID, METRIC, TASK, CLASSES, WS
  load_train() -> DataFrame incl. target      load_test() -> DataFrame | None
  get_y() -> numpy target (classification: ints 0..k-1 in CLASSES order; regression: floats)
  folds() -> [{"train": [idx], "valid": [idx]}]     holdout_idx() -> [idx]   (row positions of load_train())
  score(y_true, y_pred) -> float   (task metric; classification expects PROBABILITIES, not labels)
  simple_prep(X_tr, *others) -> numeric DataFrames (median impute + ordinal encode, fit on X_tr only; None passes through)
  predict_any(model, Z) -> binary: P(class 1) 1-D | multiclass: (n,k) probs | regression: values
  run_cv(exp_id, fit_predict, X, y, X_test=None, artifact=None, final=True) -> dict
      fit_predict(X_tr, y_tr, X_va, X_te) -> (va_pred, te_pred_or_None)   # X_te may be None
      final=True : fixed folds + holdout, saves artifacts, prints the RESULT_JSON line.
      final=False: returns {"cv_mean","cv_std"} only (use for tuning objectives).
  emit(dict) -> prints the RESULT_JSON line (end every analysis script with it)
"""

RULES = """\
Contract & Non-Negotiables:
- Reply with exactly ONE ```python block and nothing else (no text outside the block).
- Begin the script with a concise plan docstring (HYPOTHESIS, CHANGE, LEAKAGE CHECK, EXPECTED RUNTIME, FALLBACKS).
- Use the project's helpers (`from mlkit import *`). Never re-implement fold splitting; always score via `run_cv(...)`.
- No leakage: Never use TARGET or ID as features. Never touch test labels. Fit every imputer, encoder, scaler, and target-encoder INSIDE fit_predict on X_tr only.
- Test predictions: Ensure predictions match test set structure (probabilities for classification metrics, values for regression).
- Implement exactly the declared change.
- Engineering habits: Fail loudly (no blanket try/except hiding crashes). Quiet output (disable verbose logs/progress bars). Respect the time limit. Fix randomness with SEED.
- Do not edit mlkit.py or ledger.json.
"""

SKELETON_EXPERIMENT = '''\
"""
HYPOTHESIS: Baseline Random Forest pipeline
CHANGE: Minimal preprocessing + RandomForest
LEAKAGE CHECK: Preprocessing fit on X_tr only
EXPECTED RUNTIME: < 1 minute
FALLBACKS / DOWNGRADE: None
"""
from mlkit import *
from sklearn.ensemble import RandomForestClassifier   # RandomForestRegressor for regression

EXP_ID = "__EXP__"
df = load_train(); y = get_y()
X = df.drop(columns=[c for c in [TARGET, ID] if c and c in df.columns])
test = load_test()
X_test = test[X.columns] if test is not None else None

def fit_predict(X_tr, y_tr, X_va, X_te):
    A, B, C = simple_prep(X_tr, X_va, X_te)
    m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(A, y_tr)
    return predict_any(m, B), (predict_any(m, C) if C is not None else None)

run_cv(EXP_ID, fit_predict, X, y, X_test)
'''

SKELETON_ANALYSIS = '''\
from mlkit import *
df = load_train(); y = get_y()
out = {}
# ... compute small facts into `out` (rounded numbers / short strings) ...
emit(out)
'''

CODER_SYSTEM = f"""\
You are a senior ML engineer writing ONE self-contained Python 3 script evaluated on fixed cross-validation folds. Correctness and honesty come first, then speed, then cleverness.

{KIT_API}

{RULES}\
"""

PROMPTS = {
    "kit_api": KIT_API,
    "rules": RULES,
    "system": CODER_SYSTEM,
    "skeleton_experiment": SKELETON_EXPERIMENT,
    "skeleton_analysis": SKELETON_ANALYSIS,
}


class HypoOut(Base):
    """What an LLM proposes for the queue (lenient); converted to a strict QueueItem in code."""
    hypothesis: str = ""
    change: str = ""
    kind: str = "feature"
    params: dict[str, Any] = Field(default_factory=dict)
    base: str | None = None

    def to_item(self) -> QueueItem:
        kind = self.kind if self.kind in ("feature", "model", "tune") else "feature"
        base = self.base if self.base and (self.base == "best" or re.fullmatch(r"e\d+", self.base)) else None
        return QueueItem(hypothesis=str(self.hypothesis).strip(), change=str(self.change).strip(),
                         kind=kind, params=self.params, base=base)  # type: ignore[arg-type]


def write_and_run(ws: Workspace, name: str, task: str, context: str = "", kind: str = "experiment",
                  timeout: int = 900, attempts: int = 1) -> tuple[Path, ExecResult]:
    """attempts=1: a failure is returned to the caller (the controller routes it to Docs/Analyzer)."""
    from ..executor import parse_result
    llm, L = get_llm(), load(ws)
    skeleton = SKELETON_EXPERIMENT.replace("__EXP__", name) if kind == "experiment" else SKELETON_ANALYSIS
    user = (f"Installed libraries: {L.env.library_versions}\n\nSkeleton to follow:\n```python\n{skeleton}```\n\n"
            f"TASK:\n{task}\n\n{context}\n\nScript file: {name}.py")
    code = llm.code(CODER_SYSTEM, user)
    path = write_script(ws, name, code)
    res = run_script(ws, path, timeout)
    for _ in range(attempts - 1):
        if res.ok:
            break
        code = llm.code(CODER_SYSTEM, f"{user}\n\nYour previous script:\n```python\n{code}\n```\n"
                                       f"failed with:\n{tail(res.stderr)}\nReturn the full corrected script.")
        path = write_script(ws, name, code)
        res = run_script(ws, path, timeout)
    # A script that runs without error but emits no RESULT_JSON: {} line is a logic error.
    if res.ok and kind == "experiment" and parse_result(res.stdout) is None:
        res = ExecResult(ok=False, returncode=res.returncode, stdout=res.stdout,
                         stderr=res.stderr + "\nlogic error: script ran but printed no RESULT_JSON line",
                         seconds=res.seconds)
    return path, res

