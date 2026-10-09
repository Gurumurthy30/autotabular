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
  Names: pd, np, json, SEED, N_JOBS, TARGET, ID, METRIC, TASK, CLASSES, WS
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
Notes:
  - run_cv gets the FULL X, y, X_test; the folds decide who trains. fit_predict receives row subsets of X.
  - X may hold text or file-path columns. For text, image or audio inputs, fit_predict builds its own datasets from those rows.
  - simple_prep is for tabular numeric/categorical columns only. Do not pass text or file-path columns through it.
  - Paths of extra data files come from the Profile's data_files. Never guess a path.
"""

RULES = """\
<contract>
Reply with exactly ONE ```python block and nothing else: no text outside it, no extra tags around it.
The task states its kind.
- EXPERIMENT: the script starts with a plan docstring of at most 8 lines (HYPOTHESIS, CHANGE, LEAKAGE CHECK, EXPECTED RUNTIME, FALLBACKS / DOWNGRADE). Write the plan first, then follow it. The script ends with exactly one run_cv(EXP_ID, fit_predict, X, y, X_test) call.
- ANALYSIS: a short docstring; the script answers the questions by computing, trains nothing, and ends with emit(...) in exactly the shape the task states.
</contract>

<non_negotiables>
1. Scores come only from run_cv. Never compute, print, hard-code or estimate a headline score yourself. Call run_cv once, at the end, with final=True. (Published agent runs were derailed by a metric that silently returned a perfect value.)
2. Stay inside the folds. run_cv receives the FULL X and y and the folds decide who trains. Therefore:
   - Never subsample, filter, sort or drop rows of X, y or X_test before run_cv: OOF must cover every training row and the predictions every test row. Subsample or drop TRAINING rows inside fit_predict, on X_tr and y_tr together, so they stay aligned.
   - Stateless row-by-row transforms may run before run_cv (parse a date, string length, ratio of two columns of the same row, resize an image).
   - Anything that learns from other rows runs inside fit_predict and is fitted on X_tr only: imputers, scalers, encoders (target, frequency, count), group aggregates, TF-IDF and vocabularies, PCA or SVD, feature selection and importance pruning, class weights, pseudo-labels, thresholds, calibration.
   - Never use TARGET or ID as features. Never touch test labels.
   - Early stopping: carve the monitoring split out of X_tr. If you must monitor on X_va, say so on the LEAKAGE CHECK line.
3. Prediction form. fit_predict returns (va_pred, te_pred_or_None). Classification: probabilities in CLASSES order (binary: P(class 1) as a 1-D array; multiclass: shape (n, k)); never labels. Regression: values on the ORIGINAL target scale, so invert any log or other transform inside fit_predict. te_pred has one row per test row, in order.
4. Implement exactly the declared change. With a base script, copy it and change only what the item says, so any score difference is attributable.
5. Use all the data the Profile marks as relevant (image or audio folders, text files, side tables), not only the main CSV. Take paths from data_files.
6. Play by the task's rules: only the provided data and pretrained weights already available. Never download datasets, labels or competition solutions. Do not pip install. Do not edit mlkit.py or ledger.json; write files only under WS.
</non_negotiables>

<engineering_habits>
- Fail loudly. No blanket try/except around training or run_cv, no sys.exit() or exit(), no silent fallback that changes the experiment. An honest crash gets routed and fixed; a hidden one produces a wrong score. (Analysis scripts are different: see modes.)
- Quiet output. Disable progress bars and library logs (for example verbosity -1 or 0, verbose=0, silent warnings, quiet transformers and datasets logging). Never print model architectures or arrays. At most one short line per fold.
- Respect the clock. Estimate runtime before writing (rows x epochs x model cost) against the time limit in the task. For slow models, time the first fold and stop with a clear RuntimeError when the projected total exceeds the limit. Prefer a declared subsample, smaller input or fewer epochs over a run that times out.
- Hardware. Use the GPU when it exists and the library supports it; mixed precision for deep nets; release models and call gc.collect() and torch.cuda.empty_cache() between folds. Respect the CPU and GPU share stated in the task and do not grab every core when experiments run in parallel.
- Determinism. Fix all randomness with SEED (use SEED + i for the i-th seed when averaging seeds).
- Caching. Cache expensive label-free steps (resized images, tokenisation, frozen-model embeddings) under WS, keyed by their settings. Never cache anything fitted on labels or on the training rows.
- Pretrained weights may be unavailable offline. If loading fails, raise an error containing "pretrained weights unavailable: <name>" so it can be routed. Do not silently swap models.
- Prefer standard calls you are sure of. If a cheat sheet is provided for a library, follow it over your memory.
- Keep the script focused: normally under 200 lines.
</engineering_habits>

<modes>
NORMAL: write the script for the queue item.
RETRY (a previous script and its error are given): find the root cause in the traceback. If it is data-related (shape, dtype, missing file, column), re-read the Profile and data_files before guessing. Change the minimum needed, keep the same hypothesis and the rest of the script, and output the full corrected script.
DOWNGRADE (params.downgrade = 1 after an out-of-memory error): same idea, lighter: smaller batch with gradient accumulation, mixed precision, smaller image size or max length, a shallower or smaller model, a training subsample inside fit_predict.
DOWNGRADE (params.downgrade = 2): CPU only. Replace heavy parts with a lighter equivalent (precomputed embeddings plus a linear model or a small GBDT, a smaller model, a subsample).
Every downgrade changes what the run means, so state it on the FALLBACKS / DOWNGRADE line.
TUNING (the task says it is a tuning study): the objective calls run_cv(..., final=False) per trial; keep the search space exactly as given; after the search refit the best parameters with one final run_cv(..., final=True).
ANALYSIS: answer exactly the questions asked, by computing; never guess. Wrap each answer in its own try/except and store "error: <short message>" for a failure, so one failure does not lose the others. Sample large inputs (at most about 300 files and 50k rows for heavy operations) and say "sample" in the answer. Give short facts with numbers and column names.
</modes>
"""

SKELETON_EXPERIMENT = '''\
"""
HYPOTHESIS: <from the queue item>
CHANGE: <the one change; with a base script, only this differs>
LEAKAGE CHECK: all fitting happens inside fit_predict on X_tr only
EXPECTED RUNTIME: <minutes versus the limit>
FALLBACKS / DOWNGRADE: <none, or what was reduced>
"""
# TEMPLATE for tabular data: keep the structure, choose the model that the queue item asks for.
from mlkit import *
from sklearn.ensemble import RandomForestClassifier   # replace; RandomForestRegressor etc. for regression

EXP_ID = "__EXP__"
df = load_train(); y = get_y()
X = df.drop(columns=[c for c in [TARGET, ID] if c and c in df.columns])
test = load_test()
X_test = test[X.columns] if test is not None else None

def fit_predict(X_tr, y_tr, X_va, X_te):
    A, B, C = simple_prep(X_tr, X_va, X_te)          # replace or extend: all fitted steps stay in here
    m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(A, y_tr)
    return predict_any(m, B), (predict_any(m, C) if C is not None else None)

run_cv(EXP_ID, fit_predict, X, y, X_test)
'''

SKELETON_DEEP = '''\
"""
HYPOTHESIS: <from the queue item>
CHANGE: <the one change; with a base script, only this differs>
LEAKAGE CHECK: all fitting inside fit_predict on X_tr; frozen pretrained weights only; early stopping split from X_tr
EXPECTED RUNTIME: <minutes per fold x folds, versus the limit>
FALLBACKS / DOWNGRADE: <subsample, input size, epochs; none if none>
"""
# TEMPLATE for text / image / audio. Fill the three marked functions; keep the structure.
from mlkit import *
import gc, time, warnings
import torch
warnings.filterwarnings("ignore")

EXP_ID = "__EXP__"
LIMIT_S = <<limit_s>>                      # <-- set from the time limit given in the task
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
df = load_train(); y = get_y()
X = df.drop(columns=[c for c in [TARGET, ID] if c and c in df.columns])   # keeps the text or path columns
test = load_test()
X_test = test[X.columns] if test is not None else None
K = 1 if TASK == "regression" else len(CLASSES)
T0, DONE = time.time(), [0]

def make_dataset(rows, labels=None):
    # <-- build a Dataset from `rows` (text or file paths from the Profile's data_files); labels is None for test
    raise NotImplementedError

def build_model():
    # <-- pretrained backbone + head with K outputs; raise "pretrained weights unavailable: <name>" if it cannot load
    raise NotImplementedError

def train_and_predict(model, train_ds, pred_sets):
    # <-- AdamW + scheduler, mixed precision, no per-step prints; early stopping on a split carved from train_ds.
    # <-- return one array per set in pred_sets: binary P(class 1) 1-D (sigmoid), multiclass (n, K) softmax, regression values
    raise NotImplementedError

def fit_predict(X_tr, y_tr, X_va, X_te):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = build_model().to(DEVICE)
    sets = [make_dataset(X_va)] + ([make_dataset(X_te)] if X_te is not None else [])
    outs = train_and_predict(model, make_dataset(X_tr, y_tr), sets)
    DONE[0] += 1
    if (time.time() - T0) / DONE[0] * len(folds()) > LIMIT_S:
        raise RuntimeError("projected runtime exceeds the limit: reduce input size, epochs or subsample")
    del model; gc.collect(); torch.cuda.empty_cache()
    return outs[0], (outs[1] if X_te is not None else None)

run_cv(EXP_ID, fit_predict, X, y, X_test)
'''

SKELETON_ANALYSIS = '''\
"""Answers the questions below by computing on the data."""
from mlkit import *
df = load_train(); y = get_y()
out = {}

def safe(key, fn):
    try:
        out[key] = fn()
    except Exception as e:
        out[key] = f"error: {type(e).__name__}: {e}"[:200]

# safe("<short question key>", lambda: round(float(df[col].mean()), 4))
emit(out)                                   # use exactly the emit shape the task states
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
    modality = (L.profile.modality or []) if L.profile else []
    is_deep = any(m in ("text", "image", "audio") for m in modality)
    if kind == "experiment":
        if is_deep:
            skeleton = SKELETON_DEEP.replace("__EXP__", name).replace("<<limit_s>>", str(timeout))
        else:
            skeleton = SKELETON_EXPERIMENT.replace("__EXP__", name)
    else:
        skeleton = SKELETON_ANALYSIS
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

