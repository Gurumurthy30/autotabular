"""mlkit — copied into every workspace as mlkit.py. Generated scripts do `from mlkit import *`.

It makes comparisons valid (same folds, same metric, same artifact layout) and keeps
LLM-written scripts short: the script only defines features + a model, run_cv does the rest.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(os.environ.get("MLKIT_WS") or Path(__file__).resolve().parent)
def _read_ledger():
    for i in range(40):                      # Windows: the orchestrator may be replacing the file right now
        try:
            return json.loads((WS / "ledger.json").read_text(encoding="utf-8"))
        except (PermissionError, FileNotFoundError, json.JSONDecodeError):
            if i == 39:
                raise
            time.sleep(0.05)


_L = _read_ledger()
PROBLEM = _L["problem"]
SEED = _L["env"]["seed"]
N_JOBS = int(os.environ.get("MLAGENT_N_JOBS", "-1"))   # threads per worker (parallel runs share the CPU)
TARGET, ID, METRIC, TASK = PROBLEM["target"], PROBLEM.get("id_column"), PROBLEM["metric"], PROBLEM["task_type"]

_LABEL_METRICS = {"accuracy", "acc", "f1", "f1_macro", "f1_weighted", "f1_binary"}


def _mk(m):
    return str(m).lower().replace(" ", "_").replace("-", "_")


def _classes():
    if TASK == "regression":
        return []
    u = pd.read_csv(PROBLEM["train_path"], usecols=[TARGET])[TARGET].dropna().unique().tolist()
    try:
        return sorted(u)
    except TypeError:
        return sorted(u, key=str)


CLASSES = _classes()


def load_train():
    return pd.read_csv(PROBLEM["train_path"])


def load_test():
    p = PROBLEM.get("test_path")
    return pd.read_csv(p) if p else None


def get_y():
    s = load_train()[TARGET]
    if TASK == "regression":
        return pd.to_numeric(s, errors="coerce").to_numpy(float)
    return s.map({c: i for i, c in enumerate(CLASSES)}).fillna(-1).astype(int).to_numpy()


def folds():
    return json.loads((WS / "folds.json").read_text(encoding="utf-8"))["folds"]


def holdout_idx():
    p = WS / "holdout.json"
    return json.loads(p.read_text(encoding="utf-8"))["indices"] if p.exists() else []


# ---- metric ---------------------------------------------------------------

def _lab(p):
    p = np.asarray(p)
    if p.ndim == 2:
        return p.argmax(1)
    return (p >= 0.5).astype(int) if TASK == "binary" else p.astype(int)


def score(y_true, y_pred, metric=None):
    from sklearn import metrics as M
    m, yt, yp = _mk(metric or METRIC), np.asarray(y_true), np.asarray(y_pred, dtype=float)
    if m in ("accuracy", "acc"):
        return float(M.accuracy_score(yt, _lab(yp)))
    if m in ("f1", "f1_binary"):
        return float(M.f1_score(yt, _lab(yp), average="binary" if TASK == "binary" else "macro"))
    if m in ("f1_macro", "f1_weighted"):
        return float(M.f1_score(yt, _lab(yp), average=m.split("_")[1]))
    if m in ("roc_auc", "auc"):
        return float(M.roc_auc_score(yt, yp, multi_class="ovr") if yp.ndim == 2 else M.roc_auc_score(yt, yp))
    if m in ("logloss", "log_loss"):
        return float(M.log_loss(yt, np.clip(yp, 1e-15, 1 - 1e-15), labels=list(range(max(2, len(CLASSES))))))
    if m == "rmse":
        return float(np.sqrt(M.mean_squared_error(yt, yp)))
    if m == "mse":
        return float(M.mean_squared_error(yt, yp))
    if m == "mae":
        return float(M.mean_absolute_error(yt, yp))
    if m == "r2":
        return float(M.r2_score(yt, yp))
    if m == "rmsle":
        return float(np.sqrt(M.mean_squared_log_error(yt, np.clip(yp, 0, None))))
    raise ValueError(f"unsupported metric: {m}")


# ---- helpers for scripts -----------------------------------------------------

def simple_prep(X_tr, *others):
    """Median-impute numerics, ordinal-encode the rest. Fitted on X_tr ONLY. None passes through."""
    cat = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    num = [c for c in X_tr.columns if c not in cat]
    med = X_tr[num].median()
    maps = {c: {v: i for i, v in enumerate(sorted(X_tr[c].dropna().astype(str).unique()))} for c in cat}

    def tf(Z):
        if Z is None:
            return None
        Z = Z.copy()
        if num:
            Z[num] = Z[num].fillna(med).astype(float)
        for c in cat:
            Z[c] = Z[c].astype(str).map(maps[c]).fillna(-1).astype(int)
        return Z

    return (tf(X_tr), *[tf(o) for o in others])


def predict_any(model, Z):
    if hasattr(model, "predict_proba") and TASK != "regression":
        p = model.predict_proba(Z)
        return p[:, 1] if TASK == "binary" else p
    return model.predict(Z)


def emit(d):
    def _j(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    print("RESULT_JSON: " + json.dumps(d, default=_j))


def run_cv(exp_id, fit_predict, X, y, X_test=None, artifact=None, final=True):
    """Fixed folds + holdout. fit_predict(X_tr, y_tr, X_va, X_te) -> (va_pred, te_pred | None)."""
    t0, fl, y = time.time(), folds(), np.asarray(y)
    oof, scores, tes = None, [], []
    for f in fl:
        tr, va = np.asarray(f["train"]), np.asarray(f["valid"])
        pv, pt = fit_predict(X.iloc[tr], y[tr], X.iloc[va], X_test if final else None)
        pv = np.asarray(pv, dtype=float)
        if oof is None:
            oof = np.full((len(X),) + pv.shape[1:], np.nan)
        oof[va] = pv
        scores.append(score(y[va], pv))
        if pt is not None:
            tes.append(np.asarray(pt, dtype=float))
    res = {"cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores)),
           "fold_scores": [round(s, 5) for s in scores]}
    if not final:
        return res
    hidx, ho = np.asarray(holdout_idx(), dtype=int), None
    if len(hidx):
        pool = np.setdiff1d(np.arange(len(X)), hidx)
        ph, _ = fit_predict(X.iloc[pool], y[pool], X.iloc[hidx], None)
        ho = score(y[hidx], np.asarray(ph, dtype=float))
    art = WS / "artifacts"
    art.mkdir(exist_ok=True)
    np.save(art / f"{exp_id}_oof.npy", oof)
    res.update(holdout=ho, oof_path=str(art / f"{exp_id}_oof.npy"), artifact=artifact,
               seconds=round(time.time() - t0, 1))
    if tes:
        raw = np.mean(tes, axis=0)
        np.save(art / f"{exp_id}_test_raw.npy", raw)
        to_submission(raw).to_csv(art / f"{exp_id}_test.csv", index=False)
        res["test_pred_path"] = str(art / f"{exp_id}_test.csv")
    emit(res)
    return res


def to_submission(raw):
    """raw = probabilities / regression values for the test rows -> DataFrame in submission format."""
    raw, t = np.asarray(raw, dtype=float), load_test()
    sp = PROBLEM.get("sample_submission_path")
    sample = pd.read_csv(sp) if sp and Path(sp).exists() else None
    idcol, tcols = (sample.columns[0], list(sample.columns[1:])) if sample is not None else (ID, [TARGET])
    out = pd.DataFrame()
    if idcol:
        if t is not None and idcol in t.columns:
            out[idcol] = t[idcol].to_numpy()
        else:
            # ID column configured but absent from test — use a sequential index so the
            # submission has the right number of rows and Kaggle's row-count check passes.
            out[idcol] = np.arange(len(raw))
    if TASK == "regression":
        out[tcols[0]] = raw
    elif _mk(METRIC) in _LABEL_METRICS:
        out[tcols[0]] = np.asarray(CLASSES)[_lab(raw)]
    elif raw.ndim == 1:
        out[tcols[0]] = raw
    else:
        names = tcols if len(tcols) == raw.shape[1] else [str(c) for c in CLASSES]
        for i, c in enumerate(names):
            out[c] = raw[:, i]
    return out



__all__ = ["pd", "np", "json", "WS", "SEED", "N_JOBS", "TARGET", "ID", "METRIC", "TASK", "CLASSES", "PROBLEM",
           "load_train", "load_test", "get_y", "folds", "holdout_idx", "score", "simple_prep",
           "predict_any", "run_cv", "emit", "to_submission"]
