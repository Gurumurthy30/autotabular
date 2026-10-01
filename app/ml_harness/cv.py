"""Leak-free Cross Validation Engine (Phase 3.6).

Fits the entire feature pipeline + model inside each fold. Caches fold indices
to guarantee fair, paired comparisons across model candidates.
"""

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline

_FOLD_CACHE: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

METRIC_DIRECTIONS: dict[str, str] = {
    "roc_auc": "higher",
    "auc": "higher",
    "f1": "higher",
    "accuracy": "higher",
    "r2": "higher",
    "precision": "higher",
    "recall": "higher",
    "balanced_accuracy": "higher",
    "rmse": "lower",
    "mae": "lower",
    "logloss": "lower",
    "log_loss": "lower",
    "mse": "lower",
}


def get_metric_direction(metric: str) -> str:
    """Returns 'higher' or 'lower' indicating whether larger score is better."""
    return METRIC_DIRECTIONS.get(metric.lower(), "higher")


def get_fold_cache_key(n_rows: int, y: Any, strategy: str, seed: int, n_splits: int, n_repeats: int) -> str:
    """Computes a deterministic cache key for cross-validation fold splits."""
    y_arr = np.asarray(y)
    y_bytes = y_arr.tobytes() if hasattr(y_arr, "tobytes") else str(y_arr).encode("utf-8")
    y_hash = hashlib.sha256(y_bytes).hexdigest()[:12]
    return f"{n_rows}_{y_hash}_{strategy}_{seed}_{n_splits}_{n_repeats}"


def get_cached_splits(
    strategy: str,
    n_splits: int,
    n_repeats: int,
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    groups: Any = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns cached or freshly computed fold splits."""
    n_rows = len(X)
    key = get_fold_cache_key(n_rows, y, strategy, seed, n_splits, n_repeats)
    if key in _FOLD_CACHE:
        return _FOLD_CACHE[key]

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    if strategy == "stratified":
        if n_repeats > 1:
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X, y))
    elif strategy == "time":
        cv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(cv.split(X, y))
    elif strategy == "group" and groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups=groups))
    else:
        # Default KFold
        if n_repeats > 1:
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X, y))

    _FOLD_CACHE[key] = splits
    return splits


def compute_metric(
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
    metric: str,
    task_type: str = "binary_classification",
) -> float:
    """Computes target metric handling probability, continuous, or discrete predictions."""
    m = metric.lower()
    y_t = np.asarray(y_true)

    if m in ("roc_auc", "auc"):
        if y_prob is not None:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                return float(roc_auc_score(y_t, y_prob[:, 1]))
            elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                return float(roc_auc_score(y_t, y_prob, multi_class="ovr", average="macro"))
            return float(roc_auc_score(y_t, y_prob))
        return float(roc_auc_score(y_t, y_pred))

    if m in ("logloss", "log_loss"):
        if y_prob is not None:
            return float(log_loss(y_t, y_prob))
        return float(log_loss(y_t, y_pred))

    if m == "f1":
        if "multiclass" in task_type:
            return float(f1_score(y_t, y_pred, average="macro", zero_division=0))
        return float(f1_score(y_t, y_pred, average="binary", zero_division=0))

    if m == "accuracy":
        return float(accuracy_score(y_t, y_pred))

    if m == "r2":
        return float(r2_score(y_t, y_pred))

    if m == "rmse":
        return float(root_mean_squared_error(y_t, y_pred))

    if m == "mae":
        return float(mean_absolute_error(y_t, y_pred))

    return float(accuracy_score(y_t, y_pred))


def cv_evaluate(
    pipeline_factory: Callable[[], Any],
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "binary_classification",
    metric: str = "roc_auc",
    split_strategy: str = "stratified",
    seed: int = 42,
    n_splits: int = 5,
    n_repeats: int | str = "auto",
    groups: Any = None,
) -> dict[str, Any]:
    """Fits the entire pipeline (features + model) inside each CV fold.

    Returns:
        {
            "fold_scores": list[float],
            "cv_mean": float,
            "cv_std": float,
            "train_score": float,
            "train_val_gap": float,
            "oof_predictions": np.ndarray,
        }
    """
    n_rows = len(X)
    if n_repeats == "auto":
        repeats = 2 if n_rows < 1000 and split_strategy not in ("time", "group") else 1
    else:
        repeats = int(n_repeats)

    if "regression" in task_type and split_strategy == "stratified":
        split_strategy = "kfold"

    splits = get_cached_splits(split_strategy, n_splits, repeats, seed, X, y, groups=groups)

    fold_scores = []
    train_scores = []
    is_classification = "classification" in task_type
    oof_predictions = np.zeros(n_rows)
    oof_probabilities = np.zeros(n_rows) if is_classification else None

    for train_idx, val_idx in splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Build fresh pipeline for this fold
        fe_pipeline = pipeline_factory()
        est = clone(estimator)
        full_pipeline = Pipeline([
            ("features", fe_pipeline),
            ("model", est),
        ])

        full_pipeline.fit(X_train, y_train)

        # Predict
        val_pred = full_pipeline.predict(X_val)
        val_prob = None
        if hasattr(full_pipeline, "predict_proba") and is_classification:
            try:
                val_prob = full_pipeline.predict_proba(X_val)
            except Exception:
                pass

        fold_score = compute_metric(y_val, val_pred, val_prob, metric, task_type=task_type)
        fold_scores.append(fold_score)

        # Train score for gap calculation
        train_pred = full_pipeline.predict(X_train)
        train_prob = None
        if hasattr(full_pipeline, "predict_proba") and is_classification:
            try:
                train_prob = full_pipeline.predict_proba(X_train)
            except Exception:
                pass
        t_score = compute_metric(y_train, train_pred, train_prob, metric, task_type=task_type)
        train_scores.append(t_score)

        # Store OOF
        oof_predictions[val_idx] = val_pred
        if oof_probabilities is not None and val_prob is not None:
            if val_prob.ndim == 2 and val_prob.shape[1] >= 2:
                oof_probabilities[val_idx] = val_prob[:, 1]
            else:
                oof_probabilities[val_idx] = val_prob.ravel()

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    mean_train = float(np.mean(train_scores))
    gap = float(abs(mean_train - cv_mean))

    return {
        "fold_scores": fold_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "train_score": mean_train,
        "train_val_gap": gap,
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities,
    }
