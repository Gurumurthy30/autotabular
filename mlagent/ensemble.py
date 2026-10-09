"""Ensembling on OOF predictions, aligned through the fixed folds.

Methods (simplest first, which is also the tie-break order):
  single    best approved run
  greedy    Caruana-style greedy weighted average (selection with replacement)
  voting    top-k members; HARD majority vote when the metric is label-based (accuracy/F1), else soft mean
  stacking  LogisticRegression / Ridge meta-model on member OOFs, scored with an honest CV over the
            same fixed folds (the meta-model never sees the rows it predicts)

Everything is scored on the SAME rows (the intersection where every member and the stacker have an
OOF value), so time-series folds (early rows have no OOF) compare fairly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

ORDER = ["single", "greedy", "voting", "stacking"]


@dataclass
class Result:
    name: str
    members: list[str]
    weights: dict[str, float]
    oof: np.ndarray                                   # full length, NaN where not scored
    predict: Callable[[dict[str, np.ndarray]], np.ndarray]   # {member: array} -> blended array (OOF or test)
    score: float = float("nan")


def finite_rows(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a).all(axis=tuple(range(1, a.ndim))) if a.ndim > 1 else np.isfinite(a)


def common_mask(arrs: dict[str, np.ndarray]) -> np.ndarray:
    m = None
    for a in arrs.values():
        f = finite_rows(a)
        m = f if m is None else (m & f)
    return m


def _better(a: float, b: float, direction: str, eps: float = 1e-9) -> bool:
    return a > b + eps if direction == "maximize" else a < b - eps


def _best_id(scores: dict[str, float], direction: str) -> str:
    return (max if direction == "maximize" else min)(scores, key=scores.get)


def _single(oofs, y, kit, d, mask) -> Result:
    scores = {k: kit.score(y[mask], o[mask]) for k, o in oofs.items()}
    k = _best_id(scores, d)
    return Result("single", [k], {k: 1.0}, oofs[k], lambda arrs, k=k: arrs[k])


def _greedy(oofs, y, kit, d, mask, rounds: int = 10) -> Result:
    sign, counts, cur, best = (1 if d == "maximize" else -1), {}, None, None
    for r in range(rounds):
        cand = None
        for n, o in oofs.items():
            trial = o if cur is None else (cur * r + o) / (r + 1)
            sc = kit.score(y[mask], trial[mask])
            if cand is None or sign * sc > sign * cand[0]:
                cand = (sc, n, trial)
        if best is not None and sign * cand[0] <= sign * best:
            break
        best, cur = cand[0], cand[2]
        counts[cand[1]] = counts.get(cand[1], 0) + 1
    tot = sum(counts.values())
    w = {k: v / tot for k, v in counts.items()}
    predict = lambda arrs, w=w: sum(wt * arrs[k] for k, wt in w.items())
    return Result("greedy", list(w), w, predict(oofs), predict)


def _voting(oofs, y, kit, d, mask, top_k: int) -> Result:
    scores = {k: kit.score(y[mask], o[mask]) for k, o in oofs.items()}
    ranked = sorted(scores, key=scores.get, reverse=(d == "maximize"))[:max(2, top_k)]
    hard = kit.TASK != "regression" and kit._mk(kit.METRIC) in kit._LABEL_METRICS

    def predict(arrs, ids=ranked):
        stack = [arrs[i] for i in ids]
        if not hard:
            return np.mean(stack, axis=0)
        ok = np.logical_and.reduce([finite_rows(a) for a in stack])
        if stack[0].ndim == 1:                                  # binary: share of members voting class 1
            out = np.mean([(a >= 0.5).astype(float) for a in stack], axis=0)
        else:
            k = stack[0].shape[1]
            out = np.mean([np.eye(k)[np.nan_to_num(a).argmax(1)] for a in stack], axis=0)
        out[~ok] = np.nan
        return out

    return Result("voting", ranked, {i: round(1 / len(ranked), 4) for i in ranked}, predict(oofs), predict)


def _stacking(oofs, y, kit, d, mask, folds) -> Result | None:
    ids, n = list(oofs), len(y)
    clf, k = kit.TASK != "regression", len(kit.CLASSES)
    feats = lambda arrs: np.hstack([arrs[i] if arrs[i].ndim == 2 else arrs[i][:, None] for i in ids])
    mk = (lambda: LogisticRegression(C=1.0, max_iter=1000)) if clf else (lambda: Ridge(alpha=1.0))

    def pred(model, Z):
        if not clf:
            return model.predict(Z)
        p = model.predict_proba(Z)
        if kit.TASK == "binary":
            return p[:, 1]
        full = np.zeros((len(Z), k))
        full[:, model.classes_] = p
        return full

    X, idx = feats(oofs), np.where(mask)[0]
    shape = (n, k) if clf and kit.TASK != "binary" else (n,)
    stack_oof = np.full(shape, np.nan)
    for f in folds:
        tr, va = np.intersect1d(f["train"], idx), np.intersect1d(f["valid"], idx)
        if len(tr) < 30 or len(va) == 0 or (clf and len(np.unique(y[tr])) < 2):
            continue
        stack_oof[va] = pred(mk().fit(X[tr], y[tr]), X[va])
    if not finite_rows(stack_oof).any() or (clf and len(np.unique(y[idx])) < 2):
        return None
    final = mk().fit(X[idx], y[idx])
    return Result("stacking", ids, {}, stack_oof, lambda arrs: pred(final, feats(arrs)))


def evaluate(oofs: dict[str, np.ndarray], y, kit, direction: str, folds: list[dict], methods: list[str],
             top_k: int = 5) -> tuple[list[Result], np.ndarray]:
    """Build every enabled method. Returns (results, common_mask_used_for_scoring)."""
    mask = common_mask(oofs)
    results = [_single(oofs, y, kit, direction, mask)]
    if len(oofs) >= 2:
        for name, fn in (("greedy", lambda: _greedy(oofs, y, kit, direction, mask)),
                         ("voting", lambda: _voting(oofs, y, kit, direction, mask, top_k)),
                         ("stacking", lambda: _stacking(oofs, y, kit, direction, mask, folds))):
            if name not in methods:
                continue
            try:
                r = fn()
            except Exception:                                      # noqa: BLE001 — a failing method is skipped
                r = None
            if r is not None:
                results.append(r)
    common = mask.copy()
    for r in results:
        common &= finite_rows(r.oof)
    return results, common


def choose(results: list[Result], y, kit, direction: str, common: np.ndarray) -> tuple[Result, dict[str, float]]:
    """Score all methods on the same rows; a more complex method must beat the simpler best."""
    for r in results:
        r.score = float(kit.score(y[common], r.oof[common]))
    best = results[0]
    for r in sorted(results[1:], key=lambda r: ORDER.index(r.name)):
        if _better(r.score, best.score, direction):
            best = r
    return best, {r.name: round(r.score, 5) for r in results}
