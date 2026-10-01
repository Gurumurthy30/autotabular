"""Noise floor and statistical significance calculation for paired CV scores (Phase 3.7)."""

from collections.abc import Sequence

import numpy as np


def noise_floor(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    """Computes the paired standard error of the fold-wise score difference.

    SE_diff = std(diffs, ddof=1) / sqrt(N).
    This measures the true resolution limit of the paired test folds.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return 0.0

    diffs = a - b
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    return se


def is_significant_gain(
    candidate_scores: Sequence[float],
    best_scores: Sequence[float],
    direction: str = "higher",
) -> tuple[bool, float, float]:
    """Determines whether candidate's paired improvement exceeds the noise floor.

    Returns:
        (significant: bool, paired_diff: float, nf: float)
    """
    cand = np.asarray(candidate_scores, dtype=float)
    best = np.asarray(best_scores, dtype=float)
    if len(cand) != len(best) or len(cand) < 2:
        return False, 0.0, 0.0

    nf = noise_floor(cand, best)
    mean_diff = float(np.mean(cand - best))

    if direction == "lower":
        # e.g. rmse: candidate should be lower than best (mean_diff < 0)
        significant = bool((mean_diff < -1.0 * nf) and (mean_diff < 0))
    else:
        # e.g. roc_auc, f1: candidate should be higher than best (mean_diff > 0)
        significant = bool((mean_diff > 1.0 * nf) and (mean_diff > 0))

    return significant, mean_diff, nf
