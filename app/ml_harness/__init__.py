"""Machine Learning Execution Harness.

Ensures strict correctness, leak-free cross-validation, picklable custom transformers,
and standardized evaluation across models.
"""

from app.ml_harness.base import SafeTransformer
from app.ml_harness.cv import cv_evaluate, get_metric_direction
from app.ml_harness.libs import get_available_libs
from app.ml_harness.noise import is_significant_gain, noise_floor
from app.ml_harness.paths import RunPaths
from app.ml_harness.preprocess import make_basic_preprocessor
from app.ml_harness.transformers import (
    ClipQuantiles,
    CyclicEncoder,
    DateParts,
    DropColumns,
    FrequencyEncoder,
    LogPower,
    PairOps,
    RollingLag,
    TargetEncoderCV,
)

__all__ = [
    "ClipQuantiles",
    "CyclicEncoder",
    "DateParts",
    "DropColumns",
    "FrequencyEncoder",
    "LogPower",
    "PairOps",
    "RollingLag",
    "RunPaths",
    "SafeTransformer",
    "TargetEncoderCV",
    "cv_evaluate",
    "get_available_libs",
    "get_metric_direction",
    "is_significant_gain",
    "make_basic_preprocessor",
    "noise_floor",
]
