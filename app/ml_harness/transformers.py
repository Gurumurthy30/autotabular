"""Top-level picklable feature engineering transformers."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from app.ml_harness.base import SafeTransformer


class DropColumns(SafeTransformer):
    """Drops specified columns safely."""

    def __init__(self, columns: Sequence[str] | None = None, **kwargs: Any):
        super().__init__()
        cols = columns or kwargs.get("cols") or kwargs.get("drop_cols") or kwargs.get("drop") or kwargs.get("columns_to_drop")
        self.columns = list(cols) if cols else []

    def fit(self, X: Any, y: Any = None) -> "DropColumns":
        super().fit(X, y)
        if isinstance(X, pd.DataFrame):
            self.remaining_cols_ = [c for c in X.columns if c not in self.columns]
        else:
            self.remaining_cols_ = []
        return self

    def transform(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            cols_to_drop = [c for c in self.columns if c in X.columns]
            return X.drop(columns=cols_to_drop)
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.array([c for c in input_features if c not in self.columns], dtype=object)
        return np.array(self.remaining_cols_, dtype=object)


class DateParts(SafeTransformer):
    """Extracts date parts from datetime columns (month, day, dayofweek, dayofyear, is_weekend)."""

    def __init__(self, columns: Sequence[str] | None = None, **kwargs: Any):
        super().__init__()
        cols = columns or kwargs.get("cols") or kwargs.get("date_cols") or kwargs.get("datetime_cols") or kwargs.get("features")
        self.columns = list(cols) if cols else []
        self.out_features_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> "DateParts":
        super().fit(X, y)
        self.out_features_ = []
        target_cols = self.columns or (list(X.columns) if isinstance(X, pd.DataFrame) else [])
        for col in target_cols:
            self.out_features_.extend([
                f"{col}_month",
                f"{col}_day",
                f"{col}_dayofweek",
                f"{col}_dayofyear",
                f"{col}_is_weekend",
            ])
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        target_cols = self.columns or list(df.columns)
        new_cols = {}
        for col in target_cols:
            if col in df.columns:
                dt_series = pd.to_datetime(df[col], errors="coerce")
                new_cols[f"{col}_month"] = dt_series.dt.month.fillna(0).astype(int)
                new_cols[f"{col}_day"] = dt_series.dt.day.fillna(0).astype(int)
                new_cols[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
                new_cols[f"{col}_dayofyear"] = dt_series.dt.dayofyear.fillna(0).astype(int)
                new_cols[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)

        out_df = pd.DataFrame(new_cols, index=df.index)
        return out_df

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.out_features_, dtype=object)


class CyclicEncoder(SafeTransformer):
    """Encodes periodic features into sin/cos coordinates."""

    def __init__(self, columns: Sequence[str] | None = None, period: float = 24.0, **kwargs: Any):
        super().__init__()
        cols = columns or kwargs.get("cols") or kwargs.get("features")
        self.columns = list(cols) if cols else []
        p = kwargs.get("period", kwargs.get("periods", kwargs.get("cycle", period)))
        self.period = float(p)
        self.out_features_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> "CyclicEncoder":
        super().fit(X, y)
        self.out_features_ = []
        target_cols = self.columns or (list(X.columns) if isinstance(X, pd.DataFrame) else [])
        for col in target_cols:
            self.out_features_.append(f"{col}_sin")
            self.out_features_.append(f"{col}_cos")
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        target_cols = self.columns or list(df.columns)
        new_cols = {}
        for col in target_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                radians = 2 * np.pi * vals / self.period
                new_cols[f"{col}_sin"] = np.sin(radians)
                new_cols[f"{col}_cos"] = np.cos(radians)

        return pd.DataFrame(new_cols, index=df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.out_features_, dtype=object)


class PairOps(SafeTransformer):
    """Computes difference, ratio, sum, or product between pairs of columns with safe division."""

    def __init__(
        self,
        pairs: list[tuple[str, str]] | None = None,
        ops: Sequence[str] = ("diff", "ratio"),
        **kwargs: Any,
    ):
        super().__init__()
        p = pairs or kwargs.get("col_pairs") or kwargs.get("pairs")
        self.pairs = list(p) if p else []
        op_list = kwargs.get("ops", kwargs.get("operations", kwargs.get("op", ops)))
        self.ops = list(op_list) if isinstance(op_list, (list, tuple)) else [str(op_list)]
        self.out_features_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> "PairOps":
        super().fit(X, y)
        self.out_features_ = []
        for c1, c2 in self.pairs:
            for op in self.ops:
                self.out_features_.append(f"{c1}_{op}_{c2}")
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        new_cols = {}
        for c1, c2 in self.pairs:
            if c1 in df.columns and c2 in df.columns:
                s1 = pd.to_numeric(df[c1], errors="coerce").fillna(0)
                s2 = pd.to_numeric(df[c2], errors="coerce").fillna(0)
                if "diff" in self.ops:
                    new_cols[f"{c1}_diff_{c2}"] = s1 - s2
                if "sum" in self.ops:
                    new_cols[f"{c1}_sum_{c2}"] = s1 + s2
                if "prod" in self.ops:
                    new_cols[f"{c1}_prod_{c2}"] = s1 * s2
                if "ratio" in self.ops:
                    eps = 1e-6
                    sign_s2 = np.where(s2 >= 0, 1.0, -1.0)
                    safe_s2 = np.where(np.abs(s2) < eps, eps * sign_s2, s2)
                    new_cols[f"{c1}_ratio_{c2}"] = s1 / safe_s2

        return pd.DataFrame(new_cols, index=df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.out_features_, dtype=object)


class LogPower(SafeTransformer):
    """Applies log1p or power transformation on specified skewed columns."""

    def __init__(self, columns: Sequence[str] | None = None, method: str = "log1p", **kwargs: Any):
        super().__init__()
        cols = columns or kwargs.get("cols") or kwargs.get("features")
        self.columns = list(cols) if cols else []
        self.method = str(kwargs.get("method", kwargs.get("func", kwargs.get("mode", method))))

    def fit(self, X: Any, y: Any = None) -> "LogPower":
        super().fit(X, y)
        target_cols = self.columns or (list(X.columns) if isinstance(X, pd.DataFrame) else [])
        self.out_features_ = [f"{c}_{self.method}" for c in target_cols]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        target_cols = self.columns or list(df.columns)
        new_cols = {}
        for col in target_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                min_val = vals.min()
                if min_val < 0:
                    vals = vals - min_val
                new_cols[f"{col}_{self.method}"] = np.log1p(vals)

        return pd.DataFrame(new_cols, index=df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.out_features_, dtype=object)


class ClipQuantiles(SafeTransformer):
    """Clips continuous column values to lower and upper quantiles fitted on training data."""

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        lower: float = 0.01,
        upper: float = 0.99,
        **kwargs: Any,
    ):
        super().__init__()
        cols = columns or kwargs.get("cols") or kwargs.get("features") or kwargs.get("column_names")
        self.columns = list(cols) if cols else []
        low = kwargs.get(
            "lower_quantile",
            kwargs.get("lower_percentile", kwargs.get("q_low", kwargs.get("min_quantile", lower)))
        )
        high = kwargs.get(
            "upper_quantile",
            kwargs.get("upper_percentile", kwargs.get("q_high", kwargs.get("max_quantile", upper)))
        )
        low_f = float(low)
        high_f = float(high)
        if low_f > 1.0:
            low_f /= 100.0
        if high_f > 1.0:
            high_f /= 100.0
        self.lower = low_f
        self.upper = high_f
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X: Any, y: Any = None) -> "ClipQuantiles":
        super().fit(X, y)
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        target_cols = self.columns or list(df.columns)
        self.bounds_ = {}
        for col in target_cols:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) > 0:
                    q_low = float(s.quantile(self.lower))
                    q_high = float(s.quantile(self.upper))
                    self.bounds_[col] = (q_low, q_high)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, (q_low, q_high) in self.bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=q_low, upper=q_high)
        return df

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return self.feature_names_in_


class FrequencyEncoder(SafeTransformer):
    """Encodes categorical columns by frequency of categories learned in fit."""

    def __init__(self, columns: Sequence[str] | None = None, **kwargs: Any):
        super().__init__()
        cols = columns or kwargs.get("cols") or kwargs.get("cat_cols") or kwargs.get("features")
        self.columns = list(cols) if cols else []
        self.freq_maps_: dict[str, dict[Any, float]] = {}

    def fit(self, X: Any, y: Any = None) -> "FrequencyEncoder":
        super().fit(X, y)
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        target_cols = self.columns or list(df.columns)
        self.freq_maps_ = {}
        for col in target_cols:
            if col in df.columns:
                vc = df[col].value_counts(normalize=True).to_dict()
                self.freq_maps_[col] = vc
        self.out_features_ = [f"{c}_freq" for c in self.freq_maps_]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        new_cols = {}
        for col, freq_map in self.freq_maps_.items():
            if col in df.columns:
                new_cols[f"{col}_freq"] = df[col].map(freq_map).fillna(0.0)
        return pd.DataFrame(new_cols, index=df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.out_features_, dtype=object)


class TargetEncoderCV(SafeTransformer):
    """Smoothed target encoding fit inside folds to prevent data leakage."""

    def __init__(self, columns: Sequence[str] | None = None, smoothing: float = 10.0, **kwargs: Any):
        super().__init__()
        cols = columns or kwargs.get("cols") or kwargs.get("cat_cols") or kwargs.get("features")
        self.columns = list(cols) if cols else []
        s = kwargs.get("smoothing", kwargs.get("smooth", kwargs.get("alpha", smoothing)))
        self.smoothing = float(s)
        self.target_maps_: dict[str, dict[Any, float]] = {}
        self.global_mean_ = 0.0

    def fit(self, X: Any, y: Any = None) -> "TargetEncoderCV":
        super().fit(X, y)
        if y is None:
            return self
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_ser = pd.Series(y, index=df.index).astype(float)
        self.global_mean_ = float(y_ser.mean())
        target_cols = self.columns or list(df.columns)
        self.target_maps_ = {}

        for col in target_cols:
            if col in df.columns:
                grouped = y_ser.groupby(df[col])
                counts = grouped.count()
                means = grouped.mean()
                smooth = (counts * means + self.smoothing * self.global_mean_) / (counts + self.smoothing)
                self.target_maps_[col] = smooth.to_dict()

        self.out_features_ = [f"{c}_te" for c in self.target_maps_]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        new_cols = {}
        for col, t_map in self.target_maps_.items():
            if col in df.columns:
                new_cols[f"{col}_te"] = df[col].map(t_map).fillna(self.global_mean_)
        return pd.DataFrame(new_cols, index=df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.out_features_, dtype=object)


class RollingLag(SafeTransformer):
    """Computes lags and rolling statistics ONLY for time-split datasets."""

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        lags: Sequence[int] = (1, 2),
        roll_windows: Sequence[int] = (3,),
        **kwargs: Any,
    ):
        super().__init__()
        cols = columns or kwargs.get("cols") or kwargs.get("features")
        self.columns = list(cols) if cols else []
        lag_val = kwargs.get("lags", kwargs.get("lag", lags))
        self.lags = list(lag_val) if isinstance(lag_val, (list, tuple, range)) else [int(lag_val)]
        win_val = kwargs.get("roll_windows", kwargs.get("rolling_windows", kwargs.get("windows", roll_windows)))
        self.roll_windows = list(win_val) if isinstance(win_val, (list, tuple, range)) else [int(win_val)]
        self.out_features_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> "RollingLag":
        super().fit(X, y)
        self.out_features_ = []
        target_cols = self.columns or (list(X.columns) if isinstance(X, pd.DataFrame) else [])
        for col in target_cols:
            for lag in self.lags:
                self.out_features_.append(f"{col}_lag{lag}")
            for w in self.roll_windows:
                self.out_features_.append(f"{col}_roll_mean_{w}")
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        new_cols = {}
        target_cols = self.columns or list(df.columns)
        for col in target_cols:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                for lag in self.lags:
                    new_cols[f"{col}_lag{lag}"] = s.shift(lag).bfill().fillna(0)
                for w in self.roll_windows:
                    new_cols[f"{col}_roll_mean_{w}"] = s.rolling(window=w, min_periods=1).mean().bfill().fillna(0)
        return pd.DataFrame(new_cols, index=df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.out_features_, dtype=object)
