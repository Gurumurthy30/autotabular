"""Top-level picklable feature engineering transformers."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from app.ml_harness.base import SafeTransformer


class DropColumns(SafeTransformer):
    """Drops specified columns safely."""

    def __init__(self, columns: Sequence[str] | None = None, cols: Sequence[str] | None = None):
        super().__init__()
        self.columns = columns
        self.cols = cols

    def fit(self, X: Any, y: Any = None) -> "DropColumns":
        super().fit(X, y)
        c = self.columns if self.columns is not None else self.cols
        drop_list = list(c) if c is not None else []
        if isinstance(X, pd.DataFrame):
            self.remaining_cols_ = [col for col in X.columns if col not in drop_list]
        else:
            self.remaining_cols_ = []
        return self

    def transform(self, X: Any) -> Any:
        c = self.columns if self.columns is not None else self.cols
        drop_list = list(c) if c is not None else []
        if isinstance(X, pd.DataFrame):
            cols_to_drop = [col for col in drop_list if col in X.columns]
            return X.drop(columns=cols_to_drop)
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        c = self.columns if self.columns is not None else self.cols
        drop_list = list(c) if c is not None else []
        if input_features is not None:
            return np.array([col for col in input_features if col not in drop_list], dtype=object)
        return np.array(self.remaining_cols_, dtype=object)


class DateParts(SafeTransformer):
    """Extracts date parts from datetime columns (month, day, dayofweek, dayofyear, is_weekend)."""

    def __init__(self, columns: Sequence[str] | None = None, cols: Sequence[str] | None = None):
        super().__init__()
        self.columns = columns
        self.cols = cols

    def fit(self, X: Any, y: Any = None) -> "DateParts":
        super().fit(X, y)
        self.out_features_ = []
        c = self.columns if self.columns is not None else self.cols
        target_cols = list(c) if c is not None else (list(X.columns) if isinstance(X, pd.DataFrame) else [])
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
        c = self.columns if self.columns is not None else self.cols
        target_cols = list(c) if c is not None else list(df.columns)
        new_cols = {}
        for col in target_cols:
            if col in df.columns:
                dt_series = pd.to_datetime(df[col], errors="coerce")
                new_cols[f"{col}_month"] = dt_series.dt.month.fillna(0).astype(int)
                new_cols[f"{col}_day"] = dt_series.dt.day.fillna(0).astype(int)
                new_cols[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
                new_cols[f"{col}_dayofyear"] = dt_series.dt.dayofyear.fillna(0).astype(int)
                new_cols[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)

        return pd.DataFrame(new_cols, index=df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.out_features_, dtype=object)


class CyclicEncoder(SafeTransformer):
    """Encodes periodic features into sin/cos coordinates."""

    def __init__(self, columns: Sequence[str] | None = None, period: float = 24.0, cols: Sequence[str] | None = None):
        super().__init__()
        self.columns = columns
        self.period = period
        self.cols = cols

    def fit(self, X: Any, y: Any = None) -> "CyclicEncoder":
        super().fit(X, y)
        self.out_features_ = []
        c = self.columns if self.columns is not None else self.cols
        target_cols = list(c) if c is not None else (list(X.columns) if isinstance(X, pd.DataFrame) else [])
        for col in target_cols:
            self.out_features_.append(f"{col}_sin")
            self.out_features_.append(f"{col}_cos")
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        c = self.columns if self.columns is not None else self.cols
        target_cols = list(c) if c is not None else list(df.columns)
        p = float(self.period) if self.period is not None else 24.0
        new_cols = {}
        for col in target_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                radians = 2 * np.pi * vals / p
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
        col_pairs: list[tuple[str, str]] | None = None,
    ):
        super().__init__()
        self.pairs = pairs
        self.ops = ops
        self.col_pairs = col_pairs

    def fit(self, X: Any, y: Any = None) -> "PairOps":
        super().fit(X, y)
        self.out_features_ = []
        p_eff = self.pairs if self.pairs is not None else self.col_pairs
        p_list = list(p_eff) if p_eff is not None else []
        op_list = list(self.ops) if self.ops is not None else ["diff", "ratio"]
        for c1, c2 in p_list:
            for op in op_list:
                self.out_features_.append(f"{c1}_{op}_{c2}")
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        p_eff = self.pairs if self.pairs is not None else self.col_pairs
        p_list = list(p_eff) if p_eff is not None else []
        op_list = list(self.ops) if self.ops is not None else ["diff", "ratio"]
        new_cols = {}
        for c1, c2 in p_list:
            if c1 in df.columns and c2 in df.columns:
                s1 = pd.to_numeric(df[c1], errors="coerce").fillna(0)
                s2 = pd.to_numeric(df[c2], errors="coerce").fillna(0)
                if "diff" in op_list:
                    new_cols[f"{c1}_diff_{c2}"] = s1 - s2
                if "sum" in op_list:
                    new_cols[f"{c1}_sum_{c2}"] = s1 + s2
                if "prod" in op_list:
                    new_cols[f"{c1}_prod_{c2}"] = s1 * s2
                if "ratio" in op_list:
                    eps = 1e-6
                    sign_s2 = np.where(s2 >= 0, 1.0, -1.0)
                    safe_s2 = np.where(np.abs(s2) < eps, eps * sign_s2, s2)
                    new_cols[f"{c1}_ratio_{c2}"] = s1 / safe_s2

        return pd.DataFrame(new_cols, index=df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.out_features_, dtype=object)


class LogPower(SafeTransformer):
    """Applies log1p or power transformation on specified skewed columns."""

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        method: str = "log1p",
        cols: Sequence[str] | None = None,
    ):
        super().__init__()
        self.columns = columns
        self.method = method
        self.cols = cols

    def fit(self, X: Any, y: Any = None) -> "LogPower":
        super().fit(X, y)
        c = self.columns if self.columns is not None else self.cols
        target_cols = list(c) if c is not None else (list(X.columns) if isinstance(X, pd.DataFrame) else [])
        m = str(self.method) if self.method is not None else "log1p"
        self.out_features_ = [f"{col}_{m}" for col in target_cols]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        c = self.columns if self.columns is not None else self.cols
        target_cols = list(c) if c is not None else list(df.columns)
        m = str(self.method) if self.method is not None else "log1p"
        new_cols = {}
        for col in target_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                min_val = vals.min()
                if min_val < 0:
                    vals = vals - min_val
                new_cols[f"{col}_{m}"] = np.log1p(vals)

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
        cols: Sequence[str] | None = None,
        lower_quantile: float | None = None,
        upper_quantile: float | None = None,
        lower_percentile: float | None = None,
        upper_percentile: float | None = None,
    ):
        super().__init__()
        self.columns = columns
        self.lower = lower
        self.upper = upper
        self.cols = cols
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X: Any, y: Any = None) -> "ClipQuantiles":
        super().fit(X, y)
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        c = self.columns if self.columns is not None else self.cols
        target_cols = list(c) if c is not None else list(df.columns)

        low_val = self.lower_quantile if self.lower_quantile is not None else (
            self.lower_percentile if self.lower_percentile is not None else self.lower
        )
        high_val = self.upper_quantile if self.upper_quantile is not None else (
            self.upper_percentile if self.upper_percentile is not None else self.upper
        )
        low = float(low_val) if low_val is not None else 0.01
        high = float(high_val) if high_val is not None else 0.99
        if low > 1.0:
            low /= 100.0
        if high > 1.0:
            high /= 100.0

        self.bounds_: dict[str, tuple[float, float]] = {}
        for col in target_cols:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) > 0:
                    q_low = float(s.quantile(low))
                    q_high = float(s.quantile(high))
                    self.bounds_[col] = (q_low, q_high)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        bounds = getattr(self, "bounds_", {})
        for col, (q_low, q_high) in bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=q_low, upper=q_high)
        return df

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return self.feature_names_in_


class FrequencyEncoder(SafeTransformer):
    """Encodes categorical columns by frequency of categories learned in fit."""

    def __init__(self, columns: Sequence[str] | None = None, cols: Sequence[str] | None = None):
        super().__init__()
        self.columns = columns
        self.cols = cols

    def fit(self, X: Any, y: Any = None) -> "FrequencyEncoder":
        super().fit(X, y)
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        c = self.columns if self.columns is not None else self.cols
        target_cols = list(c) if c is not None else list(df.columns)
        self.freq_maps_: dict[str, dict[Any, float]] = {}
        for col in target_cols:
            if col in df.columns:
                vc = df[col].value_counts(normalize=True).to_dict()
                self.freq_maps_[col] = vc
        self.out_features_ = [f"{col}_freq" for col in self.freq_maps_]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        freq_maps = getattr(self, "freq_maps_", {})
        new_cols = {}
        for col, freq_map in freq_maps.items():
            if col in df.columns:
                new_cols[f"{col}_freq"] = df[col].map(freq_map).fillna(0.0)
        return pd.DataFrame(new_cols, index=df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.out_features_, dtype=object)


class TargetEncoderCV(SafeTransformer):
    """Smoothed target encoding fit inside folds to prevent data leakage."""

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        smoothing: float = 10.0,
        cols: Sequence[str] | None = None,
    ):
        super().__init__()
        self.columns = columns
        self.smoothing = smoothing
        self.cols = cols

    def fit(self, X: Any, y: Any = None) -> "TargetEncoderCV":
        super().fit(X, y)
        self.target_maps_: dict[str, dict[Any, float]] = {}
        self.global_mean_ = 0.0
        if y is None:
            return self
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_ser = pd.Series(y, index=df.index).astype(float)
        self.global_mean_ = float(y_ser.mean())
        c = self.columns if self.columns is not None else self.cols
        target_cols = list(c) if c is not None else list(df.columns)
        s_val = float(self.smoothing) if self.smoothing is not None else 10.0

        for col in target_cols:
            if col in df.columns:
                grouped = y_ser.groupby(df[col])
                counts = grouped.count()
                means = grouped.mean()
                smooth = (counts * means + s_val * self.global_mean_) / (counts + s_val)
                self.target_maps_[col] = smooth.to_dict()

        self.out_features_ = [f"{col}_te" for col in self.target_maps_]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        target_maps = getattr(self, "target_maps_", {})
        mean_val = getattr(self, "global_mean_", 0.0)
        new_cols = {}
        for col, t_map in target_maps.items():
            if col in df.columns:
                new_cols[f"{col}_te"] = df[col].map(t_map).fillna(mean_val)
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
        cols: Sequence[str] | None = None,
    ):
        super().__init__()
        self.columns = columns
        self.lags = lags
        self.roll_windows = roll_windows
        self.cols = cols

    def fit(self, X: Any, y: Any = None) -> "RollingLag":
        super().fit(X, y)
        self.out_features_ = []
        c = self.columns if self.columns is not None else self.cols
        target_cols = list(c) if c is not None else (list(X.columns) if isinstance(X, pd.DataFrame) else [])
        l_list = list(self.lags) if self.lags is not None else [1, 2]
        w_list = list(self.roll_windows) if self.roll_windows is not None else [3]

        for col in target_cols:
            for lag in l_list:
                self.out_features_.append(f"{col}_lag{lag}")
            for w in w_list:
                self.out_features_.append(f"{col}_roll_mean_{w}")
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        c = self.columns if self.columns is not None else self.cols
        target_cols = list(c) if c is not None else list(df.columns)
        l_list = list(self.lags) if self.lags is not None else [1, 2]
        w_list = list(self.roll_windows) if self.roll_windows is not None else [3]
        new_cols = {}
        for col in target_cols:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                for lag in l_list:
                    new_cols[f"{col}_lag{lag}"] = s.shift(lag).bfill().fillna(0)
                for w in w_list:
                    new_cols[f"{col}_roll_mean_{w}"] = s.rolling(window=w, min_periods=1).mean().bfill().fillna(0)
        return pd.DataFrame(new_cols, index=df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.out_features_, dtype=object)
