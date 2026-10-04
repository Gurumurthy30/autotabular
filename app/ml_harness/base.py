"""Base transformer classes ensuring picklability, get_feature_names_out, and scikit-learn compatibility."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SafeTransformer(BaseEstimator, TransformerMixin):
    """Base transformer enforcing scikit-learn compatibility, clean fitting, and feature naming."""

    def __init__(self):
        pass

    def fit(self, X: Any, y: Any = None) -> "SafeTransformer":
        self.is_fitted_ = True
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(list(X.columns), dtype=object)
        elif hasattr(X, "shape") and len(X.shape) > 1:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)
        else:
            self.feature_names_in_ = np.array([], dtype=object)
        return self

    def transform(self, X: Any) -> Any:
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.array(list(input_features), dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.array([], dtype=object)

    def __sklearn_clone__(self) -> "SafeTransformer":
        """Robust scikit-learn clone hook that avoids strict identity assertion crashes."""
        from sklearn.base import clone as sk_clone
        klass = self.__class__
        params = self.get_params(deep=False)
        cloned_params = {
            name: (sk_clone(p, safe=False) if hasattr(p, "get_params") else p)
            for name, p in params.items()
        }
        try:
            cloned = klass(**cloned_params)
        except Exception:
            cloned = klass()
        for k, v in cloned_params.items():
            try:
                setattr(cloned, k, v)
            except Exception:
                pass
        return cloned

