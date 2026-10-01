"""Standard baseline preprocessor factory (Phase 3.3).

Returns an unfitted ColumnTransformer conforming to column roles from ProfileSummary.
"""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from app.ml_harness.transformers import DateParts


def make_basic_preprocessor(profile: dict[str, Any] | Any) -> ColumnTransformer:
    """Builds an unfitted sklearn ColumnTransformer based on detected column roles.

    - numeric: SimpleImputer(median), optional missing indicator
    - categorical: SimpleImputer(most_frequent) + OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    - high_card: SimpleImputer(most_frequent) + OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    - datetime: DateParts
    - id, row_counter, constant: dropped
    """
    if hasattr(profile, "model_dump"):
        p_dict = profile.model_dump()
    elif isinstance(profile, dict):
        p_dict = profile
    else:
        p_dict = {}

    column_roles: dict[str, str] = p_dict.get("column_roles", {})
    columns_info = p_dict.get("columns", [])
    missing_pcts = {c.get("name"): c.get("missing_pct", 0.0) for c in columns_info if isinstance(c, dict)}

    numeric_cols = []
    categorical_cols = []
    high_card_cols = []
    datetime_cols = []
    drop_cols = []

    for col, role in column_roles.items():
        if role in ("id", "row_counter", "constant", "target"):
            drop_cols.append(col)
        elif role == "datetime":
            datetime_cols.append(col)
        elif role == "high_card":
            high_card_cols.append(col)
        elif role == "categorical":
            categorical_cols.append(col)
        elif role == "numeric":
            numeric_cols.append(col)

    transformers = []

    # Numeric pipeline
    if numeric_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("numeric", num_pipeline, numeric_cols))

    # Low-cardinality categorical pipeline
    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ])
        transformers.append(("categorical", cat_pipeline, categorical_cols))

    # High-cardinality categorical pipeline
    if high_card_cols:
        hc_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("high_card", hc_pipeline, high_card_cols))

    # Datetime pipeline
    if datetime_cols:
        transformers.append(("datetime", DateParts(columns=datetime_cols), datetime_cols))

    # If no transformers matched, create a passthrough
    if not transformers:
        transformers = [("passthrough", "passthrough", list(column_roles.keys()))]

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    try:
        ct.set_output(transform="pandas")
    except Exception:
        pass

    return ct
