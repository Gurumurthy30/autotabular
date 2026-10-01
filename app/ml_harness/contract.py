"""Feature engineering script contract verification and postprocessing (Phase 3.4 & 3.5)."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ContractViolationError(Exception):
    """Raised when an FE script violates the harness contract."""


def load_pipeline_from_script(script_path: Path | str) -> Any:
    """Dynamically imports make_feature_pipeline() from a python script and returns the unfitted pipeline."""
    p = Path(script_path)
    if not p.exists():
        raise ContractViolationError(f"Script file not found: {p}")

    code = p.read_text(encoding="utf-8")
    if "train_test_split" in code:
        raise ContractViolationError(
            "Contract violation: feature_pipeline.py must NEVER call train_test_split. "
            "The harness handles all cross-validation splitting inside each fold."
        )

    module_name = f"fe_contract_module_{p.stem}"
    spec = importlib.util.spec_from_file_location(module_name, p)
    if spec is None or spec.loader is None:
        raise ContractViolationError(f"Could not load module spec from {p}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise ContractViolationError(f"Error executing feature_pipeline.py: {e}")

    if not hasattr(mod, "make_feature_pipeline"):
        raise ContractViolationError(
            "Contract violation: feature_pipeline.py must define a function 'make_feature_pipeline()' "
            "that returns an unfitted sklearn Transformer / Pipeline / ColumnTransformer."
        )

    pipeline = mod.make_feature_pipeline()
    if not hasattr(pipeline, "fit") or not hasattr(pipeline, "transform"):
        raise ContractViolationError(
            f"Contract violation: make_feature_pipeline() returned {type(pipeline)}, "
            "which does not implement fit() and transform(). Must be a valid sklearn transformer."
        )

    # Check that pipeline is NOT pre-fitted
    if getattr(pipeline, "is_fitted_", False) or hasattr(pipeline, "classes_"):
        raise ContractViolationError(
            "Contract violation: make_feature_pipeline() returned an already fitted transformer. "
            "It must return a fresh, unfitted transformer."
        )

    return pipeline


def postprocess_features(
    pipeline: Any,
    sample_X: pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Runs a dry-run transform on sample data to:
    (a) detect and remove exact-duplicate or |corr| > 0.999 columns
    (b) verify no target-derived column leaked
    (c) extract feature names and counts

    Returns:
        (transformed_sample, kept_features, removed_features)
    """
    fitted = pipeline.fit(sample_X)
    out = fitted.transform(sample_X)

    if isinstance(out, pd.DataFrame):
        df_out = out.copy()
    else:
        names = getattr(pipeline, "get_feature_names_out", None)
        if callable(names):
            try:
                col_names = names()
            except Exception:
                col_names = [f"f{i}" for i in range(out.shape[1])]
        else:
            col_names = [f"f{i}" for i in range(out.shape[1])]
        df_out = pd.DataFrame(out, columns=col_names, index=sample_X.index)

    # (b) Leakage check: ensure target_col is not in features
    if target_col and target_col in df_out.columns:
        raise ContractViolationError(
            f"Target leakage detected! Feature pipeline produced output column '{target_col}' matching target."
        )

    # (a) Check duplicate and high correlation columns
    removed_features = []
    # Drop exact constant columns
    for col in list(df_out.columns):
        if df_out[col].nunique(dropna=True) <= 1:
            removed_features.append(f"{col} (constant)")
            df_out.drop(columns=[col], inplace=True)

    # Drop exact duplicates
    dup_cols = df_out.T.duplicated()
    for col, is_dup in dup_cols.items():
        if is_dup:
            removed_features.append(f"{col} (exact duplicate)")
            df_out.drop(columns=[col], inplace=True)

    # Check |corr| > 0.999 on numeric sample
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    if len(num_cols) >= 2 and len(df_out) > 5:
        try:
            corr_mat = df_out[num_cols].corr().abs()
            to_drop = set()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    if c1 not in to_drop and c2 not in to_drop:
                        if corr_mat.loc[c1, c2] > 0.999:
                            to_drop.add(c2)
                            removed_features.append(f"{c2} (correlated with {c1} > 0.999)")
            df_out.drop(columns=list(to_drop), inplace=True, errors="ignore")
        except Exception:
            pass

    kept_features = list(df_out.columns)
    return df_out, kept_features, removed_features
