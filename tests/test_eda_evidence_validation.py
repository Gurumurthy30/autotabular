"""
Comprehensive tests for EDA/FE evidence validation and logging.

Tests cover:
- to_evidence_str with all dict shapes from the original Pydantic errors
- numpy/pandas type coercion
- EDAFindingItem / EDAOutput validation with dict evidence
- FeatureMetadataItem eda_evidence coercion
- EVIDENCE_COERCION WARNING is written to the log file
"""

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.core.state import (
    EDAFindingItem,
    EDAOutput,
    FeatureMetadataItem,
    to_evidence_str,
)


# ─────────────────────────────────────────────────────────────────────────────
# to_evidence_str: primitive types
# ─────────────────────────────────────────────────────────────────────────────

class TestToEvidenceStrPrimitives:
    def test_none_returns_empty(self):
        assert to_evidence_str(None) == ""

    def test_plain_string_passthrough(self):
        assert to_evidence_str("already a string") == "already a string"

    def test_integer(self):
        assert to_evidence_str(123) == "123"

    def test_float(self):
        assert to_evidence_str(45.67) == "45.67"

    def test_bool(self):
        assert to_evidence_str(True) == "True"


# ─────────────────────────────────────────────────────────────────────────────
# to_evidence_str: all dict shapes from the original Pydantic errors
# ─────────────────────────────────────────────────────────────────────────────

class TestToEvidenceStrDictShapes:
    def test_iqr_std(self):
        iqr_dict = {"iqr": [7990000.0, 9290000.0], "std": 1385217.90}
        result = to_evidence_str(iqr_dict)
        assert "iqr=[7990000.0, 9290000.0]" in result
        assert "std=1385217.9" in result

    def test_skewness_distribution(self):
        skew_dict = {"skewness": 1.25, "distribution": "right-skewed"}
        result = to_evidence_str(skew_dict)
        assert "skewness=1.25" in result
        assert "distribution=right-skewed" in result

    def test_min_max_mean(self):
        minmax = {"min": 10, "max": 500, "mean": 234.5}
        result = to_evidence_str(minmax)
        assert "min=10" in result
        assert "max=500" in result
        assert "mean=234.5" in result

    def test_correlation_matrix_nested(self):
        corr = {"colA": {"colB": 0.85, "colC": -0.12}}
        result = to_evidence_str(corr)
        parsed = json.loads(result)
        assert parsed["colA"]["colB"] == 0.85
        assert parsed["colA"]["colC"] == -0.12

    def test_correlation_pairs_flat(self):
        pairs = {"feature_a": "col1", "feature_b": "col2", "correlation": 0.95}
        result = to_evidence_str(pairs)
        assert "correlation=0.95" in result
        assert "feature_a=col1" in result
        assert "feature_b=col2" in result

    def test_missing_duplicates_counts(self):
        dq = {"missing_count": 15, "missing_pct": 0.05, "duplicates": 0}
        result = to_evidence_str(dq)
        assert "missing_count=15" in result
        assert "missing_pct=0.05" in result
        assert "duplicates=0" in result

    def test_list_input(self):
        result = to_evidence_str([1.0, 2.5, 3.7])
        assert isinstance(result, str)
        assert "1.0" in result


# ─────────────────────────────────────────────────────────────────────────────
# to_evidence_str: numpy and pandas types inside dict
# ─────────────────────────────────────────────────────────────────────────────

class TestToEvidenceStrNumpyPandas:
    def test_numpy_scalars_inside_dict(self):
        d = {
            "count": np.int64(100),
            "std": np.float64(1385217.90),
            "flag": np.bool_(True),
        }
        result = to_evidence_str(d)
        assert isinstance(result, str)
        assert "count=100" in result
        assert "std=1385217.9" in result
        assert "flag=True" in result

    def test_numpy_array_inside_dict(self):
        d = {"arr": np.array([np.float64(1.5), np.float64(2.5)])}
        result = to_evidence_str(d)
        assert "[1.5, 2.5]" in result

    def test_pandas_timestamp_inside_dict(self):
        d = {"ts": pd.Timestamp("2026-09-26 11:00:00")}
        result = to_evidence_str(d)
        assert "2026-09-26 11:00:00" in result

    def test_numpy_int64_top_level(self):
        assert to_evidence_str(np.int64(42)) == "42"

    def test_numpy_float64_top_level(self):
        result = to_evidence_str(np.float64(3.14))
        assert "3.14" in result

    def test_numpy_bool_top_level(self):
        assert to_evidence_str(np.bool_(False)) == "False"


# ─────────────────────────────────────────────────────────────────────────────
# EDAFindingItem: dict evidence coercion on all shapes
# ─────────────────────────────────────────────────────────────────────────────

class TestEDAFindingItemCoercion:
    @pytest.mark.parametrize("shape", [
        {"iqr": [7990000.0, 9290000.0], "std": 1385217.90},
        {"skewness": 1.25, "distribution": "right-skewed"},
        {"min": 10, "max": 500, "mean": 234.5},
        {"colA": {"colB": 0.85, "colC": -0.12}},   # nested (correlation matrix)
        {"feature_a": "col1", "feature_b": "col2", "correlation": 0.95},
        {"missing_count": 15, "missing_pct": 0.05, "duplicates": 0},
    ])
    def test_dict_evidence_coerced_to_str(self, shape):
        item = EDAFindingItem(
            category="test",
            finding="Some finding",
            evidence=shape,   # raw dict — validator must coerce
            implication="Impacts modeling",
            recommendation="Apply transform",
        )
        assert isinstance(item.evidence, str)
        assert len(item.evidence) > 0

    def test_string_evidence_passthrough(self):
        item = EDAFindingItem(
            category="outliers",
            finding="High variance",
            evidence="std=1.39M, IQR=[7.99M, 9.29M]",
            implication="Distorts linear models",
            recommendation="Use RobustScaler",
        )
        assert item.evidence == "std=1.39M, IQR=[7.99M, 9.29M]"

    def test_numpy_types_inside_evidence_dict(self):
        item = EDAFindingItem(
            category="data_quality",
            finding="High missing rate",
            evidence={"missing_count": np.int64(687), "missing_pct": np.float64(77.1)},
            implication="Sparse signal",
            recommendation="Impute with indicator",
        )
        assert isinstance(item.evidence, str)
        assert "missing_count=687" in item.evidence
        assert "missing_pct=77.1" in item.evidence


# ─────────────────────────────────────────────────────────────────────────────
# EDAOutput: end-to-end validation with all shapes in one payload
# ─────────────────────────────────────────────────────────────────────────────

class TestEDAOutputEndToEnd:
    def test_all_evidence_shapes_in_one_payload(self):
        raw = {
            "executive_summary": "Dataset exhibits high skewness and outliers.",
            "findings": [
                {
                    "category": "outliers",
                    "finding": "High variance in price",
                    "evidence": {"iqr": [np.float64(7990000.0), np.float64(9290000.0)], "std": np.float64(1385217.90)},
                    "implication": "Distorts linear model weights",
                    "recommendation": "Use RobustScaler or Log transform",
                },
                {
                    "category": "skewness",
                    "finding": "Right-skewed target",
                    "evidence": {"skewness": np.float64(3.42), "distribution": "heavy_right_tail"},
                    "implication": "Residuals non-normal",
                    "recommendation": "Apply log1p",
                },
                {
                    "category": "correlation",
                    "finding": "Strong multicollinearity",
                    "evidence": {"rooms": {"beds": np.float64(0.92)}},
                    "implication": "Redundant feature",
                    "recommendation": "Drop one or compute ratio",
                },
                {
                    "category": "data_quality",
                    "finding": "High missing rate in cabin",
                    "evidence": {"missing_count": np.int64(687), "missing_pct": np.float64(77.1)},
                    "implication": "Sparse signal",
                    "recommendation": "Impute with missing indicator",
                },
                {
                    "category": "data_quality",
                    "finding": "Duplicate rows present",
                    "evidence": {"duplicates": np.int64(12), "duplicate_pct": np.float64(1.2)},
                    "implication": "Data quality issue",
                    "recommendation": "Drop duplicates before training",
                },
            ],
            "leakage_risks": ["ID columns match row index"],
            "suggested_feature_ideas": ["rooms_per_household"],
        }

        output = EDAOutput.model_validate(raw)
        assert len(output.findings) == 5
        for finding in output.findings:
            assert isinstance(finding.evidence, str), \
                f"evidence for '{finding.category}' is not a string: {type(finding.evidence)}"

        # Must round-trip through JSON without TypeError
        dumped = output.model_dump()
        serialized = json.dumps(dumped)
        assert isinstance(serialized, str)
        assert "iqr" in serialized
        assert "skewness" in serialized

    def test_single_finding_as_dict_coerced_to_list(self):
        """Regression: findings passed as a dict (not list) must be wrapped in a list."""
        raw = {
            "executive_summary": "Single finding test",
            "findings": {
                "category": "outliers",
                "finding": "Extreme values in lot_size",
                "evidence": {"std": 450.0},
                "implication": "Impacts distance calcs",
                "recommendation": "Clip outliers",
            },
        }
        output = EDAOutput.model_validate(raw)
        assert len(output.findings) == 1
        assert isinstance(output.findings[0].evidence, str)
        assert "std=450" in output.findings[0].evidence


# ─────────────────────────────────────────────────────────────────────────────
# FeatureMetadataItem: eda_evidence coercion
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureMetadataItemCoercion:
    def test_dict_eda_evidence_coerced(self):
        feat = FeatureMetadataItem(
            feature_name="log_price",
            source_columns=["price"],
            transformation="np.log1p",
            reason="normalize skewness",
            eda_evidence={"skewness": np.float64(2.85)},
            leakage_check="clean",
            inference_available=True,
        )
        assert isinstance(feat.eda_evidence, str)
        assert "skewness=2.85" in feat.eda_evidence

    def test_string_eda_evidence_passthrough(self):
        feat = FeatureMetadataItem(
            feature_name="age_scaled",
            source_columns=["age"],
            transformation="StandardScaler",
            reason="linear model needs scaling",
            eda_evidence="std=12.3, mean=38.5",
            leakage_check="clean",
        )
        assert feat.eda_evidence == "std=12.3, mean=38.5"


# ─────────────────────────────────────────────────────────────────────────────
# Logging: EVIDENCE_COERCION WARNING written to log
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidenceCoercionLogging:
    def test_warning_logged_when_dict_evidence_coerced(self, caplog):
        """When evidence is a dict, a WARNING with EVIDENCE_COERCION tag must be logged."""
        with caplog.at_level(logging.WARNING, logger="app.core.state"):
            EDAFindingItem(
                category="outliers",
                finding="High variance",
                evidence={"std": 1.39, "iqr": [7.99, 9.29]},
                implication="Distorts weights",
                recommendation="Use RobustScaler",
            )

        coercion_records = [r for r in caplog.records if "EVIDENCE_COERCION" in r.message]
        assert len(coercion_records) >= 1, \
            "Expected at least one EVIDENCE_COERCION WARNING but found none. " \
            f"Records: {[r.message for r in caplog.records]}"
        assert coercion_records[0].levelno == logging.WARNING

    def test_no_warning_for_string_evidence(self, caplog):
        """Plain string evidence must NOT trigger an EVIDENCE_COERCION warning."""
        with caplog.at_level(logging.WARNING, logger="app.core.state"):
            EDAFindingItem(
                category="skewness",
                finding="Right-skewed",
                evidence="skewness=2.4, std=1.2",
                implication="Log transform needed",
                recommendation="Apply log1p",
            )

        coercion_records = [r for r in caplog.records if "EVIDENCE_COERCION" in r.message]
        assert len(coercion_records) == 0

    def test_warning_logged_for_numpy_dict_evidence(self, caplog):
        """Numpy types inside dict evidence must also trigger EVIDENCE_COERCION."""
        with caplog.at_level(logging.WARNING, logger="app.core.state"):
            EDAFindingItem(
                category="data_quality",
                finding="High missing",
                evidence={"missing_count": np.int64(100), "pct": np.float64(15.0)},
                implication="Sparse signal",
                recommendation="Impute",
            )

        coercion_records = [r for r in caplog.records if "EVIDENCE_COERCION" in r.message]
        assert len(coercion_records) >= 1
        # The log message must include the field name for traceability
        assert "EDAFindingItem.evidence" in coercion_records[0].message

    def test_warning_includes_input_type(self, caplog):
        """The EVIDENCE_COERCION log must mention the input type."""
        with caplog.at_level(logging.WARNING, logger="app.core.state"):
            to_evidence_str({"key": "val"})

        coercion_records = [r for r in caplog.records if "EVIDENCE_COERCION" in r.message]
        assert any("dict" in r.message for r in coercion_records)
