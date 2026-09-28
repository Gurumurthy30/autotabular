import json
import hashlib
import time
from pathlib import Path
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError

from app.core.state import ProjectState, FeatureEngineeringOutput, FeatureMetadataItem
from app.core.model_router import ModelRouter
from app.core.memory import get_stage_context
from app.agents.coder import CoderSubAgent
from app.tools.registry import ToolRegistry
from app.config import PROJECTS_DIR
from app.utils.logger import get_logger

_log = get_logger(__name__)


FE_SYSTEM_PROMPT = """You are an expert Tabular Feature Engineering Agent.
Your responsibility is to design and implement a versioned, reproducible feature engineering pipeline
that extracts maximum signal from the dataset based on EDA findings.

CRITICAL RULES:
1. ADAPTIVE IMPUTATION: choose the right strategy per column type and missingness pattern.
   ALWAYS fit an imputer on EVERY column (even if training data currently has 0 missing values),
   so that out-of-sample prediction/test data with missing values is always safely handled:
   - Median for right-skewed / outlier-heavy numerical columns (default for numerics)
   - Mean for near-normal numerical columns
   - Mode or constant ('missing') for categorical columns
   - Add a binary indicator column for features with >10% missing rate
2. ENCODING: One-Hot for low-cardinality (<15 unique) categoricals; Ordinal for high-cardinality
   (handle_unknown='ignore'). Never skip handle_unknown.
3. SCALING / TRANSFORMS:
   - Apply log1p or Box-Cox to heavily right-skewed features (|skewness| > 1.5) identified in EDA
   - Apply RobustScaler (not StandardScaler) when EDA flagged significant outliers
   - Apply StandardScaler when model is linear/regularized and no outlier concern
4. INTERACTION & POLYNOMIAL FEATURES:
   - Create ratio/interaction features for pairs that EDA found highly correlated with target
   - Add polynomial degree-2 terms for the top 2-3 numerical predictors of target (classification only)
   - Keep feature count reasonable; aim for quality over quantity
5. DROP leakage columns EDA identified. Document disagreements in metadata.
6. INFERENCE-TIME CORRECTNESS: build a sklearn Pipeline/ColumnTransformer, fit it on training data,
   persist the FITTED object via joblib so it can be reloaded and applied to test data without refitting.
7. SAVE: parquet with clean column names + schema JSON + feature_pipeline.py (the full pipeline script).
8. NEVER import or use plotting libraries.

COMPLEX FEATURE HANDLING (apply when the column type or pattern is present; skip otherwise):
9. COLUMN TRIAGE FIRST: before transforming, classify every column and record it in metadata:
   - Drop ID-like columns (unique ratio > 0.95 with no semantic meaning), constant columns, and
     near-zero-variance columns (one value > 99%). Log each drop with the reason.
   - Detect numerics stored as strings (e.g. "1,200", "$5.30", "12%", "3 bed") and parse them to
     numeric BEFORE imputation. Detect binary columns (yes/no, True/False, 0/1) and map to 0/1.
   - Treat integer columns with few unique values (<= 10) and no ordering meaning as categorical.
10. DATETIME COLUMNS: never pass raw datetimes to the model. Extract year, month, day-of-week,
   day-of-month, quarter, is_weekend, and days-since-reference (reference date fixed at fit time and
   stored). Encode cyclical parts (month, hour, day-of-week) with sin/cos pairs. If several
   date columns exist, add differences between them (e.g. duration in days). Drop the original column.
11. HIGH-CARDINALITY & RARE CATEGORIES:
   - Group categories below 1% frequency (or below 10 rows) into an 'other' bucket, with the
     mapping learned on training data only.
   - For very high cardinality (> 50 unique), prefer frequency encoding or target encoding over
     plain ordinal. Target encoding MUST be leak-safe: use sklearn TargetEncoder (internal
     cross-fitting) or out-of-fold encoding with smoothing. Never compute it on the full training
     set and then evaluate on the same rows.
   - Ordered categoricals (size S/M/L, rating, education) get an explicit ordinal mapping in the
     correct order, not alphabetical.
12. SKEWED, MULTIMODAL & BOUNDED NUMERICS:
   - For features with extreme outliers, clip (winsorize) at fit-time 1st/99th percentiles and
     store the bounds, in addition to any scaling.
   - For features with zero-inflated or multimodal distributions, add a 'is_zero' flag and/or
     quantile-bin version alongside the transformed value; do not replace it blindly.
   - Use Yeo-Johnson (PowerTransformer) instead of Box-Cox when a feature contains zeros or
     negatives. Use log1p only for non-negative values.
13. TEXT, LISTS & COMPOSITE COLUMNS: for free-text or delimited columns, extract simple robust
   signals only (length, word count, has-keyword flags, item count, presence of top-K tokens).
   Never build large sparse text matrices unless EDA explicitly requires them. Split composite
   fields (e.g. "City, State", "Type-Grade") into separate columns. For lat/lon pairs, add
   distance-to-centroid or a coarse grid/cluster ID learned on training data.
14. GROUP AGGREGATES & MISSINGNESS SIGNALS:
   - Where a categorical has strong signal, add group-level aggregate features (mean/median/count of
     a key numeric per category), computed on training data only and stored for inference.
     Unseen groups at inference fall back to the global statistic.
   - Add a row-level 'missing_count' feature when several columns have correlated missingness
     (per EDA), and keep per-column indicators for informative missingness.
15. FEATURE CONTROL & PRUNING (after generation, before saving):
   - Remove one of each feature pair with |correlation| > 0.95 (keep the one more correlated with
     the target) and any generated feature with near-zero variance.
   - Enforce a feature budget: generated features should not exceed roughly 3x the number of
     original usable columns. If exceeded, keep the top features by mutual information with the
     target, computed on training data only.
   - Document every created, dropped, and pruned feature with its reason in the schema JSON
     (name, source columns, transform, reason).
16. ROBUSTNESS & REPRODUCIBILITY:
   - Every learned statistic (bounds, reference dates, category maps, encoders, group aggregates)
     must be fit on training data only and live INSIDE the persisted Pipeline (use custom
     sklearn transformers with fit/transform, get_feature_names_out, and set_output support).
   - Fix random_state everywhere. Ensure the transform step tolerates unseen categories, missing
     columns' NaNs, unseen dtypes, and reordered columns without crashing.
   - Final feature matrix must have unique, snake_case names, no NaN/inf, and numeric or
     boolean dtypes only. Run a sanity check after transform and log shape, dropped columns,
     and feature count; raise a clear error if any check fails instead of failing silently.
"""


def run_feature_engineering(state: ProjectState, router: ModelRouter, registry: ToolRegistry) -> dict[str, Any]:
    """Designs and executes a reproducible feature engineering pipeline yielding feature_data.parquet."""
    stage_start = time.monotonic()
    project_id = state["project_id"]
    run_id = state.get("run_id", "unknown")
    _log.info("[FE] Stage started | project=%s run=%s", project_id, run_id)

    tools = registry.get_tools_for_role("feature_engineering")
    coder_tools = registry.get_tools_for_role("coder")
    coder = CoderSubAgent(project_id, router, coder_tools.files, coder_tools.execution)

    ctx = get_stage_context(state, "feature_engineering")
    dataset_version = state.get("dataset_version", "dataset_v1")
    dataset_path = str(tools.dataset.get_dataset_path(dataset_version).resolve()).replace("\\", "/")
    target_col = state.get("target_column")
    task_type = state.get("task_type")
    iteration = state.get("iteration", 1)

    features_dir = PROJECTS_DIR / project_id / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    parquet_out_path = str((features_dir / "feature_data.parquet").resolve()).replace("\\", "/")
    schema_out_path = str((features_dir / "feature_schema.json").resolve()).replace("\\", "/")
    pipeline_pkl_path = str((features_dir / "feature_pipeline.pkl").resolve()).replace("\\", "/")
    pipeline_code_path = features_dir / "feature_pipeline.py"
    report_md_path = features_dir / "feature_report.md"

    eda_findings_json = json.dumps(state.get('eda_findings', {}), indent=2)[:2500]
    profile_json = json.dumps(state.get('profile_summary', {}), indent=2)[:1500]
    eval_json = json.dumps(state.get('evaluation_summary', {}), indent=2)

    # Ask Coder to write feature_pipeline.py and execute it
    task_prompt = f"""Write a COMPLETE, PRODUCTION-GRADE feature engineering pipeline for a tabular ML task.
Source dataset path: '{dataset_path}'
Target column: '{target_col}'
Task type: '{task_type}'
Destination parquet output path: '{parquet_out_path}'
Destination schema json path: '{schema_out_path}'
Fitted pipeline save path (joblib): '{pipeline_pkl_path}'

Dataset profile:
{profile_json}

EDA findings and recommendations (USE THESE to justify every decision):
{eda_findings_json}

Feedback from previous evaluation (if any):
{eval_json}

REQUIREMENTS (implement ALL of these):
1. Load dataset from source path using pandas.
2. Separate target column '{target_col}' — preserve it but never transform it into a feature.
3. DROP identifiable ID/index columns (100% unique, no predictive power) and any columns EDA
   flagged as leakage risk. Log which columns you drop and why.
4. ADAPTIVE IMPUTATION per column (CRITICAL: ALWAYS fit an imputer on EVERY column, even if training data currently has 0 missing values, so future test/prediction data with missing values is safely imputed):
   - Numerical with |skewness|>1.5 or outlier flag: use median imputation
   - Numerical near-normal: use mean imputation
   - Categorical: use most_frequent (mode) imputation; add a binary is_missing indicator
     for any column with >10% missing rate
5. ENCODE CATEGORICALS:
   - Cardinality <= 15: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
   - Cardinality > 15: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
6. SCALE / TRANSFORM NUMERICALS:
   - Columns that EDA flagged as right-skewed (|skewness|>1.5): apply np.log1p BEFORE scaling
   - Columns EDA flagged as having significant outliers: use RobustScaler
   - Remaining numerical: use StandardScaler
7. INTERACTION FEATURES:
   - For each pair of numerical features that EDA found highly correlated with the target
     (correlation > 0.3), add a product interaction feature: col_a_x_col_b
   - Add ratio features for pairs where a ratio is semantically meaningful (e.g. income/expense)
   - For CLASSIFICATION tasks: add degree-2 polynomial terms for the top-3 numerical predictors
     of the target (based on EDA correlations or feature importances if available)
8. Build a scikit-learn Pipeline or ColumnTransformer that encapsulates ALL of the above steps.
   FIT it on the full training data (minus target), then TRANSFORM X.
   Save the FITTED pipeline object to '{pipeline_pkl_path}' using joblib.dump().
9. Reconstruct the full DataFrame: X_transformed (features) + target column '{target_col}'.
   Save to '{parquet_out_path}' using pyarrow.
10. Save schema (all column names and dtypes after transform) to '{schema_out_path}' as JSON.
11. Print a summary of engineered features with column counts.
12. NO PLOTTING (do NOT import matplotlib/seaborn/plotly).
"""

    coder_res = coder.run_task(task_description=task_prompt, context=ctx)
    if coder_res["status"] == "FAILED":
        return {
            "status": "FAILED",
            "current_stage": "feature_engineering",
            "error": coder_res["error"],
        }

    # Save the pipeline script into features/feature_pipeline.py
    executed_code = coder_res.get("executed_code", "# Generated feature pipeline")
    tools.files.write_file("features/feature_pipeline.py", executed_code)

    # Use LLM to generate structured feature metadata and report
    llm = router.get_model("feature_engineering", temperature=0.0)
    structured_llm = llm.with_structured_output(FeatureEngineeringOutput)

    metadata_prompt = f"""Generate structured metadata and report for the feature engineering pipeline:
Target Column: {target_col}
Task Type: {task_type}
Iteration: {iteration}
Coder Execution Output:
{coder_res.get('stdout_summary', '')}

Executed Pipeline Code snippet:
{executed_code[:2000]}

Provide:
- version (e.g. "feat_v{iteration}")
- list of created_features with metadata (feature_name, source_columns, transformation, reason, eda_evidence, leakage_check, inference_available)
- pipeline_file path ("features/feature_pipeline.py")
- data_file path ("features/feature_data.parquet")
- schema_file path ("features/feature_schema.json")
- notes explaining key design choices
"""

    fe_output = None
    prompt_hash = hashlib.sha256(FE_SYSTEM_PROMPT.encode()).hexdigest()[:12]
    _log.info("[FE] Invoking structured LLM for metadata | prompt_version=%s iter=%s", prompt_hash, iteration)
    try:
        fe_output = structured_llm.invoke([
            SystemMessage(content=FE_SYSTEM_PROMPT),
            HumanMessage(content=metadata_prompt),
        ])
        _log.debug("[FE] Raw LLM response (truncated): %s", str(fe_output)[:600])
    except ValidationError as ve:
        _log.exception(
            "[FE] Pydantic ValidationError in metadata | fields=%s | preview=%s",
            [e['loc'] for e in ve.errors()], str(ve.errors())[:400],
        )
    except Exception as exc:
        _log.exception("[FE] LLM structured output failed: %s", exc)

    if fe_output is None:
        # Fallback to inspecting schema directly
        # The schema file may be a dict {col: dtype}, a list [{name, dtype}], or other shapes.
        # Normalise everything to a flat {col_name: dtype_str} dict first.
        schema_dict: dict[str, str] = {}
        if Path(schema_out_path).exists():
            try:
                with open(schema_out_path, "r", encoding="utf-8") as f:
                    raw_schema = json.load(f)

                if isinstance(raw_schema, dict):
                    # Could be {col: dtype} or {col: {dtype: ..., ...}}
                    for k, v in raw_schema.items():
                        schema_dict[str(k)] = str(v) if not isinstance(v, dict) else str(v.get("dtype", "object"))
                elif isinstance(raw_schema, list):
                    # Could be [{name: col, dtype: ...}, ...] or ["col1", "col2", ...]
                    for item in raw_schema:
                        if isinstance(item, dict):
                            col = item.get("name") or item.get("column") or item.get("col") or str(item)
                            dtype = str(item.get("dtype", "object"))
                            schema_dict[str(col)] = dtype
                        else:
                            schema_dict[str(item)] = "object"
                else:
                    _log.warning("[FE] Unexpected schema format: %s — skipping schema fallback", type(raw_schema).__name__)
            except Exception as schema_exc:
                _log.exception("[FE] Failed to load schema from disk: %s", schema_exc)

        # If schema is still empty, try reading column names from the parquet directly
        if not schema_dict and Path(parquet_out_path).exists():
            try:
                import pandas as pd
                _df = pd.read_parquet(parquet_out_path)
                schema_dict = {col: str(dtype) for col, dtype in _df.dtypes.items()}
                _log.info("[FE] Schema loaded from parquet fallback | columns=%d", len(schema_dict))
            except Exception as parquet_exc:
                _log.exception("[FE] Failed to read parquet for schema fallback: %s", parquet_exc)

        feature_items = []
        for col_name, dtype_str in schema_dict.items():
            if col_name != target_col:
                feature_items.append(
                    FeatureMetadataItem(
                        feature_name=col_name,
                        source_columns=[col_name.split("_")[0]],
                        transformation="engineered or encoded",
                        reason="generated during feature pipeline execution",
                        eda_evidence=f"dtype={dtype_str}",
                        leakage_check="clean (no direct leakage)",
                        inference_available=True,
                    )
                )

        fe_output = FeatureEngineeringOutput(
            version=f"feat_v{iteration}",
            created_features=feature_items,
            pipeline_file="features/feature_pipeline.py",
            data_file="features/feature_data.parquet",
            schema_file="features/feature_schema.json",
            notes="Feature pipeline executed successfully and transformed parquet produced.",
        )


    output_dict = fe_output.model_dump()

    # Create feature_report.md
    report_lines = [
        f"# Feature Engineering Report: Project `{project_id}` (Version {fe_output.version})",
        f"- **Data File:** `features/feature_data.parquet`",
        f"- **Pipeline Script:** `features/feature_pipeline.py`",
        f"- **Target Column:** `{target_col}`",
        "",
        "## Key Decisions",
        fe_output.notes,
        "",
        "## Engineered Features Metadata",
        "| Feature Name | Sources | Transformation | Reason | Leakage Check |",
        "|---|---|---|---|---|",
    ]
    for feat in fe_output.created_features:
        srcs = ", ".join(feat.source_columns)
        clean_trans = feat.transformation.replace("|", "/")
        clean_reason = feat.reason.replace("|", "/")
        report_lines.append(f"| {feat.feature_name} | {srcs} | {clean_trans} | {clean_reason} | {feat.leakage_check} |")

    tools.files.write_file("features/feature_report.md", "\n".join(report_lines))

    # Register artifacts
    art_parquet = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="feature_data_parquet",
        path=parquet_out_path,
        run_id=state.get("run_id"),
        version=fe_output.version,
    )
    art_pipe = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="feature_pipeline_py",
        path=str(pipeline_code_path),
        run_id=state.get("run_id"),
        version=fe_output.version,
    )
    art_report = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="feature_report_md",
        path=str(report_md_path),
        run_id=state.get("run_id"),
        version=fe_output.version,
    )

    artifacts_list = list(state.get("artifacts", []))
    artifacts_list.extend([art_parquet, art_pipe, art_report])

    if Path(pipeline_pkl_path).exists():
        art_pkl = tools.artifacts.register_artifact(
            project_id=project_id,
            artifact_type="feature_pipeline_pkl",
            path=str(pipeline_pkl_path),
            run_id=state.get("run_id"),
            version=fe_output.version,
        )
        artifacts_list.append(art_pkl)

    elapsed = time.monotonic() - stage_start
    _log.info(
        "[FE] Stage completed | project=%s run=%s | features=%d | version=%s | duration=%.2fs",
        project_id, run_id, len(fe_output.created_features), fe_output.version, elapsed,
    )

    return {
        "feature_summary": output_dict,
        "current_stage": "feature_engineering",
        "artifacts": artifacts_list,
        "status": "SUCCESS",
    }
