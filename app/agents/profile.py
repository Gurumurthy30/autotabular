"""Deterministic, comprehensive tabular dataset profiling (Phase 2).

Computes column roles, target statistics, data hygiene metrics, and leak/id suspects
without LLM calls. Produces both profile.json and a ~1500-char compact representation.
"""

import json
import re
from typing import Any

import pandas as pd

from app.config import PROJECTS_DIR
from app.core.state import ColumnProfile, ProfileSummary, ProjectState, TaskType
from app.tools.registry import ToolRegistry
from app.utils.logger import get_logger

_log = get_logger(__name__)

ID_PATTERN = re.compile(
    r"(?i)^(id|_id|uuid|guid|key|code|index|hash|ref)$|(_id|_uuid|_guid|_key|_code|_hash|_ref)$"
)


def is_datetime_series(series: pd.Series) -> bool:
    """Checks if a series is datetime or >= 95% of non-null string values parse as datetime."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    # Sample up to 100 values to avoid slow date parsing on huge data
    sample = non_null.head(100)
    # Check if sample strings look like dates (at least 6 chars and contains separator or numbers)
    first_val = str(sample.iloc[0]).strip()
    if len(first_val) < 6:
        return False

    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        valid_ratio = parsed.notna().sum() / len(sample)
        return bool(valid_ratio >= 0.95)
    except Exception:
        return False


def is_row_counter(series: pd.Series, row_count: int) -> bool:
    """Checks if series is integer, strictly unique, sorted with constant step (diff == 1)."""
    if not pd.api.types.is_integer_dtype(series) and not pd.api.types.is_float_dtype(series):
        return False
    if series.nunique(dropna=True) != row_count:
        return False

    # Check if diff == 1 everywhere
    diffs = series.diff().dropna()
    if len(diffs) > 0 and (diffs == 1).all():
        return True
    return False


def profile_dataset(state: ProjectState, registry: ToolRegistry) -> dict[str, Any]:
    """Deterministic tabular profiling node computing column roles and compact representations."""
    project_id = state["project_id"]
    tools = registry.get_tools_for_role("profile")
    dataset_version = state.get("dataset_version", "dataset_v1")

    # Load dataset
    df: pd.DataFrame = tools.dataset.load_dataset(dataset_version)
    target_col = state.get("target_column")

    row_count, col_count = df.shape
    duplicates_count = int(df.duplicated().sum())
    total_cells = row_count * col_count
    total_missing = int(df.isna().sum().sum())
    missing_total_pct = round((total_missing / total_cells * 100) if total_cells > 0 else 0.0, 2)
    memory_size_bytes = int(df.memory_usage(deep=True).sum())

    column_roles: dict[str, str] = {}
    id_suspects: list[dict[str, Any]] = []
    columns_profile: list[ColumnProfile] = []
    row_order_meaningful = False
    possible_time_col = None
    possible_group_cols: list[str] = []

    # 1. Inspect each column and determine role
    for col in df.columns:
        series = df[col]
        missing_cnt = int(series.isna().sum())
        missing_pct = round((missing_cnt / row_count * 100) if row_count > 0 else 0.0, 2)
        n_unique = int(series.nunique(dropna=True))
        is_const = bool(n_unique <= 1)
        samples = series.dropna().head(3).tolist()

        columns_profile.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                missing_count=missing_cnt,
                missing_pct=missing_pct,
                unique_count=n_unique,
                is_constant=is_const,
                sample_values=samples,
            )
        )

        if col == target_col:
            column_roles[col] = "target"
            continue

        if is_const:
            column_roles[col] = "constant"
            continue

        # Check datetime
        if is_datetime_series(series):
            column_roles[col] = "datetime"
            if possible_time_col is None:
                possible_time_col = col
            continue

        # Check row_counter (strictly increasing integer with diff == 1)
        if is_row_counter(series, row_count):
            column_roles[col] = "row_counter"
            row_order_meaningful = True
            id_suspects.append({
                "column": col,
                "role": "row_counter",
                "reason": "Sequential integer counter (diff == 1); excluded from features, row order meaningful.",
            })
            continue

        # Check true_id
        is_name_id = bool(ID_PATTERN.search(col))
        unique_ratio = n_unique / row_count if row_count > 0 else 0.0

        if is_name_id and unique_ratio >= 0.99:
            column_roles[col] = "id"
            id_suspects.append({
                "column": col,
                "role": "id",
                "reason": f"Column name matches ID pattern and is {unique_ratio:.1%} unique.",
            })
            continue

        # Long unique string identifier check
        if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and unique_ratio >= 0.99:
            avg_len = series.dropna().astype(str).str.len().mean() if n_unique > 0 else 0
            if avg_len > 20:
                column_roles[col] = "id"
                id_suspects.append({
                    "column": col,
                    "role": "id",
                    "reason": f"Long unique string (avg len {avg_len:.1f}) and {unique_ratio:.1%} unique.",
                })
                continue

        # Check group column (repeating IDs e.g. customer_id, store_id)
        if is_name_id and 1 < n_unique < 0.8 * row_count:
            possible_group_cols.append(col)

        # High cardinality categoricals
        if (pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series)) and n_unique > 50:
            column_roles[col] = "high_card"
            continue

        if pd.api.types.is_numeric_dtype(series):
            column_roles[col] = "numeric"
        else:
            column_roles[col] = "categorical"

    # Near-duplicate column pairs (|Pearson r| > 0.98)
    near_duplicate_pairs: list[list[str]] = []
    num_cols = [c for c, r in column_roles.items() if r in ("numeric", "row_counter")]
    if len(num_cols) >= 2 and row_count > 5:
        try:
            sample_df = df[num_cols].dropna()
            if len(sample_df) > 5000:
                sample_df = sample_df.sample(5000, random_state=42)
            corr_mat = sample_df.corr().abs()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    val = corr_mat.loc[c1, c2]
                    if val > 0.98:
                        near_duplicate_pairs.append([c1, c2])
        except Exception as e:
            _log.debug("[PROFILE] Correlation check skipped: %s", e)

    # 2. Analyze target stats & detect task type
    task_guess = TaskType.AMBIGUOUS
    target_dist: dict[str, Any] = {}
    notes = []

    if target_col and target_col in df.columns:
        t_series = df[target_col].dropna()
        t_unique = int(t_series.nunique())

        if t_unique == 2:
            task_guess = TaskType.BINARY_CLASSIFICATION
            val_counts = t_series.value_counts()
            target_dist = {
                "class_counts": {str(k): int(v) for k, v in val_counts.items()},
                "minority_pct": float(round(val_counts.min() / len(t_series) * 100, 2)),
                "num_classes": 2,
            }
            notes.append(f"Binary classification: minority class is {target_dist['minority_pct']}%.")
        elif 2 < t_unique <= 20 and (t_series.dtype == "object" or pd.api.types.is_categorical_dtype(t_series) or pd.api.types.is_integer_dtype(t_series)):
            task_guess = TaskType.MULTICLASS_CLASSIFICATION
            val_counts = t_series.value_counts().head(20)
            target_dist = {
                "class_counts": {str(k): int(v) for k, v in val_counts.items()},
                "num_classes": t_unique,
            }
            notes.append(f"Multiclass classification with {t_unique} classes.")
        elif pd.api.types.is_numeric_dtype(t_series) and t_unique > 20:
            task_guess = TaskType.REGRESSION
            skew_val = float(round(float(t_series.skew()), 4)) if len(t_series) > 2 else 0.0
            target_dist = {
                "min": float(t_series.min()),
                "max": float(t_series.max()),
                "mean": float(round(t_series.mean(), 4)),
                "std": float(round(t_series.std(), 4)),
                "median": float(round(t_series.median(), 4)),
                "skew": skew_val,
            }
            notes.append(f"Regression: range=[{target_dist['min']}, {target_dist['max']}], skew={skew_val}.")
        else:
            task_guess = TaskType.AMBIGUOUS
            notes.append(f"Target '{target_col}' could not be definitively classified.")
    else:
        notes.append("No valid target column found in dataset.")

    # 3. Create ~1500 char compact profile for briefs
    numeric_count = sum(1 for r in column_roles.values() if r == "numeric")
    cat_count = sum(1 for r in column_roles.values() if r == "categorical")
    high_card_count = sum(1 for r in column_roles.values() if r == "high_card")
    date_cols = [c for c, r in column_roles.items() if r == "datetime"]
    id_cols = [c for c, r in column_roles.items() if r == "id"]
    row_counters = [c for c, r in column_roles.items() if r == "row_counter"]

    compact_lines = [
        f"Rows: {row_count} | Cols: {col_count} | Mem: {memory_size_bytes / 1024:.1f} KB | Missing: {missing_total_pct}%",
        f"Target: {target_col} | Task: {task_guess.value} | Stats: {json.dumps(target_dist)}",
        f"Roles: numeric={numeric_count}, cat={cat_count}, high_card={high_card_count}, datetime={len(date_cols)}",
    ]
    if row_counters:
        compact_lines.append(f"Row Counters (excluded, order meaningful): {', '.join(row_counters)}")
    if id_cols:
        compact_lines.append(f"True IDs (dropped): {', '.join(id_cols)}")
    if date_cols:
        compact_lines.append(f"Date Columns: {', '.join(date_cols)}")
    if possible_group_cols:
        compact_lines.append(f"Possible Group Columns: {', '.join(possible_group_cols)}")
    if near_duplicate_pairs:
        compact_lines.append(f"Near-Duplicate Pairs (|corr| > 0.98): {near_duplicate_pairs[:3]}")

    compact_profile = "\n".join(compact_lines)[:1500]

    profile_summary = ProfileSummary(
        row_count=row_count,
        column_count=col_count,
        columns=columns_profile,
        target_column=target_col,
        task_type_guess=task_guess,
        duplicates_count=duplicates_count,
        missing_total_pct=missing_total_pct,
        target_distribution=target_dist,
        notes=notes,
        column_roles=column_roles,
        id_suspects=id_suspects,
        row_order_meaningful=row_order_meaningful,
        possible_time_column=possible_time_col,
        possible_group_columns=possible_group_cols,
        near_duplicate_pairs=near_duplicate_pairs,
        compact=compact_profile,
    )

    # Save artifacts on disk
    profile_dir = PROJECTS_DIR / project_id / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    json_path = profile_dir / "profile.json"
    summary_md_path = profile_dir / "summary.md"

    summary_dict = profile_summary.model_dump()
    tools.files.write_file("profile/profile.json", json.dumps(summary_dict, indent=2))

    # Generate Markdown Summary (NO plots)
    md_lines = [
        f"# Dataset Profile: Project `{project_id}`",
        f"- **Rows:** {row_count}",
        f"- **Columns:** {col_count}",
        f"- **Duplicates:** {duplicates_count}",
        f"- **Total Missing Cells:** {missing_total_pct}%",
        f"- **Guessed Task Type:** `{task_guess.value}`",
        f"- **Target Column:** `{target_col}`",
        f"- **Row Order Meaningful:** {row_order_meaningful}",
        "",
        "## Column Roles",
        "| Column | Role | Dtype | Missing % | Unique |",
        "|---|---|---|---|---|",
    ]
    for col in columns_profile:
        role = column_roles.get(col.name, "unknown")
        md_lines.append(f"| {col.name} | {role} | {col.dtype} | {col.missing_pct}% | {col.unique_count} |")

    if id_suspects:
        md_lines.append("\n## ID & Counter Suspects")
        for susp in id_suspects:
            md_lines.append(f"- **{susp['column']}** ({susp['role']}): {susp['reason']}")

    tools.files.write_file("profile/summary.md", "\n".join(md_lines))

    # Register artifacts
    art_json = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="profile_json",
        path=str(json_path),
        run_id=state.get("run_id"),
        version=dataset_version,
    )
    art_md = tools.artifacts.register_artifact(
        project_id=project_id,
        artifact_type="profile_summary_md",
        path=str(summary_md_path),
        run_id=state.get("run_id"),
        version=dataset_version,
    )

    artifacts_list = list(state.get("artifacts", []))
    artifacts_list.extend([art_json, art_md])

    assigned_task_type = state.get("task_type") or task_guess

    worker_report = {
        "status": "ok",
        "result_summary": f"Profiled {row_count} rows, {col_count} cols. Task: {task_guess.value}. Order meaningful: {row_order_meaningful}."[:300],
        "evidence": {
            "row_count": row_count,
            "column_count": col_count,
            "missing_total_pct": missing_total_pct,
            "task_type": task_guess.value,
            "column_roles": column_roles,
            "id_suspects": id_suspects,
            "row_order_meaningful": row_order_meaningful,
            "possible_time_column": possible_time_col,
            "imbalance": target_dist,
        },
        "concern": None,
        "suggestion": None,
        "notebook": {
            "step": state.get("step", 0),
            "tried": "Deterministic tabular profiling",
            "outcome": f"Identified roles for {col_count} columns across {row_count} rows",
            "lesson": f"Task: {task_guess.value}, Row order meaningful: {row_order_meaningful}",
            "score_impact": None,
            "errors": [],
        },
        "artifacts": [str(json_path), str(summary_md_path)],
    }

    return {
        "profile_summary": summary_dict,
        "task_type": assigned_task_type,
        "current_stage": "profile",
        "artifacts": artifacts_list,
        "status": "SUCCESS",
        "report": worker_report,
    }
