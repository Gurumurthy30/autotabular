"""Data Profiler: Inspects raw datasets, classifies feature roles, and detects leakage.

Purpose:
  Studies the data and problem structure at the beginning of the pipeline.
  Constructs a deterministic baseline Profile (data types, roles, target statistics,
  missingness, leakage flags, tagged notes, file inventory, and modality),
  prompts an LLM to formulate 3-6 targeted analytical questions about unknowns,
  and executes an analytical Python script to answer them with hard data.

What it reads:
  - Training dataset (and optional test / sample submission files).
  - Problem configuration (target, metric, task type, id column).

What it writes:
  - Ledger Profile (L.profile) with column_roles, target_stats, missingness,
    leakage_flags, questions, answers, tagged notes, modality, and data_files inventory.

Who calls it:
  - Pipeline Graph data_profiler node at the start of a pipeline run.
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field, field_validator
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .. import ui
from ..executor import parse_result, tail
from ..ledger import Workspace, compact, load, session
from ..llm import LLMFormatError, get_llm
from ..prompts import SHARED_BASE, render
from ..state import Base, Problem, Profile
from .script_writer import write_and_run

# Maximum number of rows sampled for mixed datetime heuristic.
SAMPLE_ROWS_DATETIME: int = 200

# Minimum parse success ratio required to classify a column as datetime.
MIN_DATETIME_PARSE_RATIO: float = 0.8

# Maximum number of files scanned under the dataset folder to prevent unbounded traversal.
MAX_SCAN_FILES: int = 100_000

# Number of top extensions retained in the file inventory.
TOP_EXTENSIONS_LIMIT: int = 8

# Maximum rows and columns evaluated in the decision tree single-feature leak check.
MAX_LEAK_ROWS: int = 20_000
MAX_LEAK_COLUMNS: int = 60
LEAK_DECISION_TREE_DEPTH: int = 6

# Maximum number of columns reported in the missingness dictionary.
MAX_MISSING_REPORT_COLUMNS: int = 40

# Media file extensions for path detection.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"})


PROMPT_SYSTEM = f"""\
{SHARED_BASE}

# Role: DATA PROFILER

You are the team's data scientist for the first hour. Before anyone models this dataset you build the understanding that every later agent depends on. A shallow or wrong profile makes every plan worse; a sharp one makes the right plan almost obvious. A script will answer your questions by computing on the training file.

<mission>
Turn the raw files and the problem statement into an accurate Profile: what the task really is, what the data really contains, how train and test relate, and where the traps are.
</mission>

<method>
Think like a data scientist who has been burned by bad validation. Choose 3-6 questions whose answers would CHANGE a modeling decision. Priorities, in order:

1. TASK & METRIC
- What does one row/sample represent, and what exactly is predicted: binary class, multiclass, multilabel, regression, ordinal grade, ranking?
- Do stated target and metric agree with the data and sample submission?
- What does the metric reward? (log loss/Brier reward calibrated probabilities; AUC/MAP reward ranking; F1/MCC make thresholds matter; RMSE punishes outliers, MAE does not; RMSLE implies log-scale target; QWK implies ordinal structure).

2. LEAKAGE, IDENTITY & TRAPS
- Id-like columns, columns that look like or encode the target, columns only known after outcome, near-duplicates across train and test, constant or all-unique columns.

3. STRUCTURE & SPLITS
- Are rows independent, or grouped (same user, patient, site, device), time-ordered, or hierarchical? (Changes the CV scheme).

4. DATA & TARGET DISTRIBUTION
- Balance, skew, outliers, missingness (which columns, how much, whether missing is itself informative), high cardinality.

5. TRAIN / TEST SHIFT
- Does test exist? Do test columns, ranges, category coverage, or periods differ from train?

6. INPUT FILES
- For text, image or audio inputs, check file formats, resolutions/durations, missing or unreadable files, label alignment. Read at most 300 files for any summary statistic.

Notice: <already_computed> already contains roles, missing values, target stats, leak flags, split/shift/format notes and the file inventory. Ask only questions that cannot be answered from <already_computed>. When a role is image_path, audio_path or text, ask at least one question about those files.
Skip generic questions (row counts are reported anyway). Each question must be answerable by one short computation on the training data; name columns when the goal text lets you infer them.
</method>

OUTPUT (JSON): "reasoning" FIRST (<= 80 words: what you suspect about this dataset and why), then "questions" (3-6 specific computation questions as strings).
Example: {{"reasoning": "Daily weather data with potential time ordering and class imbalance. Need to verify target distribution and temporal stability...", "questions": ["What fraction of the target is positive and does it drift across the row order?", "Which columns have missing values and does missingness correlate with the target?", "Are there id-like or near-constant columns?"]}}
"""

PROMPT_STATS_TASK = """\
Answer these questions about the training data. Compute, do not guess; give one fact per answer with the number(s) and column name(s), e.g. "target positive rate 0.38 (mild imbalance)".
If a question concerns media or text files, sample at most 300 files and say that the number is a sample.
<<questions>>
Also add two answers under the keys "extra_1" and "extra_2" for anything surprising you notice while computing: constant or near-duplicate columns, impossible values, a column that matches the target, heavy skew, rows duplicated across train/test.
Wrap each computation in try/except so one failure does not lose the others (store "error: ..." for that answer).
End with: emit({"answers": {"<question>": "<short answer>", "extra_1": "...", "extra_2": "..."}})\
"""

PROMPTS = {
    "system": PROMPT_SYSTEM,
    "stats_task": PROMPT_STATS_TASK,
}

SYSTEM = PROMPT_SYSTEM
DEFAULT_QUESTIONS = [
    "How is the target distributed?",
    "Which columns have the most missing values?",
    "Are there id-like, constant, or leakage-suspect columns?",
    "Is there a time or group structure in the rows?",
]


class Questions(Base):
    questions: list[str] = Field(default_factory=list)

    @field_validator("questions", mode="before")
    @classmethod
    def _clean_questions(cls, value):
        """Accept a string or a list; drop empty items, cut long ones, remove duplicates, keep at most 6."""
        items = [value] if isinstance(value, str) else (value or [])
        cleaned = [str(q).strip()[:200] for q in items if str(q).strip()]
        return list(dict.fromkeys(cleaned))[:6]


def skip_on_error(fn: Callable[[], Any], default: Any = None, log_msg: str = "") -> Any:
    """Executes a diagnostic helper safely; logs warning and returns default on failure."""
    try:
        return fn()
    except Exception as exc:
        if log_msg:
            ui.warn(f"{log_msg}: {exc}")
        return default


def read_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Reads a dataset supporting CSV, TSV, Parquet, and JSONL formats."""
    table_path = Path(path)
    extension = table_path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(table_path, nrows=nrows)
    if extension == ".tsv":
        return pd.read_csv(table_path, sep="\t", nrows=nrows)
    if extension in (".parquet", ".pq"):
        df = pd.read_parquet(table_path)
        return df.head(nrows) if nrows is not None else df
    if extension in (".jsonl", ".ndjson"):
        return pd.read_json(table_path, lines=True, nrows=nrows)

    return pd.read_csv(table_path, nrows=nrows)


def check_target_exists(df: pd.DataFrame, target: str) -> None:
    """Verifies that the problem target column exists in the training dataset."""
    if target not in df.columns:
        columns = list(df.columns)
        closest = difflib.get_close_matches(target, columns, n=5)
        closest_hint = f" Closest matches: {closest}." if closest else ""
        first_30 = columns[:30]
        raise ValueError(
            f"Target column '{target}' not found in dataset.{closest_hint} First 30 columns: {first_30}"
        )


def is_datetime_series(s: pd.Series) -> bool:
    """Determines whether a series contains timestamps, rejecting pure numbers stored as strings."""
    non_null = s.dropna().head(SAMPLE_ROWS_DATETIME)
    if len(non_null) == 0:
        return False

    str_sample = non_null.astype(str)
    # Reject pure numbers formatted as strings like '123' or '45.67'
    if all(re.match(r"^-?\d+(?:\.\d+)?$", val.strip()) for val in str_sample.head(20)):
        return False

    try:
        parsed = pd.to_datetime(non_null, format="mixed", errors="coerce")
        return bool(parsed.notna().mean() >= MIN_DATETIME_PARSE_RATIO)
    except Exception:
        return False


def detect_file_inventory(data_dir: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Inventories file types and stems in data folder, skipping mlagent workspaces."""
    ext_counts: dict[str, int] = {}
    known_stems: set[str] = set()
    total_files = 0

    if not data_dir.exists() or not data_dir.is_dir():
        return [], known_stems

    for root, dirs, files in os.walk(data_dir):
        # Exclude internal mlagent workspaces and caches
        dirs[:] = [d for d in dirs if not d.startswith("mlagent_") and d not in (".git", ".venv", "__pycache__")]
        for f in files:
            total_files += 1
            if total_files > MAX_SCAN_FILES:
                break
            file_path = Path(f)
            ext = file_path.suffix.lower()
            if ext:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
            known_stems.add(file_path.stem.lower())
        if total_files > MAX_SCAN_FILES:
            break

    sorted_exts = sorted(ext_counts.items(), key=lambda item: item[1], reverse=True)[:TOP_EXTENSIONS_LIMIT]
    inventory = [{"extension": ext, "count": count} for ext, count in sorted_exts]
    return inventory, known_stems


def classify_column_role(s: pd.Series, name: str, id_col: str | None, n_rows: int, known_stems: set[str]) -> str:
    """Classifies a column into its operational role."""
    nunique = s.nunique(dropna=True)
    if nunique <= 1:
        return "constant"
    if name == id_col:
        return "id"

    # 1. Media paths check (must run before id detection so unique image paths are never marked as id)
    non_null_strs = s.dropna().astype(str)
    if len(non_null_strs) > 0:
        sample_head = non_null_strs.head(100)
        img_match = sample_head.apply(lambda v: Path(v).suffix.lower() in IMAGE_EXTENSIONS or Path(v).stem.lower() in known_stems)
        if img_match.mean() >= 0.5:
            return "image_path"
        audio_match = sample_head.apply(lambda v: Path(v).suffix.lower() in AUDIO_EXTENSIONS or Path(v).stem.lower() in known_stems)
        if audio_match.mean() >= 0.5:
            return "audio_path"

    # 2. Boolean & Numeric
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and nunique == n_rows:
            s_clean = s.dropna()
            # Monotonic integer columns spanning all rows are IDs
            if s_clean.is_monotonic_increasing or s_clean.is_monotonic_decreasing or name.lower().endswith("id"):
                return "id"
        return "numeric"

    # 3. Datetime
    if is_datetime_series(s):
        return "datetime"

    # 4. Text vs ID vs Categorical
    avg_len = non_null_strs.str.len().mean() if len(non_null_strs) else 0.0
    if nunique == n_rows:
        return "text" if avg_len > 15 else "id"
    if avg_len > 30 and (nunique / max(1, n_rows)) > 0.3:
        return "text"

    return "categorical"


def detect_leakage(df: pd.DataFrame, target_name: str, task_type: str, roles: dict[str, str], test_path: str | None) -> list[str]:
    """Identifies potential target leaks via single-feature decision trees and column naming."""
    flags: list[str] = []
    n_rows = len(df)
    target_clean = target_name.lower().strip()
    target_tokens = set(re.findall(r"\w+", target_clean))

    # Name-contains-target check
    for col in df.columns:
        if col == target_name:
            continue
        col_tokens = set(re.findall(r"\w+", col.lower()))
        if len(target_clean) >= 3 and (target_tokens & col_tokens):
            flags.append(f"{col}: name contains target name")
        if roles.get(col) == "id":
            flags.append(f"{col}: id-like, never use as a feature")

    # In train but not in test check
    if test_path and Path(test_path).exists():
        try:
            test_cols = set(read_table(test_path, nrows=5).columns)
            for col in sorted(set(df.columns) - test_cols - {target_name}):
                flags.append(f"{col}: in train but not in test (cannot be a feature)")
        except Exception:
            pass

    # Single-feature decision tree check
    eval_cols = [c for c, r in roles.items() if r in ("numeric", "categorical", "id") and c in df.columns][:MAX_LEAK_COLUMNS]
    if not eval_cols:
        return flags

    sub_df = df.sample(min(n_rows, MAX_LEAK_ROWS), random_state=42) if n_rows > MAX_LEAK_ROWS else df
    y_raw = sub_df[target_name]

    if task_type == "regression":
        y_vals = pd.to_numeric(y_raw, errors="coerce").fillna(0.0).to_numpy()
        cv_splitter = KFold(n_splits=3, shuffle=True, random_state=42)
    else:
        y_codes, uniques = pd.factorize(y_raw)
        y_vals = y_codes
        n_classes = len(uniques)
        cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if n_classes > 1 else None

    if cv_splitter is None:
        return flags

    for col in eval_cols:
        series = sub_df[col]
        if pd.api.types.is_numeric_dtype(series):
            X_col = series.fillna(series.median() if len(series) else 0.0).to_numpy().reshape(-1, 1)
        else:
            X_col = pd.factorize(series)[0].reshape(-1, 1)

        oof_preds = np.zeros_like(y_vals, dtype=float)
        try:
            for train_idx, val_idx in cv_splitter.split(X_col, y_vals):
                if task_type == "regression":
                    tree = DecisionTreeRegressor(max_depth=LEAK_DECISION_TREE_DEPTH, random_state=42)
                    tree.fit(X_col[train_idx], y_vals[train_idx])
                    oof_preds[val_idx] = tree.predict(X_col[val_idx])
                else:
                    tree = DecisionTreeClassifier(max_depth=LEAK_DECISION_TREE_DEPTH, random_state=42)
                    tree.fit(X_col[train_idx], y_vals[train_idx])
                    if task_type == "binary" and n_classes == 2:
                        probs = tree.predict_proba(X_col[val_idx])
                        oof_preds[val_idx] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                    else:
                        oof_preds[val_idx] = tree.predict(X_col[val_idx])

            if task_type == "regression":
                score = r2_score(y_vals, oof_preds)
                if score > 0.95:
                    flags.append(f"{col}: decision tree R2 = {score:.3f} > 0.95 (possible leak)")
            elif task_type == "binary" and n_classes == 2:
                auc = roc_auc_score(y_vals, oof_preds)
                if auc > 0.97:
                    flags.append(f"{col}: decision tree AUC = {auc:.3f} > 0.97 (possible leak)")
            else:
                acc = accuracy_score(y_vals, oof_preds)
                majority_share = float(pd.Series(y_vals).value_counts(normalize=True).max())
                threshold = max(0.95, majority_share + 0.25)
                if acc > threshold:
                    flags.append(f"{col}: decision tree accuracy = {acc:.3f} > {threshold:.2f} (possible leak)")
        except Exception:
            continue

    return flags


def generate_tagged_notes(
    df: pd.DataFrame,
    test_path: str | None,
    sub_path: str | None,
    p: Problem,
    roles: dict[str, str],
    inventory: list[dict[str, Any]],
    modality: list[str],
) -> list[str]:
    """Generates tagged notes prefixed with split:, shift:, format:, ambiguity:, data:."""
    notes: list[str] = []
    n_rows = len(df)

    # 1. data: note
    files_summary = ", ".join(f"{item['extension']} ({item['count']})" for item in inventory[:5]) or "none"
    notes.append(f"data: modality={'+'.join(modality)}; files={files_summary}")

    # 2. split: notes
    # Duplicated rows ignoring ID
    feature_cols = [c for c in df.columns if roles.get(c) != "id" and c != p.target]
    if feature_cols:
        dup_count = int(df.duplicated(subset=feature_cols).sum())
        if dup_count > 0:
            notes.append(f"split: {dup_count} duplicate rows ({dup_count / n_rows:.1%}) ignoring id columns")

    # Group key candidates
    group_pattern = re.compile(r"user|patient|customer|session|store|site|device|series|group", re.IGNORECASE)
    for col in df.columns:
        if col != p.target and group_pattern.search(col):
            nunique_groups = df[col].nunique()
            median_rows = float(df[col].value_counts().median())
            notes.append(f"split: candidate group column '{col}' has {nunique_groups} groups, median {median_rows:.1f} rows/group")

    # Target drift across row order
    if n_rows >= 1000 and p.task_type in ("binary", "regression"):
        try:
            target_numeric = pd.to_numeric(df[p.target], errors="coerce").fillna(0.0)
            order_corr = target_numeric.corr(pd.Series(range(n_rows)), method="spearman")
            if pd.notna(order_corr) and abs(order_corr) > 0.1:
                notes.append(f"split: target drift versus row order (|spearman| = {abs(order_corr):.3f} > 0.1)")
        except Exception:
            pass

    # Datetime spans and time-split alignment
    dt_cols = [c for c, r in roles.items() if r == "datetime"]
    test_df: pd.DataFrame | None = None
    if test_path and Path(test_path).exists():
        test_df = skip_on_error(lambda: read_table(test_path, nrows=10_000))

    if dt_cols:
        primary_dt = dt_cols[0]
        try:
            tr_times = pd.to_datetime(df[primary_dt], errors="coerce").dropna()
            if len(tr_times):
                notes.append(f"split: train '{primary_dt}' span {tr_times.min()} to {tr_times.max()}")
                if test_df is not None and primary_dt in test_df.columns:
                    te_times = pd.to_datetime(test_df[primary_dt], errors="coerce").dropna()
                    if len(te_times):
                        notes.append(f"split: test '{primary_dt}' span {te_times.min()} to {te_times.max()}")
                        if te_times.min() > tr_times.max():
                            notes.append(f"split: test is LATER: time-based CV (train ends {tr_times.max()}, test starts {te_times.min()})")
        except Exception:
            pass

    # 3. shift: notes
    if test_df is not None:
        try:
            from scipy.stats import ks_2samp
            ks_results = []
            for col, r in roles.items():
                if r == "numeric" and col in test_df.columns:
                    s_tr = df[col].dropna().sample(min(5000, len(df)), random_state=42)
                    s_te = test_df[col].dropna().sample(min(5000, len(test_df)), random_state=42)
                    if len(s_tr) > 20 and len(s_te) > 20:
                        stat = ks_2samp(s_tr, s_te).statistic
                        if stat > 0.2:
                            ks_results.append((col, stat))
            ks_results.sort(key=lambda item: item[1], reverse=True)
            for col, stat in ks_results[:5]:
                notes.append(f"shift: covariate shift in numeric '{col}': KS={stat:.3f}")
        except Exception:
            pass

        # Categorical shift: share of unseen categories in test
        for col, r in roles.items():
            if r == "categorical" and col in test_df.columns:
                tr_cats = set(df[col].dropna().astype(str).unique())
                te_series = test_df[col].dropna().astype(str)
                if len(te_series):
                    unseen_count = int((~te_series.isin(tr_cats)).sum())
                    unseen_share = unseen_count / len(te_series)
                    if unseen_share > 0.05:
                        notes.append(f"shift: category shift in '{col}': {unseen_share:.1%} test values unseen in train")

    # 4. format: and ambiguity: notes
    if sub_path and Path(sub_path).exists():
        try:
            sub_df = read_table(sub_path, nrows=5000)
            val_cols = [c for c in sub_df.columns if c != p.id_column]
            notes.append(f"format: sample submission has {len(sub_df)} rows, {len(sub_df.columns)} columns ({len(val_cols)} prediction columns)")
            if test_df is not None and len(sub_df) != len(test_df):
                notes.append(f"ambiguity: sample submission row count ({len(sub_df)}) differs from test row count ({len(test_df)})")
            if p.target not in sub_df.columns and len(val_cols) == 1:
                notes.append(f"format: submission column '{val_cols[0]}' replaces target '{p.target}'")
            elif p.target not in sub_df.columns and len(val_cols) > 1:
                notes.append(f"ambiguity: target '{p.target}' is not in submission columns ({val_cols})")
        except Exception:
            pass

    return notes


def basic_profile(p: Problem) -> Profile:
    """Builds the comprehensive deterministic Profile from raw data and problem settings."""
    df = read_table(p.train_path)
    check_target_exists(df, p.target)

    n_rows = len(df)
    y = df[p.target]
    train_dir = Path(p.train_path).parent
    inventory, known_stems = detect_file_inventory(train_dir)

    # Classify column roles
    roles = {
        col: classify_column_role(df[col], col, p.id_column, n_rows, known_stems)
        for col in df.columns if col != p.target
    }

    # Infer modalities
    active_roles = set(roles.values())
    modality: list[str] = []
    if "image_path" in active_roles:
        modality.append("image")
    if "audio_path" in active_roles:
        modality.append("audio")
    if "text" in active_roles:
        modality.append("text")
    if any(r in ("numeric", "categorical", "datetime") for r in active_roles):
        modality.append("tabular")
    if not modality:
        modality = ["tabular"]

    # Missing values (capped to worst 40)
    missing_all = {col: round(float(f), 4) for col, f in df.isna().mean().items() if f > 0 and col != p.target}
    sorted_missing = dict(sorted(missing_all.items(), key=lambda item: item[1], reverse=True)[:MAX_MISSING_REPORT_COLUMNS])

    # Target statistics
    target_missing = round(float(y.isna().mean()), 4)
    if p.task_type == "regression":
        yn = pd.to_numeric(y, errors="coerce")
        ts: dict[str, Any] = {
            "mean": round(float(yn.mean()), 4),
            "std": round(float(yn.std()), 4),
            "min": float(yn.min()) if yn.notna().any() else 0.0,
            "max": float(yn.max()) if yn.notna().any() else 0.0,
            "skew": round(float(yn.skew()), 3) if yn.notna().any() else 0.0,
            "target_missing": target_missing,
        }
    else:
        vc = y.value_counts(normalize=True)
        ts = {
            "n_classes": len(vc),
            "distribution": {str(k): round(float(v), 3) for k, v in vc.head(10).items()},
            "imbalance_ratio": round(float(vc.max() / max(vc.min(), 1e-9)), 2) if len(vc) else 1.0,
            "target_missing": target_missing,
        }

    # Detect leaks and generate tagged notes
    leakage_flags = detect_leakage(df, p.target, p.task_type, roles, p.test_path)
    tagged_notes = generate_tagged_notes(df, p.test_path, p.sample_submission_path, p, roles, inventory, modality)

    return Profile(
        column_roles=roles,
        target_stats={"rows": n_rows, **ts},
        missing=sorted_missing,
        leakage_flags=leakage_flags,
        notes=tagged_notes,
        modality=modality,
        data_files=inventory,
    )


def run(ws: Workspace) -> None:
    """Executes the Data Profiler workflow: deterministic profile, LLM inquiries, and stats script."""
    L = load(ws)
    ui.say("Data Profiler", f"profiling {Path(L.problem.train_path).name}")

    # 1. Deterministic profile computed first
    prof = basic_profile(L.problem)

    # 2. LLM sees <already_computed> and asks 3-6 non-redundant questions
    already_computed_text = compact({
        "roles": prof.column_roles,
        "target": prof.target_stats,
        "missing": prof.missing,
        "leakage_flags": prof.leakage_flags,
        "modality": prof.modality,
        "notes": prof.notes,
        "data_files": prof.data_files,
    }, 4500)

    user_prompt = f"<already_computed>\n{already_computed_text}\n</already_computed>\n\nPropose 3-6 targeted analytical questions."
    try:
        inquiry_out = get_llm().json(SYSTEM, user_prompt, Questions)
        questions = inquiry_out.questions or DEFAULT_QUESTIONS
    except LLMFormatError:
        questions = DEFAULT_QUESTIONS

    prof.questions = questions

    # 3. Analytic stats script answers the proposed questions
    questions_block = "\n".join(f"- {q}" for q in questions)
    task_instructions = render(PROMPTS["stats_task"], questions=questions_block)
    _, exec_res = write_and_run(ws, "profile_stats", task_instructions, kind="analysis", timeout=300)

    script_result = parse_result(exec_res.stdout) if exec_res.ok else None
    if script_result and isinstance(script_result.get("answers"), dict):
        computed_answers = [f"{q} -> {a}"[:300] for q, a in script_result["answers"].items()]
        # Append stats script answers to notes
        prof.notes = prof.notes + computed_answers
    else:
        err_msg = re.sub(r"\s+", " ", tail(exec_res.stderr or exec_res.stdout, 200))
        prof.notes.append(f"quality: stats script failed: {err_msg}")
        ui.warn("stats script failed; keeping deterministic profile and notes")

    with session(ws) as L_curr:
        L_curr.profile = prof

    ui.say("Data Profiler", f"{len(prof.column_roles)} columns, {len(prof.leakage_flags)} flags, modality={'+'.join(prof.modality)}")
