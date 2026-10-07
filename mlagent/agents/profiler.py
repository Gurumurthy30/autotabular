"""Profiler: 3-6 questions -> Coder runs ONE stats script. A deterministic profile is always built
in code too (roles, missing, target stats, leakage flags), so a weak model can't leave it empty."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pydantic import Field

from .. import ui
from ..executor import parse_result, tail
from ..ledger import Workspace, compact, digest, load, session
from ..llm import LLMFormatError, get_llm
from ..state import Base, Problem, Profile
from ..prompts import SHARED_BASE
from .coder import write_and_run

PROMPT_SYSTEM = f"""\
{SHARED_BASE}

# Role: PROFILER

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

Skip generic questions (row counts are reported anyway). Each question must be answerable by one short computation on the training data; name columns when the goal text lets you infer them.
</method>

OUTPUT (JSON): "reasoning" FIRST (<= 80 words: what you suspect about this dataset and why), then "questions" (3-6 specific computation questions as strings).
Example: {{"reasoning": "Daily weather data with potential time ordering and class imbalance. Need to verify target distribution and temporal stability...", "questions": ["What fraction of the target is positive and does it drift across the row order?", "Which columns have missing values and does missingness correlate with the target?", "Are there id-like or near-constant columns?"]}}
"""

PROMPT_STATS_TASK = """\
Answer these questions about the training data. Compute, do not guess; give one fact per answer with the number(s) and column name(s), e.g. "target positive rate 0.38 (mild imbalance)".
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
DEFAULT_QUESTIONS = ["How is the target distributed?", "Which columns have the most missing values?",
                     "Are there id-like, constant, or leakage-suspect columns?",
                     "Is there a time or group structure in the rows?"]


class Questions(Base):
    questions: list[str] = Field(default_factory=list)


def _role(s: pd.Series, name: str, id_col: str | None, n: int) -> str:
    nun = s.nunique(dropna=True)
    if nun <= 1:
        return "constant"
    if name == id_col:
        return "id"
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        return "id" if (nun == n and pd.api.types.is_integer_dtype(s) and name.lower().endswith("id")) else "numeric"
    txt = s.dropna().astype(str)
    if len(txt) and txt.head(50).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.8:
        return "datetime"
    if nun == n:
        return "text" if txt.str.len().mean() > 15 else "id"
    if txt.str.len().mean() > 30 and nun / n > 0.5:
        return "text"
    return "categorical"


def basic_profile(p: Problem) -> Profile:
    df = pd.read_csv(p.train_path)
    n, y = len(df), df[p.target]
    roles = {c: _role(df[c], c, p.id_column, n) for c in df.columns if c != p.target}
    missing = {c: round(float(f), 4) for c, f in df.isna().mean().items() if f > 0 and c != p.target}
    if p.task_type == "regression":
        yn = pd.to_numeric(y, errors="coerce")
        ts = {"mean": round(float(yn.mean()), 4), "std": round(float(yn.std()), 4), "min": float(yn.min()),
              "max": float(yn.max()), "skew": round(float(yn.skew()), 3)}
    else:
        vc = y.value_counts(normalize=True)
        yn = pd.Series(pd.factorize(y)[0], index=df.index)
        ts = {"n_classes": int(len(vc)), "distribution": {str(k): round(float(v), 3) for k, v in vc.head(10).items()},
              "imbalance_ratio": round(float(vc.max() / max(vc.min(), 1e-9)), 2)}
    flags: list[str] = []
    for c, r in roles.items():
        if r == "numeric":
            try:
                cc = df[c].corr(yn, method="spearman")
                if pd.notna(cc) and abs(cc) > 0.95:
                    flags.append(f"{c}: |spearman| with target = {cc:.2f} (possible leak)")
            except Exception:                                    # noqa: BLE001
                pass
        if p.target.lower() in c.lower():
            flags.append(f"{c}: name contains the target name")
        if r == "id":
            flags.append(f"{c}: id-like, never use as a feature")
    if p.test_path and Path(p.test_path).exists():
        tcols = set(pd.read_csv(p.test_path, nrows=5).columns)
        for c in sorted(set(df.columns) - tcols - {p.target}):
            flags.append(f"{c}: in train but not in test (cannot be a feature)")
    return Profile(column_roles=roles, target_stats={"rows": n, **ts}, missing=missing, leakage_flags=flags)


def run(ws: Workspace) -> None:
    L = load(ws)
    ui.say("Profiler", f"profiling {Path(L.problem.train_path).name}")
    try:
        qs = get_llm().json(SYSTEM, compact(digest(L, "profiler")), Questions).questions[:6] or DEFAULT_QUESTIONS
    except LLMFormatError:
        qs = DEFAULT_QUESTIONS
    prof = basic_profile(L.problem)
    prof.questions = qs
    task = ("Answer these questions about the training data with short factual strings. Compute, don't guess.\n"
            + "\n".join(f"- {q}" for q in qs)
            + '\nEnd with: emit({"answers": {"<question>": "<short answer>"}})')
    _, res = write_and_run(ws, "profile_stats", task, kind="analysis", timeout=300)
    out = parse_result(res.stdout) if res.ok else None
    if out and isinstance(out.get("answers"), dict):
        prof.notes = [f"{q} -> {a}"[:300] for q, a in out["answers"].items()]
    else:
        prof.notes = ["stats script failed: " + re.sub(r"\s+", " ", tail(res.stderr, 200))]
        ui.warn("stats script failed; using the built-in profile only")
    with session(ws) as L:
        L.profile = prof
    ui.say("Profiler", f"{len(prof.column_roles)} columns, {len(prof.leakage_flags)} flags")
