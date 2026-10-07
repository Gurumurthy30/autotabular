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
from .coder import write_and_run

SYSTEM = ("You are a data scientist about to model a tabular dataset. Write 3-6 sharp, answerable questions "
          "about the data: target balance/distribution, missingness, leakage risks, id/time/group columns, "
          "high-cardinality columns, train-vs-test shift. Fields: questions (list of strings).")
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
