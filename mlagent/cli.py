"""mlagent CLI.

  mlagent                         interactive session (type a CSV path or a request, /help for commands)
  mlagent run data/train.csv      one-shot, prompts only for what it cannot infer (--yes skips prompts)
  mlagent resume <path>           continue a stopped run (path = data file, workspace, or its parent folder)
  mlagent resume <path> --more 5  re-open a FINISHED run for 5 more experiments (also --minutes M)
  mlagent status <path>           runs / queue / result of a workspace
  mlagent lb <path> --score 0.77  record a public-LB score and check the CV/LB gap (or --submit to send it)
  mlagent config init [path]      write a commented mlagent.yaml
  mlagent graph                   print the LangGraph as mermaid

The workspace is created NEXT TO THE DATA:  <data_dir>/mlagent_<train_stem>/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from . import config as cfgmod
from . import doctor, lb, llm, ui
from .build_graph import mermaid, run_pipeline
from .ledger import Workspace, experiments_used, load, load_control, save, session
from .state import Env, Ledger, Problem

LIBS = ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost", "optuna")
MIN_METRICS = {"rmse", "mae", "mse", "logloss", "log_loss", "rmsle", "bce", "cross_entropy"}
METRICS = {"binary": ["accuracy", "roc_auc", "f1", "logloss"],
           "multiclass": ["accuracy", "f1_macro", "logloss"],
           "regression": ["rmse", "mae", "r2", "rmsle"]}

METRIC_ALIASES = {
    "roc": "roc_auc",
    "auc": "roc_auc",
    "roc_auc": "roc_auc",
    "rocauc": "roc_auc",
    "auc_roc": "roc_auc",
    "acc": "accuracy",
    "accuracy": "accuracy",
    "f1": "f1",
    "f1_score": "f1",
    "f1_binary": "f1",
    "f1_macro": "f1_macro",
    "macro_f1": "f1_macro",
    "f1_weighted": "f1_weighted",
    "weighted_f1": "f1_weighted",
    "f1_micro": "f1_micro",
    "micro_f1": "f1_micro",
    "logloss": "logloss",
    "log_loss": "logloss",
    "bce": "logloss",
    "cross_entropy": "logloss",
    "rmse": "rmse",
    "root_mean_squared_error": "rmse",
    "mse": "mse",
    "mean_squared_error": "mse",
    "mae": "mae",
    "mean_absolute_error": "mae",
    "r2": "r2",
    "r_squared": "r2",
    "rmsle": "rmsle",
    "root_mean_squared_log_error": "rmsle",
}


def normalize_metric(m: str) -> str:
    s = str(m).strip().lower().replace("-", "_").replace(" ", "_")
    return METRIC_ALIASES.get(s, s)


# ---- inference helpers -------------------------------------------------------------------------

def _versions() -> dict[str, str]:
    out = {}
    for lib in LIBS:
        try:
            out[lib if lib != "scikit-learn" else "sklearn"] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    return out


def _sibling(folder: Path, patterns: list[str], exclude: Path) -> str | None:
    for pat in patterns:
        for f in sorted(folder.glob(pat)):
            if f.resolve() != exclude.resolve():
                return str(f.resolve())
    return None


def infer(train: Path) -> dict:
    folder = train.parent
    test = _sibling(folder, ["test.csv", "*test*.csv"], train)
    sample = _sibling(folder, ["sample_submission*.csv", "*submission*.csv"], train)
    df = pd.read_csv(train, nrows=20000)
    cols = list(df.columns)
    target, id_col = cols[-1], None
    if sample:
        sc = list(pd.read_csv(sample, nrows=2).columns)
        id_col = sc[0] if sc[0] in cols else None
        t = [c for c in sc[1:] if c in cols]
        target = t[0] if t else target
    elif test:                                       # target = the column train has and test lacks
        extra = [c for c in cols if c not in pd.read_csv(test, nrows=2).columns]
        target = extra[0] if len(extra) == 1 else target
    if id_col is None:
        id_col = next((c for c in cols if c != target and (c.lower() in ("id", "passengerid", "row_id")
                                                              or c.lower().endswith("_id"))), None)
    return {"test": test, "sample": sample, "target": target, "id": id_col, "cols": cols, "df": df}


def task_type(y: pd.Series) -> str:
    n = y.nunique(dropna=True)
    if n == 2:
        return "binary"
    if not pd.api.types.is_numeric_dtype(y) or (pd.api.types.is_integer_dtype(y) and n <= 20):
        return "multiclass"
    return "regression"


def cli_overrides(args) -> dict:
    """Only flags the user actually passed (None = not passed) — they beat env and config.yaml."""
    return {k: getattr(args, k, None) for k in cfgmod.CLI_MAP}


def apply_llm(cfg: cfgmod.Config) -> None:
    llm.configure(cfg.model.tag, cfg.model.host, cfg.model.temperature)


# ---- problem / workspace setup ---------------------------------------------------------------------

def build_problem(train: Path, args, goal: str | None = None) -> Problem:
    info = infer(train)
    interactive = not getattr(args, "yes", False)
    target = getattr(args, "target", None) or info["target"]
    if interactive and not getattr(args, "target", None):
        target = Prompt.ask("Target column", choices=info["cols"], default=target, show_choices=False)
    tt = task_type(info["df"][target])
    metric = getattr(args, "metric", None) or METRICS[tt][0]
    if interactive and not getattr(args, "metric", None):
        metric = Prompt.ask(f"Metric ({tt})", default=metric)
    metric = normalize_metric(metric)
    goal = goal or getattr(args, "goal", None) or f"Predict {target} from {train.name}"
    return Problem(goal=goal, target=target, metric=metric, direction="minimize" if metric.lower() in MIN_METRICS else "maximize",
                   task_type=tt, train_path=str(train.resolve()),  # type: ignore[arg-type]
                   test_path=getattr(args, "test", None) or info["test"],
                   id_column=getattr(args, "id_col", None) or info["id"],
                   sample_submission_path=getattr(args, "sample_submission", None) or info["sample"])


def init_workspace(problem: Problem, cfg: cfgmod.Config) -> Workspace:
    ws = Workspace.for_data(problem.train_path).make()
    shutil.copy(Path(__file__).with_name("mlkit_template.py"), ws.root / "mlkit.py")
    for stale in (ws.control_path, ws.lb_path):                  # a fresh run must not inherit old routing state
        stale.unlink(missing_ok=True)
    save(ws, Ledger(problem=problem, env=Env(seed=cfg.seed, started_at=time.time(),
                                             library_versions=_versions(), budget=cfg.budget)))
    cfgmod.save(ws.root, cfg)
    return ws


def summarize(p: Problem, ws: Workspace, cfg: cfgmod.Config) -> None:
    extras = [f"workers={cfg.parallel.workers}", f"model={cfg.model.tag}",
              f"budget={cfg.budget.max_experiments} exp / {cfg.budget.max_minutes} min"]
    if cfg.kaggle.competition:
        extras.append(f"kaggle={cfg.kaggle.competition}{' (submit)' if cfg.kaggle.submit else ''}")
    rows = [f"[bold]goal[/]      {p.goal}", f"[bold]train[/]     {p.train_path}",
            f"[bold]test[/]      {p.test_path or '-'}", f"[bold]target[/]    {p.target}   ({p.task_type}, {p.metric}, {p.direction})",
            f"[bold]id column[/] {p.id_column or '-'}", f"[bold]workspace[/] {ws.root}", f"[bold]config[/]    {', '.join(extras)}"]
    ui.console.print(Panel("\n".join(rows), title="plan", border_style="cyan"))


def start(train: Path, args, goal: str | None = None) -> Workspace | None:
    if not train.exists():
        ui.error(f"file not found: {train}")
        return None
    ws = Workspace.for_data(train)
    if ws.ledger_path.exists() and not getattr(args, "fresh", False):
        L = load(ws)
        if L.status != "done" and (getattr(args, "yes", False) or Confirm.ask(
                f"A previous run exists here ({L.status}). Resume it?", default=True)):
            return resume(ws.root, args)
        if L.status == "done" and not getattr(args, "yes", False) and not Confirm.ask(
                "A finished run exists here. Start a fresh one (the old ledger is overwritten)?", default=False):
            return ws
    cfg = cfgmod.resolve(cli_overrides(args), data_dir=train.parent, explicit=getattr(args, "config", None))
    apply_llm(cfg)
    problem = build_problem(train, args, goal)
    if not shutil.which("ollama") and not cfg.model.host and not os.getenv("OLLAMA_HOST"):
        ui.warn("no local `ollama` found and no host configured — set model.host / OLLAMA_HOST (and OLLAMA_API_KEY) for a remote model")
    ws = init_workspace(problem, cfg)
    summarize(problem, ws, cfg)
    if not getattr(args, "yes", False) and not Confirm.ask("Start?", default=True):
        return ws
    run_pipeline(ws)
    status(ws.root)
    return ws


def resume(path, args=None, more: int | None = None, minutes: int | None = None) -> Workspace | None:
    try:
        ws = Workspace.resolve(path)
    except FileNotFoundError as e:
        ui.error(str(e))
        return None
    more = more if more is not None else getattr(args, "more", None)
    minutes = minutes if minutes is not None else getattr(args, "minutes", None)
    cfg = cfgmod.override(cfgmod.load_workspace(ws.root), cli_overrides(args) if args else None)
    apply_llm(cfg)
    L = load(ws)
    if L.status == "done" and not (more or minutes):
        ui.warn("this run already finished. Re-open it with:  mlagent resume <path> --more 5")
        return ws
    shutil.copy(Path(__file__).with_name("mlkit_template.py"), ws.root / "mlkit.py")
    with session(ws) as L:
        L.env.budget = cfg.budget                                   # config.yaml edits take effect on resume
        if more:
            L.env.budget.max_experiments = experiments_used(L) + more
        if minutes:
            L.env.budget.max_minutes = minutes
        L.status, L.stop_reason, L.env.started_at = "running", None, time.time()   # fresh clock
        cfg.budget = L.env.budget
    cfgmod.save(ws.root, cfg)
    if more or minutes:                                              # re-open the loop: forget the tuner stage
        ctl = load_control(ws)
        ctl.update(tuned=False, tuner_ids=[], tuner_retry_used=False, reentry_left=0, analyzer_empty=False,
                   runs_since_analysis=0, crashes=[])
        ws.control_path.write_text(json.dumps(ctl), encoding="utf-8")
    ui.say("Controller", f"resuming {ws.root}")
    run_pipeline(ws)
    status(ws.root)
    return ws


def lb_command(path, score: float | None, args) -> None:
    ws = Workspace.resolve(path)
    if score is not None:
        lb.record(ws, score, note="manual")
        return
    if getattr(args, "submit", None):
        cfg = cfgmod.override(cfgmod.load_workspace(ws.root), {"kaggle": getattr(args, "kaggle", None), "submit": True})
        if not cfg.kaggle.competition:
            ui.error("pass --kaggle <competition> (or set kaggle.competition in config.yaml)")
            return
        cfgmod.save(ws.root, cfg)
        lb.run(ws)
        return
    ui.console.print(ws.lb_path.read_text(encoding="utf-8") if ws.lb_path.exists() else "no leaderboard results yet")


def status(path) -> None:
    try:
        ws = Workspace.resolve(path)
        L = load(ws)
    except Exception as e:                                       # noqa: BLE001
        ui.error(f"cannot read workspace: {e}")
        return
    ui.console.print(f"[bold]{ws.root}[/]\nstatus: {L.status}   stop: {L.stop_reason}   queue: "
                     f"{sum(q.status == 'pending' for q in L.queue)} pending / {len(L.queue)} total")
    ui.runs_table(L)
    if L.final:
        ui.console.print(f"ensemble: {L.final.ensemble}\nsubmission: {L.final.submission_path}  checks: {L.final.checks}\n"
                         f"report: {L.final.report_path}")
    if ws.lb_path.exists():
        ui.console.print(f"leaderboard: {ws.lb_path.read_text(encoding="utf-8")[:600]}")


# ---- interactive session --------------------------------------------------------------------------------

HELP = """[bold]Commands[/]
  /run <train.csv> [goal...]   start on a dataset (workspace is created beside it)
  /resume [path] [--more N]    continue the last or a given workspace (--more re-opens a finished run)
  /status [path]               runs, queue, final result
  /report                      print report.md of the current workspace
  /lb <score>                  record the public leaderboard score of the last run and check the CV/LB gap
  /config                      show the resolved config of the current workspace
  /graph                       print the agent graph (mermaid)
  /model                       show the LLM model in use
  /exit
You can also just type a CSV path, or a request such as: [dim]predict Survived using data/train.csv, maximize accuracy[/]"""


def _parse_free_text(text: str) -> tuple[Path | None, str | None]:
    m = re.search(r'["\']([^"\']+\.csv)["\']', text) or re.search(r"([^\s\"']+\.csv)", text)
    if not m:
        return None, None
    path = Path(m.group(1)).expanduser()
    goal = text.replace(m.group(0), path.name).strip()
    return path, (goal if goal != path.name else None)


def repl() -> None:
    from .llm import get_llm
    cfg = cfgmod.resolve()
    apply_llm(cfg)
    ui.banner(get_llm().model)
    args = argparse.Namespace(yes=False)
    current: Workspace | None = None
    while True:
        try:
            line = Prompt.ask("[bold cyan]mlagent[/]").strip()
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            return
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            return
        if cmd == "/help":
            ui.console.print(HELP)
        elif cmd == "/model":
            ui.console.print(get_llm().model)
        elif cmd == "/graph":
            ui.console.print(mermaid())
        elif cmd == "/config":
            ui.console.print((current.root / "config.yaml").read_text(encoding="utf-8") if current else cfgmod.TEMPLATE)
        elif cmd == "/status":
            status(rest or (current.root if current else "."))
        elif cmd == "/report":
            p = (current.report_path if current else None) or Path(rest or ".") / "report.md"
            ui.console.print(p.read_text(encoding="utf-8") if p.exists() else "no report yet")
        elif cmd == "/lb":
            try:
                lb_command(current.root if current else ".", float(rest), args)
            except (ValueError, RuntimeError, AttributeError) as e:
                ui.error(f"usage: /lb <score> after a finished run ({e})")
        elif cmd == "/resume":
            m = re.search(r"--more\s+(\d+)", rest)
            p = re.sub(r"--more\s+\d+", "", rest).strip()
            current = resume(p or (current.root if current else "."), None, more=int(m.group(1)) if m else None) or current
        else:
            text = rest if cmd == "/run" else line
            path, goal = _parse_free_text(text)
            if path is None:
                ui.warn("I need a CSV path, e.g.  /run data/train.csv")
                continue
            current = start(path, args, goal) or current


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mlagent", description="CLI multi-agent ML engineer")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--config", help="path to a config yaml")
        p.add_argument("--model", help="Ollama model tag (overrides env + config)")
        p.add_argument("--workers", type=int, help="run N experiments concurrently")
        p.add_argument("--max-experiments", type=int, dest="max_experiments")
        p.add_argument("--max-minutes", type=int, dest="max_minutes")
        p.add_argument("--seed", type=int)
        p.add_argument("--target-score", type=float, dest="target_score", help="stop early when the best CV reaches it")
        p.add_argument("--kaggle", help="competition slug, e.g. titanic")
        p.add_argument("--submit", action="store_true", default=None, help="submit to Kaggle and check the CV/LB gap")

    r = sub.add_parser("run", help="run on a dataset")
    r.add_argument("train")
    r.add_argument("--goal")
    r.add_argument("--target")
    r.add_argument("--metric")
    r.add_argument("--test")
    r.add_argument("--id-col", dest="id_col")
    r.add_argument("--sample-submission", dest="sample_submission")
    r.add_argument("--yes", "-y", action="store_true", help="never prompt; use inferred values")
    r.add_argument("--fresh", action="store_true", help="ignore an existing workspace and start over")
    common(r)
    rs = sub.add_parser("resume", help="continue a stopped run, or extend a finished one")
    rs.add_argument("path", nargs="?", default=".")
    rs.add_argument("--more", type=int, help="allow N more experiments (re-opens a finished run)")
    rs.add_argument("--minutes", type=int, help="new time budget in minutes, counted from now")
    common(rs)
    sub.add_parser("status").add_argument("path", nargs="?", default=".")
    lp = sub.add_parser("lb", help="CV vs public leaderboard check")
    lp.add_argument("path", nargs="?", default=".")
    lp.add_argument("--score", type=float, help="public LB score to record")
    lp.add_argument("--submit", action="store_true", default=None)
    lp.add_argument("--kaggle")
    cp = sub.add_parser("config", help="config helpers")
    cp.add_argument("action", choices=["init"])
    cp.add_argument("path", nargs="?", default="mlagent.yaml")
    doc = sub.add_parser("doctor", help="diagnose environment, dependencies, GPU, Ollama")
    doc.add_argument("path", nargs="?", default=None, help="optional path to verify write access")
    doc.add_argument("--model", help="model tag to check (e.g. gemma4:31b-cloud)")
    doc.add_argument("--check-json", action="store_true", help="test JSON round-trip with LLM")
    sub.add_parser("graph")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        start(Path(a.train).expanduser(), a)
    elif a.cmd == "resume":
        resume(a.path, a)
    elif a.cmd == "status":
        status(a.path)
    elif a.cmd == "lb":
        lb_command(a.path, a.score, a)
    elif a.cmd == "doctor":
        ok = doctor.run_doctor(a.path, a.model, a.check_json)
        sys.exit(0 if ok else 1)
    elif a.cmd == "config":
        p = Path(a.path)
        if p.exists():
            ui.error(f"{p} already exists")
        else:
            p.write_text(cfgmod.TEMPLATE, encoding="utf-8")
            ui.ok(f"wrote {p}")
    elif a.cmd == "graph":
        print(mermaid())
    else:
        repl()


if __name__ == "__main__":
    main(sys.argv[1:])
