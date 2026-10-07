import os
import sys

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

# Ensure UTF-8 output on Windows.  On a legacy cmd (cp850/cp1252) the symbols
# ● ✓ ✗ would raise UnicodeEncodeError; setting PYTHONIOENCODING here covers
# the current process.  The subprocess env already sets PYTHONUTF8=1 via executor.py.
if sys.platform == "win32" and os.environ.get("PYTHONIOENCODING", "").lower() not in ("utf-8", "utf8"):
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

def _make_console() -> Console:
    try:
        c = Console()
        # Quick smoke-test: can the underlying stream encode the bullet symbol?
        if hasattr(c.file, "encoding") and c.file.encoding:
            "●".encode(c.file.encoding)
        return c
    except (UnicodeEncodeError, LookupError):
        # Fallback: ASCII-safe console (no colour loss, just safe box characters)
        return Console(safe_box=True, highlight=False)


console = _make_console()

_COL = {"Profiler": "cyan", "Strategist": "magenta", "Experimenter": "green", "Validator": "yellow",
        "Analyzer": "blue", "Docs": "bright_black", "Tuner": "bright_magenta",
        "Finisher": "bright_green", "Controller": "white"}


def say(agent: str, msg: str) -> None:
    console.print(f"[bold {_COL.get(agent, 'white')}]●[/] [bold]{agent:<12}[/] {escape(msg)}")


def ok(msg: str) -> None:
    console.print(f"[green]✓[/] {escape(msg)}")


def warn(msg: str) -> None:
    console.print(f"[yellow]![/] {escape(msg)}")


def error(msg: str) -> None:
    console.print(f"[red]✗[/] {escape(msg)}")


def banner(model: str) -> None:
    console.print(Panel.fit("[bold]mlagent[/]  multi-agent ML engineer\n"
                            f"[dim]model: {escape(model)}  ·  /help for commands[/]", border_style="cyan"))


def runs_table(L) -> None:
    from .ledger import verdicts
    v = verdicts(L)
    t = Table(title=f"Runs  ({L.problem.metric}, {L.problem.direction})", header_style="bold")
    for c in ("exp", "src", "cv_mean", "std", "holdout", "verdict", "error"):
        t.add_column(c)
    for r in L.runs:
        fmt = lambda x: "" if x is None else f"{x:.4f}"
        t.add_row(r.exp_id, r.source, fmt(r.cv_mean), fmt(r.cv_std), fmt(r.holdout),
                  "" if r.error else v.get(r.exp_id, ""), ((r.error or "").strip().splitlines() or [""])[-1][:60])
    console.print(t)
