"""Terminal UI for mlagent (Rich).

Backward compatible: console, say, ok, warn, error, banner, runs_table and _COL
keep their names and signatures.  New, purely additive helpers at the bottom:
section(), status(), progress(), kv_panel().

Environment switches
    MLAGENT_TIMESTAMPS=0   hide the dim elapsed-time column in say()
    NO_COLOR=1             honoured natively by Rich
"""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from rich.box import ASCII2, ROUNDED, SIMPLE_HEAD
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# --------------------------------------------------------------------------- #
# Encoding safety (Windows)
# --------------------------------------------------------------------------- #
# Setting PYTHONIOENCODING after start-up does not affect the already-open
# stdout, so also reconfigure the live streams where possible.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

# --------------------------------------------------------------------------- #
# Theme: semantic style names instead of colour strings scattered in the code
# --------------------------------------------------------------------------- #
THEME = Theme(
    {
        "ok": "bold green",
        "warn": "bold yellow",
        "err": "bold red",
        "muted": "dim",
        "accent": "bold cyan",
        "best": "bold green",
        "bad": "red",
        "key": "dim",
    }
)


def _unicode_ok(c: Console) -> bool:
    enc = getattr(c.file, "encoding", None)
    if not enc:
        return True
    try:
        "●✓✗·★…".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _make_console() -> Console:
    c = Console(theme=THEME)
    if _unicode_ok(c):
        return c
    # ASCII-safe fallback: safe box characters, no highlighter surprises.
    return Console(theme=THEME, safe_box=True, highlight=False)


console = _make_console()
_UNICODE = _unicode_ok(console)

# Symbols degrade to plain ASCII on legacy terminals instead of crashing.
SYM = (
    {"dot": "●", "ok": "✓", "err": "✗", "warn": "!", "sep": "·", "star": "★", "ell": "…"}
    if _UNICODE
    else {"dot": "*", "ok": "OK", "err": "x", "warn": "!", "sep": "-", "star": "*", "ell": "..."}
)
_BOX = ROUNDED if _UNICODE else ASCII2
# Rich's own "ellipsis" overflow draws "…", which crashes on ASCII-only streams.
_OVF = "ellipsis" if _UNICODE else "crop"

_COL = {
    "Data Profiler": "cyan",
    "Experiment Planner": "magenta",
    "Experiment Runner": "green",
    "Run Validator": "yellow",
    "Results Analyzer": "blue",
    "API Docs Lookup": "bright_black",
    "Hyperparameter Tuner": "bright_magenta",
    "Final Submission": "bright_green",
    "Pipeline Controller": "white",
}

_T0 = time.monotonic()
_SHOW_TS = os.environ.get("MLAGENT_TIMESTAMPS", "1") not in ("0", "false", "no", "")


def _elapsed() -> str:
    s = int(time.monotonic() - _T0)
    return f"{s // 60:02d}:{s % 60:02d}"


# --------------------------------------------------------------------------- #
# Core messages
# --------------------------------------------------------------------------- #
def say(agent: str, msg: str) -> None:
    """One agent line.  Uses a grid so long messages wrap under the message
    column instead of under the bullet."""
    style = _COL.get(agent, "white")
    grid = Table.grid(padding=(0, 1), expand=True)
    if _SHOW_TS:
        grid.add_column(width=5, no_wrap=True, style="muted")
    grid.add_column(width=1, no_wrap=True)
    grid.add_column(width=20, no_wrap=True, overflow=_OVF)
    grid.add_column(ratio=1)

    cells = []
    if _SHOW_TS:
        cells.append(_elapsed())
    cells.append(Text(SYM["dot"], style=f"bold {style}"))
    cells.append(Text(agent, style="bold"))
    cells.append(console.render_str(escape(msg), emoji=False, highlight=True))
    grid.add_row(*cells)
    console.print(grid)


def ok(msg: str) -> None:
    console.print(f"[ok]{SYM['ok']}[/] {escape(msg)}")


def warn(msg: str) -> None:
    console.print(f"[warn]{SYM['warn']}[/] {escape(msg)}")


def error(msg: str) -> None:
    console.print(f"[err]{SYM['err']}[/] {escape(msg)}")


def banner(model: str, subtitle: str | None = None) -> None:
    body = Group(
        Text.assemble(("mlagent", "bold"), "  multi-agent ML engineer"),
        Text.assemble(
            (f"model: {model}  {SYM['sep']}  /help for commands", "muted"),
        ),
        *([Text(subtitle, style="muted")] if subtitle else []),
    )
    console.print(
        Panel.fit(body, border_style="cyan", box=_BOX, padding=(0, 2), title="[accent]ML[/]", title_align="left")
    )


# --------------------------------------------------------------------------- #
# Runs table
# --------------------------------------------------------------------------- #
_GOOD = ("best", "accept", "keep", "improv", "win", "pass", "good", "promot")
_BAD = ("reject", "worse", "fail", "regress", "drop", "bad", "discard")


def _verdict_text(s: str) -> Text:
    low = s.lower()
    if any(k in low for k in _GOOD):
        return Text(s, style="ok")
    if any(k in low for k in _BAD):
        return Text(s, style="bad")
    return Text(s)


def _minimise(direction: object) -> bool:
    d = str(direction).lower()
    return d.startswith(("min", "low")) or d in ("asc", "down")


def runs_table(L) -> None:
    from .ledger import verdicts

    v = verdicts(L)
    runs = list(L.runs)

    if not runs:
        console.print(f"[muted]No runs yet ({escape(str(L.problem.metric))}, {escape(str(L.problem.direction))}).[/]")
        return

    # Best successful run by cv_mean, respecting the metric direction.
    scored = [r for r in runs if r.cv_mean is not None and not r.error]
    best = None
    if scored:
        pick = min if _minimise(L.problem.direction) else max
        best = pick(scored, key=lambda r: r.cv_mean)
    n_err = sum(1 for r in runs if r.error)

    caption = f"{len(runs)} runs"
    if n_err:
        caption += f"  {SYM['sep']}  {n_err} failed"
    if best is not None:
        caption += f"  {SYM['sep']}  best {best.exp_id} = {best.cv_mean:.4f}"

    t = Table(
        title=f"Runs  ({escape(str(L.problem.metric))}, {escape(str(L.problem.direction))})",
        caption=caption,
        caption_style="muted",
        header_style="bold",
        box=SIMPLE_HEAD if _UNICODE else ASCII2,
        row_styles=["", "dim"] if len(runs) > 8 else None,
        pad_edge=False,
    )
    # Only the error column may shrink; metric columns stay readable on narrow terminals.
    t.add_column("exp", no_wrap=True)
    t.add_column("src", no_wrap=True)
    t.add_column("cv_mean", justify="right", no_wrap=True, min_width=7)
    t.add_column("std", justify="right", style="muted", no_wrap=True, min_width=6)
    t.add_column("holdout", justify="right", no_wrap=True, min_width=7)
    t.add_column("verdict", no_wrap=True)
    t.add_column("error", max_width=40, min_width=10)

    def fmt(x) -> str:
        return "" if x is None else f"{x:.4f}"

    for r in runs:
        is_best = best is not None and r.exp_id == best.exp_id
        exp = Text(f"{SYM['star']} {r.exp_id}" if is_best else f"  {r.exp_id}", style="best" if is_best else "")
        cv = Text(fmt(r.cv_mean), style="best" if is_best else "")
        last_err = ((r.error or "").strip().splitlines() or [""])[-1]
        err = Text(last_err, style="bad", no_wrap=True, overflow=_OVF)
        verdict = "" if r.error else v.get(r.exp_id, "")
        t.add_row(exp, str(r.source), cv, fmt(r.cv_std), fmt(r.holdout), _verdict_text(verdict), err)
    console.print(t)


# --------------------------------------------------------------------------- #
# New, additive helpers
# --------------------------------------------------------------------------- #
def section(title: str, agent: str | None = None) -> None:
    """Labelled divider between pipeline stages."""
    style = _COL.get(agent, "cyan") if agent else "cyan"
    console.print(Rule(Text(title, style=f"bold {style}"), style=style, characters="─" if _UNICODE else "-"))


@contextmanager
def status(msg: str, agent: str | None = None) -> Iterator[None]:
    """Spinner for long steps:  with status("Running CV", "Experiment Runner"): ..."""
    style = _COL.get(agent, "cyan") if agent else "cyan"
    with console.status(
        f"[{style}]{escape(msg)}[/]",
        spinner="dots" if _UNICODE else "line",
        spinner_style=style,
    ):
        yield


def progress(transient: bool = True) -> Progress:
    """Multi-task progress bar bound to the shared console (no output tearing)."""
    return Progress(
        SpinnerColumn("dots" if _UNICODE else "line"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=transient,
    )


def kv_panel(title: str, data: Mapping[str, object], border: str = "cyan") -> None:
    """Compact key/value box, e.g. data profile or final summary."""
    g = Table.grid(padding=(0, 2))
    g.add_column(style="key", no_wrap=True)
    g.add_column()
    for k, val in data.items():
        g.add_row(str(k), console.render_str(escape(str(val)), emoji=False, highlight=True))
    console.print(Panel.fit(g, title=f"[bold]{escape(title)}[/]", title_align="left", border_style=border, box=_BOX))