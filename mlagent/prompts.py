"""Prompt loader and exporter.

Each agent module in mlagent/agents/<agent>.py defines its own prompts directly in code via PROMPTS.
Workspace overrides (optional): <workspace>/prompts/<agent>.md can still override specific sections.
"""

from __future__ import annotations

import importlib
import re
from functools import lru_cache
from pathlib import Path

_HDR = re.compile(r"^=== (\w+) ===[ \t]*$", re.M)

AGENT_NAMES = (
    "profiler",
    "strategist",
    "coder",
    "experimenter",
    "validator",
    "analyzer",
    "tuner",
    "docs",
    "finisher",
)


def _parse(text: str) -> dict[str, str]:
    parts = _HDR.split(text)                      # [preamble, name, body, name, body, ...]
    return {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}


@lru_cache(maxsize=None)
def _default(agent: str) -> dict[str, str]:
    try:
        mod = importlib.import_module(f"mlagent.agents.{agent}")
        return dict(getattr(mod, "PROMPTS", {}))
    except Exception:
        return {}


def sections(agent: str, ws=None) -> dict[str, str]:
    out = dict(_default(agent))
    if ws is not None:
        p = Path(ws.root) / "prompts" / f"{agent}.md"
        if p.exists():
            out.update(_parse(p.read_text(encoding="utf-8")))
    return out


def get(agent: str, section: str, ws=None, **kw) -> str:
    text = sections(agent, ws)[section]
    for k, v in kw.items():
        text = text.replace(f"<<{k}>>", str(v))
    return text.strip()


def export(ws) -> list[Path]:
    dest = Path(ws.root) / "prompts"
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for agent in AGENT_NAMES:
        secs = _default(agent)
        text = "\n".join(f"=== {k} ===\n{v}" for k, v in secs.items()) + "\n"
        dest_file = dest / f"{agent}.md"
        dest_file.write_text(text, encoding="utf-8")
        out.append(dest_file)
    return out
