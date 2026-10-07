"""Prompt loader. One markdown file per agent in mlagent/prompts/, split into `=== name ===` sections.

Override per project: put <workspace>/prompts/<agent>.md next to the data — only the sections you
define there replace the defaults (the rest fall back).  `mlagent prompts export <path>` copies the
defaults into the workspace so you can edit them.  Placeholders look like <<name>>.
"""

from __future__ import annotations

import re
import shutil
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).with_name("prompts")
_HDR = re.compile(r"^=== (\w+) ===[ \t]*$", re.M)


def _parse(text: str) -> dict[str, str]:
    parts = _HDR.split(text)                      # [preamble, name, body, name, body, ...]
    return {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}


@lru_cache(maxsize=None)
def _default(agent: str) -> dict[str, str]:
    return _parse((_DIR / f"{agent}.md").read_text(encoding="utf-8"))


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
    for f in sorted(_DIR.glob("*.md")):
        shutil.copy(f, dest / f.name)
        out.append(dest / f.name)
    return out
