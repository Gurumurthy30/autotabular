"""Docs agent: local introspection (help/signature/docstring of the INSTALLED version) -> cheat sheet in
ledger.docs[library]. Reactive on API errors; proactive only for rare libs. Official web docs are a hook
for later (local introspection is always tried first)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import Field

from .. import ui, webdocs
from ..executor import parse_result, run_script, tail, write_script
from ..ledger import RARE_LIBS, Workspace, load, session
from ..llm import LLMFormatError, RateLimitError, get_llm
from ..state import Base

RARE_OBJECTS = {
    "lightgbm": ["lightgbm.LGBMClassifier", "lightgbm.LGBMClassifier.fit", "lightgbm.early_stopping"],
    "xgboost": ["xgboost.XGBClassifier", "xgboost.XGBClassifier.fit"],
    "catboost": ["catboost.CatBoostClassifier", "catboost.CatBoostClassifier.fit"],
    "optuna": ["optuna.create_study", "optuna.study.Study.optimize"],
}

SYSTEM_Q = ("An ML script failed. From the error and script, name the Python library at fault and the dotted "
            "paths of the objects whose signature/docstring we should inspect (e.g. lightgbm.LGBMClassifier.fit). "
            "Fields: library (string), objects (list of dotted paths).")
SYSTEM_SHEET = ("Write a cheat sheet (max 25 lines, plain text) for a coder using this library version. "
                "ONLY facts present in the inspected signatures/docstrings: correct import, constructor/fit "
                "kwargs, removed or renamed arguments, one minimal usage example. No speculation.")

INTROSPECT = '''
import importlib, inspect, json
out = {}
for path in __OBJECTS__:
    try:
        parts = path.split(".")
        obj = importlib.import_module(parts[0])
        out["__version__:" + parts[0]] = str(getattr(obj, "__version__", "?"))
        for p in parts[1:]:
            obj = getattr(obj, p)
        try: sig = str(inspect.signature(obj))
        except Exception: sig = "?"
        out[path] = {"signature": sig[:1500], "doc": (inspect.getdoc(obj) or "")[:1000]}
    except Exception as e:
        out[path] = {"error": repr(e)}
print("RESULT_JSON: " + json.dumps(out))
'''



class DocQuery(Base):
    library: str = ""
    objects: list[str] = Field(default_factory=list)


def _introspect(ws: Workspace, objects: list[str], tag: str) -> dict:
    code = INTROSPECT.replace("__OBJECTS__", json.dumps(objects))
    res = run_script(ws, write_script(ws, f"docs_{tag}", code), timeout=120)
    return parse_result(res.stdout) or {"error": tail(res.stderr, 300)}


def insufficient(info: dict) -> bool:
    """True if local introspection returned errors or couldn't resolve objects."""
    if not info:
        return True
    obj_items = {k: v for k, v in info.items() if not k.startswith("__")}
    if not obj_items:
        return True
    return any(isinstance(v, dict) and ("error" in v or v.get("signature") == "?") for v in obj_items.values())


def _gather(ws: Workspace, lib: str, objects: list[str], tag: str, force_web: bool = False) -> dict:
    info = _introspect(ws, objects, tag)
    cfg = ws.config().docs
    if cfg.web and (force_web or insufficient(info)):
        for u in webdocs.candidate_urls(lib, objects):
            txt = webdocs.fetch_text(u, timeout=cfg.web_timeout)
            if txt:
                info[f"web:{u}"] = txt
    return info


def _sheet(lib: str, info: dict) -> str:
    raw = json.dumps(info)[:6000]
    try:
        return get_llm().chat(SYSTEM_SHEET, f"Library: {lib}\nInspected:\n{raw}").strip()[:3500]
    except RateLimitError:
        raise
    except Exception:                                                  # noqa: BLE001
        return f"(raw introspection, unsummarised)\n{raw[:2500]}"


def lookup(ws: Workspace, error: str, exp_id: str | None = None) -> str | None:
    """Reactive: an API error happened. Returns the library whose sheet was written/extended."""
    L = load(ws)
    script = ws.scripts / f"{exp_id}.py" if exp_id else None
    code = script.read_text()[-2500:] if script and script.exists() else ""
    try:
        q = get_llm().json(SYSTEM_Q, f"ERROR:\n{tail(error, 1200)}\n\nSCRIPT (tail):\n{code}", DocQuery)
    except LLMFormatError:
        q = DocQuery()
    lib = (q.library or "").split(".")[0].strip()
    if not lib:
        m = re.findall(r"site-packages/(\w+)/", error)
        lib = m[-1] if m else "sklearn"
    objs = [o for o in q.objects if o.startswith(lib)][:5] or [lib]
    ui.say("Docs", f"inspecting installed {lib}: {', '.join(objs)}")
    prior_crashes = [r for r in L.runs if r.exp_id == exp_id and r.error] if exp_id else []
    force_web = len(prior_crashes) >= 2
    sheet = _sheet(lib, _gather(ws, lib, objs, lib, force_web=force_web))
    with session(ws) as L:
        L.docs[lib] = (L.docs[lib] + "\n--- extra lookup ---\n" + sheet)[-4500:] if lib in L.docs else sheet
    return lib


def ensure(ws: Workspace, libs: list[str]) -> None:
    """Proactive: rare libs get a cheat sheet before the first script that uses them."""
    L = load(ws)
    for lib in libs:
        if lib in RARE_LIBS and lib in L.env.library_versions and lib not in L.docs:
            ui.say("Docs", f"building cheat sheet for {lib} (first use)")
            sheet = _sheet(lib, _gather(ws, lib, RARE_OBJECTS[lib], lib))
            with session(ws) as L2:
                L2.docs[lib] = sheet

