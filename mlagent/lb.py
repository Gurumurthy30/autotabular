"""CV/LB gap check. Submits the final submission.csv through the Kaggle CLI (or takes a score you
paste in with `mlagent lb <path> --score X`), compares the public LB with our CV and flags a gap larger
than max(3 * cv_std, gap_rel * |cv|). A flagged gap runs the Analyzer (leakage / folds / target
encoding / shift hypotheses) and is written to the report + final.checks['cv_lb_gap_ok'].

Kaggle allows only a few submissions a day, so only the FINAL submission is sent."""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import time
from datetime import datetime
from typing import Protocol

from . import ui
from .agents import analyzer
from .executor import tail
from .ledger import Workspace, best_run, load, session
from .llm import RateLimitError

MARK = "<!-- lb -->"


class Client(Protocol):
    def submit(self, comp: str, path: str, message: str) -> None: ...
    def submissions(self, comp: str) -> list[dict]: ...


class KaggleCLI:
    """Thin wrapper over the official `kaggle` command (needs ~/.kaggle/kaggle.json)."""

    def _run(self, args: list[str]) -> str:
        p = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        if p.returncode != 0:
            raise RuntimeError(tail(p.stderr or p.stdout, 400))
        return p.stdout

    def submit(self, comp, path, message):
        self._run(["competitions", "submit", "-c", comp, "-f", str(path), "-m", message])

    def submissions(self, comp):
        return list(csv.DictReader(io.StringIO(self._run(["competitions", "submissions", "-c", comp, "-v"]))))


def parse_score(row: dict) -> float | None:
    for k, v in row.items():
        if k and k.replace("_", "").lower() == "publicscore" and v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                return None
    return None


def wait_for_score(client: Client, comp: str, message: str, wait: int, interval: int = 10,
                   sleep=time.sleep) -> float | None:
    deadline = time.time() + wait
    while True:
        rows = [r for r in client.submissions(comp) if (r.get("description") or r.get("Description") or "") == message]
        for r in rows:
            s = parse_score(r)
            status = (r.get("status") or r.get("Status") or "").lower()
            if s is not None and "error" not in status:
                return s
        if time.time() >= deadline:
            return None
        sleep(interval)


def gap_check(cv: float, std: float, lb: float, direction: str, rel: float) -> dict:
    """gap > 0 means the LB is WORSE than CV (the dangerous direction: leakage / overfit to CV)."""
    gap = (cv - lb) if direction == "maximize" else (lb - cv)
    thr = max(3 * (std or 0.0), rel * abs(cv))
    return {"cv": round(cv, 5), "lb": round(lb, 5), "gap_worse": round(gap, 5), "threshold": round(thr, 5),
            "flagged": abs(gap) > thr, "direction_of_gap": "LB worse than CV" if gap > 0 else "LB better than CV"}


def _read(ws: Workspace) -> list[dict]:
    try:
        return json.loads(ws.lb_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _report_section(ws: Workspace, e: dict) -> None:
    body = [MARK, "## Leaderboard check",
            f"- CV **{e['cv']}** vs public LB **{e['lb']}** ({e['direction_of_gap']}, gap {e['gap_worse']}, threshold {e['threshold']})",
            f"- verdict: {'**FLAGGED** — re-check leakage, folds, target encoding, train/test shift' if e['flagged'] else 'consistent'}"]
    old = ws.report_path.read_text(encoding="utf-8") if ws.report_path.exists() else ""
    ws.report_path.write_text(old.split(MARK)[0].rstrip() + "\n\n" + "\n".join(body) + "\n", encoding="utf-8")


def record(ws: Workspace, lb_score: float, note: str = "") -> dict:
    """Compare a leaderboard score (submitted or pasted) with the final CV and persist the outcome."""
    L = load(ws)
    cv = (L.final.ensemble or {}).get("cv") if L.final and L.final.ensemble else None
    b = best_run(L)
    if cv is None and b is None:
        raise RuntimeError("no CV score to compare with: the run has no finished result yet")
    cv = cv if cv is not None else b.cv_mean
    entry = {**gap_check(cv, (b.cv_std if b else 0.0) or 0.0, lb_score, L.problem.direction, ws.config().kaggle.gap_rel),
             "note": note, "time": datetime.now().isoformat(timespec="seconds")}
    ws.lb_path.write_text(json.dumps(_read(ws) + [entry], indent=2), encoding="utf-8")
    with session(ws) as L:
        if L.final:
            L.final.checks["cv_lb_gap_ok"] = not entry["flagged"]
    _report_section(ws, entry)
    (ui.warn if entry["flagged"] else ui.ok)(
        f"CV {entry['cv']} vs LB {entry['lb']}: gap {entry['gap_worse']} (threshold {entry['threshold']}) "
        f"→ {'FLAGGED' if entry['flagged'] else 'consistent'}")
    if entry["flagged"]:
        try:
            analyzer.run(ws, "cv_lb_gap", error=json.dumps(entry))
            ui.say("Analyzer", "cv/lb gap hypotheses queued — `mlagent resume --more 5` to test them")
        except RateLimitError:
            ui.warn("rate limited: could not run the Analyzer on the gap")
    return entry


def run(ws: Workspace, client: Client | None = None, sleep=time.sleep) -> dict | None:
    """Submit the final submission.csv and check the gap. No-op unless kaggle.submit is on."""
    cfg, L = ws.config().kaggle, load(ws)
    if not (cfg.submit and cfg.competition):
        return None
    if L.status == "stopped" or not (L.final and L.final.submission_path):
        ui.warn("skipping Kaggle submission: run stopped early or no submission.csv")
        return None
    client = client or KaggleCLI()
    msg = f"mlagent {(L.final.ensemble or {}).get('method', 'run')} cv={(L.final.ensemble or {}).get('cv')}"
    msg = re.sub(r"\s+", " ", msg)
    try:
        ui.say("Finisher", f"submitting to Kaggle ({cfg.competition}) and waiting up to {cfg.wait_seconds}s for the score")
        client.submit(cfg.competition, L.final.submission_path, msg)
        score = wait_for_score(client, cfg.competition, msg, cfg.wait_seconds, sleep=sleep)
    except Exception as e:                                           # noqa: BLE001 — never lose the run to Kaggle
        ui.warn(f"Kaggle step failed: {e}")
        return None
    if score is None:
        ui.warn("no public score yet; record it later with: mlagent lb <path> --score <value>")
        return None
    return record(ws, score, note=f"kaggle:{cfg.competition}")
