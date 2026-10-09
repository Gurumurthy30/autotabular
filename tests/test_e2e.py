"""End-to-end tests: the whole LangGraph with a scripted fake LLM (no model needed).  pytest -q tests/"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import FakeLLM, make_data  # noqa: E402

from mlagent import cli, llm  # noqa: E402
from mlagent import config as cfgmod  # noqa: E402
from mlagent import leaderboard_check as lb
from mlagent.ledger import Workspace, experiments_used, load, load_control  # noqa: E402


def ns(**kw):
    base = {
        "yes": True, "fresh": True, "max_experiments": 6, "max_minutes": 30, "seed": None, "workers": None,
        "model": None, "target": None, "metric": None, "goal": None, "test": None, "id_col": None,
        "sample_submission": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def run(tmp_path, kind="plain", fake=None, **kw):
    train = make_data(tmp_path / "d", kind)
    llm.set_llm(fake or FakeLLM())
    cli.start(train, ns(**kw))
    ws = Workspace.for_data(train)
    return ws, load(ws)


@pytest.fixture(autouse=True)
def _reset():
    yield
    llm.set_llm(None)


@pytest.mark.slow
def test_full_pipeline_docs_lightgbm_optuna_ensembles(tmp_path):
    fake = FakeLLM(api_exp="e003", lightgbm_hyp=True, optuna_tuner=True)
    ws, L = run(tmp_path, fake=fake)
    assert L.status == "done" and L.final.submission_path
    assert all(L.final.checks.values()), L.final.checks
    # docs: reactive sheet for the API error + proactive sheet for lightgbm (real introspection of the installed lib)
    assert "sklearn" in L.docs and "lightgbm" in L.docs and "optuna" in L.docs
    # real Optuna tuner path
    tuned = [r for r in L.runs if r.source == "tuner"]
    assert tuned and all(r.error is None for r in tuned), [r.error for r in tuned]
    assert any(v.exp_id.startswith("t_") and v.verdict == "approve" for v in L.validation)
    # real LightGBM experiment ran
    assert any(r.exp_id == "e002" and r.error is None for r in L.runs)
    # ensemble: all enabled methods were scored on the same rows
    cand = L.final.ensemble["candidates"]
    assert {"single", "greedy"} <= set(cand), cand
    # persisted routing state: the API retry is remembered
    assert load_control(ws)["retries"].get("e003") == 1
    # crashed rows carry no verdict in the table data
    from mlagent.ledger import run_rows
    assert all(r["verdict"] is None for r in run_rows(L) if r["error"])


@pytest.mark.parametrize("kind,cv", [
    ("time", {"kind": "time", "n_splits": 4, "time_col": "date"}),
    ("group", {"kind": "group", "n_splits": 4, "group_col": "grp"}),
])
def test_time_and_group_cv_end_to_end(tmp_path, kind, cv):
    ws, L = run(tmp_path, kind, FakeLLM(cv=cv), max_experiments=4)
    assert L.status == "done" and L.final.submission_path and all(L.final.checks.values())
    assert L.strategy.validation.kind == kind
    df = pd.read_csv(L.problem.train_path)
    folds = json.loads(ws.folds_path.read_text())["folds"]
    ho = np.array(json.loads(ws.holdout_path.read_text())["indices"])
    pool = np.setdiff1d(np.arange(len(df)), ho)
    for f in folds:
        tr, va = np.array(f["train"]), np.array(f["valid"])
        assert not set(tr) & set(va) and not set(tr) & set(ho) and not set(va) & set(ho)
        if kind == "time":                       # no peeking into the future, holdout is the latest slice
            assert tr.max() < va.min() and va.max() < ho.min()
        else:                                    # groups never straddle train / valid / holdout
            g = df.grp.to_numpy()
            assert not set(g[tr]) & set(g[va]) and not set(g[pool]) & set(g[ho])
    assert all(v.verdict == "approve" for v in L.validation if not v.exp_id.startswith("t_")), \
        [(v.exp_id, v.reasons) for v in L.validation]


def test_parallel_workers_share_one_ledger(tmp_path, capsys):
    ws, L = run(tmp_path, fake=FakeLLM(), workers=2, max_experiments=6)
    assert L.status == "done"
    assert "in parallel" in capsys.readouterr().out
    ok = [r.exp_id for r in L.runs if r.source == "experimenter" and not r.error]
    assert len(ok) == len(set(ok)) >= 4                       # no lost / duplicated writes under threads
    assert json.loads(ws.ledger_path.read_text())["runs"]       # file is valid JSON
    assert cfgmod.load_workspace(ws.root).parallel.workers == 2


def test_oom_downgrades_to_level_two_then_succeeds(tmp_path):
    ws, L = run(tmp_path, fake=FakeLLM(oom_exp="e002", oom_until=2), max_experiments=3)
    e2 = [r for r in L.runs if r.exp_id == "e002"]
    assert [bool(r.error) for r in e2] == [True, True, False], [r.error for r in e2]
    assert next(q for q in L.queue if q.id == "e002").params["downgrade"] == 2
    assert load_control(ws)["oom_retries"] == {"e002": 2}
    assert L.status == "done"


def test_rate_limit_stops_gracefully_then_resume_finishes(tmp_path):
    ws, L = run(tmp_path, fake=FakeLLM(rate_limit_after=13), max_experiments=6)
    assert L.status == "stopped" and L.stop_reason == "rate_limit"
    assert ws.report_path.exists() and L.final is not None            # Finisher still reported what existed
    before = experiments_used(L)
    assert 1 <= before < 6
    llm.set_llm(FakeLLM())                                            # quota is back
    cli.resume(ws.root, ns(more=None, minutes=None))
    L2 = load(ws)
    assert L2.status == "done" and experiments_used(L2) > before
    assert not any(q.status == "running" for q in L2.queue)


@pytest.mark.slow
def test_resume_more_reopens_a_finished_run(tmp_path):
    ws, L = run(tmp_path, max_experiments=3)
    n = experiments_used(L)
    assert L.status == "done" and load_control(ws)["tuned"] is True
    llm.set_llm(FakeLLM())
    cli.resume(ws.root, None, more=2)
    L2 = load(ws)
    assert L2.status == "done" and experiments_used(L2) >= n + 1       # the loop really re-ran
    assert any(q.status == "done" for q in L2.queue[n:])


def test_repl_session_free_text_and_commands(tmp_path, monkeypatch, capsys):
    train = make_data(tmp_path / "d")
    (train.parent / "mlagent.yaml").write_text("budget:\n  max_experiments: 3\n  tune_top_n: 1\nparallel:\n  workers: 1\n")
    llm.set_llm(FakeLLM())
    cmds = ["/help", f"predict Survived using {train} maximize accuracy", "/status", "/report", "/config", "/exit"]

    class P:
        @staticmethod
        def ask(prompt, **kw):
            return cmds.pop(0) if "mlagent" in prompt else kw.get("default")

    class C:
        @staticmethod
        def ask(*a, **kw):
            return True

    monkeypatch.setattr(cli, "Prompt", P)
    monkeypatch.setattr(cli, "Confirm", C)
    cli.repl()
    out = capsys.readouterr().out
    ws = Workspace.for_data(train)
    assert ws.root == train.parent / "mlagent_train"                    # workspace beside the data
    L = load(ws)
    assert L.status == "done" and "maximize accuracy" in L.problem.goal
    assert L.env.budget.max_experiments == 3                            # picked up mlagent.yaml from the data folder
    assert "Commands" in out and "workers" in out and not cmds


class FakeKaggle:
    def __init__(self, score):
        self.score, self.polls, self.sent = score, 0, []

    def submit(self, comp, path, message):
        self.sent.append((comp, path, message))

    def submissions(self, comp):
        self.polls += 1
        row = {"description": self.sent[-1][2], "status": "complete", "publicScore": ""}
        if self.polls >= 2:
            row["publicScore"] = str(self.score)
        return [row]


def test_cv_lb_gap_flag_then_consistent(tmp_path):
    ws, L = run(tmp_path, max_experiments=3)
    cfg = cfgmod.override(cfgmod.load_workspace(ws.root), {"kaggle": "titanic", "submit": True})
    cfgmod.save(ws.root, cfg)
    cv = L.final.ensemble["cv"]
    fk = FakeKaggle(score=0.30)
    e = lb.run(ws, fk, sleep=lambda s: None)
    assert fk.sent and fk.polls == 2 and e["flagged"] and e["direction_of_gap"] == "LB worse than CV"
    L = load(ws)
    assert L.final.checks["cv_lb_gap_ok"] is False and L.analysis[-1].after_exp          # Analyzer ran on the gap
    assert "FLAGGED" in ws.report_path.read_text()
    e2 = lb.run(ws, FakeKaggle(score=cv - 0.005), sleep=lambda s: None)
    assert not e2["flagged"] and load(ws).final.checks["cv_lb_gap_ok"] is True
    assert ws.report_path.read_text().count(lb.MARK) == 1                                 # section replaced, not stacked
    assert len(json.loads(ws.lb_path.read_text())) == 2
    manual = lb.record(ws, 0.99, note="manual")                                           # offline path
    assert manual["flagged"] and manual["direction_of_gap"] == "LB better than CV"
