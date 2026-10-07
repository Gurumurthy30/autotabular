"""Unit tests for the rule-bearing modules.  pytest -q tests/test_units.py"""

import argparse
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import FakeLLM, make_data  # noqa: E402
from mlagent import cli, ensemble, hw, llm, prompts, webdocs  # noqa: E402
from mlagent import config as cfgmod  # noqa: E402
from mlagent.agents import docs  # noqa: E402
from mlagent.agents.strategist import CVOut, _sanitize_cv, goal_target, make_folds  # noqa: E402
from mlagent.ledger import (add_queue_items, beats, consecutive_crashes, counted_gain_flags,  # noqa: E402
                            is_api_error, is_oom_error, load, plateau, session, stop_reason)
from mlagent.state import (Budget, CVScheme, Env, Ledger, Problem, QueueItem, Run, Strategy,  # noqa: E402
                           ValidationRecord)


@pytest.fixture(autouse=True)
def _reset():
    yield
    llm.set_llm(None)


def mini_ws(tmp_path, **cli_over):
    train = make_data(tmp_path / "d")
    cfg = cfgmod.resolve(cli_over)
    p = cli.build_problem(train, argparse.Namespace(yes=True))
    ws = cli.init_workspace(p, cfg)
    df = pd.read_csv(train)
    folds, ho = make_folds(df, p, CVScheme(kind="stratified", n_splits=4), 42)
    ws.folds_path.write_text(json.dumps({"folds": folds}))
    ws.holdout_path.write_text(json.dumps({"indices": ho}))
    return ws


# ---- config ---------------------------------------------------------------------------------------------

def test_config_precedence_yaml_env_cli(tmp_path, monkeypatch):
    y = tmp_path / "c.yaml"
    y.write_text("model: {tag: from-yaml}\nseed: 7\nbudget: {max_experiments: 9}\nparallel: {workers: 3}\n")
    monkeypatch.delenv("MLAGENT_MODEL", raising=False)
    c = cfgmod.resolve({}, explicit=y)
    assert (c.model.tag, c.seed, c.budget.max_experiments, c.parallel.workers) == ("from-yaml", 7, 9, 3)
    monkeypatch.setenv("MLAGENT_MODEL", "from-env")
    assert cfgmod.resolve({}, explicit=y).model.tag == "from-env"                       # env > yaml
    c = cfgmod.resolve({"model": "from-cli", "max_experiments": 2, "workers": 0}, explicit=y)
    assert c.model.tag == "from-cli" and c.budget.max_experiments == 2                  # cli > env
    assert c.parallel.workers == 1                                                      # clamped
    assert yaml_defaults_match_template()


def yaml_defaults_match_template():
    import yaml
    return cfgmod.Config.model_validate(yaml.safe_load(cfgmod.TEMPLATE)) == cfgmod.Config()


def test_config_is_frozen_into_workspace_and_found_in_data_dir(tmp_path):
    train = make_data(tmp_path / "d")
    (train.parent / "mlagent.yaml").write_text("seed: 5\nkaggle: {competition: titanic}\n")
    cfg = cfgmod.resolve({}, data_dir=train.parent)
    assert cfg.seed == 5
    ws = cli.init_workspace(cli.build_problem(train, argparse.Namespace(yes=True)), cfg)
    assert ws.config().kaggle.competition == "titanic" and load(ws).env.seed == 5


# ---- prompts --------------------------------------------------------------------------------------------

def test_prompts_defaults_placeholders_and_workspace_override(tmp_path):
    ws = mini_ws(tmp_path)
    assert "KIT API" in prompts.get("coder", "kit_api", ws)
    t = prompts.get("experimenter", "task_fresh", ws, exp_id="e9", hypothesis="h", change="c", params={})
    assert "Experiment e9" in t and "<<" not in t
    for agent in ("coder", "profiler", "strategist", "experimenter", "validator", "analyzer", "docs", "tuner", "finisher"):
        assert prompts.sections(agent), agent
    (ws.root / "prompts").mkdir()
    (ws.root / "prompts" / "finisher.md").write_text("=== summary_system ===\nCUSTOM SUMMARY PROMPT\n")
    assert prompts.get("finisher", "summary_system", ws) == "CUSTOM SUMMARY PROMPT"
    assert "Fields:" in prompts.get("analyzer", "system", ws)                           # untouched agents fall back
    assert len(prompts.export(ws)) == 9


# ---- ledger rules ----------------------------------------------------------------------------------------

def _ledger(direction="maximize"):
    p = Problem(goal="g", target="y", metric="accuracy", direction=direction, task_type="binary", train_path="x.csv")
    return Ledger(problem=p, env=Env(budget=Budget(max_experiments=5, plateau_n=2)),
                  strategy=Strategy(validation=CVScheme(kind="kfold")))


def _add_run(L, eid, cv, std=0.01, error=None, verdict="approve", src="experimenter"):
    L.runs.append(Run(exp_id=eid, cv_mean=None if error else cv, cv_std=std, error=error, source=src))
    if not error:
        L.validation.append(ValidationRecord(exp_id=eid, verdict=verdict))


def test_dedupe_gain_plateau_budget_and_crash_rules():
    L = _ledger()
    mk = lambda h, c, **p: QueueItem(hypothesis=h, change=c, kind="model", params=p)           # noqa: E731
    ids = add_queue_items(L, [mk("a", "x", n=1), mk("A!", "y", n=1), mk("b", "x", n=1), mk("c", "z", n=2)], "strategist")
    assert ids == ["e001", "e002"]                                           # "A!" (same hypothesis) and "b" (same change+params) dropped
    assert beats(0.80, 0.78, 0.01, "maximize") and not beats(0.785, 0.78, 0.01, "maximize")
    assert beats(0.50, 0.55, 0.01, "minimize")
    for e, cv in [("e1", 0.70), ("e2", 0.705), ("e3", 0.706)]:
        _add_run(L, e, cv)
    assert counted_gain_flags(L) == [True, False, False] and plateau(L, 2)
    _add_run(L, "e4", 0.80)
    assert counted_gain_flags(L)[-1] and not plateau(L, 2)
    _add_run(L, "e5", 0.99, verdict="reject")                               # rejected runs never count
    assert counted_gain_flags(L)[-1] is True and len(counted_gain_flags(L)) == 4
    assert stop_reason(L) == "budget"                                       # 5 runs used = max_experiments
    L2 = _ledger()
    for e in ("a", "a", "b"):                                               # Docs-assisted retries of one experiment count once
        L2.runs.append(Run(exp_id=e, error="boom"))
    assert consecutive_crashes(L2) == 2


def test_error_classifiers():
    assert is_oom_error("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate")
    assert is_oom_error("MemoryError: Unable to allocate 9 GiB") and not is_oom_error("KeyError: 'Age'")
    assert is_api_error("TypeError: fit() got an unexpected keyword argument 'verbose'")
    assert not is_api_error("ValueError: could not convert string to float")


def test_goal_target_parser():
    assert goal_target("reach 0.85 accuracy") == 0.85 and goal_target("at least 85%") == 0.85
    assert goal_target("rmse below 0.4") == 0.4 and goal_target("predict survival") is None


def test_sanitize_cv_overrides_invalid_llm_choices():
    p = Problem(goal="g", target="y", metric="accuracy", direction="maximize", task_type="binary", train_path="x")
    assert _sanitize_cv(CVOut(kind="group", group_col="nope"), p, ["a"], 600).kind == "stratified"
    assert _sanitize_cv(CVOut(kind="kfold"), p, ["a"], 600).kind == "stratified"
    assert _sanitize_cv(CVOut(kind="time", time_col="t"), p, ["t"], 600).kind == "time"
    assert _sanitize_cv(CVOut(n_splits=99), p, ["a"], 600).n_splits == 10


# ---- ensembling ------------------------------------------------------------------------------------------

def test_all_ensemble_methods_on_synthetic_oofs(tmp_path):
    ws = mini_ws(tmp_path)
    kit = ws.load_mlkit()
    y, rng = kit.get_y(), np.random.default_rng(3)
    ho = np.array(json.loads(ws.holdout_path.read_text())["indices"])
    oofs = {}
    for name, noise in (("a", 0.9), ("b", 0.9), ("c", 1.4)):               # independent noisy views of the target
        p = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, noise * 0.25, len(y)), 0.01, 0.99)
        p[ho] = np.nan
        oofs[name] = p
    res, common = ensemble.evaluate(oofs, y, kit, "maximize", kit.folds(), ["greedy", "voting", "stacking"], top_k=3)
    assert {r.name for r in res} == {"single", "greedy", "voting", "stacking"}
    choice, table = ensemble.choose(res, y, kit, "maximize", common)
    assert table[choice.name] >= table["single"] and common.sum() == len(y) - len(ho)
    test = {k: rng.random(25) for k in oofs}
    for r in res:                                                           # every method can score unseen rows
        out = r.predict(test) if r.name != "single" else r.predict(test)
        assert out.shape == (25,) and np.isfinite(out).all()
    vote = next(r for r in res if r.name == "voting")
    assert set(np.unique(vote.predict({k: (v > 0.5).astype(float) for k, v in test.items()}))) <= {0.0, 1/3, 2/3, 1.0}  # hard votes (accuracy)


# ---- web docs --------------------------------------------------------------------------------------------

@pytest.fixture
def docs_server(tmp_path, monkeypatch):
    page = tmp_path / "stable" / "modules" / "generated"
    page.mkdir(parents=True)
    body = ("<html><nav>MENU MENU</nav><main><h1>NoSuchClass</h1><p>NoSuchClass(max_depth=None) — the real signature "
            "documented upstream. Use fit(X, y) then predict(X). " + "More documentation text. " * 10 +
            "</p></main><script>var x=1;</script></html>")
    (page / "sklearn.ensemble.NoSuchClass.html").write_text(body)
    (page / "sklearn.ensemble.RandomForestClassifier.html").write_text(body.replace("NoSuchClass", "RandomForestClassifier"))
    hits = []

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *a):
            hits.append(self.path)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(H, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setitem(webdocs.BASES, "sklearn", f"http://127.0.0.1:{srv.server_port}/stable")
    yield hits
    srv.shutdown()


def test_webdocs_allowlist_and_html_cleaning(docs_server):
    assert not webdocs.host_allowed("https://evil.example.com/x") and not webdocs.host_allowed("http://scikit-learn.org/x")
    assert webdocs.host_allowed("https://scikit-learn.org/stable/x.html") and webdocs.host_allowed("https://optuna.readthedocs.io/a")
    assert webdocs.fetch_text("https://evil.example.com/x") is None
    url = webdocs.candidate_urls("sklearn", ["sklearn.ensemble.NoSuchClass.fit"])[0]       # method -> class page
    assert url.endswith("/modules/generated/sklearn.ensemble.NoSuchClass.html")
    text = webdocs.fetch_text(url)
    assert "real signature documented upstream" in text and "MENU" not in text and "var x" not in text


def test_docs_agent_goes_to_official_docs_only_when_local_is_not_enough(tmp_path, docs_server):
    ws = mini_ws(tmp_path)
    llm.set_llm(FakeLLM())
    docs.lookup(ws, "AttributeError: module 'sklearn.ensemble' has no attribute 'NoSuchClass'")
    assert load(ws).docs["sklearn"] and any("NoSuchClass" in h for h in docs_server)         # local failed -> web used
    docs_server.clear()
    info = docs._gather(ws, "sklearn", ["sklearn.ensemble.RandomForestClassifier"], "t", force_web=False)
    assert not docs.insufficient(info) and not docs_server                                   # local enough -> no web
    info = docs._gather(ws, "sklearn", ["sklearn.ensemble.RandomForestClassifier"], "t", force_web=True)
    assert any(k.startswith("web:") for k in info) and docs_server                           # error persisted -> escalate
    with session(ws) as L:
        pass
    cfg = cfgmod.override(ws.config(), {})
    cfg.docs.web = False
    cfgmod.save(ws.root, cfg)
    docs_server.clear()
    assert not any(k.startswith("web:") for k in docs._gather(ws, "sklearn", ["sklearn.ensemble.NoSuchClass"], "t", True))


# ---- hardware --------------------------------------------------------------------------------------------

def test_device_env_assigns_gpus_and_thread_shares(monkeypatch):
    monkeypatch.setattr(hw, "gpus", lambda: [{"name": "T4", "total_mb": 15360, "free_mb": 15000}] * 2)
    monkeypatch.setattr(hw, "cores", lambda: 8)
    assert hw.device_env(0, 2)["CUDA_VISIBLE_DEVICES"] == "0" and hw.device_env(1, 2)["CUDA_VISIBLE_DEVICES"] == "1"
    assert hw.device_env(2, 3)["CUDA_VISIBLE_DEVICES"] == ""                  # more workers than GPUs -> CPU
    assert hw.device_env(0, 4)["MLAGENT_N_JOBS"] == "2" and "MLAGENT_N_JOBS" not in hw.device_env(0, 1)
    assert "T4" in hw.describe()
    monkeypatch.setattr(hw, "gpus", lambda: [])
    assert "none (CPU only)" in hw.describe() and hw.device_env(0, 1) == {}


# ---- think-block stripping & LLM helpers -------------------------------------------------

def test_think_block_stripping_in_extract_json_and_code():
    text_with_think = "<think>Let me reason first: {fake_json: 123}</think>\n```json\n{\"real_key\": 42}\n```"
    extracted = llm.extract_json(text_with_think)
    assert extracted == {"real_key": 42}

    code_with_think = "<think>Planning: ```python\n# wrong code\n```</think>\n```python\nx = 10\nprint(x)\n```"
    code = llm.extract_code(code_with_think)
    assert "print(x)" in code and "# wrong code" not in code


# ---- to_submission edge cases ------------------------------------------------------------

def test_to_submission_missing_id_column_fallback(tmp_path):
    ws = mini_ws(tmp_path)
    kit = ws.load_mlkit()
    t = kit.load_test()
    n = len(t) if t is not None else 10
    raw = np.linspace(0.1, 0.9, n)
    sub = kit.to_submission(raw)
    assert len(sub) == n
    assert kit.TARGET in sub.columns
    if kit.ID:
        assert kit.ID in sub.columns


# ---- doctor checks -----------------------------------------------------------------------

def test_doctor_diagnostics(tmp_path):
    from mlagent import doctor
    # Smoke test doctor checks
    ok_py, _ = doctor.check_python()
    assert ok_py is True
    ok_venv, _ = doctor.check_virtualenv()
    assert ok_venv is True
    ok_write, _ = doctor.check_write_permission(tmp_path)
    assert ok_write is True
    ok_enc, _ = doctor.check_console_encoding()
    assert ok_enc is True

