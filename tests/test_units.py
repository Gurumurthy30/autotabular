"""Unit tests for the rule-bearing modules.  pytest -q tests/test_units.py"""

import argparse
import functools
import http.server
import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import FakeLLM, make_data  # noqa: E402

from mlagent import cli, ensemble, hw, llm, webdocs  # noqa: E402
from mlagent import config as cfgmod  # noqa: E402
from mlagent.agents import api_docs_lookup as docs  # noqa: E402
from mlagent.agents.experiment_planner import CVOut, _sanitize_cv, goal_target, make_folds  # noqa: E402
from mlagent.ledger import (  # noqa: E402
    add_queue_items,
    beats,
    consecutive_crashes,
    counted_gain_flags,
    is_api_error,
    is_oom_error,
    load,
    plateau,
    session,
    stop_reason,
)
from mlagent.state import (  # noqa: E402
    Budget,
    CVScheme,
    Env,
    Ledger,
    Problem,
    QueueItem,
    Run,
    Strategy,
    ValidationRecord,
)


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


# ---- agent embedded prompts -------------------------------------------------------------

def test_agent_prompts_embedded_in_modules():
    import importlib

    from mlagent.prompts import AGENT_NAMES
    for name in AGENT_NAMES:
        mod = importlib.import_module(f"mlagent.agents.{name}")
        assert hasattr(mod, "PROMPTS"), f"{name} must define PROMPTS"
        assert isinstance(mod.PROMPTS, dict)
        assert len(mod.PROMPTS) > 0, f"{name}.PROMPTS is empty"

    from mlagent.agents import (
        experiment_planner,
        experiment_runner,
        script_writer,
    )
    assert "kit_api" in script_writer.PROMPTS and "KIT API" in script_writer.PROMPTS["kit_api"]
    assert "system" in experiment_planner.PROMPTS and "experiment planner" in experiment_planner.PROMPTS["system"].lower()
    assert "task_fresh" in experiment_runner.PROMPTS and "<experiment id=" in experiment_runner.PROMPTS["task_fresh"]


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
        out = r.predict(test)
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
    with session(ws):
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
    monkeypatch.setattr(hw, "gpus", list)
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
    assert isinstance(ok_venv, bool)
    ok_write, _ = doctor.check_write_permission(tmp_path)
    assert ok_write is True
    # Check that passing a file path also works cleanly
    sample_file = tmp_path / "train.csv"
    sample_file.write_text("a,b\n1,2", encoding="utf-8")
    ok_file_write, msg = doctor.check_write_permission(sample_file)
    assert ok_file_write is True
    assert "Dataset readable" in msg
    ok_enc, _ = doctor.check_console_encoding()
    assert ok_enc is True


# ---- metric aliases & roc score ----------------------------------------------------------

def test_metric_aliases_and_roc_score(tmp_path):
    assert cli.normalize_metric("ROC") == "roc_auc"
    assert cli.normalize_metric("auc") == "roc_auc"
    assert cli.normalize_metric("roc-auc") == "roc_auc"
    assert cli.normalize_metric("ACC") == "accuracy"
    assert cli.normalize_metric("F1") == "f1"

    ws = mini_ws(tmp_path)
    kit = ws.load_mlkit()
    y_true = np.array([0, 0, 1, 1])
    y_pred_prob = np.array([0.1, 0.2, 0.8, 0.9])
    y_pred_2d = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

    # Both "roc", "roc_auc", "auc" should evaluate without error
    s_roc = kit.score(y_true, y_pred_prob, metric="roc")
    s_auc = kit.score(y_true, y_pred_prob, metric="roc_auc")
    assert s_roc == 1.0
    assert s_auc == 1.0

    # 2D probabilities should be handled properly
    s_2d = kit.score(y_true, y_pred_2d, metric="roc")
    assert s_2d == 1.0


# ---- profiler cleaning & rate limit -----------------------------------------------------

def test_profiler_questions_cleaning_and_rate_limit(tmp_path):
    from mlagent.agents import data_profiler as profiler
    from mlagent.llm import RateLimitError

    # Test field_validator directly on Questions model
    q1 = profiler.Questions.model_validate({"questions": "one question"})
    assert q1.questions == ["one question"]

    q12 = profiler.Questions.model_validate({"questions": [f"question {i}" for i in range(12)]})
    assert len(q12.questions) == 6

    # Test profiler.run with RateLimitError propagation
    ws = mini_ws(tmp_path)

    class RateLimitLLM:
        def json(self, *a, **kw):
            raise RateLimitError("rate limited")

    llm.set_llm(RateLimitLLM())
    with pytest.raises(RateLimitError):
        profiler.run(ws)


# ---- validator code_truncated signal ----------------------------------------------------

def test_validator_code_truncated_signal(tmp_path, monkeypatch):
    from mlagent.agents import run_validator as validator
    from mlagent.executor import ExecResult
    from mlagent.state import Run

    ws = mini_ws(tmp_path)
    long_script = tmp_path / "long_script.py"
    long_script.write_text("x = 1\n" * 5000, encoding="utf-8")  # ~30k chars

    short_script = tmp_path / "short_script.py"
    short_script.write_text("x = 1\n", encoding="utf-8")

    captured_sigs = []

    def mock_write_and_run(ws, name, task, kind="analysis", timeout=300):
        if "e_long" in name:
            assert "NOTE: script longer than 24000 characters" in task
        return tmp_path / f"{name}.py", ExecResult(ok=True, returncode=0, stdout='RESULT_JSON: {"notes": "ok"}', stderr="", seconds=1.0)

    monkeypatch.setattr(validator, "write_and_run", mock_write_and_run)
    monkeypatch.setattr(validator, "core_signals", lambda _L, _r, _kit: ({}, []))

    class MockJudgeLLM:
        def json(self, system, user, schema):
            data = json.loads(user) if isinstance(user, str) else user
            captured_sigs.append(data.get("signals", {}))
            return validator.Verdict(verdict="approve", reasons=[])

    llm.set_llm(MockJudgeLLM())

    with session(ws) as L:
        L.runs.append(Run(exp_id="e_long", cv_mean=0.8, cv_std=0.01, script_path=str(long_script)))
    validator.validate_one(ws, "e_long")
    assert captured_sigs[-1].get("code_truncated") is True

    with session(ws) as L:
        L.runs.append(Run(exp_id="e_short", cv_mean=0.8, cv_std=0.01, script_path=str(short_script)))
    validator.validate_one(ws, "e_short")
    assert captured_sigs[-1].get("code_truncated") is False


def test_base_code_resolution(tmp_path):
    """Verify how _base_code resolves 'best' (no approved run), experiment id, None, and nonexistent id."""
    from mlagent.agents.experiment_runner import _base_code
    from mlagent.state import Run, ValidationRecord

    ws = mini_ws(tmp_path)
    script_file = tmp_path / "e001.py"
    script_file.write_text("# e001 script code", encoding="utf-8")

    # Case 1: "best" with no approved run
    with session(ws) as L:
        L.runs.append(Run(exp_id="e001", cv_mean=0.85, cv_std=0.01, script_path=str(script_file)))
        # No validation record -> not approved
    L = load(ws)
    bid, bcode = _base_code(L, "best")
    assert bid is None and bcode is None, "'best' with no approved run must resolve to (None, None)"

    # Case 2: an experiment id that exists with valid script
    bid, bcode = _base_code(L, "e001")
    assert bid == "e001" and bcode == "# e001 script code", "Existing exp_id must resolve to (exp_id, code)"

    # Case 3: null / None
    bid, bcode = _base_code(L, None)
    assert bid is None and bcode is None, "None base must resolve to (None, None)"

    # Case 4: an id that does not exist
    bid, bcode = _base_code(L, "e999_nonexistent")
    assert bid is None and bcode is None, "Nonexistent exp_id must resolve to (None, None)"

    # Also verify: "best" with an approved run resolves correctly
    with session(ws) as L:
        L.validation.append(ValidationRecord(exp_id="e001", verdict="approve"))
    L = load(ws)
    bid, bcode = _base_code(L, "best")
    assert bid == "e001" and bcode == "# e001 script code", "'best' with approved run must resolve to (exp_id, code)"

    # Case 5: an explicit base id that was rejected by validator -> must ignore and return (None, None)
    rej_file = tmp_path / "e_rej.py"
    rej_file.write_text("# rejected code", encoding="utf-8")
    with session(ws) as L:
        L.runs.append(Run(exp_id="e_rej", cv_mean=0.90, cv_std=0.01, script_path=str(rej_file)))
        L.validation.append(ValidationRecord(exp_id="e_rej", verdict="reject", reasons=["leakage detected"]))
    L = load(ws)
    bid, bcode = _base_code(L, "e_rej")
    assert bid is None and bcode is None, "Explicit base id rejected by validator must resolve to (None, None)"


def test_budget_run_and_tune_minutes_and_skeleton_deep(tmp_path, monkeypatch):
    """Verify configurable max_run_minutes, tune_minutes_per_model and SKELETON_DEEP selection."""
    from mlagent.agents import script_writer as coder
    from mlagent.executor import ExecResult
    from mlagent.state import Budget, Profile

    # 1. Budget defaults
    b = Budget()
    assert b.max_run_minutes == 15
    assert b.tune_minutes_per_model == 15

    # 2. SKELETON_DEEP sent when modality contains text/image/audio
    ws = mini_ws(tmp_path)
    with session(ws) as L:
        L.profile = Profile(modality=["text", "tabular"])

    captured_users = []

    class MockCoderLLM:
        def code(self, system, user):
            captured_users.append(user)
            return "print('ok')\nemit({'cv_mean': 0.8})"

    llm.set_llm(MockCoderLLM())
    monkeypatch.setattr(coder, "run_script", lambda _ws, _p, _t: ExecResult(ok=True, returncode=0, stdout='RESULT_JSON: {"cv_mean": 0.8}', stderr="", seconds=1.0))

    coder.write_and_run(ws, "e_deep", "train deep", timeout=1200)
    assert len(captured_users) == 1
    assert "LIMIT_S = 1200" in captured_users[0]
    assert "<<limit_s>>" not in captured_users[0]
    assert "make_dataset" in captured_users[0]  # Part of SKELETON_DEEP


def test_analyzer_trigger_instructions_and_context_keys(tmp_path, monkeypatch):
    """Verify that analyzer injects exact instruction text and context keys per trigger."""
    from mlagent.agents import results_analyzer as analyzer
    from mlagent.state import Run, ValidationRecord

    ws = mini_ws(tmp_path)
    with session(ws) as L:
        L.runs.append(Run(exp_id="t_e001", source="tuner", cv_mean=0.8, script_path="s.py"))
        L.validation.append(ValidationRecord(exp_id="t_e001", verdict="reject", reasons=["holdout drops"]))

    captured_users = {}

    class MockAnalyzerLLM:
        def json(self, system, user, schema):
            # Record current trigger's user payload
            current_trigger = getattr(self, "current_trigger", "unknown")
            captured_users[current_trigger] = user
            return analyzer.AnalysisOut(bottleneck="saturated", evidence="done", hypotheses=[])

    mock_llm = MockAnalyzerLLM()
    llm.set_llm(mock_llm)
    monkeypatch.setattr(analyzer, "write_and_run", lambda _ws, _name, _task, kind="analysis", timeout=300: (None, MagicMock(ok=True, stdout='RESULT_JSON: {"signals": "ok"}')))

    triggers = ["periodic", "empty_queue", "crash", "tuner_failure", "cv_lb_gap"]
    for trig in triggers:
        mock_llm.current_trigger = trig
        if trig == "crash":
            analyzer.run(ws, trigger=trig, error="KeyError: 'target'")
        elif trig == "cv_lb_gap":
            analyzer.run(ws, trigger=trig, error=json.dumps({"flagged": True, "cv": 0.8, "lb": 0.6}))
        else:
            analyzer.run(ws, trigger=trig)

    instr_crash = analyzer.PROMPTS["instr_crash"]
    instr_tuner = analyzer.PROMPTS["instr_tuner_failure"]
    instr_gap = analyzer.PROMPTS["instr_cv_lb_gap"]

    # 1. crash trigger
    assert instr_crash in captured_users["crash"]
    assert "crash_error" in captured_users["crash"]
    assert "KeyError: 'target'" in captured_users["crash"]

    # 2. tuner_failure trigger
    assert instr_tuner in captured_users["tuner_failure"]
    assert "validation" in captured_users["tuner_failure"]
    assert "holdout drops" in captured_users["tuner_failure"]

    # 3. cv_lb_gap trigger
    assert instr_gap in captured_users["cv_lb_gap"]
    assert "lb_gap" in captured_users["cv_lb_gap"]

    # 4. periodic and empty_queue contain NONE of the three instructions
    for neutral in ("periodic", "empty_queue"):
        text = captured_users[neutral]
        assert instr_crash not in text
        assert instr_tuner not in text
        assert instr_gap not in text
        assert "crash_error" not in text
        assert "lb_gap" not in text


def test_digest_additions_and_curated_runs(tmp_path):
    """Verify digest additions (budget, hardware, seed, earlier_analyses, gains_by_kind, curated runs, notes)."""
    from mlagent.ledger import curated_run_rows, digest
    from mlagent.state import Analysis, Profile, QueueItem, Run, ValidationRecord

    ws = mini_ws(tmp_path)
    with session(ws) as L:
        # Notes with tagged and untagged
        notes = [f"note_{i}" for i in range(15)]
        notes.append("split: stratified 5-fold")
        notes.append("metric: logloss")
        L.profile = Profile(notes=notes)

        # 24 runs fixture
        for i in range(24):
            eid = f"e{i:03d}"
            # e000 is baseline (cv=0.70)
            # e003 is a counted gain (cv=0.85 > 0.70 + 0.01)
            # e005 is best run (cv=0.95 > 0.85 + 0.01)
            # Others have lower cv (0.60)
            cv = 0.95 if i == 5 else (0.85 if i == 3 else (0.70 if i == 0 else 0.60))
            L.runs.append(Run(exp_id=eid, cv_mean=cv, cv_std=0.01, source="experimenter"))
            L.validation.append(ValidationRecord(exp_id=eid, verdict="approve"))
            L.queue.append(QueueItem(id=eid, hypothesis=f"hypo {i}", kind="feature" if i % 2 == 0 else "model", change=f"change {i}", status="done"))

        # Add some pending queue items
        L.queue.append(QueueItem(id="e024", hypothesis="pending hypo 1", kind="feature", change="pending feat", status="pending"))
        L.queue.append(QueueItem(id="e025", hypothesis="pending hypo 2", kind="model", change="pending model", status="pending"))

        # Add earlier analyses
        for a_idx in range(6):
            L.analysis.append(Analysis(after_exp=f"e{a_idx}", bottleneck=f"bottleneck_{a_idx}", evidence=f"evidence_{a_idx}", new_ids=["q1", "q2"]))

    L = load(ws)

    # 1. Experiment Planner digest checks
    strat_d = digest(L, "experiment_planner")
    assert "budget" in strat_d
    assert strat_d["budget"]["max_experiments"] == 15
    assert strat_d["budget"]["max_run_minutes"] == 15
    assert strat_d["budget"]["tune_minutes_per_model"] == 15
    assert "hardware" in strat_d
    assert "seed" in strat_d
    # notes[:20] with tagged first
    assert len(strat_d["profile"]["notes"]) <= 20
    assert strat_d["profile"]["notes"][0].startswith("split:") or strat_d["profile"]["notes"][0].startswith("metric:")

    # 2. Results Analyzer digest checks
    analyzer_d = digest(L, "results_analyzer")
    assert "budget" in analyzer_d
    assert "hardware" in analyzer_d
    assert "seed" in analyzer_d
    assert "earlier_analyses" in analyzer_d
    assert len(analyzer_d["earlier_analyses"]) == 4
    assert analyzer_d["earlier_analyses"][-1]["bottleneck"] == "bottleneck_5"
    assert analyzer_d["earlier_analyses"][-1]["hypothesis_count"] == 2

    # Gains by kind
    assert "gains_by_kind" in analyzer_d
    gk = analyzer_d["gains_by_kind"]
    assert "feature" in gk and "model" in gk
    assert gk["feature"]["approved"] > 0
    assert gk["model"]["approved"] > 0

    # Queue contains ONLY pending items
    assert len(analyzer_d["queue"]) == 2
    assert all(q["status"] == "pending" for q in analyzer_d["queue"])

    # 3. Curated runs on 24-run fixture
    # Must keep: e000 (baseline / first), e003 (gain), e005 (best), and the 8 most recent runs (e016..e023)
    c_runs = curated_run_rows(L)
    c_ids = [r["exp"] for r in c_runs]
    assert "e000" in c_ids, "Curated runs must keep baseline / first run"
    assert "e003" in c_ids, "Curated runs must keep counted gain run"
    assert "e005" in c_ids, "Curated runs must keep best approved run"
    for recent_idx in range(16, 24):
        assert f"e{recent_idx:03d}" in c_ids, f"Curated runs must keep recent run e{recent_idx:03d}"
    # Verify time order is preserved
    indices = [int(eid[1:]) for eid in c_ids]
    assert indices == sorted(indices), "Curated runs must preserve time order"


# ---- C11 & D1/D3 upgrades ----------------------------------------------------------------

def test_profiler_upgrades_offline_fixtures(tmp_path):
    """C11: Offline fixtures asserting roles, modality, notes, and flags without external dependencies."""
    import pandas as pd

    from mlagent.agents import data_profiler
    from mlagent.state import Problem

    # 1. Titanic-like CSV
    t_dir = tmp_path / "titanic"
    t_dir.mkdir()
    t_csv = t_dir / "train.csv"
    t_df = pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [0, 1, 1, 0, 1],
        "Pclass": [3, 1, 3, 1, 3],
        "Name": ["Braund, Mr. Owen", "Cumings, Mrs. John", "Heikkinen, Miss. Laina", "Futrelle, Mrs. Jacques", "Allen, Mr. William"],
        "Sex": ["male", "female", "female", "female", "male"],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
        "Fare": [7.25, 71.28, 7.92, 53.1, 8.05],
    })
    t_csv.write_text(t_df.to_csv(index=False), encoding="utf-8")
    p1 = Problem(goal="predict", train_path=str(t_csv), target="Survived", metric="accuracy", direction="maximize", task_type="binary", id_col="PassengerId")
    prof1 = data_profiler.basic_profile(p1)
    assert prof1.column_roles["PassengerId"] == "id"
    assert "tabular" in prof1.modality
    assert prof1.target_stats["target_missing"] == 0

    # 2. Text CSV
    txt_dir = tmp_path / "text"
    txt_dir.mkdir()
    txt_csv = txt_dir / "train.csv"
    long_texts = [
        "This is an extensive review of the machine learning agent capabilities. It tests whether token lengths and NLP playbooks are triggered properly when column text exceeds word thresholds.",
        "Another very long and detailed descriptive text that contains enough words to be definitively classified as full natural language text rather than a short categorical string.",
        "A third extensive review text ensuring that the average word count and character count across the entire column firmly trigger text modality."
    ] * 2
    txt_df = pd.DataFrame({
        "review_id": [1, 2, 3, 4, 5, 6],
        "review_text": long_texts,
        "sentiment": [1, 0, 1, 0, 1, 0],
    })
    txt_csv.write_text(txt_df.to_csv(index=False), encoding="utf-8")
    p2 = Problem(goal="predict", train_path=str(txt_csv), target="sentiment", metric="accuracy", direction="maximize", task_type="binary", id_col="review_id")
    prof2 = data_profiler.basic_profile(p2)
    assert prof2.column_roles["review_text"] == "text"
    assert "text" in prof2.modality

    # 3. CSV with an image-path column plus a folder of tiny PNG files
    img_dir = tmp_path / "image_data"
    img_dir.mkdir()
    sub_imgs = img_dir / "images"
    sub_imgs.mkdir()
    minimal_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    for img_name in ("img1.png", "img2.png", "img3.png"):
        (sub_imgs / img_name).write_bytes(minimal_png)
    img_csv = img_dir / "train.csv"
    img_df = pd.DataFrame({
        "id": [101, 102, 103],
        "file_name": ["images/img1.png", "images/img2.png", "images/img3.png"],
        "target": [0, 1, 0],
    })
    img_csv.write_text(img_df.to_csv(index=False), encoding="utf-8")
    p3 = Problem(goal="predict", train_path=str(img_csv), target="target", metric="accuracy", direction="maximize", task_type="binary", id_col="id")
    prof3 = data_profiler.basic_profile(p3)
    assert prof3.column_roles["file_name"] == "image_path"
    assert "image" in prof3.modality
    assert prof3.column_roles["file_name"] != "id"
    png_inv = [f for f in prof3.data_files if f.get("extension") == ".png"]
    assert len(png_inv) == 1 and png_inv[0]["count"] >= 3

    # 4. CSV with group column and date column
    grp_dir = tmp_path / "group_date"
    grp_dir.mkdir()
    grp_csv = grp_dir / "train.csv"
    grp_df = pd.DataFrame({
        "row_id": list(range(1, 21)),
        "customer_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 2,
        "transaction_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"] * 5,
        "amount": [10.5, 20.0, 15.2, 8.0] * 5,
        "is_fraud": [0, 0, 0, 1] * 5,
    })
    grp_csv.write_text(grp_df.to_csv(index=False), encoding="utf-8")
    p4 = Problem(goal="predict", train_path=str(grp_csv), target="is_fraud", metric="accuracy", direction="maximize", task_type="binary", id_col="row_id")
    prof4 = data_profiler.basic_profile(p4)
    assert prof4.column_roles["transaction_date"] == "datetime"
    assert prof4.column_roles["customer_id"] != "group"  # Group role not assigned automatically
    assert any("candidate group column 'customer_id'" in n for n in prof4.notes)

    # 5. CSV with leaked column (copy of target)
    leak_dir = tmp_path / "leak"
    leak_dir.mkdir()
    leak_csv = leak_dir / "train.csv"
    y_vals = [0, 1, 0, 1, 0, 1] * 10
    leak_df = pd.DataFrame({
        "row_id": list(range(1, len(y_vals) + 1)),
        "feature_x": [float(i) for i in range(len(y_vals))],
        "target_leak": y_vals,
        "target": y_vals,
    })
    leak_csv.write_text(leak_df.to_csv(index=False), encoding="utf-8")
    p5 = Problem(goal="predict", train_path=str(leak_csv), target="target", metric="accuracy", direction="maximize", task_type="binary", id_col="row_id")
    prof5 = data_profiler.basic_profile(p5)
    assert any("target_leak" in f for f in prof5.leakage_flags)


def test_analyzer_signals_and_run_fold_scores(tmp_path):
    """D1 and D3: fold_scores, train_score, and analyzer signals on a 3-run fixture."""
    import numpy as np

    from mlagent.ledger import digest, load, session
    from mlagent.state import Run, ValidationRecord

    ws = mini_ws(tmp_path)
    # D1: Old ledger without fold_scores / train_score loads with defaults None
    old_run_json = '{"exp_id": "e_old", "cv_mean": 0.8, "cv_std": 0.02, "source": "experimenter"}'
    run_obj = Run.model_validate_json(old_run_json)
    assert run_obj.fold_scores is None
    assert run_obj.train_score is None

    # D3: Fixture with 3 approved runs
    oof1 = np.array([0.2, 0.8, 0.4, 0.9])
    oof2 = np.array([0.25, 0.75, 0.45, 0.85])
    oof3 = np.array([0.1, 0.9, 0.3, 0.95])

    oof1_path = tmp_path / "e001_oof.npy"
    oof2_path = tmp_path / "e002_oof.npy"
    oof3_path = tmp_path / "e003_oof.npy"
    np.save(oof1_path, oof1)
    np.save(oof2_path, oof2)
    np.save(oof3_path, oof3)

    with session(ws) as L:
        L.problem.direction = "maximize"
        L.runs.append(Run(
            exp_id="e001", cv_mean=0.81, cv_std=0.02, fold_scores=[0.79, 0.81, 0.83],
            train_score=0.89, oof_path=str(oof1_path), source="experimenter"
        ))
        L.validation.append(ValidationRecord(exp_id="e001", verdict="approve"))

        L.runs.append(Run(
            exp_id="e002", cv_mean=0.88, cv_std=0.015, fold_scores=[0.86, 0.88, 0.90],
            train_score=0.96, oof_path=str(oof2_path), source="experimenter"
        ))
        L.validation.append(ValidationRecord(exp_id="e002", verdict="approve"))

        L.runs.append(Run(
            exp_id="e003", cv_mean=0.84, cv_std=0.02, fold_scores=[0.82, 0.84, 0.86],
            train_score=0.91, oof_path=str(oof3_path), source="experimenter"
        ))
        L.validation.append(ValidationRecord(exp_id="e003", verdict="approve"))

    L = load(ws)
    diag = digest(L, "results_analyzer")

    # 1. per-fold scores of best approved run (e002)
    assert "per_fold_scores" in diag
    pfs = diag["per_fold_scores"]
    assert pfs["exp_id"] == "e002"
    assert pfs["fold_scores"] == [0.86, 0.88, 0.90]
    assert pfs["spread"] == 0.04

    # 2. train vs validation gap
    assert "train_versus_validation_gap" in diag
    tvg = diag["train_versus_validation_gap"]
    assert tvg["exp_id"] == "e002"
    assert tvg["train_score"] == 0.96
    assert tvg["validation_score"] == 0.88
    assert tvg["gap"] == 0.08

    # 3. OOF correlation matrix
    assert "oof_correlation_between_runs" in diag
    oof_corr = diag["oof_correlation_between_runs"]
    assert set(oof_corr["runs"]) == {"e001", "e002", "e003"}
    matrix = oof_corr["matrix"]
    assert matrix["e001"]["e001"] == 1.0
    assert 0.0 < matrix["e001"]["e002"] <= 1.0



