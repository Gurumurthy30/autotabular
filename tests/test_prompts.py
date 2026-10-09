"""Tests for prompt contracts, template rendering, and API signatures.
pytest -q tests/test_prompts.py
"""

import ast
import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlagent.agents import (
    api_docs_lookup as docs,
)
from mlagent.agents import (
    data_profiler as profiler,
)
from mlagent.agents import (
    experiment_planner as strategist,
)
from mlagent.agents import (
    experiment_runner as experimenter,
)
from mlagent.agents import (
    final_submission as finisher,
)
from mlagent.agents import (
    hyperparameter_tuner as tuner,
)
from mlagent.agents import (
    results_analyzer as analyzer,
)
from mlagent.agents import (
    run_validator as validator,
)
from mlagent.agents import (
    script_writer as coder,
)
from mlagent.state import Budget, Env, Ledger, Problem, Run, ValidationRecord  # noqa: E402

# ---- KIT_API symbols ----------------------------------------------------------------------

def _get_mlkit_all() -> set[str]:
    """Parse __all__ from mlkit_template.py using AST without executing module."""
    path = Path(__file__).resolve().parents[1] / "mlagent" / "mlkit_template.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    return {elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)}
    return set()


def test_all_kit_api_names_exist_in_mlkit():
    """Verify all names declared in coder.KIT_API exist in mlkit_template.__all__."""
    kit_api = coder.KIT_API
    # Extract line with "Names: pd, np, ..."
    m = re.search(r"Names:\s*([^\n]+)", kit_api)
    assert m is not None, "KIT_API must contain a 'Names:' line"
    names = [n.strip() for n in m.group(1).split(",") if n.strip()]
    assert len(names) >= 10

    mlkit_all = _get_mlkit_all()
    assert len(mlkit_all) > 0, "Failed to parse __all__ from mlkit_template.py"

    for name in names:
        assert name in mlkit_all, f"Name '{name}' from KIT_API missing in mlkit_template.__all__"

    # Verify functions explicitly documented in KIT_API
    func_names = [
        "load_train", "load_test", "get_y", "folds", "holdout_idx",
        "score", "simple_prep", "predict_any", "run_cv", "emit"
    ]
    for fn in func_names:
        assert fn in mlkit_all, f"Function '{fn}' from KIT_API missing in mlkit_template.__all__"


# ---- Code fence and XML tag balancing ---------------------------------------------------

def test_prompt_code_fences_are_balanced():
    """Verify code fences (```) count is even across all prompt constants."""
    modules = [profiler, strategist, coder, experimenter, validator, analyzer, tuner, docs, finisher]
    for mod in modules:
        for k, v in mod.PROMPTS.items():
            if isinstance(v, str):
                # Instructions mention "```python block" inline as text; remove inline mentions to check fence blocks
                cleaned = re.sub(r"```python block\b", "", v)
                count = cleaned.count("```")
                assert count % 2 == 0, f"{mod.__name__}.PROMPTS['{k}'] has unbalanced code fences ({count} ```)"


def test_prompt_xml_tags_match():
    """Verify major XML-style tags in prompts are properly opened and closed."""
    checked_tags = ["team", "principles", "output_protocol", "how_to_think", "playbooks",
                    "contract", "study", "base_script", "previous_script", "error"]
    modules = [profiler, strategist, coder, experimenter, validator, analyzer, tuner, docs, finisher]
    for mod in modules:
        for k, v in mod.PROMPTS.items():
            if not isinstance(v, str):
                continue
            for tag in checked_tags:
                open_pattern = rf"(?<!<)<{tag}(?:[\s>])"
                close_pattern = rf"</{tag}>"
                if re.search(open_pattern, v):
                    open_count = len(re.findall(open_pattern, v))
                    close_count = len(re.findall(close_pattern, v))
                    assert open_count == close_count, (
                        f"{mod.__name__}.PROMPTS['{k}'] has mismatched <{tag}> tags: "
                        f"{open_count} open vs {close_count} close"
                    )
            # Check <experiment id="...">...</experiment> tags
            if '<experiment id="' in v:
                open_exp = len(re.findall(r'<experiment\s+id="', v))
                close_exp = v.count("</experiment>")
                assert open_exp == close_exp, (
                    f"{mod.__name__}.PROMPTS['{k}'] has mismatched <experiment> tags: "
                    f"{open_exp} open vs {close_exp} close"
                )


# ---- JSON Example Validation -------------------------------------------------------------

VALID_BOTTLENECKS = {
    "process_or_leakage", "validation_noise", "representation_or_features",
    "model_capacity", "variance_or_regularisation", "metric_alignment",
    "unused_data", "feature_noise_or_width", "label_or_row_quality",
    "compute_or_time", "saturation"
}


def test_prompt_json_examples_parse_valid():
    """Verify JSON examples embedded in prompt docstrings parse as valid JSON and use legal enums."""
    # Validator example
    val_m = re.search(r"Example:\s*(\{.*?\})\n", validator.PROMPT_SYSTEM, re.DOTALL)
    assert val_m is not None
    val_data = json.loads(val_m.group(1))
    assert val_data["verdict"] in ("approve", "reject")
    assert isinstance(val_data["reasons"], list)
    assert isinstance(val_data["signals_checked"], list)

    # Strategist examples
    strat_examples = [
        json.loads(line.strip())
        for line in strategist.PROMPT_SYSTEM.splitlines()
        if line.strip().startswith('{"reasoning":')
    ]
    assert len(strat_examples) >= 2
    for data in strat_examples:
        assert "reasoning" in data
        assert data["validation"]["kind"] in ("kfold", "stratified", "group", "time")
        assert isinstance(data["hypotheses"], list)
        for h in data["hypotheses"]:
            assert h["kind"] in ("feature", "model")
            assert h.get("base") in ("best", None) or re.fullmatch(r"e\d+", str(h.get("base")))

    # Analyzer examples
    analyzer_examples = [
        json.loads(line.strip())
        for line in analyzer.PROMPT_SYSTEM.splitlines()
        if line.strip().startswith('{"reasoning":')
    ]
    assert len(analyzer_examples) >= 3
    for data in analyzer_examples:
        assert "reasoning" in data
        assert "bottleneck" in data
        b_label = data["bottleneck"].split(":")[0].strip()
        assert b_label in VALID_BOTTLENECKS, f"Bottleneck '{b_label}' not in VALID_BOTTLENECKS"
        assert "evidence" in data
        assert isinstance(data["hypotheses"], list)
        for h in data["hypotheses"]:
            assert h["kind"] in ("feature", "model")
            assert h.get("base") in ("best", None) or re.fullmatch(r"e\d+", str(h.get("base")))


# ---- Output Schema & Dropped Keys Contract -----------------------------------------------

def test_agent_output_contracts_and_dropped_keys():
    """Validate that prompt output keys either match the Pydantic model or are in LOGGED_ONLY_KEYS."""
    LOGGED_ONLY_KEYS = {"reasoning", "focus", "levers", "self_check"}

    # Strategist
    strat_model_fields = set(strategist.StrategyOut.model_fields.keys())
    # Prompt asks for: reasoning, validation, target_score, risks, domain_notes, hypotheses, plan_summary, levers, self_check
    strat_prompt_keys = {
        "reasoning", "validation", "target_score", "risks", "domain_notes",
        "hypotheses", "plan_summary", "levers", "self_check"
    }
    unaccounted_strat = strat_prompt_keys - strat_model_fields - LOGGED_ONLY_KEYS
    assert not unaccounted_strat, f"Strategist prompt keys {unaccounted_strat} not in model or LOGGED_ONLY_KEYS"

    # Analyzer
    analyzer_model_fields = set(analyzer.AnalysisOut.model_fields.keys())
    # Prompt asks for: reasoning, focus, bottleneck, evidence, hypotheses, self_check
    analyzer_prompt_keys = {"reasoning", "focus", "bottleneck", "evidence", "hypotheses", "self_check"}
    unaccounted_analyzer = analyzer_prompt_keys - analyzer_model_fields - LOGGED_ONLY_KEYS
    assert not unaccounted_analyzer, f"Analyzer prompt keys {unaccounted_analyzer} not in model or LOGGED_ONLY_KEYS"

    # Validator
    val_model_fields = set(validator.Verdict.model_fields.keys())
    val_prompt_keys = {"reasoning", "verdict", "reasons", "signals_checked"}
    unaccounted_val = val_prompt_keys - val_model_fields - LOGGED_ONLY_KEYS
    assert not unaccounted_val, f"Validator prompt keys {unaccounted_val} not in model or LOGGED_ONLY_KEYS"


# ---- Template Rendering & Placeholder Validation ----------------------------------------

def test_prompt_templates_render_cleanly():
    """Verify all templates render with dummy values and leave no unrendered placeholders."""
    # profiler stats_task
    rendered_prof = profiler.PROMPTS["stats_task"].replace("<<questions>>", "- question 1\n- question 2")
    assert "<<" not in rendered_prof and ">>" not in rendered_prof

    # experimenter task_fresh
    rendered_fresh = (
        experimenter.PROMPTS["task_fresh"]
        .replace("<<exp_id>>", "e001")
        .replace("<<hypothesis>>", "test hypo")
        .replace("<<change>>", "test change")
        .replace("<<params>>", "{}")
        .replace("<<time_limit>>", "300s")
        .replace("<<hardware>>", "CPU 4 cores")
    )
    assert "<<" not in rendered_fresh and ">>" not in rendered_fresh

    # experimenter task_from_base
    rendered_base = (
        experimenter.PROMPTS["task_from_base"]
        .replace("<<exp_id>>", "e002")
        .replace("<<hypothesis>>", "test hypo 2")
        .replace("<<change>>", "test change 2")
        .replace("<<params>>", "{}")
        .replace("<<base_id>>", "e001")
        .replace("<<base_seconds>>", "45")
        .replace("<<time_limit>>", "300s")
        .replace("<<hardware>>", "CPU 4 cores")
        .replace("<<base_code>>", "print('hello')")
    )
    assert "<<" not in rendered_base and ">>" not in rendered_base

    # experimenter task_retry
    rendered_retry = (
        experimenter.PROMPTS["task_retry"]
        .replace("<<exp_id>>", "e001")
        .replace("<<attempt>>", "2")
        .replace("<<previous_code>>", "print('prev')")
        .replace("<<error_kind>>", "logic")
        .replace("<<error>>", "KeyError: 'target'")
        .replace("<<time_limit>>", "300s")
        .replace("<<hardware>>", "CPU 4 cores")
    )
    assert "<<" not in rendered_retry and ">>" not in rendered_retry

    # validator script_task
    rendered_val = (
        validator.PROMPTS["script_task"]
        .replace("<<exp>>", "e001")
        .replace("<<code>>", "print('validated')")
    )
    assert "<<" not in rendered_val and ">>" not in rendered_val

    # analyzer signals
    rendered_sig = analyzer.PROMPTS["signals"].format(best="e001")
    assert "{best}" not in rendered_sig


# ---- Tuner Tests -------------------------------------------------------------------------

def test_tuner_task_formatting_with_braces_in_code():
    """Verify Tuner PROMPT_TASK renders without KeyError when script contains braces."""
    code_with_braces = '''
import json
d = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
f_str = f"Formatted: {d['a']} with {len(d['b'])} items"
set_lit = {1, 2, 3, 4}
print(f_str, set_lit)
'''
    task = tuner.PROMPT_TASK.format(
        base="e001",
        cv=0.8523,
        cv_std=0.0120,
        base_seconds=42.0,
        direction="maximize",
        trials=15,
        secs=450,
        max_params=6,
        hardware="1x NVIDIA T4 (15GB)",
        eid="t_e001",
        code=code_with_braces,
    )
    assert "KeyError" not in task
    assert 'd = {"a": 1' in task
    assert "f_str = f\"Formatted:" in task
    assert "Search at most 6 parameters" in task


def test_tuner_skips_script_longer_than_max_chars(tmp_path, monkeypatch):
    """Verify that a script exceeding MAX_SCRIPT_CHARS (24,000) is skipped without tuning."""
    script_file = tmp_path / "long_script.py"
    # Create 30,000 character script
    script_file.write_text("# padding comment\n" * 1500 + "print('done')\n", encoding="utf-8")
    assert script_file.stat().st_size > 24_000

    ws = MagicMock()
    ws.root = tmp_path

    # Construct mock ledger with an approved candidate
    r = Run(exp_id="e001", cv_mean=0.85, cv_std=0.01, seconds=20.0, script_path=str(script_file))
    v = ValidationRecord(exp_id="e001", verdict="approve")
    L = Ledger(
        problem=Problem(goal="g", target="y", metric="accuracy", direction="maximize",
                        task_type="binary", train_path=str(tmp_path / "train.csv")),
        env=Env(budget=Budget(max_experiments=10, max_minutes=60, tune_trials=10)),
        runs=[r],
        validation=[v],
    )

    monkeypatch.setattr(tuner, "load", lambda _ws: L)
    monkeypatch.setattr(docs, "ensure", lambda _ws, _libs: None)
    monkeypatch.setattr(tuner, "minutes_elapsed", lambda _L: 10.0)

    mock_write_and_run = MagicMock()
    monkeypatch.setattr(tuner, "write_and_run", mock_write_and_run)

    warned = []
    monkeypatch.setattr(tuner.ui, "warn", lambda msg: warned.append(msg))

    tuner.run(ws)

    # write_and_run must NOT have been called because script > 24,000 chars was skipped
    assert not mock_write_and_run.called
    assert any("script longer than 24000 chars; skipping tuning" in w for w in warned)


# ---- Recorded Prompts Integration Test ---------------------------------------------------

class RecordingFakeLLM:
    def __init__(self, inner):
        self.inner = inner
        self.recorded_texts = []

    def _record(self, *args):
        for a in args:
            if isinstance(a, str):
                self.recorded_texts.append(a)

    def json(self, system, user, schema):
        self._record(system, user)
        return self.inner.json(system, user, schema)

    def code(self, system, user):
        self._record(system, user)
        return self.inner.code(system, user)

    def chat(self, system, user):
        self._record(system, user)
        return self.inner.chat(system, user)


def check_no_unrendered_placeholders(raw_text: str) -> None:
    tag_pattern = re.compile(
        r"<(?:script|base_script|previous_script)\b[^>]*>.*?</(?:script|base_script|previous_script)>",
        re.DOTALL
    )
    fence_pattern = re.compile(r"```.*?```", re.DOTALL)

    cleaned = tag_pattern.sub("", raw_text)
    cleaned = fence_pattern.sub("", cleaned)

    # 1. No <<word>> placeholders remain
    placeholder_m = re.search(r"<<([a-zA-Z0-9_]+)>>", cleaned)
    if placeholder_m:
        raise AssertionError(f"Contains unrendered placeholder: <<{placeholder_m.group(1)}>>")

    # Remove only brace groups that contain a double quote (JSON or dict literals with strings)
    cur = cleaned
    prev = None
    while prev != cur:
        prev = cur
        cur = re.sub(r'\{[^{}]*"[^{}]*\}', '', cur)

    # A '}}' is allowed only on lines that contain emit(
    for line in cur.splitlines():
        if "}}" in line and "emit(" not in line:
            raise AssertionError(f"Line contains '}}' without emit(: {line.strip()!r}")

    # 2. No {identifier} or {identifier:format} format placeholder remains
    bare_m = re.search(r"\{([a-zA-Z_][a-zA-Z0-9_]*(?::[^\s\(\)\{\},]+)?)\}", cur)
    if bare_m:
        raise AssertionError(f"Contains unrendered format placeholder: {{{bare_m.group(1)}}}")


def test_check_no_unrendered_placeholders_negative():
    """Verify negative cases are caught by check_no_unrendered_placeholders."""
    import pytest

    # Deliberate negative test: render a template with one missing placeholder
    template_tag = "System prompt with <<required_info>> and <<missing_info>>"
    partial_tag = template_tag.replace("<<required_info>>", "Done")
    with pytest.raises(AssertionError, match=r"unrendered placeholder: <<missing_info>>"):
        check_no_unrendered_placeholders(partial_tag)

    template_fmt = "Model evaluation result: {model} scored {cv:.4f}"
    partial_fmt = template_fmt.replace("{model}", "LGBM")
    with pytest.raises(AssertionError, match=r"unrendered format placeholder: \{cv:\.4f\}"):
        check_no_unrendered_placeholders(partial_fmt)

    with pytest.raises(AssertionError, match="without emit"):
        check_no_unrendered_placeholders("Random line with }} closing braces")

    # Positive sanity checks that must NOT raise:
    check_no_unrendered_placeholders("End with: emit({'answers': {'<question>': 'ans'}})")
    check_no_unrendered_placeholders('{"problem": {"goal": "Predict", "target": "y"}}')


def test_render_single_pass_no_rescan_and_strict_keys():
    """Verify render() single-pass behavior, no nested rescan, and strict keys validation."""
    import pytest

    from mlagent.prompts import render

    template = "Exp <<exp_id>>: change is <<change>>, code is <<code>>"
    rendered = render(
        template,
        exp_id="e001",
        change="add feature",
        code="print('<<injected_placeholder>>')",
    )
    # 1. Injected placeholder in value is NOT rescanned
    assert "<<injected_placeholder>>" in rendered
    assert "Exp e001: change is add feature" in rendered

    # 2. Missing key raises KeyError
    with pytest.raises(KeyError, match="Missing required placeholder"):
        render(template, exp_id="e001", change="add feature")

    # 3. Unknown key raises KeyError
    with pytest.raises(KeyError, match="Unknown placeholder"):
        render(template, exp_id="e001", change="add feature", code="c", extra_key="bad")


def test_recorded_prompts_have_no_unrendered_placeholders(tmp_path):
    import argparse
    import json

    from fakes import FakeLLM, make_data

    from mlagent import cli, llm
    from mlagent.agents import experiment_runner, results_analyzer
    from mlagent.ledger import session
    from mlagent.state import QueueItem

    train = make_data(tmp_path / "d")
    fake = RecordingFakeLLM(FakeLLM(api_exp="e001"))
    llm.set_llm(fake)
    ns = argparse.Namespace(
        yes=True, fresh=True, max_experiments=2, max_minutes=5, seed=42, workers=1,
        model=None, target=None, metric=None, goal=None, test=None, id_col=None, sample_submission=None
    )
    # 1. Runs pipeline covering data_profiler, experiment_planner, retry path, docs, validator, tuner, finisher
    ws = cli.start(train, ns)

    # 2. Runs downgrades 1 and 2
    with session(ws) as L:
        L.queue.append(QueueItem(id="e_d1", hypothesis="h1", change="c1", kind="model", params={"downgrade": 1}, status="pending"))
        L.queue.append(QueueItem(id="e_d2", hypothesis="h2", change="c2", kind="model", params={"downgrade": 2}, status="pending"))
    experiment_runner.run(ws, "e_d1")
    experiment_runner.run(ws, "e_d2")

    # 3. Runs all analyzer triggers (periodic, empty_queue, crash, tuner_failure, cv_lb_gap)
    triggers = ["periodic", "empty_queue", "crash", "tuner_failure", "cv_lb_gap"]
    for trig in triggers:
        if trig == "crash":
            results_analyzer.run(ws, trigger=trig, error="RuntimeError: logic bug in script")
        elif trig == "cv_lb_gap":
            results_analyzer.run(ws, trigger=trig, error=json.dumps({"flagged": True, "cv": 0.85, "lb": 0.65}))
        else:
            results_analyzer.run(ws, trigger=trig)

    assert len(fake.recorded_texts) > 0, "No prompts were recorded during the run"

    for i, raw_text in enumerate(fake.recorded_texts):
        try:
            check_no_unrendered_placeholders(raw_text)
        except AssertionError as e:
            raise AssertionError(f"Recorded text {i} failed check: {e}\nText preview: {raw_text[:200]}") from e


