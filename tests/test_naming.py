"""Tests for naming compliance, graph node names, control migration, and prompt overrides.
pytest -q tests/test_naming.py
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from fakes import FakeLLM, make_data

from mlagent import cli, llm, pipeline_graph, prompts
from mlagent.ledger import Workspace, load, load_control, migrate_control_state

OLDER_FILENAMES = {
    "profiler.py",
    "strategist.py",
    "coder.py",
    "experimenter.py",
    "validator.py",
    "analyzer.py",
    "docs.py",
    "tuner.py",
    "finisher.py",
    "lb.py",
    "build_graph.py",
}

OLD_NODE_NAMES = {
    "profiler",
    "strategist",
    "controller",
    "experimenter",
    "validator",
    "analyzer",
    "tuner",
    "finisher",
}

NEW_NODE_NAMES = {
    "data_profiler",
    "experiment_planner",
    "pipeline_controller",
    "experiment_runner",
    "run_validator",
    "results_analyzer",
    "hyperparameter_tuner",
    "final_submission",
}


def test_no_older_files_in_agents_or_package():
    """Verify that all older shim files have been completely removed from agents/ and mlagent/."""
    repo_root = Path(__file__).resolve().parents[1]
    agents_dir = repo_root / "mlagent" / "agents"
    mlagent_dir = repo_root / "mlagent"

    for fname in OLDER_FILENAMES:
        assert not (agents_dir / fname).exists(), f"Older file still exists: agents/{fname}"
        assert not (mlagent_dir / fname).exists(), f"Older file still exists: mlagent/{fname}"


def test_no_old_module_imports_in_source_or_tests():
    """Verify that no active source file or test file imports legacy module names."""
    repo_root = Path(__file__).resolve().parents[1]
    checked_dirs = [repo_root / "mlagent", repo_root / "tests"]
    old_module_tails = {f.removesuffix(".py") for f in OLDER_FILENAMES}

    violations: list[str] = []
    for directory in checked_dirs:
        for py_file in directory.rglob("*.py"):
            if py_file.name == "__init__.py" or py_file.name == "test_naming.py":
                continue
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        parts = alias.name.split(".")
                        if any(p in old_module_tails for p in parts if p not in ("docs",)):
                            violations.append(f"{py_file.relative_to(repo_root)}:{node.lineno} imports {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    parts = mod.split(".")
                    if any(p in old_module_tails for p in parts if p not in ("docs",)):
                        violations.append(f"{py_file.relative_to(repo_root)}:{node.lineno} imports from {mod}")

    assert not violations, "Found legacy module imports in active code:\n" + "\n".join(violations)


def test_graph_has_only_new_node_names():
    """Verify that the compiled LangGraph pipeline contains only new node names and no old ones."""
    app = pipeline_graph.build_graph()
    nodes = set(app.get_graph().nodes.keys())

    assert NEW_NODE_NAMES.issubset(nodes), f"Missing new nodes: {NEW_NODE_NAMES - nodes}"
    assert not (nodes & OLD_NODE_NAMES), f"Found old nodes in graph: {nodes & OLD_NODE_NAMES}"


def test_old_format_control_json_migrates_all_nodes():
    """Verify migrate_control_state maps every legacy node name to its new counterpart."""
    from mlagent.ledger import OLD_TO_NEW_GRAPH_NODES

    for old_name, new_name in OLD_TO_NEW_GRAPH_NODES.items():
        state = {"next": old_name}
        migrated = migrate_control_state(state)
        assert migrated["next"] == new_name, f"Expected {old_name} -> {new_name}, got {migrated['next']}"


def test_old_format_control_json_fixture_resumes_to_end(tmp_path):
    """Load an OLD-format control.json fixture (with old node names) and resume a mocked run to the end."""
    train = make_data(tmp_path / "d")
    llm.set_llm(FakeLLM())
    ns = argparse.Namespace(
        yes=True, fresh=True, max_experiments=2, max_minutes=5, seed=42, workers=1,
        model=None, target=None, metric=None, goal=None, test=None, id_col=None, sample_submission=None
    )
    ws = cli.start(train, ns)

    # Overwrite control.json with old-format node names to simulate resuming an old workspace
    old_control = {
        "next": "controller",
        "retries": {},
        "oom_retries": {},
        "crashes": [],
        "pending_validation": [],
        "runs_since_analysis": 1,
        "analysis_trigger": "periodic",
        "analyzer_empty": False,
        "tuned": False,
        "tuner_ids": [],
        "tuner_retry_used": False,
        "reentry_left": 2,
        "resume_tuner": False,
        "abort": False,
    }
    (ws.root / "control.json").write_text(json.dumps(old_control, indent=2), encoding="utf-8")

    # Verify load_control migrates "controller" -> "pipeline_controller"
    loaded = load_control(ws)
    assert loaded["next"] == "pipeline_controller"

    # Resume the mocked run to completion
    cli.resume(ws.root)
    L = load(ws)
    assert L.status == "done"
    assert L.final.submission_path is not None


def test_prompt_override_file_loads_for_canonical_name(tmp_path):
    """Workspaces with <workspace>/prompts/experiment_planner.md load custom override text."""
    train = make_data(tmp_path / "d")
    ws = Workspace.for_data(train)
    prompts_dir = ws.root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Write prompt override file for experiment_planner
    override_text = "=== system ===\nCustom override text for experiment_planner.\n"
    (prompts_dir / "experiment_planner.md").write_text(override_text, encoding="utf-8")

    # Read sections for canonical new agent name
    secs = prompts.sections("experiment_planner", ws)
    assert secs["system"] == "Custom override text for experiment_planner."

    # Exporting prompts writes canonical file names
    exported_files = prompts.export(ws)
    exported_names = {f.name for f in exported_files}
    assert "experiment_planner.md" in exported_names
    assert "data_profiler.md" in exported_names
    assert len(exported_names) == len(prompts.AGENT_NAMES)
