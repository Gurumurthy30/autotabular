"""Unit tests for briefs, context budgets, compact_eda, and concern deduplication."""

import hashlib

from app.core.briefs import (
    JUDGE_CONTEXT_LIMIT,
    REPORT_CONTEXT_LIMIT,
    SUPERVISOR_CONTEXT_LIMIT,
    WORKER_BRIEF_LIMIT,
    build_worker_brief,
    check_and_log_budget,
    compact_eda,
    render_supervisor_context,
)
from app.core.run_memory import RunMemory
from app.core.schemas import LedgerRow


def test_compact_eda_top_12_priority_order(tmp_path, monkeypatch):
    """Tests compact_eda selects top 12 ordered high > med > low with expected schema."""
    monkeypatch.setattr("app.core.briefs.PROJECTS_DIR", tmp_path)

    # Create 20 findings across low, med, high priorities
    raw_findings = []
    for i in range(5):
        raw_findings.append({
            "id": f"low_{i}",
            "category": "skew",
            "finding": f"Low priority finding {i}",
            "priority": "low",
            "action": f"Action for low {i}",
        })
    for i in range(10):
        raw_findings.append({
            "id": f"high_{i}",
            "category": "leakage",
            "finding": f"High priority finding {i}",
            "priority": "high",
            "action": f"Action for high {i}",
        })
    for i in range(5):
        raw_findings.append({
            "id": f"med_{i}",
            "category": "missingness",
            "finding": f"Medium priority finding {i}",
            "priority": "medium",
            "action": f"Action for med {i}",
        })

    eda_data = {"findings": raw_findings}
    compact_str = compact_eda(eda_data, project_id="p_test")

    lines = compact_str.strip().split("\n")
    # Up to 12 items + 1 overflow note
    assert len(lines) == 13
    
    # 10 high + 2 medium = 12 findings
    assert any("high_0 | leakage | High priority finding 0 | Action for high 0" in l for l in lines)
    assert any("high_9 | leakage | High priority finding 9 | Action for high 9" in l for l in lines)
    assert any("med_0" in l for l in lines)
    assert any("8 more in" in l for l in lines)

    # High items must appear before medium items
    high_idx = [i for i, l in enumerate(lines) if "high_" in l]
    med_idx = [i for i, l in enumerate(lines) if "med_" in l]
    assert max(high_idx) < min(med_idx)


def test_briefs_and_context_under_budgets(tmp_path, monkeypatch):
    """Tests that Supervisor, Worker, Judge, and Report briefs stay strictly under budget limits."""
    monkeypatch.setattr("app.core.briefs.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)

    mem = RunMemory("p_test", "r_test")
    # Populate large ledger (25 rows)
    for i in range(25):
        mem.append_ledger(LedgerRow(
            step=i,
            ts="2026-09-29T12:00:00Z",
            agent="model" if i % 2 == 0 else "fe",
            action="model" if i % 2 == 0 else "fe",
            brief_summary=f"Brief for step {i}",
            status="ok",
            result_summary=f"Result for step {i} with details",
            version_id="v1",
            score=0.85,
            delta_vs_best=0.01,
            judge_overall=None,
            issues=[],
            concern=None,
            decision_reason="Standard supervisor reasoning",
            duration_s=3.0,
        ))

    state = {
        "project_id": "p_test",
        "run_id": "r_test",
        "step": 25,
        "target_column": "target",
        "target_metric": "roc_auc",
        "worker_runs": {"profile": 1, "eda": 2, "fe": 3, "model": 3},
        "profile_summary": {"rows": 1000, "cols": 15, "missing_pct": 0.05, "target_type": "binary"},
        "eda_findings": {"findings": [{"id": f"f_{i}", "category": "c", "finding": "f", "priority": "high", "action": "a"} for i in range(15)]},
        "current_version": "v1",
    }

    # 1. Supervisor context budget (<= 6000)
    sup_ctx = render_supervisor_context(
        mem=mem,
        state=state,
        guard_note="Sample guard note",
        open_concern={"claim": "Possible leakage", "evidence": "col_a", "suggestion": "drop col_a"},
    )
    assert len(sup_ctx) <= SUPERVISOR_CONTEXT_LIMIT
    assert check_and_log_budget("supervisor", sup_ctx, SUPERVISOR_CONTEXT_LIMIT) is True

    # 2. Worker briefs (<= 5000 for each agent)
    decision = {
        "thought": "Let's build models",
        "action": "model",
        "brief": {
            "objective": "Train gradient boosting and random forest candidates",
            "focus_points": ["Balance class weights", "Check train/val gap"],
            "constraints": ["Diverse families"],
            "mode": "add_candidates",
        },
        "reason": "EDA confirmed no leakage and FE created 12 features",
    }

    for agent_name in ("profile", "eda", "fe", "model"):
        worker_brief = build_worker_brief(agent_name, decision, mem, state)
        assert len(worker_brief) <= WORKER_BRIEF_LIMIT
        assert check_and_log_budget(agent_name, worker_brief, WORKER_BRIEF_LIMIT) is True

    # 3. Judge context budget (<= 7000)
    judge_brief = build_worker_brief("judge", decision, mem, state)
    assert len(judge_brief) <= JUDGE_CONTEXT_LIMIT
    assert check_and_log_budget("judge", judge_brief, JUDGE_CONTEXT_LIMIT) is True

    # 4. Report context budget (<= 6000)
    report_brief = build_worker_brief("report", decision, mem, state)
    assert len(report_brief) <= REPORT_CONTEXT_LIMIT
    assert check_and_log_budget("report", report_brief, REPORT_CONTEXT_LIMIT) is True


def test_concern_deduplication():
    """Tests concern deduplication: normalized text hash already raised is ignored."""
    name = "fe"
    concern1 = {"claim": "Target column appears correlated with ID column", "evidence": "r=0.99", "suggestion": "drop ID"}
    concern2 = {"claim": "Target column appears correlated with ID column", "evidence": "r=0.99", "suggestion": "drop ID"}
    concern3 = {"claim": "High missingness in Age", "evidence": "pct=0.45", "suggestion": "impute median"}

    def hash_concern(agent, c):
        norm = f"{agent}:{c.get('claim', '')}".strip().lower()
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    h1 = hash_concern(name, concern1)
    h2 = hash_concern(name, concern2)
    h3 = hash_concern(name, concern3)

    assert h1 == h2
    assert h1 != h3

    seen_concerns = []
    # First time: accepted
    if h1 not in seen_concerns:
        seen_concerns.append(h1)
        accepted_1 = True
    else:
        accepted_1 = False
    assert accepted_1 is True

    # Second time (identical concern): rejected
    if h2 not in seen_concerns:
        seen_concerns.append(h2)
        accepted_2 = True
    else:
        accepted_2 = False
    assert accepted_2 is False

    # Third time (different concern): accepted
    if h3 not in seen_concerns:
        seen_concerns.append(h3)
        accepted_3 = True
    else:
        accepted_3 = False
    assert accepted_3 is True


def test_fe_dtype_check_rejects_unencoded_string_columns(tmp_path, monkeypatch):
    """Tests that FE contract check rejects transformed sample containing object/string columns."""
    from unittest.mock import MagicMock
    import pandas as pd
    from app.agents.feature_engineering import run_feature_engineering

    # Setup project dir
    proj_dir = tmp_path / "test_fe_proj"
    data_dir = proj_dir / "datasets" / "dataset_v1"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "embarked": ["S", "C", "Q"],
        "target": [0, 1, 0],
    })
    df.to_csv(data_dir / "data.csv", index=False)

    monkeypatch.setattr("app.agents.feature_engineering.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.ml_harness.paths.PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)

    state = {
        "project_id": "test_fe_proj",
        "run_id": "r1",
        "dataset_version": "dataset_v1",
        "target_column": "target",
        "task_type": "binary_classification",
        "step": 2,
        "supervisor_decision": {"brief": {"mode": "new"}},
    }

    # Pipeline script that leaves 'embarked' as raw string
    pipeline_code = '''import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class BadTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.copy()

def make_feature_pipeline():
    return Pipeline([("bad", BadTransformer())])
'''

    mock_coder = MagicMock()
    mock_coder.run_task.return_value = {
        "status": "SUCCESS",
        "success": True,
        "code": pipeline_code,
        "full_stdout": "",
        "files_created": [],
    }

    mock_router = MagicMock()
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = MagicMock(dataset=None, files=MagicMock(), execution=MagicMock())

    monkeypatch.setattr("app.agents.feature_engineering.CoderSubAgent", lambda *args, **kwargs: mock_coder)

    # When BadTransformer is returned, it fails dry run because 'embarked' is object/string,
    # and falls back to make_basic_preprocessor which encodes all categoricals.
    res = run_feature_engineering(state, mock_router, mock_registry)
    assert res["status"] == "SUCCESS"
    # Basic preprocessor was used as fallback, converting all to numeric
    assert "remainder__embarked" not in res["feature_summary"].get("kept_features", [])


def test_resolve_target_column_case_insensitive_and_missing(tmp_path, monkeypatch):
    """Tests case-insensitive resolution of target column against dataset headers, and error on missing."""
    import pandas as pd
    import pytest
    from app.tools.dataset_tools import DatasetTools

    proj_dir = tmp_path / "test_target_proj"
    data_dir = proj_dir / "datasets" / "dataset_v1"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"PassengerId": [1, 2], "Survived": [0, 1], "Age": [22, 38]})
    df.to_csv(data_dir / "data.csv", index=False)

    monkeypatch.setattr("app.tools.dataset_tools.PROJECTS_DIR", tmp_path)

    tools = DatasetTools("test_target_proj")

    # 1. Exact match
    assert tools.resolve_target_column("dataset_v1", "Survived") == "Survived"

    # 2. Case-insensitive match (lowercase)
    assert tools.resolve_target_column("dataset_v1", "survived") == "Survived"

    # 3. Case-insensitive match (mixed case)
    assert tools.resolve_target_column("dataset_v1", "sUrViVeD") == "Survived"

    # 4. Nonexistent column raises ValueError naming headers
    with pytest.raises(ValueError, match="Target column 'nonexistent' not found in dataset headers"):
        tools.resolve_target_column("dataset_v1", "nonexistent")

    # 5. Empty or whitespace target raises ValueError
    with pytest.raises(ValueError, match="Target column must be specified and non-empty"):
        tools.resolve_target_column("dataset_v1", "  ")


def test_model_plan_validation():
    """Tests Pydantic schema validation for model experiment plans across formats."""
    import pytest
    from app.agents.model import ModelExperimentItem, parse_and_validate_plan

    # 1. Valid list of items
    raw_list = '[{"family": "RandomForestClassifier", "params_or_search_space": {"n_estimators": 50}, "reason": "robust baseline"}]'
    items = parse_and_validate_plan(raw_list)
    assert len(items) == 1
    assert items[0].family == "RandomForestClassifier"
    assert items[0].params_or_search_space == {"n_estimators": 50}

    # 2. Valid dict with 'plan' key wrapped in markdown fences
    raw_dict = """```json
    {
      "plan": [
        {"family": "HistGradientBoostingClassifier", "params_or_search_space": {"max_iter": 100}, "reason": "handles missing values"},
        {"family": "LogisticRegression", "params_or_search_space": {"C": 1.0}, "reason": "linear baseline"}
      ]
    }
    ```"""
    items = parse_and_validate_plan(raw_dict)
    assert len(items) == 2
    assert items[0].family == "HistGradientBoostingClassifier"
    assert items[1].family == "LogisticRegression"

    # 3. Invalid JSON raises ValueError
    with pytest.raises(ValueError):
        parse_and_validate_plan("This is not JSON at all")

    # 4. Empty plan raises ValueError
    with pytest.raises(ValueError, match="Plan contains 0 candidates"):
        parse_and_validate_plan('{"plan": []}')


def test_model_duplicate_rejection():
    """Tests that filter_duplicate_candidates rejects items whose family+params already exist in models_tried."""
    from app.agents.model import ModelExperimentItem, filter_duplicate_candidates
    from app.core.knowledge import ModelTried

    models_tried = [
        ModelTried(
            id="rf_1",
            family="RandomForestClassifier",
            params_summary='{"max_depth": 10, "n_estimators": 100}',
            cv_mean=0.82,
            outcome="tested",
        ),
        ModelTried(
            id="lgb_1",
            family="LGBMClassifier",
            params_summary="{}",
            cv_mean=0.85,
            outcome="best",
        ),
    ]

    proposed = [
        # Exact duplicate of rf_1 (different param key order)
        ModelExperimentItem(family="RandomForestClassifier", params_or_search_space={"n_estimators": 100, "max_depth": 10}, reason="retry"),
        # Same family, different params -> KEPT
        ModelExperimentItem(family="RandomForestClassifier", params_or_search_space={"n_estimators": 200, "max_depth": 5}, reason="tuned"),
        # Exact duplicate of lgb_1 (empty params) -> REJECTED
        ModelExperimentItem(family="LGBMClassifier", params_or_search_space={}, reason="retry"),
        # New family -> KEPT
        ModelExperimentItem(family="CatBoostClassifier", params_or_search_space={"iterations": 100}, reason="new"),
    ]

    kept, rejected = filter_duplicate_candidates(proposed, models_tried)
    assert len(kept) == 2
    assert kept[0].family == "RandomForestClassifier"
    assert kept[0].params_or_search_space == {"n_estimators": 200, "max_depth": 5}
    assert kept[1].family == "CatBoostClassifier"
    assert len(rejected) == 2
    assert "RandomForestClassifier" in rejected[0]
    assert "LGBMClassifier" in rejected[1]


# --- Task 2A-Fix Unit Tests ---

def test_default_metric_per_task_type():
    """Tests that get_default_metric maps binary -> roc_auc, multiclass -> f1_macro, regression -> rmse."""
    from app.core.state import TaskType, get_default_metric

    # Binary classification
    assert get_default_metric("binary_classification") == "roc_auc"
    assert get_default_metric("binary") == "roc_auc"
    assert get_default_metric(TaskType.BINARY_CLASSIFICATION) == "roc_auc"

    # Multiclass classification
    assert get_default_metric("multiclass_classification") == "f1_macro"
    assert get_default_metric("multiclass") == "f1_macro"
    assert get_default_metric(TaskType.MULTICLASS_CLASSIFICATION) == "f1_macro"

    # Regression
    assert get_default_metric("regression") == "rmse"
    assert get_default_metric(TaskType.REGRESSION) == "rmse"

    # None / unknown fallback
    assert get_default_metric(None) == "roc_auc"
    assert get_default_metric("unknown") == "roc_auc"


def test_llm_guard_429_then_429_then_error():
    """Tests fake client: 429, then 429, then error after at most 2 retries."""
    import pytest
    from unittest.mock import MagicMock
    from langchain_core.messages import HumanMessage
    from app.core.model_router import GuardedChatModel, LLMRateLimitError, circuit_breaker

    circuit_breaker.reset()

    class FakeHTTPResponse:
        def __init__(self, status_code=429, headers=None):
            self.status_code = status_code
            self.headers = headers or {"Retry-After": "1.5"}

    class FakeHTTP429(Exception):
        def __init__(self, msg="Too Many Requests", headers=None):
            super().__init__(msg)
            self.status_code = 429
            self.response = FakeHTTPResponse(429, headers)

    fake_client = MagicMock()
    fake_client.invoke.side_effect = [
        FakeHTTP429("Rate limited attempt 1", {"Retry-After": "2.0"}),
        FakeHTTP429("Rate limited attempt 2", {"Retry-After": "3.0"}),
        FakeHTTP429("Rate limited attempt 3", {"Retry-After": "4.0"}),
    ]

    sleep_calls = []
    guarded = GuardedChatModel(fake_client, sleep_fn=lambda s: sleep_calls.append(s))

    with pytest.raises(LLMRateLimitError) as exc_info:
        guarded.invoke([HumanMessage(content="test")])

    assert "HTTP 429" in str(exc_info.value)
    assert fake_client.invoke.call_count == 3  # initial + 2 retries
    assert sleep_calls == [2.0, 3.0]
    assert circuit_breaker.consecutive_rate_limits == 1


def test_llm_guard_429_then_success():
    """Tests fake client: 429, then success on 1st retry."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage, HumanMessage
    from app.core.model_router import GuardedChatModel, circuit_breaker

    circuit_breaker.reset()

    class FakeHTTPResponse:
        def __init__(self, status_code=429, headers=None):
            self.status_code = status_code
            self.headers = headers or {"Retry-After": "0.5"}

    class FakeHTTP429(Exception):
        def __init__(self, msg="Too Many Requests", headers=None):
            super().__init__(msg)
            self.status_code = 429
            self.response = FakeHTTPResponse(429, headers)

    fake_client = MagicMock()
    fake_client.invoke.side_effect = [
        FakeHTTP429("Rate limited attempt 1"),
        AIMessage(content="{'success': true}"),
    ]

    sleep_calls = []
    guarded = GuardedChatModel(fake_client, sleep_fn=lambda s: sleep_calls.append(s))

    res = guarded.invoke([HumanMessage(content="test")])
    assert res.content == "{'success': true}"
    assert fake_client.invoke.call_count == 2
    assert sleep_calls == [20.0]  # default fallback retry-after
    assert circuit_breaker.consecutive_rate_limits == 0


def test_circuit_breaker_halts_run_after_two_consecutive_errors():
    """Tests that after 2 consecutive LLMRateLimitErrors, circuit breaker halts the run."""
    import pytest
    from unittest.mock import MagicMock
    from langchain_core.messages import HumanMessage
    from app.core.model_router import GuardedChatModel, LLMRateLimitError, circuit_breaker
    from app.graph.build_graph import build_ml_graph

    circuit_breaker.reset()

    class FakeHTTPResponse:
        status_code = 429
        headers = {}

    class FakeHTTP429(Exception):
        status_code = 429
        response = FakeHTTPResponse()

    fake_model = MagicMock()
    fake_model.invoke.side_effect = FakeHTTP429("Rate limit")

    guarded = GuardedChatModel(fake_model, sleep_fn=lambda s: None)

    # 1st call fails
    with pytest.raises(LLMRateLimitError):
        guarded.invoke([HumanMessage(content="call 1")])
    assert circuit_breaker.consecutive_rate_limits == 1
    assert not circuit_breaker.tripped

    # 2nd call fails -> circuit breaker trips!
    with pytest.raises(LLMRateLimitError):
        guarded.invoke([HumanMessage(content="call 2")])
    assert circuit_breaker.consecutive_rate_limits == 2
    assert circuit_breaker.tripped

    # Graph supervisor execution must halt immediately without looping
    mock_router = MagicMock()
    mock_router.get_model.return_value = guarded

    graph = build_ml_graph("test_cb_proj", router=mock_router)
    initial_state = {
        "project_id": "test_cb_proj",
        "run_id": "r_cb",
        "step": 1,
        "status": "RUNNING",
    }
    final = graph.invoke(initial_state)
    assert final["status"] == "FAILED"
    assert "LLM rate limited" in final["error"]


def test_worker_try_except_records_traceback_and_error_text(tmp_path, monkeypatch):
    """Tests that unhandled worker exceptions log full traceback and store last 1500 chars in report.error_text."""
    from unittest.mock import MagicMock
    from app.graph.build_graph import make_worker_node
    from app.core.run_memory import RunMemory

    proj_dir = tmp_path / "test_tb_proj"
    run_dir = proj_dir / "runs" / "r_tb"
    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.core.run_memory.PROJECTS_DIR", tmp_path)

    mem = RunMemory("test_tb_proj", "r_tb")

    def failing_worker(state, router, registry, brief=None):
        def deep_function():
            raise RuntimeError("Deep nested transformer calculation failed completely with detailed error message")
        deep_function()

    mock_router = MagicMock()
    mock_registry = MagicMock()

    node = make_worker_node("model", failing_worker, mock_router, mock_registry)
    state = {
        "project_id": "test_tb_proj",
        "run_id": "r_tb",
        "step": 3,
        "supervisor_decision": {"brief": {"objective": "train model"}},
    }

    result = node(state)

    assert result["status"] == "RUNNING"  # worker failure does not crash workflow
    report = result["report"]
    assert report["status"] == "failed"
    assert "error_text" in report
    assert "Deep nested transformer calculation failed" in report["error_text"]
    assert len(report["error_text"]) <= 1500
    assert "Traceback" in report["evidence"]["traceback"]

    # Verify ledger entry received error_text
    ledger = mem.read_ledger()
    assert len(ledger) == 1
    assert ledger[0].error_text is not None
    assert "Deep nested transformer" in ledger[0].error_text



