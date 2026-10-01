"""Unit tests for Coder retry optimization, duplicate code hash detection, and bounded history."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from app.agents.coder import (
    CoderSubAgent,
    extract_error_summary,
)


def test_extract_error_summary():
    """Tests concise one-line extraction of python traceback errors."""
    stderr1 = """Traceback (most recent call last):
  File "train.py", line 42, in <module>
    X = df['survived']
KeyError: 'survived'"""
    s1 = extract_error_summary(stderr1, "")
    assert "KeyError" in s1
    assert "'survived'" in s1

    stderr2 = """Traceback (most recent call last):
  File "script.py", line 12, in <module>
    pipe.predict(X_test)
sklearn.exceptions.NotFittedError: This StandardScaler instance is not fitted yet."""
    s2 = extract_error_summary(stderr2, "")
    assert "NotFittedError" in s2

    stderr_empty = ""
    assert extract_error_summary(stderr_empty, "") == "Unknown execution error"


def test_identical_code_detection_and_bounded_history():
    """Tests that returning identical code does not re-execute, counts as attempt, and does not grow history."""
    mock_router = MagicMock()
    mock_file_tools = MagicMock()
    mock_execution = MagicMock()

    mock_llm = MagicMock()
    mock_router.get_model.return_value = mock_llm

    # LLM returns identical code twice
    identical_code = "```python\nimport pandas as pd\nprint(1)\n```"
    mock_llm.invoke.return_value = AIMessage(content=identical_code)

    # First execution fails with an error
    mock_execution.run_script.return_value = MagicMock(
        success=False,
        stdout="Running...",
        stderr="ZeroDivisionError: division by zero",
        exit_code=1,
    )

    coder = CoderSubAgent(
        project_id="test_proj",
        model_router=mock_router,
        file_tools=mock_file_tools,
        execution_manager=mock_execution,
    )

    result = coder.run_task(
        task_description="Compute statistics",
        context={"data": "test"},
        max_retries=2,
    )

    assert result["status"] == "FAILED"
    # Runner should only execute once because the second attempt returned identical code and was intercepted
    assert mock_execution.run_script.call_count == 1

    # Check error summaries recorded
    summaries = result.get("error_summaries", [])
    assert len(summaries) >= 1
    assert any("ZeroDivisionError" in s for s in summaries)

    # Verify bounded history: mock_llm.invoke should have received exactly 2 messages in each call
    for call_args in mock_llm.invoke.call_args_list:
        messages = call_args[0][0]
        # Must be bounded: exactly [SystemMessage, HumanMessage] (never growing list of N messages)
        assert len(messages) == 2
