import pytest

from infra.models import ModelTask
from infra.suggestor import (
    resolve_use_case,
    suggest_use_case,
    supported_use_cases,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("chat", ModelTask.CHAT),
        ("conversation", ModelTask.CHAT),
        ("assistant", ModelTask.CHAT),
        ("code", ModelTask.CODING),
        ("coding", ModelTask.CODING),
        ("programming", ModelTask.CODING),
        ("developer", ModelTask.CODING),
        ("reasoning", ModelTask.REASONING),
        ("math", ModelTask.REASONING),
        ("rag", ModelTask.RAG),
        ("retrieval", ModelTask.RAG),
        ("summary", ModelTask.SUMMARIZATION),
        ("summarization", ModelTask.SUMMARIZATION),
        ("general", ModelTask.GENERAL),
    ],
)
def test_resolve_use_case(value, expected):
    assert resolve_use_case(value) == expected


def test_resolve_use_case_is_case_insensitive():
    assert resolve_use_case("CODING") == ModelTask.CODING
    assert resolve_use_case("Chat") == ModelTask.CHAT


def test_resolve_use_case_strips_whitespace():
    assert resolve_use_case("  coding  ") == ModelTask.CODING


def test_resolve_use_case_rejects_unknown_use_case():
    with pytest.raises(ValueError, match="Unknown use case"):
        resolve_use_case("unknown-task")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I want to build a Python application", ModelTask.CODING),
        ("Help me write Rust code", ModelTask.CODING),
        ("I need a programming assistant", ModelTask.CODING),
        ("Solve this mathematical problem", ModelTask.REASONING),
        ("I need logical reasoning", ModelTask.REASONING),
        ("Summarize this document", ModelTask.SUMMARIZATION),
        ("Give me a short summary", ModelTask.SUMMARIZATION),
        ("Search my documents using retrieval", ModelTask.RAG),
        ("Build a RAG system", ModelTask.RAG),
        ("I want a conversational assistant", ModelTask.CHAT),
        ("Let's have a conversation", ModelTask.CHAT),
        ("Do something with text", ModelTask.GENERAL),
    ],
)
def test_suggest_use_case(text, expected):
    result = suggest_use_case(text)

    assert result.task == expected
    assert result.reason


def test_suggest_use_case_strips_whitespace():
    result = suggest_use_case("   write Python code   ")

    assert result.task == ModelTask.CODING


def test_suggest_use_case_returns_reason():
    result = suggest_use_case("Build a Python application")

    assert isinstance(result.reason, str)
    assert len(result.reason) > 0


def test_supported_use_cases_contains_expected_values():
    values = supported_use_cases()

    assert "chat" in values
    assert "coding" in values
    assert "reasoning" in values
    assert "rag" in values
    assert "summarization" in values
    assert "general" in values


def test_supported_use_cases_is_sorted():
    values = supported_use_cases()

    assert values == sorted(values)


def test_supported_use_cases_contains_no_duplicates():
    values = supported_use_cases()

    assert len(values) == len(set(values))


def test_suggest_use_case_basic():
    result = suggest_use_case("write code")
    assert result.task == ModelTask.CODING
    assert result.source == "heuristic"
