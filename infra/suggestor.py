from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from infra.models import ModelTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UseCaseSuggestion:
    task: ModelTask
    reason: str
    confidence: float | None = None
    source: str = "heuristic"


ALIASES: dict[str, ModelTask] = {
    "chat": ModelTask.CHAT,
    "conversation": ModelTask.CHAT,
    "assistant": ModelTask.CHAT,
    "code": ModelTask.CODING,
    "coding": ModelTask.CODING,
    "programming": ModelTask.CODING,
    "developer": ModelTask.CODING,
    "reasoning": ModelTask.REASONING,
    "math": ModelTask.REASONING,
    "rag": ModelTask.RAG,
    "retrieval": ModelTask.RAG,
    "summary": ModelTask.SUMMARIZATION,
    "summarization": ModelTask.SUMMARIZATION,
    "general": ModelTask.GENERAL,
}


def resolve_use_case(value: str) -> ModelTask:
    normalized = value.strip().lower()

    task = ALIASES.get(normalized)

    if task is None:
        supported = ", ".join(sorted(ALIASES))

        raise ValueError(f"Unknown use case '{value}'. Supported values: {supported}")

    return task


def suggest_use_case_heuristic(text: str) -> UseCaseSuggestion:
    normalized = text.strip().lower()

    if any(
        word in normalized
        for word in (
            "code",
            "coding",
            "program",
            "python",
            "rust",
            "developer",
        )
    ):
        return UseCaseSuggestion(
            task=ModelTask.CODING,
            reason="The request appears programming-oriented.",
        )

    if any(
        word in normalized
        for word in (
            "reason",
            "math",
            "solve",
            "logic",
        )
    ):
        return UseCaseSuggestion(
            task=ModelTask.REASONING,
            reason="The request appears reasoning-oriented.",
        )

    if any(
        word in normalized
        for word in (
            "summar",
            "shorten",
            "summary",
        )
    ):
        return UseCaseSuggestion(
            task=ModelTask.SUMMARIZATION,
            reason="The request appears summarization-oriented.",
        )

    if any(
        word in normalized
        for word in (
            "rag",
            "documents",
            "retrieval",
            "knowledge base",
        )
    ):
        return UseCaseSuggestion(
            task=ModelTask.RAG,
            reason="The request appears retrieval-oriented.",
        )

    if any(
        word in normalized
        for word in (
            "chat",
            "conversation",
            "assistant",
        )
    ):
        return UseCaseSuggestion(
            task=ModelTask.CHAT,
            reason="The request appears conversational.",
        )

    return UseCaseSuggestion(
        task=ModelTask.GENERAL,
        reason="No specific workload was detected.",
    )


def suggest_use_case(
    text: str,
    *args: Any,
    **kwargs: Any,
) -> UseCaseSuggestion:
    return suggest_use_case_heuristic(text)


def supported_use_cases() -> list[str]:
    return sorted(ALIASES.keys())
