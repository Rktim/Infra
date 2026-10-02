from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass(frozen=True)
class GenerationResult:
    text: str


@dataclass(frozen=True)
class ModelLoadConfig:
    model_path: Path
    dtype: str = "auto"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.85


class InferenceBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, config: ModelLoadConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def unload_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError
