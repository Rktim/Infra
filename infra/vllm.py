from __future__ import annotations

import gc
import importlib.util

from infra.backend import (
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
    ModelLoadConfig,
)


class VLLMBackend(InferenceBackend):
    """
    vLLM inference backend.

    This backend runs vLLM inside the current Python process.

    It is different from the HTTP server mode implemented by
    infra.server.
    """

    def __init__(self) -> None:
        self._engine = None
        self._model_config: ModelLoadConfig | None = None

    @property
    def name(self) -> str:
        return "vLLM"

    def is_available(self) -> bool:
        """
        Check whether vLLM is installed.

        Do not import vLLM here because importing it can be
        expensive and may initialize runtime components.
        """

        return importlib.util.find_spec("vllm") is not None

    def load_model(
        self,
        config: ModelLoadConfig,
    ) -> None:
        """
        Load a local model into vLLM.

        This is an in-process inference backend.
        """

        if not self.is_available():
            raise RuntimeError("vLLM is not installed in the current environment")

        if self.is_loaded():
            raise RuntimeError("A model is already loaded. Call unload_model() first.")

        model_path = config.model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        if not model_path.is_dir():
            raise NotADirectoryError(f"Model path is not a directory: {model_path}")

        gpu_memory_utilization = config.gpu_memory_utilization

        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")

        max_model_len = config.max_model_len

        if max_model_len is not None and max_model_len <= 0:
            raise ValueError("max_model_len must be greater than zero")

        from vllm import LLM

        kwargs: dict[str, object] = {
            "model": str(model_path),
            "dtype": config.dtype,
            "gpu_memory_utilization": (gpu_memory_utilization),
        }

        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len

        try:
            self._engine = LLM(**kwargs)
            self._model_config = config

        except Exception:
            self._engine = None
            self._model_config = None

            self._cleanup_cuda()

            raise

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """
        Generate a response using the loaded vLLM engine.
        """

        if not self.is_loaded():
            raise RuntimeError("No model is loaded. Call load_model() first.")

        if not request.messages:
            raise ValueError("messages cannot be empty")

        if request.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        if request.temperature < 0:
            raise ValueError("temperature cannot be negative")

        if not 0 < request.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        try:
            outputs = self._engine.chat(
                messages,
                sampling_params=sampling_params,
            )

        except Exception as error:
            raise RuntimeError(f"vLLM generation failed: {error}") from error

        if not outputs:
            return GenerationResult(text="")

        first_output = outputs[0]

        if not first_output.outputs:
            return GenerationResult(text="")

        generated = first_output.outputs[0]

        return GenerationResult(text=generated.text)

    def unload_model(self) -> None:
        """
        Release the vLLM engine and attempt to release CUDA
        memory.
        """

        engine = self._engine

        self._engine = None
        self._model_config = None

        if engine is not None:
            del engine

        gc.collect()

        self._cleanup_cuda()

    def is_loaded(self) -> bool:
        return self._engine is not None

    @staticmethod
    def _cleanup_cuda() -> None:
        """
        Best-effort CUDA cleanup.

        CUDA cleanup must never turn a successful unload into
        an exception.
        """

        try:
            import torch
        except ImportError:
            return

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
