from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from infra.estimator import Precision
from infra.hardware import HardwareProfile
from infra.metadata import ModelMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationParams:
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 2048
    enforce_eager: bool = False
    kv_cache_dtype: str = "auto"
    max_num_seqs: int = 8
    served_model_name: str | None = None
    tuning_rationale: str = ""
    source: str = "heuristic"
    confidence: float | None = None

    def as_vllm_args(self) -> list[str]:
        args = [
            "--gpu-memory-utilization",
            f"{self.gpu_memory_utilization:.2f}",
            "--max-model-len",
            str(self.max_model_len),
        ]
        if self.enforce_eager:
            args.append("--enforce-eager")
        if self.kv_cache_dtype != "auto":
            args.extend(["--kv-cache-dtype", self.kv_cache_dtype])
        if self.max_num_seqs != 8:
            args.extend(["--max-num-seqs", str(self.max_num_seqs)])
        if self.served_model_name:
            args.extend(["--served-model-name", self.served_model_name])
        return args


def calculate_kv_bytes_per_token(
    metadata: ModelMetadata,
    precision: Precision = Precision.FP16,
) -> int:
    """Calculate the bytes required per token of KV cache."""
    layers = metadata.num_hidden_layers or 28
    heads = metadata.num_attention_heads or 28
    kv_heads = metadata.num_key_value_heads or heads
    hidden_size = metadata.hidden_size or 3584
    head_dim = hidden_size // heads if heads > 0 else 128

    values_per_token = 2 * layers * kv_heads * head_dim
    return int(values_per_token * precision.bytes_per_value)


def calculate_max_safe_context(
    metadata: ModelMetadata,
    hardware: HardwareProfile,
    gpu_memory_utilization: float = 0.85,
    precision: Precision = Precision.FP16,
    safety_margin_bytes: int = 512 * 1024 * 1024,
) -> int:
    """Calculate the maximum safe context length (in tokens) that fits in GPU VRAM."""
    if not hardware.gpus or hardware.gpus[0].total_vram_bytes is None:
        return 2048

    vram = hardware.gpus[0].total_vram_bytes
    budget = int(vram * gpu_memory_utilization)

    param_count = metadata.parameter_count or 7_000_000_000
    weight_bytes = int(param_count * precision.bytes_per_value)

    available_for_kv = budget - weight_bytes - safety_margin_bytes
    if available_for_kv <= 0:
        return 1024

    bytes_per_token = calculate_kv_bytes_per_token(metadata, precision)
    if bytes_per_token <= 0:
        return 2048

    raw_tokens = available_for_kv // bytes_per_token
    native_max = metadata.max_context_length or 32768

    # Cap to standard usable power-of-2 context windows
    standard_windows = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    usable = 1024
    for w in standard_windows:
        if w <= raw_tokens and w <= native_max:
            usable = w
        else:
            break

    return usable


def optimize_params_heuristic(
    metadata: ModelMetadata,
    hardware: HardwareProfile,
    task: str = "general",
    target_context: int = 2048,
    gpu_memory_utilization: float = 0.85,
) -> OptimizationParams:
    """Deterministic, hardware-aware heuristic optimization."""
    vram = (
        hardware.gpus[0].total_vram_bytes
        if hardware.gpus and hardware.gpus[0].total_vram_bytes
        else 16 * 1024**3
    )
    param_count = metadata.parameter_count or 7_000_000_000
    weight_bytes = int(param_count * Precision.FP16.bytes_per_value)

    budget = int(vram * gpu_memory_utilization)
    headroom_bytes = budget - weight_bytes
    headroom_gb = headroom_bytes / (1024**3)

    max_safe = calculate_max_safe_context(metadata, hardware, gpu_memory_utilization)

    # In low-headroom setups (< 2.5 GB free), CUDA Graphs warmup can trigger OOM
    enforce_eager = headroom_gb < 2.5
    max_num_seqs = 4 if headroom_gb < 2.5 else (8 if headroom_gb < 5.0 else 16)

    # Select optimal context length
    if target_context <= max_safe:
        chosen_context = target_context
    else:
        chosen_context = max_safe

    # If coding or RAG task with ample headroom, try to expand to 4096 or 8192
    if task in ("coding", "rag", "reasoning") and max_safe >= 4096 and chosen_context < 4096:
        chosen_context = 4096

    served_name = metadata.model_id.split("/")[-1].lower()

    rationale_parts = []
    if enforce_eager:
        rationale_parts.append(
            f"Tight VRAM headroom ({headroom_gb:.1f} GiB) requires eager execution to prevent graph capture OOM."
        )
    else:
        rationale_parts.append(
            f"Ample headroom ({headroom_gb:.1f} GiB) allows full CUDA Graph acceleration for minimum latency."
        )

    rationale_parts.append(
        f"Context auto-tuned to {chosen_context:,} tokens (safe upper bound: {max_safe:,})."
    )

    return OptimizationParams(
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=chosen_context,
        enforce_eager=enforce_eager,
        kv_cache_dtype="auto",
        max_num_seqs=max_num_seqs,
        served_model_name=served_name,
        tuning_rationale=" ".join(rationale_parts),
        source="heuristic",
    )


def optimize_serving_params(
    metadata: ModelMetadata,
    hardware: HardwareProfile,
    task: str = "general",
    target_context: int = 2048,
    gpu_memory_utilization: float = 0.85,
    *args: Any,
    **kwargs: Any,
) -> OptimizationParams:
    """
    Unified optimizer entry point using analytical hardware heuristics.
    """
    return optimize_params_heuristic(
        metadata=metadata,
        hardware=hardware,
        task=task,
        target_context=target_context,
        gpu_memory_utilization=gpu_memory_utilization,
    )
