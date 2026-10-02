from dataclasses import dataclass
from enum import Enum

from infra.models import ModelConfig


class Precision(Enum):
    FP32 = ("fp32", 4.0)
    FP16 = ("fp16", 2.0)
    BF16 = ("bf16", 2.0)
    INT8 = ("int8", 1.0)
    INT4 = ("int4", 0.5)

    def __init__(self, label: str, bytes_per_value: float):
        self.label = label
        self.bytes_per_value = bytes_per_value


@dataclass
class EstimationRequest:
    context_length: int
    weight_precision: Precision = Precision.FP16
    kv_cache_precision: Precision = Precision.FP16


@dataclass
class ResourceEstimate:
    weight_memory_bytes: int
    kv_cache_memory_bytes: int
    total_memory_bytes: int
    context_length: int
    weight_precision: Precision
    kv_cache_precision: Precision


@dataclass
class RuntimeAwareEstimate:
    static_estimate: ResourceEstimate
    backend: str
    overhead_bytes: int
    safety_margin_bytes: int
    deployment_memory_bytes: int

    @property
    def total_overhead_bytes(self) -> int:
        return self.overhead_bytes + self.safety_margin_bytes


def validate_estimation_request(
    request: EstimationRequest,
    config: ModelConfig,
) -> None:
    if request.context_length <= 0:
        raise ValueError("context_length must be greater than zero")

    max_context = config.max_position_embeddings

    if max_context is not None and request.context_length > max_context:
        raise ValueError(
            f"Requested context length ({request.context_length}) "
            f"exceeds detected model maximum ({max_context})"
        )


def estimate_weight_memory(
    parameter_count: int,
    precision: Precision = Precision.FP16,
) -> int:
    if parameter_count <= 0:
        raise ValueError("parameter_count must be greater than zero")

    return int(parameter_count * precision.bytes_per_value)


def estimate_kv_cache_memory(
    config: ModelConfig,
    context_length: int,
    precision: Precision = Precision.FP16,
) -> int:
    if context_length <= 0:
        raise ValueError("context_length must be greater than zero")

    required = [
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
    ]

    if any(value is None for value in required):
        raise ValueError("Model configuration is missing fields required for KV-cache estimation")

    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")

    head_dimension = config.hidden_size // config.num_attention_heads

    values_per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * head_dimension

    return int(values_per_token * context_length * precision.bytes_per_value)


def estimate_resources(
    config: ModelConfig,
    parameter_count: int,
    request: EstimationRequest,
) -> ResourceEstimate:
    validate_estimation_request(request, config)

    weight_memory = estimate_weight_memory(
        parameter_count,
        request.weight_precision,
    )

    kv_cache_memory = estimate_kv_cache_memory(
        config,
        request.context_length,
        request.kv_cache_precision,
    )

    return ResourceEstimate(
        weight_memory_bytes=weight_memory,
        kv_cache_memory_bytes=kv_cache_memory,
        total_memory_bytes=weight_memory + kv_cache_memory,
        context_length=request.context_length,
        weight_precision=request.weight_precision,
        kv_cache_precision=request.kv_cache_precision,
    )


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(value)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} GiB"


def estimate_runtime_aware_memory(
    static_estimate: ResourceEstimate,
    *,
    backend: str,
    overhead_fraction: float = 0.20,
    safety_margin_fraction: float = 0.10,
) -> RuntimeAwareEstimate:
    """
    Estimate practical deployment memory.

    This intentionally does not use observed runtime measurements directly.
    Backend overhead is modeled separately from the theoretical model
    memory estimate so the result remains explicit and explainable.
    """

    if overhead_fraction < 0:
        raise ValueError("overhead_fraction cannot be negative")

    if safety_margin_fraction < 0:
        raise ValueError("safety_margin_fraction cannot be negative")

    static_memory = static_estimate.total_memory_bytes

    overhead_bytes = int(static_memory * overhead_fraction)

    safety_margin_bytes = int(static_memory * safety_margin_fraction)

    deployment_memory_bytes = static_memory + overhead_bytes + safety_margin_bytes

    return RuntimeAwareEstimate(
        static_estimate=static_estimate,
        backend=backend,
        overhead_bytes=overhead_bytes,
        safety_margin_bytes=safety_margin_bytes,
        deployment_memory_bytes=deployment_memory_bytes,
    )
