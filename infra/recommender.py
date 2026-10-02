from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from types import SimpleNamespace

from infra.compatibility import (
    CompatibilityResult,
    check_compatibility,
)
from infra.discovery import (
    DiscoveredModel,
    DiscoveryRequest,
    HuggingFaceDiscovery,
)
from infra.estimator import (
    Precision,
    ResourceEstimate,
    RuntimeAwareEstimate,
    estimate_runtime_aware_memory,
    estimate_weight_memory,
)
from infra.hardware import (
    GPUInfo,
    HardwareProfile,
)
from infra.metadata import (
    HuggingFaceMetadata,
    ModelMetadata,
)
from infra.optimizer import OptimizationParams, optimize_serving_params

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommendationRequest:
    task: str

    context_length: int = 2048

    max_models: int = 5

    weight_precision: Precision = Precision.FP16

    kv_cache_precision: Precision = Precision.FP16

    backend: str = "vllm"

    gpu_memory_utilization: float = 0.90


@dataclass(frozen=True)
class ModelRecommendation:
    metadata: ModelMetadata

    static_estimate: ResourceEstimate

    runtime_estimate: RuntimeAwareEstimate

    static_compatibility: CompatibilityResult

    runtime_compatibility: CompatibilityResult

    score: float

    reasons: tuple[str, ...]

    optimization_params: OptimizationParams | None = None

    @property
    def status(self) -> str:
        """
        Human-readable compatibility status.
        """

        # A model that cannot statically fit into physical
        # GPU memory is not realistically deployable.
        if not self.static_compatibility.fits:
            return "✗ DOES NOT FIT"

        # Model weights fit, but the runtime-aware estimate
        # exceeds the configured serving memory budget.
        if not self.runtime_compatibility.fits:
            return "⚠ RUNTIME TIGHT"

        available = self.runtime_compatibility.available_memory_bytes

        remaining = self.runtime_compatibility.remaining_memory_bytes

        if available <= 0:
            return "UNKNOWN"

        ratio = remaining / available

        if ratio < 0.15:
            return "⚠ TIGHT"

        return "✓ FITS"


TASK_PROFILES: dict[str, dict[str, tuple[tuple[str, float], ...]]] = {
    "coding": {
        "positive": (
            ("coder", 18.0),
            ("code", 14.0),
            ("coding", 14.0),
            ("programming", 14.0),
            ("developer", 12.0),
            ("codegen", 14.0),
            ("starcoder", 18.0),
        ),
        "negative": (
            ("chat", 8.0),
            ("conversation", 8.0),
            ("reasoning", 6.0),
        ),
    },
    "chat": {
        "positive": (
            ("chat", 18.0),
            ("conversation", 16.0),
            ("conversational", 14.0),
            ("assistant", 14.0),
            ("instruct", 12.0),
        ),
        "negative": (
            ("coder", 18.0),
            ("coding", 16.0),
            ("programming", 16.0),
            ("codegen", 16.0),
        ),
    },
    "reasoning": {
        "positive": (
            ("deepseek-r1", 24.0),
            ("r1", 20.0),
            ("reasoning", 18.0),
            ("qwq", 18.0),
            ("think", 14.0),
            ("thinking", 14.0),
            ("math", 14.0),
            ("logic", 12.0),
            ("instruct", 4.0),
        ),
        "negative": (
            ("coder", 12.0),
            ("coding", 10.0),
            ("codegen", 10.0),
        ),
    },
    "rag": {
        "positive": (
            ("rag", 24.0),
            ("retrieval", 20.0),
            ("retriever", 18.0),
            ("128k", 14.0),
            ("long-context", 14.0),
            ("question-answer", 14.0),
            ("question_answer", 14.0),
            ("qa", 12.0),
            ("context", 10.0),
            ("instruct", 8.0),
        ),
        "negative": (
            ("coder", 14.0),
            ("coding", 12.0),
            ("programming", 12.0),
            ("codegen", 12.0),
        ),
    },
    "summarization": {
        "positive": (
            ("summarization", 20.0),
            ("summarisation", 20.0),
            ("summary", 18.0),
            ("summarize", 18.0),
            ("long-context", 14.0),
            ("instruct", 8.0),
        ),
        "negative": (
            ("coder", 14.0),
            ("coding", 12.0),
            ("programming", 12.0),
            ("codegen", 12.0),
        ),
    },
    "general": {
        "positive": (
            ("instruct", 10.0),
            ("assistant", 8.0),
            ("chat", 6.0),
            ("text-generation", 5.0),
            ("language-model", 5.0),
        ),
        "negative": (),
    },
}


def recommend_models(
    hardware: HardwareProfile,
    request: RecommendationRequest,
    *,
    refresh: bool = False,
) -> list[ModelRecommendation]:
    """
    Recommend models for the supplied hardware.

    Decision policy:

    1. Discover candidate models.
    2. Filter unsupported backends.
    3. Rank candidates cheaply.
    4. Fetch detailed metadata.
    5. Build static memory estimate.
    6. Reject models that cannot fit physical GPU memory.
    7. Build runtime-aware estimate.
    8. Classify runtime compatibility.
    9. Keep runtime-tight models visible.
    10. Rank recommendations.
    """

    if request.context_length <= 0:
        raise ValueError("context_length must be greater than zero")

    if request.max_models <= 0:
        raise ValueError("max_models must be greater than zero")

    if not (0 < request.gpu_memory_utilization <= 1):
        raise ValueError("gpu_memory_utilization must be between 0 and 1")

    if not hardware.gpus:
        return []

    discovery = HuggingFaceDiscovery()

    discovery_limit = max(
        100,
        request.max_models * 30,
    )

    candidates = discovery.discover(
        DiscoveryRequest(
            use_case=request.task,
            limit=discovery_limit,
        ),
        refresh=refresh,
    )

    if not candidates:
        return []

    candidates = [
        candidate
        for candidate in candidates
        if supports_backend(
            candidate.model_id,
            request.backend,
        )
    ]

    if not candidates:
        return []

    candidates.sort(
        key=lambda model: (
            cheap_relevance(
                model,
                request.task,
            ),
            math.log10(
                max(
                    1,
                    model.downloads,
                )
            ),
            math.log10(
                max(
                    1,
                    model.likes,
                )
            ),
        ),
        reverse=True,
    )

    shortlist_size = max(
        50,
        request.max_models * 15,
    )

    shortlist = candidates[:shortlist_size]

    metadata_store = HuggingFaceMetadata()

    recommendations: list[ModelRecommendation] = []

    for candidate in shortlist:
        metadata = metadata_store.get(
            candidate.model_id,
            refresh=refresh,
        )

        if metadata is None:
            continue

        if not supports_backend(
            metadata.model_id,
            request.backend,
        ):
            continue

        if not has_estimation_metadata(metadata):
            continue

        estimate = build_estimate(
            metadata,
            request,
        )

        if estimate is None:
            continue

        static_result, _ = check_compatibility(
            estimate=estimate,
            hardware=hardware,
            runtime_estimate=None,
        )

        if not static_result.fits:
            continue

        runtime_estimate = estimate_runtime_aware_memory(
            estimate,
            backend=request.backend,
            overhead_fraction=0.20,
            safety_margin_fraction=0.10,
        )

        runtime_hardware = build_runtime_hardware(
            hardware,
            request.gpu_memory_utilization,
        )

        _, runtime_result = check_compatibility(
            estimate=estimate,
            hardware=runtime_hardware,
            runtime_estimate=runtime_estimate,
        )

        if runtime_result is None:
            continue

        score = calculate_score(
            metadata,
            request,
            static_result,
            runtime_result,
        )

        reasons = build_reasons(
            metadata,
            request,
            static_result,
            runtime_result,
        )

        recommendations.append(
            ModelRecommendation(
                metadata=metadata,
                static_estimate=estimate,
                runtime_estimate=runtime_estimate,
                static_compatibility=static_result,
                runtime_compatibility=runtime_result,
                score=score,
                reasons=reasons,
            )
        )

    recommendations.sort(
        key=lambda item: (
            status_rank(item.status),
            -item.score,
            -max(
                0,
                item.metadata.downloads,
            ),
            -max(
                0,
                item.metadata.likes,
            ),
        )
    )

    top_models = recommendations[: request.max_models]
    optimized_results: list[ModelRecommendation] = []
    for item in top_models:
        params = optimize_serving_params(
            metadata=item.metadata,
            hardware=hardware,
            task=request.task,
            target_context=request.context_length,
            gpu_memory_utilization=request.gpu_memory_utilization,
        )
        optimized_results.append(replace(item, optimization_params=params))

    return optimized_results


def supports_backend(
    model_id: str,
    backend: str,
) -> bool:
    """
    Return whether a model repository is suitable for the
    selected inference backend.
    """

    normalized_model = model_id.strip().lower()

    normalized_backend = backend.strip().lower()

    if normalized_backend == "vllm":
        unsupported_markers = (
            "mlx",
            "gguf",
            "onnx",
            "coreml",
        )

        return not any(marker in normalized_model for marker in unsupported_markers)

    return True


def build_runtime_hardware(
    hardware: HardwareProfile,
    gpu_memory_utilization: float,
):
    """
    Create a hardware profile with GPU VRAM reduced to the
    memory fraction available to the serving backend.

    Supports both real HardwareProfile objects and the
    lightweight test objects used by the test suite.
    """

    if not (0 < gpu_memory_utilization <= 1):
        raise ValueError("gpu_memory_utilization must be between 0 and 1")

    runtime_gpus = []

    for gpu in hardware.gpus:
        runtime_vram = int(gpu.total_vram_bytes * gpu_memory_utilization)

        runtime_gpus.append(
            GPUInfo(
                name=gpu.name,
                total_vram_bytes=runtime_vram,
            )
        )

    # Real production hardware profile.
    if isinstance(
        hardware,
        HardwareProfile,
    ):
        return HardwareProfile(
            os_name=hardware.os_name,
            architecture=hardware.architecture,
            cpu=hardware.cpu,
            memory=hardware.memory,
            gpus=runtime_gpus,
            storage=hardware.storage,
        )

    # Lightweight test double.
    return SimpleNamespace(
        gpus=runtime_gpus,
    )


def has_estimation_metadata(
    metadata: ModelMetadata,
) -> bool:
    """
    Check whether metadata contains everything required for
    memory estimation.
    """

    required = (
        metadata.parameter_count,
        metadata.hidden_size,
        metadata.num_hidden_layers,
        metadata.num_attention_heads,
        metadata.num_key_value_heads,
        metadata.max_context_length,
    )

    return all(value is not None for value in required)


def resolve_weight_precision(
    metadata: ModelMetadata,
    requested: Precision,
) -> Precision:
    """
    Resolve effective model weight precision.

    Quantized model metadata overrides the default requested
    precision when the quantization type is known.
    """

    quantization = (metadata.quantization or "").strip().lower()

    if quantization in (
        "4bit",
        "int4",
        "nf4",
        "awq",
        "gptq",
    ):
        return Precision.INT4

    if quantization in (
        "8bit",
        "int8",
    ):
        return Precision.INT8

    return requested


def build_estimate(
    metadata: ModelMetadata,
    request: RecommendationRequest,
) -> ResourceEstimate | None:
    """
    Estimate model weight and KV-cache memory.
    """

    if not has_estimation_metadata(metadata):
        return None

    assert metadata.parameter_count is not None
    assert metadata.hidden_size is not None
    assert metadata.num_hidden_layers is not None
    assert metadata.num_attention_heads is not None
    assert metadata.num_key_value_heads is not None
    assert metadata.max_context_length is not None

    if metadata.parameter_count <= 0:
        return None

    if metadata.hidden_size <= 0:
        return None

    if metadata.num_hidden_layers <= 0:
        return None

    if metadata.num_attention_heads <= 0:
        return None

    if metadata.num_key_value_heads <= 0:
        return None

    if request.context_length <= 0:
        return None

    if request.context_length > metadata.max_context_length:
        return None

    if metadata.hidden_size % metadata.num_attention_heads != 0:
        return None

    weight_precision = resolve_weight_precision(
        metadata,
        request.weight_precision,
    )

    weight_memory = estimate_weight_memory(
        metadata.parameter_count,
        weight_precision,
    )

    head_dimension = metadata.hidden_size // metadata.num_attention_heads

    values_per_token = (
        2 * metadata.num_hidden_layers * metadata.num_key_value_heads * head_dimension
    )

    kv_memory = int(
        values_per_token * request.context_length * request.kv_cache_precision.bytes_per_value
    )

    return ResourceEstimate(
        weight_memory_bytes=weight_memory,
        kv_cache_memory_bytes=kv_memory,
        total_memory_bytes=(weight_memory + kv_memory),
        context_length=request.context_length,
        weight_precision=weight_precision,
        kv_cache_precision=(request.kv_cache_precision),
    )


def _task_profile(task: str) -> dict[str, tuple[tuple[str, float], ...]]:
    return TASK_PROFILES.get(
        task.strip().lower(),
        TASK_PROFILES["general"],
    )


def _contains_term(text: str, term: str) -> bool:
    normalized = text.replace("_", " ").replace("-", " ").lower()
    normalized_terms = normalized.split()
    term_parts = term.replace("_", " ").replace("-", " ").lower().split()
    if not term_parts:
        return False
    width = len(term_parts)
    return any(
        normalized_terms[index : index + width] == term_parts
        for index in range(len(normalized_terms) - width + 1)
    )


def cheap_relevance(
    model: DiscoveredModel,
    task: str,
) -> float:
    """Fast task-specific relevance score from discovery metadata."""

    text = " ".join(
        (
            model.model_id,
            model.pipeline_tag or "",
            model.library_name or "",
            *model.tags,
        )
    ).lower()

    profile = _task_profile(task)
    score = 0.0

    for term, weight in profile["positive"]:
        if _contains_term(text, term):
            score += weight

    for term, penalty in profile["negative"]:
        if _contains_term(text, term):
            score -= penalty

    return score


def task_relevance(
    metadata: ModelMetadata,
    task: str,
) -> float:
    """Detailed task-specific relevance score from model metadata."""

    text = " ".join(
        (
            metadata.model_id,
            metadata.architecture or "",
            metadata.model_type or "",
            metadata.pipeline_tag or "",
            *metadata.tags,
        )
    ).lower()

    profile = _task_profile(task)
    score = 0.0

    for term, weight in profile["positive"]:
        if _contains_term(text, term):
            score += weight

    for term, penalty in profile["negative"]:
        if _contains_term(text, term):
            score -= penalty

    return max(0.0, score)


def calculate_score(
    metadata: ModelMetadata,
    request: RecommendationRequest,
    static_result: CompatibilityResult,
    runtime_result: CompatibilityResult,
) -> float:
    """
    Calculate recommendation score.

    Models that statically fit are expected here.
    Runtime-tight models receive a penalty but remain visible.
    """

    score = 0.0

    relevance = task_relevance(
        metadata,
        request.task,
    )

    score += min(
        40.0,
        relevance,
    )

    if (
        metadata.max_context_length is not None
        and metadata.max_context_length >= request.context_length
    ):
        score += 15.0

    if static_result.fits:
        score += 10.0

        available = static_result.available_memory_bytes

        remaining = static_result.remaining_memory_bytes

        if available > 0:
            ratio = remaining / available

            score += min(
                10.0,
                max(
                    0.0,
                    ratio * 10.0,
                ),
            )

    if runtime_result.fits:
        score += 15.0

        available = runtime_result.available_memory_bytes

        remaining = runtime_result.remaining_memory_bytes

        if available > 0:
            ratio = remaining / available

            score += min(
                10.0,
                max(
                    0.0,
                    ratio * 10.0,
                ),
            )

    else:
        # Runtime-tight models remain visible but rank lower.
        score -= 10.0

    score += min(
        5.0,
        math.log10(
            max(
                1,
                metadata.downloads,
            )
        ),
    )

    score += min(
        5.0,
        math.log10(
            max(
                1,
                metadata.likes,
            )
        ),
    )

    return round(
        score,
        2,
    )


def build_reasons(
    metadata: ModelMetadata,
    request: RecommendationRequest,
    static_result: CompatibilityResult,
    runtime_result: CompatibilityResult,
) -> tuple[str, ...]:
    """
    Build human-readable recommendation explanations.
    """

    reasons: list[str] = []

    relevance = task_relevance(
        metadata,
        request.task,
    )

    if relevance > 0:
        reasons.append(f"relevant to {request.task}")

    if (
        metadata.max_context_length is not None
        and metadata.max_context_length >= request.context_length
    ):
        reasons.append(f"supports {request.context_length:,} token context")

    if metadata.quantization:
        reasons.append(f"detected quantization: {metadata.quantization}")

    if static_result.fits:
        reasons.append("fits the static memory estimate")

    if runtime_result.fits:
        reasons.append("fits the runtime-aware memory estimate")

        available = runtime_result.available_memory_bytes

        remaining = runtime_result.remaining_memory_bytes

        if available > 0:
            headroom = remaining / (1024**3)

            reasons.append(f"approximately {headroom:.2f} GiB runtime VRAM headroom")

    else:
        reasons.append("runtime-aware estimate exceeds the configured runtime memory budget")

        deficit = max(
            0,
            -runtime_result.remaining_memory_bytes,
        )

        if deficit > 0:
            deficit_gib = deficit / (1024**3)

            reasons.append(f"approximately {deficit_gib:.2f} GiB over the runtime budget")

    return tuple(reasons)


def status_rank(
    status: str,
) -> int:
    """
    Lower value means a better recommendation status.
    """

    ranks = {
        "✓ FITS": 0,
        "⚠ TIGHT": 1,
        "⚠ RUNTIME TIGHT": 2,
        "✗ DOES NOT FIT": 3,
        "UNKNOWN": 4,
    }

    return ranks.get(
        status,
        5,
    )
