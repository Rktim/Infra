from __future__ import annotations

from dataclasses import dataclass

from infra.estimator import (
    ResourceEstimate,
    RuntimeAwareEstimate,
)
from infra.hardware import HardwareProfile


@dataclass(frozen=True)
class CompatibilityResult:
    fits: bool

    available_memory_bytes: int

    required_memory_bytes: int

    remaining_memory_bytes: int

    message: str

    scope: str


def check_compatibility(
    estimate: ResourceEstimate,
    hardware: HardwareProfile,
    runtime_estimate: RuntimeAwareEstimate | None = None,
    memory_budget_bytes: int | None = None,
) -> tuple[
    CompatibilityResult,
    CompatibilityResult | None,
]:
    """
    Check whether a model fits the detected hardware.

    Static compatibility:
        Compares the static model estimate against
        physical GPU VRAM.

    Runtime compatibility:
        If a RuntimeAwareEstimate is provided, compares
        the runtime-aware requirement against either:

        1. memory_budget_bytes, when supplied
        2. physical GPU VRAM otherwise

    memory_budget_bytes is useful for backends such as vLLM,
    where gpu_memory_utilization limits the memory available
    to the runtime.
    """

    if not hardware.gpus:
        raise RuntimeError("No GPU detected. GPU compatibility cannot be evaluated.")

    gpu = hardware.gpus[0]

    if estimate.total_memory_bytes < 0:
        raise ValueError("estimate.total_memory_bytes cannot be negative")

    if runtime_estimate is not None and runtime_estimate.deployment_memory_bytes < 0:
        raise ValueError("runtime deployment memory cannot be negative")

    if memory_budget_bytes is not None and memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be greater than zero")

    static_available = gpu.total_vram_bytes

    static_required = estimate.total_memory_bytes

    static_remaining = static_available - static_required

    static_fits = static_required <= static_available

    if static_fits:
        static_message = (
            f"Static estimate fits on "
            f"{gpu.name}. "
            f"This result does not include "
            f"backend-specific runtime overhead."
        )

    else:
        static_message = f"Static estimate does not fit on {gpu.name}."

    static_result = CompatibilityResult(
        fits=static_fits,
        available_memory_bytes=(static_available),
        required_memory_bytes=(static_required),
        remaining_memory_bytes=max(
            0,
            static_remaining,
        ),
        message=static_message,
        scope="static_estimate",
    )

    if runtime_estimate is None:
        return (
            static_result,
            None,
        )

    if memory_budget_bytes is not None:
        runtime_available = memory_budget_bytes

        budget_description = "configured runtime memory budget"

    else:
        runtime_available = static_available

        budget_description = "physical GPU VRAM"

    runtime_required = runtime_estimate.deployment_memory_bytes

    runtime_remaining = runtime_available - runtime_required

    runtime_fits = runtime_required <= runtime_available

    if runtime_fits:
        runtime_message = (
            f"Runtime-aware estimate fits within "
            f"the {budget_description}. "
            f"Backend: "
            f"{runtime_estimate.backend}. "
            f"This is a planning estimate, "
            f"not a measured runtime result."
        )

    else:
        runtime_message = (
            f"Runtime-aware estimate does not fit "
            f"within the {budget_description}. "
            f"Backend: "
            f"{runtime_estimate.backend}. "
            f"This is a planning estimate, "
            f"not a measured runtime result."
        )

    runtime_result = CompatibilityResult(
        fits=runtime_fits,
        available_memory_bytes=(runtime_available),
        required_memory_bytes=(runtime_required),
        remaining_memory_bytes=max(
            0,
            runtime_remaining,
        ),
        message=runtime_message,
        scope="runtime_aware_estimate",
    )

    return (
        static_result,
        runtime_result,
    )
