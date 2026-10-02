from types import SimpleNamespace

import pytest

from infra.compatibility import check_compatibility
from infra.estimator import (
    EstimationRequest,
    estimate_resources,
    estimate_runtime_aware_memory,
)


def make_hardware(vram_bytes):
    gpu = SimpleNamespace(
        name="Test GPU",
        total_vram_bytes=vram_bytes,
    )

    return SimpleNamespace(
        gpus=[gpu],
    )


def make_estimates():
    config = SimpleNamespace(
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=2048,
    )

    request = EstimationRequest(
        context_length=1024,
    )

    static = estimate_resources(
        config,
        parameter_count=1_000_000,
        request=request,
    )

    runtime = estimate_runtime_aware_memory(
        static,
        backend="vllm",
        overhead_fraction=0.20,
        safety_margin_fraction=0.10,
    )

    return static, runtime


def test_static_compatibility_fits():
    static, runtime = make_estimates()

    hardware = make_hardware(static.total_memory_bytes + 1)

    static_result, runtime_result = check_compatibility(
        static,
        hardware,
        runtime,
    )

    assert static_result.fits is True
    assert static_result.scope == "static_estimate"
    assert static_result.available_memory_bytes > 0
    assert static_result.required_memory_bytes == static.total_memory_bytes
    assert static_result.remaining_memory_bytes >= 0
    assert "fits" in static_result.message.lower()

    assert runtime_result is not None


def test_static_compatibility_does_not_fit():
    static, runtime = make_estimates()

    hardware = make_hardware(static.total_memory_bytes - 1)

    static_result, _runtime_result = check_compatibility(
        static,
        hardware,
        runtime,
    )

    assert static_result.fits is False
    assert static_result.remaining_memory_bytes == 0
    assert "does not fit" in static_result.message.lower()


def test_runtime_compatibility_uses_physical_vram_by_default():
    static, runtime = make_estimates()

    hardware = make_hardware(runtime.deployment_memory_bytes + 1)

    static_result, runtime_result = check_compatibility(
        static,
        hardware,
        runtime,
    )

    assert static_result.fits is True
    assert runtime_result is not None
    assert runtime_result.fits is True
    assert runtime_result.scope == "runtime_aware_estimate"


def test_runtime_compatibility_uses_configured_budget():
    static, runtime = make_estimates()

    budget = runtime.deployment_memory_bytes + 1

    hardware = make_hardware(static.total_memory_bytes * 10)

    _, runtime_result = check_compatibility(
        static,
        hardware,
        runtime,
        memory_budget_bytes=budget,
    )

    assert runtime_result is not None
    assert runtime_result.fits is True
    assert runtime_result.available_memory_bytes == budget
    assert "configured runtime memory budget" in runtime_result.message


def test_runtime_compatibility_fails_when_budget_is_too_small():
    static, runtime = make_estimates()

    budget = runtime.deployment_memory_bytes - 1

    hardware = make_hardware(static.total_memory_bytes * 10)

    _, runtime_result = check_compatibility(
        static,
        hardware,
        runtime,
        memory_budget_bytes=budget,
    )

    assert runtime_result is not None
    assert runtime_result.fits is False
    assert runtime_result.remaining_memory_bytes == 0


def test_no_runtime_estimate():
    static, _ = make_estimates()

    hardware = make_hardware(static.total_memory_bytes + 1)

    static_result, runtime_result = check_compatibility(
        static,
        hardware,
    )

    assert static_result.fits is True
    assert runtime_result is None


def test_no_gpu_raises_error():
    static, runtime = make_estimates()

    hardware = SimpleNamespace(gpus=[])

    with pytest.raises(
        RuntimeError,
        match="No GPU detected",
    ):
        check_compatibility(
            static,
            hardware,
            runtime,
        )


def test_invalid_memory_budget():
    static, runtime = make_estimates()

    hardware = make_hardware(static.total_memory_bytes * 10)

    with pytest.raises(ValueError, match="memory_budget_bytes"):
        check_compatibility(
            static,
            hardware,
            runtime,
            memory_budget_bytes=0,
        )


def test_negative_static_estimate():
    static, runtime = make_estimates()

    static.total_memory_bytes = -1

    hardware = make_hardware(1024)

    with pytest.raises(ValueError, match="cannot be negative"):
        check_compatibility(
            static,
            hardware,
            runtime,
        )
