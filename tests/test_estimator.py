from types import SimpleNamespace

import pytest

from infra.estimator import (
    EstimationRequest,
    Precision,
    estimate_kv_cache_memory,
    estimate_resources,
    estimate_runtime_aware_memory,
    estimate_weight_memory,
    format_bytes,
    validate_estimation_request,
)


def make_config(
    *,
    hidden_size=1024,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=4,
    max_position_embeddings=2048,
):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
    )


def test_precision_values():
    assert Precision.FP32.bytes_per_value == 4.0
    assert Precision.FP16.bytes_per_value == 2.0
    assert Precision.BF16.bytes_per_value == 2.0
    assert Precision.INT8.bytes_per_value == 1.0
    assert Precision.INT4.bytes_per_value == 0.5


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        (Precision.FP32, 400),
        (Precision.FP16, 200),
        (Precision.BF16, 200),
        (Precision.INT8, 100),
        (Precision.INT4, 50),
    ],
)
def test_estimate_weight_memory(precision, expected):
    assert estimate_weight_memory(100, precision) == expected


def test_estimate_weight_memory_rejects_invalid_parameter_count():
    with pytest.raises(ValueError):
        estimate_weight_memory(0)

    with pytest.raises(ValueError):
        estimate_weight_memory(-1)


def test_validate_estimation_request_accepts_valid_context():
    config = make_config()

    request = EstimationRequest(context_length=1024)

    validate_estimation_request(request, config)


def test_validate_estimation_request_rejects_zero_context():
    config = make_config()

    request = EstimationRequest(context_length=0)

    with pytest.raises(ValueError, match="context_length"):
        validate_estimation_request(request, config)


def test_validate_estimation_request_rejects_context_above_model_limit():
    config = make_config(max_position_embeddings=2048)

    request = EstimationRequest(context_length=4096)

    with pytest.raises(ValueError, match="exceeds detected model maximum"):
        validate_estimation_request(request, config)


def test_estimate_kv_cache_memory():
    config = make_config(
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=4,
    )

    result = estimate_kv_cache_memory(
        config,
        context_length=100,
        precision=Precision.FP16,
    )

    head_dimension = 1024 // 16
    expected = 2 * 12 * 4 * head_dimension * 100 * 2

    assert result == expected


def test_estimate_kv_cache_memory_rejects_invalid_context():
    config = make_config()

    with pytest.raises(ValueError):
        estimate_kv_cache_memory(config, 0)


def test_estimate_kv_cache_memory_rejects_missing_configuration():
    config = make_config()
    config.num_key_value_heads = None

    with pytest.raises(ValueError, match="missing fields"):
        estimate_kv_cache_memory(config, 1024)


def test_estimate_kv_cache_memory_rejects_invalid_head_dimension():
    config = make_config(
        hidden_size=1000,
        num_attention_heads=16,
    )

    with pytest.raises(
        ValueError,
        match="divisible by num_attention_heads",
    ):
        estimate_kv_cache_memory(config, 1024)


def test_estimate_resources():
    config = make_config()

    request = EstimationRequest(
        context_length=1024,
        weight_precision=Precision.FP16,
        kv_cache_precision=Precision.FP16,
    )

    result = estimate_resources(
        config,
        parameter_count=1_000_000,
        request=request,
    )

    assert result.weight_memory_bytes == 2_000_000
    assert result.kv_cache_memory_bytes > 0
    assert result.total_memory_bytes == (result.weight_memory_bytes + result.kv_cache_memory_bytes)
    assert result.context_length == 1024
    assert result.weight_precision == Precision.FP16
    assert result.kv_cache_precision == Precision.FP16


def test_runtime_aware_memory():
    config = make_config()

    request = EstimationRequest(context_length=1024)

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

    assert runtime.backend == "vllm"

    assert runtime.overhead_bytes == int(static.total_memory_bytes * 0.20)

    assert runtime.safety_margin_bytes == int(static.total_memory_bytes * 0.10)

    assert runtime.total_overhead_bytes == (runtime.overhead_bytes + runtime.safety_margin_bytes)

    assert runtime.deployment_memory_bytes == (
        static.total_memory_bytes + runtime.total_overhead_bytes
    )


def test_runtime_aware_memory_rejects_negative_overhead():
    config = make_config()

    static = estimate_resources(
        config,
        parameter_count=1_000_000,
        request=EstimationRequest(context_length=1024),
    )

    with pytest.raises(ValueError):
        estimate_runtime_aware_memory(
            static,
            backend="vllm",
            overhead_fraction=-0.1,
        )


def test_runtime_aware_memory_rejects_negative_safety_margin():
    config = make_config()

    static = estimate_resources(
        config,
        parameter_count=1_000_000,
        request=EstimationRequest(context_length=1024),
    )

    with pytest.raises(ValueError):
        estimate_runtime_aware_memory(
            static,
            backend="vllm",
            safety_margin_fraction=-0.1,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0.00 B"),
        (1024, "1.00 KiB"),
        (1024**2, "1.00 MiB"),
        (1024**3, "1.00 GiB"),
    ],
)
def test_format_bytes(value, expected):
    assert format_bytes(value) == expected
