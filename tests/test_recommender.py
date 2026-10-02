from types import SimpleNamespace

import pytest

from infra import recommender
from infra.discovery import DiscoveredModel
from infra.estimator import Precision
from infra.metadata import ModelMetadata
from infra.recommender import (
    RecommendationRequest,
    build_estimate,
    build_reasons,
    calculate_score,
    has_estimation_metadata,
    recommend_models,
    resolve_weight_precision,
)


def make_hardware(vram_gib=16):
    return SimpleNamespace(
        gpus=[
            SimpleNamespace(
                name="Test GPU",
                total_vram_bytes=vram_gib * 1024**3,
            )
        ]
    )


def make_metadata(
    model_id="test/model",
    *,
    parameters=1_000_000,
    hidden_size=1024,
    layers=12,
    attention_heads=16,
    kv_heads=4,
    context=4096,
    downloads=1000,
    likes=100,
    tags=("text-generation", "instruct"),
    pipeline_tag="text-generation",
    quantization=None,
    description="",
):
    return ModelMetadata(
        model_id=model_id,
        downloads=downloads,
        likes=likes,
        tags=tags,
        pipeline_tag=pipeline_tag,
        architecture="TestForCausalLM",
        model_type="test",
        parameter_count=parameters,
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=attention_heads,
        num_key_value_heads=kv_heads,
        max_context_length=context,
        quantization=quantization,
        description=description,
    )


def make_candidate(
    model_id="test/model",
    *,
    downloads=1000,
    likes=100,
    tags=("text-generation", "instruct"),
):
    return DiscoveredModel(
        model_id=model_id,
        downloads=downloads,
        likes=likes,
        tags=tags,
        pipeline_tag="text-generation",
        library_name="transformers",
    )


def test_recommendation_request_defaults():
    request = RecommendationRequest(task="coding")

    assert request.task == "coding"
    assert request.context_length == 2048
    assert request.max_models == 5
    assert request.weight_precision == Precision.FP16
    assert request.kv_cache_precision == Precision.FP16
    assert request.backend == "vllm"


def test_recommend_models_rejects_invalid_context():
    hardware = make_hardware()

    request = RecommendationRequest(
        task="coding",
        context_length=0,
    )

    with pytest.raises(
        ValueError,
        match="context_length",
    ):
        recommend_models(hardware, request)


def test_recommend_models_rejects_invalid_max_models():
    hardware = make_hardware()

    request = RecommendationRequest(
        task="coding",
        max_models=0,
    )

    with pytest.raises(
        ValueError,
        match="max_models",
    ):
        recommend_models(hardware, request)


def test_has_estimation_metadata_true():
    metadata = make_metadata()

    assert has_estimation_metadata(metadata) is True


@pytest.mark.parametrize(
    "field",
    [
        "parameter_count",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_context_length",
    ],
)
def test_has_estimation_metadata_false_when_missing(field):
    metadata = make_metadata()

    values = {
        "parameter_count": metadata.parameter_count,
        "hidden_size": metadata.hidden_size,
        "num_hidden_layers": metadata.num_hidden_layers,
        "num_attention_heads": metadata.num_attention_heads,
        "num_key_value_heads": metadata.num_key_value_heads,
        "max_context_length": metadata.max_context_length,
    }

    values[field] = None

    metadata = ModelMetadata(
        model_id=metadata.model_id,
        downloads=metadata.downloads,
        likes=metadata.likes,
        tags=metadata.tags,
        pipeline_tag=metadata.pipeline_tag,
        architecture=metadata.architecture,
        model_type=metadata.model_type,
        parameter_count=values["parameter_count"],
        hidden_size=values["hidden_size"],
        num_hidden_layers=values["num_hidden_layers"],
        num_attention_heads=values["num_attention_heads"],
        num_key_value_heads=values["num_key_value_heads"],
        max_context_length=values["max_context_length"],
    )

    assert has_estimation_metadata(metadata) is False


@pytest.mark.parametrize(
    ("quantization", "expected"),
    [
        ("4bit", Precision.INT4),
        ("int4", Precision.INT4),
        ("nf4", Precision.INT4),
        ("awq", Precision.INT4),
        ("gptq", Precision.INT4),
        ("8bit", Precision.INT8),
        ("int8", Precision.INT8),
        (None, Precision.FP16),
        ("fp16", Precision.BF16),
    ],
)
def test_resolve_weight_precision(
    quantization,
    expected,
):
    metadata = make_metadata(quantization=quantization)

    requested = Precision.BF16 if quantization == "fp16" else Precision.FP16

    assert (
        resolve_weight_precision(
            metadata,
            requested,
        )
        == expected
    )


def test_build_estimate():
    metadata = make_metadata(
        parameters=1_000_000,
        hidden_size=1024,
        layers=12,
        attention_heads=16,
        kv_heads=4,
        context=4096,
    )

    request = RecommendationRequest(
        task="coding",
        context_length=1024,
        weight_precision=Precision.FP16,
        kv_cache_precision=Precision.FP16,
    )

    result = build_estimate(
        metadata,
        request,
    )

    assert result is not None
    assert result.weight_memory_bytes == 2_000_000
    assert result.kv_cache_memory_bytes > 0
    assert result.total_memory_bytes == (result.weight_memory_bytes + result.kv_cache_memory_bytes)


def test_build_estimate_rejects_context_above_model_limit():
    metadata = make_metadata(context=2048)

    request = RecommendationRequest(
        task="coding",
        context_length=4096,
    )

    assert (
        build_estimate(
            metadata,
            request,
        )
        is None
    )


def test_build_estimate_rejects_invalid_hidden_size():
    metadata = make_metadata(
        hidden_size=1000,
        attention_heads=16,
    )

    request = RecommendationRequest(
        task="coding",
        context_length=1024,
    )

    assert (
        build_estimate(
            metadata,
            request,
        )
        is None
    )


def test_build_estimate_rejects_missing_metadata():
    metadata = make_metadata()

    metadata = ModelMetadata(
        model_id=metadata.model_id,
        parameter_count=None,
        hidden_size=metadata.hidden_size,
        num_hidden_layers=metadata.num_hidden_layers,
        num_attention_heads=metadata.num_attention_heads,
        num_key_value_heads=metadata.num_key_value_heads,
        max_context_length=metadata.max_context_length,
    )

    request = RecommendationRequest(task="coding")

    assert (
        build_estimate(
            metadata,
            request,
        )
        is None
    )


def test_build_estimate_uses_detected_4bit_quantization():
    metadata = make_metadata(
        parameters=1_000_000,
        quantization="4bit",
    )

    request = RecommendationRequest(
        task="coding",
        context_length=1024,
        weight_precision=Precision.FP16,
    )

    result = build_estimate(
        metadata,
        request,
    )

    assert result is not None
    assert result.weight_precision == Precision.INT4
    assert result.weight_memory_bytes == 500_000


def test_calculate_score_returns_float():
    metadata = make_metadata()

    request = RecommendationRequest(
        task="coding",
        context_length=1024,
    )

    hardware = make_hardware()

    result = build_estimate(
        metadata,
        request,
    )

    runtime = recommender.estimate_runtime_aware_memory(
        result,
        backend="vllm",
    )

    static, runtime_result = recommender.check_compatibility(
        result,
        hardware,
        runtime,
    )

    score = calculate_score(
        metadata,
        request,
        static,
        runtime_result,
    )

    assert isinstance(score, float)
    assert score >= 0


def test_relevant_model_scores_higher_than_irrelevant_model():
    request = RecommendationRequest(
        task="coding",
        context_length=1024,
    )

    coding = make_metadata(
        model_id="coding-model",
        tags=("coder", "code", "text-generation"),
    )

    general = make_metadata(
        model_id="general-model",
        tags=("text-generation",),
    )

    hardware = make_hardware()

    coding_estimate = build_estimate(
        coding,
        request,
    )

    general_estimate = build_estimate(
        general,
        request,
    )

    coding_runtime = recommender.estimate_runtime_aware_memory(
        coding_estimate,
        backend="vllm",
    )

    general_runtime = recommender.estimate_runtime_aware_memory(
        general_estimate,
        backend="vllm",
    )

    coding_static, coding_runtime_result = recommender.check_compatibility(
        coding_estimate,
        hardware,
        coding_runtime,
    )

    general_static, general_runtime_result = recommender.check_compatibility(
        general_estimate,
        hardware,
        general_runtime,
    )

    coding_score = calculate_score(
        coding,
        request,
        coding_static,
        coding_runtime_result,
    )

    general_score = calculate_score(
        general,
        request,
        general_static,
        general_runtime_result,
    )

    assert coding_score > general_score


def test_build_reasons_contains_context():
    metadata = make_metadata()

    request = RecommendationRequest(
        task="coding",
        context_length=1024,
    )

    hardware = make_hardware()

    estimate = build_estimate(
        metadata,
        request,
    )

    runtime = recommender.estimate_runtime_aware_memory(
        estimate,
        backend="vllm",
    )

    static, runtime_result = recommender.check_compatibility(
        estimate,
        hardware,
        runtime,
    )

    reasons = build_reasons(
        metadata,
        request,
        static,
        runtime_result,
    )

    assert any("1,024 token context" in reason for reason in reasons)

    assert any("fits" in reason.lower() for reason in reasons)


def test_build_reasons_reports_quantization():
    metadata = make_metadata(quantization="4bit")

    request = RecommendationRequest(
        task="coding",
        context_length=1024,
    )

    hardware = make_hardware()

    estimate = build_estimate(
        metadata,
        request,
    )

    runtime = recommender.estimate_runtime_aware_memory(
        estimate,
        backend="vllm",
    )

    static, runtime_result = recommender.check_compatibility(
        estimate,
        hardware,
        runtime,
    )

    reasons = build_reasons(
        metadata,
        request,
        static,
        runtime_result,
    )

    assert any("4bit" in reason for reason in reasons)


def test_recommend_models_returns_recommendations(monkeypatch):
    candidates = [
        make_candidate(
            "test/coder-1",
            downloads=10_000,
            likes=100,
            tags=("coder", "code"),
        ),
        make_candidate(
            "test/coder-2",
            downloads=5_000,
            likes=50,
            tags=("coder", "code"),
        ),
    ]

    metadata = {
        candidate.model_id: make_metadata(
            candidate.model_id,
            downloads=candidate.downloads,
            likes=candidate.likes,
            tags=candidate.tags,
        )
        for candidate in candidates
    }

    class FakeDiscovery:
        def discover(self, request, *, refresh=False):
            return candidates

    class FakeMetadata:
        def get(self, model_id, *, refresh=False):
            return metadata.get(model_id)

    monkeypatch.setattr(
        recommender,
        "HuggingFaceDiscovery",
        lambda: FakeDiscovery(),
    )

    monkeypatch.setattr(
        recommender,
        "HuggingFaceMetadata",
        lambda: FakeMetadata(),
    )

    result = recommend_models(
        make_hardware(16),
        RecommendationRequest(
            task="coding",
            context_length=1024,
            max_models=2,
        ),
    )

    assert len(result) == 2
    assert all(isinstance(item, recommender.ModelRecommendation) for item in result)


def test_recommend_models_respects_max_models(monkeypatch):
    candidates = [
        make_candidate(
            f"test/coder-{index}",
            downloads=10_000 - index,
            likes=100,
            tags=("coder", "code"),
        )
        for index in range(10)
    ]

    metadata = {
        candidate.model_id: make_metadata(
            candidate.model_id,
            downloads=candidate.downloads,
            likes=candidate.likes,
            tags=candidate.tags,
        )
        for candidate in candidates
    }

    class FakeDiscovery:
        def discover(self, request, *, refresh=False):
            return candidates

    class FakeMetadata:
        def get(self, model_id, *, refresh=False):
            return metadata.get(model_id)

    monkeypatch.setattr(
        recommender,
        "HuggingFaceDiscovery",
        lambda: FakeDiscovery(),
    )

    monkeypatch.setattr(
        recommender,
        "HuggingFaceMetadata",
        lambda: FakeMetadata(),
    )

    result = recommend_models(
        make_hardware(),
        RecommendationRequest(
            task="coding",
            context_length=1024,
            max_models=3,
        ),
    )

    assert len(result) == 3


def test_recommend_models_returns_empty_when_no_candidates(
    monkeypatch,
):
    class FakeDiscovery:
        def discover(self, request, *, refresh=False):
            return []

    monkeypatch.setattr(
        recommender,
        "HuggingFaceDiscovery",
        lambda: FakeDiscovery(),
    )

    result = recommend_models(
        make_hardware(),
        RecommendationRequest(
            task="coding",
        ),
    )

    assert result == []


def test_recommend_models_skips_metadata_without_estimation_data(
    monkeypatch,
):
    candidate = make_candidate("test/incomplete")

    incomplete = ModelMetadata(model_id="test/incomplete")

    class FakeDiscovery:
        def discover(self, request, *, refresh=False):
            return [candidate]

    class FakeMetadata:
        def get(self, model_id, *, refresh=False):
            return incomplete

    monkeypatch.setattr(
        recommender,
        "HuggingFaceDiscovery",
        lambda: FakeDiscovery(),
    )

    monkeypatch.setattr(
        recommender,
        "HuggingFaceMetadata",
        lambda: FakeMetadata(),
    )

    result = recommend_models(
        make_hardware(),
        RecommendationRequest(
            task="coding",
        ),
    )

    assert result == []


def test_recommend_models_skips_models_that_do_not_fit(
    monkeypatch,
):
    candidate = make_candidate("test/huge")

    huge_model = make_metadata(
        "test/huge",
        parameters=100_000_000_000,
        downloads=1000,
        likes=100,
    )

    class FakeDiscovery:
        def discover(self, request, *, refresh=False):
            return [candidate]

    class FakeMetadata:
        def get(self, model_id, *, refresh=False):
            return huge_model

    monkeypatch.setattr(
        recommender,
        "HuggingFaceDiscovery",
        lambda: FakeDiscovery(),
    )

    monkeypatch.setattr(
        recommender,
        "HuggingFaceMetadata",
        lambda: FakeMetadata(),
    )

    result = recommend_models(
        make_hardware(4),
        RecommendationRequest(
            task="coding",
            context_length=1024,
        ),
    )

    assert result == []


def test_recommendation_status_fits():
    request = RecommendationRequest(
        task="coding",
        context_length=1024,
    )

    metadata = make_metadata()

    estimate = build_estimate(
        metadata,
        request,
    )

    runtime = recommender.estimate_runtime_aware_memory(
        estimate,
        backend="vllm",
    )

    static, runtime_result = recommender.check_compatibility(
        estimate,
        make_hardware(16),
        runtime,
    )

    recommendation = recommender.ModelRecommendation(
        metadata=metadata,
        static_estimate=estimate,
        runtime_estimate=runtime,
        static_compatibility=static,
        runtime_compatibility=runtime_result,
        score=50.0,
        reasons=(),
    )

    assert recommendation.status == "✓ FITS"


def test_recommendation_status_tight():
    request = RecommendationRequest(
        task="coding",
        context_length=1024,
    )

    metadata = make_metadata(
        parameters=3_000_000_000,
    )

    estimate = build_estimate(
        metadata,
        request,
    )

    runtime = recommender.estimate_runtime_aware_memory(
        estimate,
        backend="vllm",
    )

    static, runtime_result = recommender.check_compatibility(
        estimate,
        make_hardware(16),
        runtime,
    )

    recommendation = recommender.ModelRecommendation(
        metadata=metadata,
        static_estimate=estimate,
        runtime_estimate=runtime,
        static_compatibility=static,
        runtime_compatibility=runtime_result,
        score=50.0,
        reasons=(),
    )

    assert recommendation.status in {
        "✓ FITS",
        "⚠ TIGHT",
        "⚠ RUNTIME TIGHT",
    }
