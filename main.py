from pathlib import Path

from infra.backend import (
    ChatMessage,
    GenerationRequest,
    ModelLoadConfig,
)
from infra.compatibility import check_gpu_compatibility
from infra.downloader import download_model
from infra.estimator import (
    EstimationRequest,
    Precision,
    estimate_resources,
    format_bytes,
)
from infra.hardware import detect_hardware
from infra.models import (
    count_safetensor_parameters,
    load_model,
    load_model_config,
)
from infra.runtime import (
    RuntimeProfile,
    compare_runtime_to_estimate,
    create_runtime_profile,
    get_gpu_memory_snapshot,
)
from infra.server import ServerConfig, VLLMServer
from infra.vllm import VLLMBackend


def print_runtime_profile(
    profile: RuntimeProfile,
    estimated_memory_bytes: int,
) -> None:
    """Display runtime evidence and compare it with the static estimate."""

    comparison = compare_runtime_to_estimate(
        estimated_memory_bytes=estimated_memory_bytes,
        profile=profile,
    )

    print("\nRuntime Profile:")
    print(f"Backend: {profile.backend}")
    print(f"Model: {profile.model_id}")
    print(f"Success: {profile.success}")
    print(f"Baseline GPU memory: {format_bytes(profile.measurement.before.used_bytes)}")
    print(f"Observed GPU memory: {format_bytes(profile.measurement.after.used_bytes)}")
    print(f"Observed increase: {format_bytes(profile.observed_memory_bytes)}")

    print("\nEstimate vs Runtime:")
    print(f"Static estimate: {format_bytes(comparison.estimated_memory_bytes)}")
    print(f"Observed runtime: {format_bytes(comparison.observed_memory_bytes)}")

    difference = comparison.difference_bytes
    direction = "above" if difference >= 0 else "below"

    print(f"Difference: {format_bytes(abs(difference))} {direction} estimate")

    if comparison.runtime_to_estimate_ratio is not None:
        print(f"Runtime / estimate ratio: {comparison.runtime_to_estimate_ratio:.2f}x")


def run_vllm(
    model_path: Path,
    model_id: str,
    context_length: int,
    estimated_memory_bytes: int,
) -> RuntimeProfile | None:
    """
    Run direct inference through vLLM and collect runtime evidence.

    The runtime snapshot is captured while the model remains loaded.
    """

    backend = VLLMBackend()

    print("\nvLLM Direct Inference:")
    print(f"Available: {backend.is_available()}")

    if not backend.is_available():
        print("vLLM is not available in the current environment.")
        return None

    baseline = get_gpu_memory_snapshot()
    profile: RuntimeProfile | None = None
    success = False

    runtime_config = {
        "dtype": "auto",
        "max_model_len": context_length,
        "gpu_memory_utilization": 0.85,
    }

    try:
        backend.load_model(
            ModelLoadConfig(
                model_path=model_path,
                dtype="auto",
                max_model_len=context_length,
                gpu_memory_utilization=0.85,
            )
        )

        print(f"Backend: {backend.name}")
        print(f"Loaded: {backend.is_loaded()}")

        result = backend.generate(
            GenerationRequest(
                messages=[
                    ChatMessage(
                        role="user",
                        content="Explain recursion in one sentence.",
                    )
                ],
                max_tokens=64,
            )
        )

        success = True

        print("\nGeneration:")
        print(result.text)

        # Capture evidence while the model is still loaded.
        profile = create_runtime_profile(
            backend=backend.name,
            model_id=model_id,
            configuration=runtime_config,
            before=baseline,
            success=success,
        )

        print_runtime_profile(
            profile=profile,
            estimated_memory_bytes=estimated_memory_bytes,
        )

        return profile

    finally:
        if backend.is_loaded():
            backend.unload_model()

        print("\nvLLM model unloaded.")


def run_server(
    model_path: Path,
    context_length: int,
) -> None:
    """Start vLLM's OpenAI-compatible server and validate an API request."""

    server = VLLMServer()

    print("\nvLLM Server:")

    try:
        status = server.start(
            ServerConfig(
                model_path=model_path,
                host="127.0.0.1",
                port=8000,
                dtype="auto",
                max_model_len=context_length,
                gpu_memory_utilization=0.75,
                max_num_seqs=4,
                enforce_eager=True,
            )
        )

        print(f"State: {status.state.value}")
        print(f"PID: {status.pid}")
        print(f"URL: {status.url}")

        response = server.chat(
            messages=[
                {
                    "role": "user",
                    "content": "Explain recursion in one sentence.",
                }
            ],
            max_tokens=64,
            temperature=0.7,
        )

        print("\nAPI Generation:")
        print(response)

    finally:
        status = server.stop()
        print("\nvLLM server stopped.")
        print(f"State: {status.state.value}")


def main() -> None:
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    context_length = 2048

    # ------------------------------------------------------------
    # 1. Hardware inspection
    # ------------------------------------------------------------
    print("Detecting hardware...")
    hardware = detect_hardware()

    # ------------------------------------------------------------
    # 2. Model inspection
    # ------------------------------------------------------------
    print("Inspecting model...")
    model = load_model(model_id)
    config = load_model_config(model_id)
    parameters = count_safetensor_parameters(model_id)

    if parameters.count is None:
        raise ValueError("Could not determine model parameter count")

    # ------------------------------------------------------------
    # 3. Static resource estimation
    # ------------------------------------------------------------
    request = EstimationRequest(
        context_length=context_length,
        weight_precision=Precision.FP16,
        kv_cache_precision=Precision.FP16,
    )

    estimate = estimate_resources(
        config=config,
        parameter_count=parameters.count,
        request=request,
    )

    # ------------------------------------------------------------
    # 4. Static compatibility check
    # ------------------------------------------------------------
    compatibility = check_gpu_compatibility(
        estimate=estimate,
        hardware=hardware,
    )

    # ------------------------------------------------------------
    # 5. Report pre-run information
    # ------------------------------------------------------------
    print("\nHardware:")
    print(hardware)

    print("\nModel:")
    print(model)

    print("\nParameters:")
    print(f"{parameters.count:,}")
    print(f"Source: {parameters.source}")

    print("\nEstimation Request:")
    print(f"Context length: {request.context_length}")
    print(f"Weight precision: {request.weight_precision.label}")
    print(f"KV cache precision: {request.kv_cache_precision.label}")

    print("\nStatic Estimate:")
    print(f"Weights: {format_bytes(estimate.weight_memory_bytes)}")
    print(f"KV cache: {format_bytes(estimate.kv_cache_memory_bytes)}")
    print(f"Total: {format_bytes(estimate.total_memory_bytes)}")

    print("\nStatic Compatibility:")
    print(compatibility)

    if not compatibility.fits:
        print("\nThe static model estimate does not fit the available GPU.")
        return

    # ------------------------------------------------------------
    # 6. Download / resolve local model
    # ------------------------------------------------------------
    print("\nDownloading Model...")
    model_path = download_model(model_id)
    print(model_path)

    # ------------------------------------------------------------
    # 7. Direct inference + runtime observation
    # ------------------------------------------------------------
    run_vllm(
        model_path=model_path,
        model_id=model_id,
        context_length=context_length,
        estimated_memory_bytes=estimate.total_memory_bytes,
    )

    # ------------------------------------------------------------
    # 8. Server lifecycle + API validation
    # ------------------------------------------------------------
    run_server(
        model_path=model_path,
        context_length=context_length,
    )


if __name__ == "__main__":
    main()
