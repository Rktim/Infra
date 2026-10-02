from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from infra.backend import ChatMessage, GenerationRequest, ModelLoadConfig
from infra.compatibility import check_compatibility
from infra.downloader import download_model
from infra.estimator import (
    EstimationRequest,
    Precision,
    estimate_resources,
    estimate_runtime_aware_memory,
    format_bytes,
)
from infra.hardware import detect_hardware
from infra.metadata import HuggingFaceMetadata
from infra.models import (
    ModelConfig,
    ParameterInfo,
    count_safetensor_parameters,
    load_model_config,
)
from infra.optimizer import optimize_serving_params
from infra.recommender import (
    RecommendationRequest,
    recommend_models,
)
from infra.server import ServerConfig, VLLMServer
from infra.suggestor import supported_use_cases
from infra.vllm import VLLMBackend

DEFAULT_CONTEXT_LENGTH = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_OVERHEAD_FRACTION = 0.20
DEFAULT_SAFETY_MARGIN_FRACTION = 0.10


def print_header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def print_error(message: str) -> None:
    print(f"\nError: {message}")


def print_hardware(hardware) -> None:
    print("\nHardware:")

    print(f"  OS:           {hardware.os_name}")
    print(f"  Architecture: {hardware.architecture}")

    print("\n  CPU:")
    print(f"    Name:            {hardware.cpu.name}")
    print(f"    Physical cores:  {hardware.cpu.physical_cores}")
    print(f"    Logical cores:   {hardware.cpu.logical_cores}")

    print("\n  Memory:")

    total_memory = format_bytes(hardware.memory.total_bytes)
    available_memory = format_bytes(hardware.memory.available_bytes)

    print(f"    Total:     {total_memory}")
    print(f"    Available: {available_memory}")

    print("\n  GPU:")

    if not hardware.gpus:
        print("    None detected")
    else:
        for index, gpu in enumerate(hardware.gpus):
            gpu_memory = format_bytes(gpu.total_vram_bytes)

            print(f"    [{index}] {gpu.name}")
            print(f"        VRAM: {gpu_memory}")

    print("\n  Storage:")

    storage_total = format_bytes(hardware.storage.total_bytes)
    storage_free = format_bytes(hardware.storage.free_bytes)

    print(f"    Total: {storage_total}")
    print(f"    Free:  {storage_free}")


def resolve_model_path(model_id: str) -> Path:
    """
    Resolve an already-downloaded Hugging Face model snapshot.

    This function NEVER downloads model files.
    """

    from huggingface_hub import scan_cache_dir

    model_id = model_id.strip()

    if not model_id:
        raise ValueError("model_id cannot be empty")

    expected_repo = f"models--{model_id.replace('/', '--')}"

    cache_info = scan_cache_dir()

    for repo in cache_info.repos:
        if repo.repo_path.name != expected_repo:
            continue

        if not repo.revisions:
            break

        revision = max(
            repo.revisions,
            key=lambda item: item.last_modified,
        )

        snapshot_path = Path(revision.snapshot_path)

        if snapshot_path.is_dir():
            return snapshot_path

        break

    raise FileNotFoundError(
        f"Model is not available locally: {model_id}. Run `infra download {model_id}` first."
    )


def get_vllm_memory_budget(
    hardware,
    gpu_memory_utilization: float,
) -> int:
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")

    if not hardware.gpus:
        raise RuntimeError("No GPU detected.")

    gpu = hardware.gpus[0]

    return int(gpu.total_vram_bytes * gpu_memory_utilization)


def resolve_model_config_and_parameters(model_id: str):
    """
    Resolve model configuration and parameter count without downloading weights.
    Uses Hugging Face Hub metadata with fallback to config loading.
    """
    metadata_store = HuggingFaceMetadata()
    metadata = metadata_store.get(model_id)

    if (
        metadata is not None
        and metadata.parameter_count is not None
        and metadata.hidden_size is not None
    ):
        config = ModelConfig(
            model_type=metadata.model_type,
            architecture=metadata.architecture,
            hidden_size=metadata.hidden_size,
            num_hidden_layers=metadata.num_hidden_layers,
            num_attention_heads=metadata.num_attention_heads,
            num_key_value_heads=metadata.num_key_value_heads,
            max_position_embeddings=metadata.max_context_length,
        )
        parameters = ParameterInfo(
            count=metadata.parameter_count,
            source="Hugging Face Hub metadata",
        )
        return config, parameters, metadata

    config = load_model_config(model_id)
    parameters = count_safetensor_parameters(model_id)
    return config, parameters, metadata


def inspect_model(
    model_id: str,
    context_length: int,
    *,
    backend: str = "vllm",
    overhead_fraction: float = DEFAULT_OVERHEAD_FRACTION,
    safety_margin_fraction: float = DEFAULT_SAFETY_MARGIN_FRACTION,
    gpu_memory_utilization: float | None = None,
):
    if context_length <= 0:
        raise ValueError("context_length must be greater than zero")

    print("Detecting hardware...")
    hardware = detect_hardware()

    print("Inspecting model...")

    config, parameters, _metadata = resolve_model_config_and_parameters(model_id)

    if parameters.count is None:
        raise RuntimeError("Could not determine model parameter count.")

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

    runtime_estimate = estimate_runtime_aware_memory(
        estimate,
        backend=backend,
        overhead_fraction=overhead_fraction,
        safety_margin_fraction=safety_margin_fraction,
    )

    memory_budget = None

    if backend.lower() == "vllm" and gpu_memory_utilization is not None and hardware.gpus:
        memory_budget = get_vllm_memory_budget(
            hardware,
            gpu_memory_utilization,
        )

    static_result, runtime_result = check_compatibility(
        estimate=estimate,
        hardware=hardware,
        runtime_estimate=runtime_estimate,
        memory_budget_bytes=memory_budget,
    )

    return (
        hardware,
        config,
        parameters,
        estimate,
        runtime_estimate,
        static_result,
        runtime_result,
    )


def command_inspect(
    args: argparse.Namespace,
) -> int:

    (
        hardware,
        config,
        parameters,
        estimate,
        runtime_estimate,
        static_result,
        runtime_result,
    ) = inspect_model(
        args.model_id,
        args.context_length,
        backend=args.backend,
        overhead_fraction=args.overhead_fraction,
        safety_margin_fraction=args.safety_margin_fraction,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print_hardware(hardware)

    print("\nModel:")
    print(f"  ID: {args.model_id}")

    architecture = getattr(
        config,
        "architecture",
        None,
    )

    if architecture:
        print(f"  Architecture: {architecture}")

    print("\nParameters:")
    print(f"  Count: {parameters.count:,}")
    print(f"  Source: {parameters.source}")

    print("\nEstimation:")
    print(f"  Context length: {args.context_length:,}")
    print(f"  Weight precision: {estimate.weight_precision.label}")
    print(f"  KV-cache precision: {estimate.kv_cache_precision.label}")

    print_header("Static Estimate")

    print(f"Weights:  {format_bytes(estimate.weight_memory_bytes)}")

    print(f"KV cache: {format_bytes(estimate.kv_cache_memory_bytes)}")

    print(f"Total:    {format_bytes(estimate.total_memory_bytes)}")

    print_header("Runtime-Aware Estimate")

    print(f"Backend: {runtime_estimate.backend}")

    print(f"Static memory:          {format_bytes(estimate.total_memory_bytes)}")

    print(f"Backend overhead:       {format_bytes(runtime_estimate.overhead_bytes)}")

    print(f"Safety margin:          {format_bytes(runtime_estimate.safety_margin_bytes)}")

    print(f"Deployment requirement: {format_bytes(runtime_estimate.deployment_memory_bytes)}")

    print_header("Compatibility")

    print(f"Static:  {'✓ FITS' if static_result.fits else '✗ DOES NOT FIT'}")

    if runtime_result is not None:
        print(f"Runtime: {'✓ FITS' if runtime_result.fits else '✗ DOES NOT FIT'}")

        print(f"Runtime budget: {format_bytes(runtime_result.available_memory_bytes)}")

    meta = HuggingFaceMetadata().get(args.model_id)
    if meta is not None and hardware.gpus:
        opt = optimize_serving_params(
            metadata=meta,
            hardware=hardware,
            task="general",
            target_context=args.context_length,
            gpu_memory_utilization=args.gpu_memory_utilization or DEFAULT_GPU_MEMORY_UTILIZATION,
        )
        print_header("Tuned Serving Parameters")
        source_label = "Auto-Tuned"
        print(f"Profile:            {source_label}")
        print(f"Max Context:        {opt.max_model_len:,} tokens")
        print(f"GPU Utilization:    {opt.gpu_memory_utilization:.2f}")
        exec_mode = (
            "Eager Execution (Prevents Graph OOM)"
            if opt.enforce_eager
            else "CUDA Graphs (Low Latency / High Throughput)"
        )
        print(f"Execution Mode:     {exec_mode}")
        if opt.tuning_rationale:
            print(f"Insight:            {opt.tuning_rationale}")
        serve_cmd = f"infra serve {args.model_id} --context-length {opt.max_model_len} --gpu-memory-utilization {opt.gpu_memory_utilization:.2f}"
        if opt.enforce_eager:
            serve_cmd += " --enforce-eager"
        print(f"\nQuick Launch:\n  {serve_cmd}")

    return 0


def command_download(
    args: argparse.Namespace,
) -> int:

    print(f"Downloading: {args.model_id}")

    model_path = download_model(args.model_id)

    print()
    print("Download complete.")
    print(f"Model path: {model_path}")

    return 0


def command_run(
    args: argparse.Namespace,
) -> int:

    model_path = resolve_model_path(args.model_id)

    hardware = detect_hardware()

    config, parameters, _ = resolve_model_config_and_parameters(args.model_id)

    if parameters.count is None:
        raise RuntimeError("Could not determine model parameter count.")

    estimate_request = EstimationRequest(
        context_length=args.context_length,
        weight_precision=Precision.FP16,
        kv_cache_precision=Precision.FP16,
    )

    estimate = estimate_resources(
        config=config,
        parameter_count=parameters.count,
        request=estimate_request,
    )

    runtime_estimate = estimate_runtime_aware_memory(
        estimate,
        backend="vllm",
        overhead_fraction=args.overhead_fraction,
        safety_margin_fraction=args.safety_margin_fraction,
    )

    memory_budget = get_vllm_memory_budget(
        hardware,
        args.gpu_memory_utilization,
    )

    _, runtime_result = check_compatibility(
        estimate=estimate,
        hardware=hardware,
        runtime_estimate=runtime_estimate,
        memory_budget_bytes=memory_budget,
    )

    if not runtime_result.fits and not args.force:
        print_header("Preflight Failed")

        print(f"Required: {format_bytes(runtime_result.required_memory_bytes)}")

        print(f"vLLM budget: {format_bytes(runtime_result.available_memory_bytes)}")

        print("\nUse --force to run anyway.")

        return 1

    backend = VLLMBackend()

    backend.load_model(
        ModelLoadConfig(
            model_path=model_path,
            dtype=args.dtype,
            gpu_memory_utilization=(args.gpu_memory_utilization),
            max_model_len=args.context_length,
        )
    )

    try:
        result = backend.generate(
            GenerationRequest(
                messages=[
                    ChatMessage(
                        role="user",
                        content=args.prompt,
                    )
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )

        print_header("Generation")
        print(result.text)

    finally:
        backend.unload_model()

    return 0


def command_serve(
    args: argparse.Namespace,
) -> int:

    model_path = resolve_model_path(args.model_id)

    hardware = detect_hardware()

    memory_budget = get_vllm_memory_budget(
        hardware,
        args.gpu_memory_utilization,
    )

    config, parameters, _ = resolve_model_config_and_parameters(args.model_id)

    if parameters.count is None:
        raise RuntimeError("Could not determine model parameter count.")

    estimate_request = EstimationRequest(
        context_length=args.context_length,
        weight_precision=Precision.FP16,
        kv_cache_precision=Precision.FP16,
    )

    estimate = estimate_resources(
        config=config,
        parameter_count=parameters.count,
        request=estimate_request,
    )

    runtime_estimate = estimate_runtime_aware_memory(
        estimate,
        backend="vllm",
        overhead_fraction=args.overhead_fraction,
        safety_margin_fraction=args.safety_margin_fraction,
    )

    _, runtime_result = check_compatibility(
        estimate=estimate,
        hardware=hardware,
        runtime_estimate=runtime_estimate,
        memory_budget_bytes=memory_budget,
    )

    print_hardware(hardware)

    print_header("Runtime-Aware Preflight")

    print(f"Required:          {format_bytes(runtime_estimate.deployment_memory_bytes)}")

    print(f"GPU VRAM:           {format_bytes(hardware.gpus[0].total_vram_bytes)}")

    print(f"vLLM utilization:   {args.gpu_memory_utilization:.2f}")

    print(f"vLLM memory budget: {format_bytes(memory_budget)}")

    if not runtime_result.fits:
        print("\nWARNING: runtime-aware estimate does not fit the vLLM memory budget.")

        if not args.force:
            print("Refusing to start server. Use --force to override.")
            return 1

        print("Starting because --force was specified.")

    server = VLLMServer()

    server_config = ServerConfig(
        model_path=model_path,
        served_model_name=args.model_id,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        max_model_len=args.context_length,
        gpu_memory_utilization=(args.gpu_memory_utilization),
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
    )

    print_header("vLLM Server")

    try:
        status = server.start(server_config)

        print(f"State: {status.state.value}")
        print(f"PID:   {status.pid}")
        print(f"URL:   {status.url}")
        print(f"API:   {status.api_url}")
        print("\nServer is running. Press Ctrl+C to stop.")

        while server.is_running():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping server...")

    finally:
        if server.is_running():
            server.stop()

    return 0


def command_status(
    args: argparse.Namespace,
) -> int:

    server = VLLMServer()

    status = server.managed_status()

    print_header("infra Server")

    print(f"State:   {status.state.value}")

    if status.pid is not None:
        print(f"PID:     {status.pid}")

    if status.url is not None:
        print(f"URL:     {status.url}")

    if status.api_url is not None:
        print(f"API:     {status.api_url}")

    if status.message:
        print(f"Message: {status.message}")

    return 0


def command_stop(
    args: argparse.Namespace,
) -> int:

    server = VLLMServer()

    status = server.stop_managed()

    print_header("infra Server")

    print(f"State: {status.state.value}")

    if status.message:
        print(status.message)

    return 0


def command_recommend(
    args: argparse.Namespace,
) -> int:

    hardware = detect_hardware()

    request = RecommendationRequest(
        task=args.use_case,
        context_length=args.context_length,
        max_models=args.max_models,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print("Searching Hugging Face models...")

    recommendations = recommend_models(
        hardware,
        request,
        refresh=args.refresh,
    )

    print_header("Recommendations")

    if not recommendations:
        print("No compatible models found.")
        return 0

    print(f"Found {len(recommendations)} compatible model(s).")

    for index, recommendation in enumerate(
        recommendations,
        start=1,
    ):
        metadata = recommendation.metadata

        parameters = metadata.parameter_count

        if parameters is None:
            parameter_text = "unknown"
        elif parameters >= 1_000_000_000:
            parameter_text = f"{parameters / 1_000_000_000:.2f}B"
        else:
            parameter_text = f"{parameters / 1_000_000:.2f}M"

        max_context = (
            "unknown" if metadata.max_context_length is None else f"{metadata.max_context_length:,}"
        )

        quantization = metadata.quantization or "none detected"

        static_memory = format_bytes(recommendation.static_estimate.total_memory_bytes)

        runtime_memory = format_bytes(recommendation.runtime_estimate.deployment_memory_bytes)

        print()
        print(f"{index}. {metadata.model_id}")

        print(f"   Score:       {recommendation.score:.2f}")

        print(f"   Parameters:  {parameter_text}")

        print(f"   Architecture: {metadata.architecture or 'unknown'}")

        print(f"   Max context: {max_context}")

        print(f"   Quantization: {quantization}")

        print(f"   Static:      {static_memory}")

        print(f"   Runtime:     {runtime_memory}")

        print(f"   Status:      {recommendation.status}")

        print("   Why:")

        for reason in recommendation.reasons:
            print(f"      • {reason}")

        if recommendation.optimization_params is not None:
            opt = recommendation.optimization_params
            source_tag = "Auto-Tuned"
            print(f"\n   Optimization Parameters ({source_tag}):")
            print(f"      • Context length:       {opt.max_model_len:,} tokens")
            print(f"      • GPU utilization:      {opt.gpu_memory_utilization:.2f}")
            mode_desc = "Eager Mode (safe)" if opt.enforce_eager else "CUDA Graphs (fastest)"
            print(f"      • Execution mode:       {mode_desc}")
            if opt.tuning_rationale:
                print(f"      • Tuning rationale:     {opt.tuning_rationale}")
            serve_cmd = f"infra serve {metadata.model_id} --context-length {opt.max_model_len} --gpu-memory-utilization {opt.gpu_memory_utilization:.2f}"
            if opt.enforce_eager:
                serve_cmd += " --enforce-eager"
            print(f"      • Quick launch:         {serve_cmd}")

    print()
    print("Recommendation uses Hugging Face metadata only.")
    print("Model weights are NOT downloaded during recommendation.")

    return 0


def command_tui(
    args: argparse.Namespace,
) -> int:

    from infra.tui import main as tui_main

    tui_main()

    return 0


def add_estimation_arguments(
    parser: argparse.ArgumentParser,
) -> None:

    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
    )

    parser.add_argument(
        "--backend",
        default="vllm",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )

    parser.add_argument(
        "--overhead-fraction",
        type=float,
        default=DEFAULT_OVERHEAD_FRACTION,
    )

    parser.add_argument(
        "--safety-margin-fraction",
        type=float,
        default=DEFAULT_SAFETY_MARGIN_FRACTION,
    )


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="infra",
        description=(
            "Hardware-aware local model discovery, "
            "recommendation, downloading, serving, "
            "and inference."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect model resources and compatibility.",
    )

    inspect_parser.add_argument("model_id")

    add_estimation_arguments(inspect_parser)

    inspect_parser.set_defaults(handler=command_inspect)

    run_parser = subparsers.add_parser(
        "run",
        help="Run direct in-process vLLM inference.",
    )

    run_parser.add_argument("model_id")

    run_parser.add_argument(
        "--prompt",
        default="Explain recursion in one sentence.",
    )

    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
    )

    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    run_parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )

    run_parser.add_argument(
        "--dtype",
        default="auto",
    )

    add_estimation_arguments(run_parser)

    run_parser.add_argument(
        "--force",
        action="store_true",
    )

    run_parser.set_defaults(handler=command_run)

    download_parser = subparsers.add_parser(
        "download",
        help="Download a model from Hugging Face.",
    )

    download_parser.add_argument("model_id")

    download_parser.set_defaults(handler=command_download)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a managed OpenAI-compatible vLLM server.",
    )

    serve_parser.add_argument("model_id")

    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    serve_parser.add_argument(
        "--dtype",
        default="auto",
    )

    serve_parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
    )

    serve_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )

    serve_parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
    )

    serve_parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture.",
    )

    serve_parser.add_argument(
        "--overhead-fraction",
        type=float,
        default=DEFAULT_OVERHEAD_FRACTION,
    )

    serve_parser.add_argument(
        "--safety-margin-fraction",
        type=float,
        default=DEFAULT_SAFETY_MARGIN_FRACTION,
    )

    serve_parser.add_argument(
        "--force",
        action="store_true",
        help="Start despite failed preflight.",
    )

    serve_parser.set_defaults(handler=command_serve)

    status_parser = subparsers.add_parser(
        "status",
        help="Show managed vLLM server status.",
    )

    status_parser.set_defaults(handler=command_status)

    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop the managed vLLM server.",
    )

    stop_parser.set_defaults(handler=command_stop)

    recommend_parser = subparsers.add_parser(
        "recommend",
        help="Recommend compatible Hugging Face models.",
    )

    recommend_parser.add_argument(
        "--use-case",
        required=True,
        choices=supported_use_cases(),
    )

    recommend_parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
    )

    recommend_parser.add_argument(
        "--max-models",
        type=int,
        default=5,
    )

    recommend_parser.add_argument(
        "--backend",
        default="vllm",
    )

    recommend_parser.add_argument(
        "--refresh",
        action="store_true",
    )

    recommend_parser.set_defaults(handler=command_recommend)

    recommend_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help=("Fraction of GPU VRAM available to the serving backend (default: 0.90)"),
    )

    tui_parser = subparsers.add_parser(
        "tui",
        help="Launch the interactive terminal interface.",
    )

    tui_parser.set_defaults(handler=command_tui)

    return parser


def main(
    argv: list[str] | None = None,
) -> int:

    parser = build_parser()

    args = parser.parse_args(argv)

    try:
        return args.handler(args)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    except Exception as error:
        print_error(str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
