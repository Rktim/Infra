from infra.estimator import Precision
from infra.hardware import CPUInfo, GPUInfo, HardwareProfile, MemoryInfo, StorageInfo
from infra.metadata import ModelMetadata
from infra.optimizer import (
    calculate_kv_bytes_per_token,
    calculate_max_safe_context,
    optimize_params_heuristic,
    optimize_serving_params,
)


def make_test_hardware(vram_gib: float = 16.0) -> HardwareProfile:
    return HardwareProfile(
        os_name="Linux",
        architecture="x86_64",
        cpu=CPUInfo(name="Test CPU", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(total_bytes=32 * 1024**3, available_bytes=16 * 1024**3),
        gpus=[
            GPUInfo(
                name="NVIDIA RTX 4080",
                total_vram_bytes=int(vram_gib * 1024**3),
            )
        ],
        storage=StorageInfo(total_bytes=500 * 1024**3, free_bytes=200 * 1024**3),
    )


def make_test_metadata(
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    params: int = 7_615_616_512,
    hidden_size: int = 3584,
    layers: int = 28,
    heads: int = 28,
    kv_heads: int = 4,
    max_context: int = 32768,
) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        parameter_count=params,
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_context_length=max_context,
        architecture="Qwen2ForCausalLM",
        pipeline_tag="text-generation",
        tags=("coding", "instruct"),
    )


def test_calculate_kv_bytes_per_token():
    meta = make_test_metadata()
    # layers=28, kv_heads=4, head_dim=3584//28=128
    # values_per_token = 2 * 28 * 4 * 128 = 28672
    # FP16 = 28672 * 2 = 57344 bytes/token
    bytes_per_token = calculate_kv_bytes_per_token(meta, Precision.FP16)
    assert bytes_per_token == 57344


def test_calculate_max_safe_context():
    meta = make_test_metadata()
    # 16 GB * 0.85 = 13.6 GB
    # 7.6B * 2 = 15.2 GB (doesn't fit in 13.6 GB directly at FP16!)
    # Let's test with 24 GB GPU
    hw24 = make_test_hardware(vram_gib=24.0)
    max_ctx = calculate_max_safe_context(meta, hw24, gpu_memory_utilization=0.85)
    assert max_ctx >= 4096


def test_optimize_params_heuristic_ample_headroom():
    hw = make_test_hardware(vram_gib=24.0)
    meta = make_test_metadata()
    opt = optimize_params_heuristic(meta, hw, task="coding", target_context=4096)
    assert opt.max_model_len >= 4096
    assert opt.enforce_eager is False
    assert opt.source == "heuristic"
    vllm_args = opt.as_vllm_args()
    assert "--max-model-len" in vllm_args
    assert "--gpu-memory-utilization" in vllm_args
    assert "--enforce-eager" not in vllm_args


def test_optimize_params_heuristic_tight_headroom():
    # 18 GB GPU with 7.6B model (~15.2 GB weights) -> headroom is tight (< 2.5 GB)
    hw = make_test_hardware(vram_gib=18.0)
    meta = make_test_metadata()
    opt = optimize_params_heuristic(meta, hw, task="general", target_context=2048)
    assert opt.enforce_eager is True
    assert opt.max_num_seqs == 4
    vllm_args = opt.as_vllm_args()
    assert "--enforce-eager" in vllm_args


def test_optimize_serving_params_heuristic():
    hw = make_test_hardware(vram_gib=24.0)
    meta = make_test_metadata()
    opt = optimize_serving_params(meta, hw, task="coding")
    assert opt.source == "heuristic"
