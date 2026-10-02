from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUMemorySnapshot:
    used_bytes: int
    total_bytes: int

    @property
    def free_bytes(self) -> int:
        return self.total_bytes - self.used_bytes


@dataclass(frozen=True)
class RuntimeMeasurement:
    before: GPUMemorySnapshot
    after: GPUMemorySnapshot

    @property
    def memory_delta_bytes(self) -> int:
        return self.after.used_bytes - self.before.used_bytes


@dataclass(frozen=True)
class RuntimeProfile:
    backend: str
    model_id: str
    configuration: Mapping[str, object]
    measurement: RuntimeMeasurement
    success: bool

    @property
    def observed_memory_bytes(self) -> int:
        return self.measurement.memory_delta_bytes


@dataclass(frozen=True)
class RuntimeComparison:
    estimated_memory_bytes: int
    observed_memory_bytes: int

    @property
    def difference_bytes(self) -> int:
        return self.observed_memory_bytes - self.estimated_memory_bytes

    @property
    def runtime_to_estimate_ratio(self) -> float | None:
        if self.estimated_memory_bytes <= 0:
            return None

        return self.observed_memory_bytes / self.estimated_memory_bytes


def get_gpu_memory_snapshot(
    gpu_index: int = 0,
) -> GPUMemorySnapshot:

    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "nvidia-smi was not found. NVIDIA GPU monitoring is unavailable."
        ) from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or str(error)
        raise RuntimeError(f"Could not query GPU memory: {message}") from error

    values = result.stdout.strip().split(",")

    if len(values) != 2:
        raise RuntimeError(f"Unexpected nvidia-smi output: {result.stdout!r}")

    try:
        used_mib = int(values[0].strip())
        total_mib = int(values[1].strip())
    except ValueError as error:
        raise RuntimeError(f"Could not parse GPU memory values: {result.stdout!r}") from error

    return GPUMemorySnapshot(
        used_bytes=used_mib * 1024 * 1024,
        total_bytes=total_mib * 1024 * 1024,
    )


def measure_memory(
    before: GPUMemorySnapshot,
    gpu_index: int = 0,
) -> RuntimeMeasurement:

    after = get_gpu_memory_snapshot(gpu_index)

    return RuntimeMeasurement(
        before=before,
        after=after,
    )


def create_runtime_profile(
    *,
    backend: str,
    model_id: str,
    configuration: Mapping[str, object],
    before: GPUMemorySnapshot,
    success: bool,
    gpu_index: int = 0,
) -> RuntimeProfile:

    measurement = measure_memory(
        before=before,
        gpu_index=gpu_index,
    )

    return RuntimeProfile(
        backend=backend,
        model_id=model_id,
        configuration=dict(configuration),
        measurement=measurement,
        success=success,
    )


def compare_runtime_to_estimate(
    estimated_memory_bytes: int,
    profile: RuntimeProfile,
) -> RuntimeComparison:

    if estimated_memory_bytes < 0:
        raise ValueError("estimated_memory_bytes cannot be negative")

    return RuntimeComparison(
        estimated_memory_bytes=estimated_memory_bytes,
        observed_memory_bytes=profile.observed_memory_bytes,
    )
