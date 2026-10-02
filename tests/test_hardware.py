from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from infra.hardware import (
    CPUInfo,
    GPUInfo,
    HardwareProfile,
    MemoryInfo,
    StorageInfo,
    detect_apple_silicon_gpu,
    detect_cpu,
    detect_gpus,
    detect_hardware,
    detect_memory,
    detect_nvidia_gpus,
    detect_rocm_gpus,
    detect_storage,
)


def test_detect_cpu():
    cpu = detect_cpu()
    assert isinstance(cpu, CPUInfo)
    assert cpu.name
    assert isinstance(cpu.physical_cores, (int, type(None)))
    assert isinstance(cpu.logical_cores, (int, type(None)))


def test_detect_memory():
    memory = detect_memory()
    assert isinstance(memory, MemoryInfo)
    assert memory.total_bytes > 0
    assert memory.available_bytes >= 0


def test_detect_storage():
    storage = detect_storage()
    assert isinstance(storage, StorageInfo)
    assert storage.total_bytes > 0
    assert storage.free_bytes >= 0


def test_detect_hardware():
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.os_name
    assert profile.architecture
    assert isinstance(profile.cpu, CPUInfo)
    assert isinstance(profile.memory, MemoryInfo)
    assert isinstance(profile.storage, StorageInfo)
    assert isinstance(profile.gpus, list)


def test_detect_nvidia_gpus_mock():
    mock_stdout = "NVIDIA GeForce RTX 4090, 24576\n"
    with (
        patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=mock_stdout,
        )
        gpus = detect_nvidia_gpus()
        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA GeForce RTX 4090"
        assert gpus[0].total_vram_bytes == 24576 * 1024 * 1024


def test_detect_rocm_gpus_mock():
    mock_json = json.dumps(
        {
            "card0": {
                "Card series": "AMD Radeon RX 7900 XTX",
                "VRAM Total Memory (B)": "25752371200",
            }
        }
    )
    with (
        patch("shutil.which", return_value="/opt/rocm/bin/rocm-smi"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=mock_json,
        )
        gpus = detect_rocm_gpus()
        assert len(gpus) == 1
        assert "AMD Radeon RX 7900 XTX" in gpus[0].name
        assert gpus[0].total_vram_bytes == 25752371200


def test_detect_apple_silicon_gpu_mock():
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.processor", return_value="arm"),
        patch("platform.machine", return_value="arm64"),
        patch("psutil.virtual_memory") as mock_mem,
        patch("subprocess.run") as mock_run,
    ):
        mock_mem.return_value = MagicMock(total=64 * 1024 * 1024 * 1024)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Apple M2 Max\n",
        )
        gpus = detect_apple_silicon_gpu()
        assert len(gpus) == 1
        assert "Apple M2 Max" in gpus[0].name
        assert gpus[0].total_vram_bytes == int(64 * 1024 * 1024 * 1024 * 0.75)


def test_detect_gpus_precedence_nvidia():
    with (
        patch(
            "infra.hardware.detect_nvidia_gpus",
            return_value=[GPUInfo(name="NVIDIA RTX 4090", total_vram_bytes=24 * 1024**3)],
        ),
        patch("infra.hardware.detect_rocm_gpus") as mock_rocm,
    ):
        gpus = detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA RTX 4090"
        mock_rocm.assert_not_called()


def test_detect_gpus_fallback_to_rocm():
    with (
        patch("infra.hardware.detect_nvidia_gpus", return_value=[]),
        patch(
            "infra.hardware.detect_rocm_gpus",
            return_value=[GPUInfo(name="AMD RX 7900", total_vram_bytes=24 * 1024**3)],
        ),
    ):
        gpus = detect_gpus()
        assert len(gpus) == 1
        assert "AMD RX 7900" in gpus[0].name


def test_detect_gpus_fallback_to_apple():
    with (
        patch("infra.hardware.detect_nvidia_gpus", return_value=[]),
        patch("infra.hardware.detect_rocm_gpus", return_value=[]),
        patch(
            "infra.hardware.detect_apple_silicon_gpu",
            return_value=[
                GPUInfo(name="Apple M3 Max (Unified Memory)", total_vram_bytes=36 * 1024**3)
            ],
        ),
    ):
        gpus = detect_gpus()
        assert len(gpus) == 1
        assert "Apple M3 Max" in gpus[0].name


def test_detect_gpus_cpu_only():
    with (
        patch("infra.hardware.detect_nvidia_gpus", return_value=[]),
        patch("infra.hardware.detect_rocm_gpus", return_value=[]),
        patch("infra.hardware.detect_apple_silicon_gpu", return_value=[]),
    ):
        gpus = detect_gpus()
        assert gpus == []
