from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass
class CPUInfo:
    name: str
    physical_cores: int | None
    logical_cores: int | None


@dataclass
class MemoryInfo:
    total_bytes: int
    available_bytes: int


@dataclass
class GPUInfo:
    name: str
    total_vram_bytes: int | None


@dataclass
class StorageInfo:
    total_bytes: int
    free_bytes: int


@dataclass
class HardwareProfile:
    os_name: str
    architecture: str
    cpu: CPUInfo
    memory: MemoryInfo
    gpus: list[GPUInfo]
    storage: StorageInfo


def _get_linux_cpu_name() -> str | None:
    """Read the CPU model name exposed by the Linux kernel."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                key, separator, value = line.partition(":")

                if separator and key.strip().lower() == "model name":
                    return value.strip() or None

    except OSError:
        return None

    return None


def _get_windows_cpu_name() -> str | None:

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        name = result.stdout.strip()
        return name or None

    except (OSError, subprocess.TimeoutExpired):
        return None


def detect_cpu() -> CPUInfo:
    return CPUInfo(
        name=get_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
    )


def get_cpu_name() -> str:
    system = platform.system()

    if system == "Linux":
        name = _get_linux_cpu_name()
        if name:
            return name

    elif system == "Windows":
        name = _get_windows_cpu_name()
        if name:
            return name

    return platform.processor() or "Unknown CPU"


def detect_memory() -> MemoryInfo:
    memory = psutil.virtual_memory()

    return MemoryInfo(
        total_bytes=memory.total,
        available_bytes=memory.available,
    )


def detect_nvidia_gpus() -> list[GPUInfo]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    gpus = []

    for line in result.stdout.strip().splitlines():
        parts = line.split(",")

        if len(parts) != 2:
            continue

        name = parts[0].strip()

        try:
            # nvidia-smi reports this value in MiB.
            vram_mib = float(parts[1].strip())
            vram_bytes = int(vram_mib * 1024 * 1024)
        except ValueError:
            vram_bytes = None

        gpus.append(
            GPUInfo(
                name=name,
                total_vram_bytes=vram_bytes,
            )
        )

    return gpus


def detect_rocm_gpus() -> list[GPUInfo]:
    """Detect AMD ROCm GPUs via rocm-smi."""
    if shutil.which("rocm-smi") is None:
        return []

    try:
        result = subprocess.run(
            [
                "rocm-smi",
                "--showmeminfo",
                "vram",
                "--showproductname",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []

        data = json.loads(result.stdout)
        gpus: list[GPUInfo] = []
        for card_key, card_info in data.items():
            if not isinstance(card_info, dict):
                continue
            name = (
                card_info.get("Card series")
                or card_info.get("Card model")
                or f"AMD GPU ({card_key})"
            )
            vram_raw = card_info.get("VRAM Total Memory (B)") or card_info.get("VRAM Total Memory")
            vram_bytes: int | None = None
            if vram_raw is not None:
                try:
                    vram_bytes = int(vram_raw)
                except (ValueError, TypeError):
                    pass
            gpus.append(GPUInfo(name=str(name), total_vram_bytes=vram_bytes))
        return gpus
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        OSError,
    ):
        return []


def detect_apple_silicon_gpu() -> list[GPUInfo]:
    """Detect Apple Silicon GPU with unified memory on macOS."""
    if platform.system() != "Darwin":
        return []

    try:
        proc = platform.processor() or ""
        machine = platform.machine() or ""
        if "arm" not in proc.lower() and "arm" not in machine.lower():
            return []

        chip_name = "Apple Silicon"
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                chip_name = res.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass

        total_ram = psutil.virtual_memory().total
        vram_bytes = int(total_ram * 0.75)

        return [
            GPUInfo(
                name=f"{chip_name} (Unified Memory)",
                total_vram_bytes=vram_bytes,
            )
        ]
    except Exception:
        return []


def detect_gpus() -> list[GPUInfo]:
    nvidia = detect_nvidia_gpus()
    if nvidia:
        return nvidia

    rocm = detect_rocm_gpus()
    if rocm:
        return rocm

    apple = detect_apple_silicon_gpu()
    if apple:
        return apple

    return []


def detect_storage() -> StorageInfo:
    usage = shutil.disk_usage(Path.cwd())

    return StorageInfo(
        total_bytes=usage.total,
        free_bytes=usage.free,
    )


def detect_hardware() -> HardwareProfile:

    return HardwareProfile(
        os_name=platform.system(),
        architecture=platform.machine(),
        cpu=detect_cpu(),
        memory=detect_memory(),
        gpus=detect_gpus(),
        storage=detect_storage(),
    )
