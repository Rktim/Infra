<p align="center">
  <img src="assets/logo_gear_iso.jpg" alt="infra logo" width="160" />
</p>

<h1 align="center">infra</h1>

<p align="center">
  <strong>Hardware-aware local model discovery, memory estimation, and vLLM serving.</strong>
</p>

---

## Overview

`infra` inspects local compute hardware (NVIDIA, AMD ROCm, Apple Silicon, CPU/RAM), calculates exact model memory and KV-cache footprints across precisions (`FP16`, `BF16`, `INT8`, `INT4`), recommends fitting models from the Hugging Face Hub for your workload, and manages local vLLM serving daemons.

## Installation

```bash
# Using uv (recommended)
git clone <repository-url>
cd infra
uv pip install -e .

# Or standard pip
pip install -e .
```

Requires Python `>= 3.12, < 3.13`.

## Quickstart

### 1. Inspect Hardware Fit
Check exact static weight and dynamic KV-cache requirements against your machine before downloading weights:
```bash
infra inspect Qwen/Qwen2.5-Coder-7B-Instruct --context-length 4096 --precision bf16
```

### 2. Discover & Recommend Models
Find models fitted to your hardware budget by use case (`coding`, `chat`, `reasoning`, `rag`, `summarization`, `general`):
```bash
infra recommend --use-case coding --context-length 2048
```

### 3. Download Model
Download verified safetensors weights directly into your local Hugging Face cache:
```bash
infra download Qwen/Qwen2.5-Coder-1.5B-Instruct
```

### 4. Manage vLLM Server
Launch, inspect, and stop a background OpenAI-compatible vLLM daemon:
```bash
# Start background server
infra serve --model Qwen/Qwen2.5-Coder-1.5B-Instruct --port 8000 --context-length 2048

# Check status
infra status

# Stop server
infra stop
```

### 5. In-Process Inference
Run a quick prompt directly through the local runtime without starting a server:
```bash
infra run --model Qwen/Qwen2.5-Coder-1.5B-Instruct --prompt "Write a Python binary search function"
```

### 6. Interactive Terminal Dashboard (TUI)
Launch the interactive terminal interface:
```bash
infra tui
```
- **Models Tab**: Real-time GPU VRAM and CPU/RAM telemetry, use-case filters, model recommendation tables with hardware fit indicators (`[OK] FITS`, `[WARN] TIGHT`, `[FAIL] DOES NOT FIT`), and one-click downloading.
- **Server Tab**: View downloaded models, configure context length and GPU memory utilization, and manage vLLM daemons.
- **API Tab**: Endpoint status and copy-pasteable client snippets (Python, cURL).

## Testing

Run the full test suite:
```bash
uv run pytest
```
