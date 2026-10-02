from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

MODEL_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
]


def download_model(
    model_id: str,
    allow_patterns: list[str] | None = None,
    progress_callback: Callable[[float, float], None] | None = None,
) -> Path:
    """
    Download a selected Hugging Face model.

    progress_callback:
        current_bytes
        total_bytes
    """

    model_id = model_id.strip()

    if not model_id:
        raise ValueError("model_id cannot be empty")

    patterns = allow_patterns if allow_patterns is not None else MODEL_PATTERNS

    if not patterns:
        raise ValueError("allow_patterns cannot be empty")

    tqdm_class = None

    if progress_callback is not None:

        class SilentProgressTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)

                progress_callback(
                    float(self.n),
                    float(self.total or 0),
                )

            def update(self, n: int = 1):
                result = super().update(n)

                progress_callback(
                    float(self.n),
                    float(self.total or 0),
                )

                return result

        tqdm_class = SilentProgressTqdm

    path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        allow_patterns=patterns,
        tqdm_class=tqdm_class,
    )

    return Path(path)
