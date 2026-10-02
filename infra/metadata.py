from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from huggingface_hub import HfApi

from infra.cache import MetadataCache


@dataclass(frozen=True)
class ArchitectureProfile:
    parameter_count: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_context_length: int


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    author: str | None = None
    downloads: int = 0
    likes: int = 0
    tags: tuple[str, ...] = ()
    pipeline_tag: str | None = None
    library_name: str | None = None
    architecture: str | None = None
    model_type: str | None = None
    parameter_count: int | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    max_context_length: int | None = None
    license: str | None = None
    quantization: str | None = None
    description: str = ""


KNOWN_ARCHITECTURES: dict[str, ArchitectureProfile] = {
    # --- Qwen 2.5 Coder ---
    "qwen2.5-coder-0.5b": ArchitectureProfile(
        parameter_count=494032768,
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_context_length=32768,
    ),
    "qwen2.5-coder-1.5b": ArchitectureProfile(
        parameter_count=1543714304,
        hidden_size=1536,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_context_length=32768,
    ),
    "qwen2.5-coder-3b": ArchitectureProfile(
        parameter_count=3085938688,
        hidden_size=2048,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        max_context_length=32768,
    ),
    "qwen2.5-coder-7b": ArchitectureProfile(
        parameter_count=7615616512,
        hidden_size=3584,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_context_length=32768,
    ),
    "qwen2.5-coder-14b": ArchitectureProfile(
        parameter_count=14770000000,
        hidden_size=5120,
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        max_context_length=32768,
    ),
    "qwen2.5-coder-32b": ArchitectureProfile(
        parameter_count=32400000000,
        hidden_size=5120,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        max_context_length=32768,
    ),
    # --- Qwen 2.5 General & Instruct ---
    "qwen2.5-0.5b": ArchitectureProfile(
        parameter_count=494032768,
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_context_length=32768,
    ),
    "qwen2.5-1.5b": ArchitectureProfile(
        parameter_count=1543714304,
        hidden_size=1536,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_context_length=32768,
    ),
    "qwen2.5-3b": ArchitectureProfile(
        parameter_count=3085938688,
        hidden_size=2048,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        max_context_length=32768,
    ),
    "qwen2.5-7b": ArchitectureProfile(
        parameter_count=7615616512,
        hidden_size=3584,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_context_length=32768,
    ),
    "qwen2.5-14b": ArchitectureProfile(
        parameter_count=14770000000,
        hidden_size=5120,
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        max_context_length=32768,
    ),
    "qwen2.5-32b": ArchitectureProfile(
        parameter_count=32400000000,
        hidden_size=5120,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        max_context_length=32768,
    ),
    "qwen2.5-72b": ArchitectureProfile(
        parameter_count=72700000000,
        hidden_size=8192,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_context_length=32768,
    ),
    # --- Qwen 2.5 Math ---
    "qwen2.5-math-1.5b": ArchitectureProfile(
        parameter_count=1543714304,
        hidden_size=1536,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_context_length=32768,
    ),
    "qwen2.5-math-7b": ArchitectureProfile(
        parameter_count=7615616512,
        hidden_size=3584,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_context_length=32768,
    ),
    # --- Qwen 2 ---
    "qwen2-0.5b": ArchitectureProfile(
        parameter_count=494032768,
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_context_length=32768,
    ),
    "qwen2-1.5b": ArchitectureProfile(
        parameter_count=1543714304,
        hidden_size=1536,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_context_length=32768,
    ),
    "qwen2-7b": ArchitectureProfile(
        parameter_count=7615616512,
        hidden_size=3584,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_context_length=32768,
    ),
    "qwen2-72b": ArchitectureProfile(
        parameter_count=72700000000,
        hidden_size=8192,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_context_length=32768,
    ),
    # --- Llama 3 / 3.1 / 3.2 ---
    "llama-3.2-1b": ArchitectureProfile(
        parameter_count=1235814400,
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_context_length=131072,
    ),
    "llama-3.2-3b": ArchitectureProfile(
        parameter_count=3212749824,
        hidden_size=3072,
        num_hidden_layers=28,
        num_attention_heads=24,
        num_key_value_heads=8,
        max_context_length=131072,
    ),
    "llama-3.1-8b": ArchitectureProfile(
        parameter_count=8030261248,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_context_length=131072,
    ),
    "llama-3-8b": ArchitectureProfile(
        parameter_count=8030261248,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_context_length=8192,
    ),
    "llama-3.1-70b": ArchitectureProfile(
        parameter_count=70553706496,
        hidden_size=8192,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_context_length=131072,
    ),
    # --- DeepSeek R1 Distill ---
    "deepseek-r1-distill-qwen-1.5b": ArchitectureProfile(
        parameter_count=1777088000,
        hidden_size=1536,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_context_length=131072,
    ),
    "deepseek-r1-distill-qwen-7b": ArchitectureProfile(
        parameter_count=7615616512,
        hidden_size=3584,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_context_length=131072,
    ),
    "deepseek-r1-distill-llama-8b": ArchitectureProfile(
        parameter_count=8030261248,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_context_length=131072,
    ),
    "deepseek-r1-distill-qwen-14b": ArchitectureProfile(
        parameter_count=14770000000,
        hidden_size=5120,
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        max_context_length=131072,
    ),
    "deepseek-r1-distill-qwen-32b": ArchitectureProfile(
        parameter_count=32400000000,
        hidden_size=5120,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        max_context_length=131072,
    ),
    # --- Mistral ---
    "mistral-7b": ArchitectureProfile(
        parameter_count=7248023552,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_context_length=32768,
    ),
    "mistral-nemo": ArchitectureProfile(
        parameter_count=12200000000,
        hidden_size=5120,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_context_length=128000,
    ),
    # --- Gemma 2 ---
    "gemma-2-2b": ArchitectureProfile(
        parameter_count=2614341888,
        hidden_size=2304,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_context_length=8192,
    ),
    "gemma-2-9b": ArchitectureProfile(
        parameter_count=9241705984,
        hidden_size=3584,
        num_hidden_layers=42,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_context_length=8192,
    ),
    # --- Phi ---
    "phi-3.5-mini": ArchitectureProfile(
        parameter_count=3821079552,
        hidden_size=3072,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_context_length=131072,
    ),
    "phi-3-mini": ArchitectureProfile(
        parameter_count=3821079552,
        hidden_size=3072,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_context_length=131072,
    ),
    "phi-4-mini": ArchitectureProfile(
        parameter_count=3821079552,
        hidden_size=3072,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_context_length=131072,
    ),
    # --- Lightweight Models ---
    "smollm2-1.7b": ArchitectureProfile(
        parameter_count=1710000000,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_context_length=8192,
    ),
    "tinyllama-1.1b": ArchitectureProfile(
        parameter_count=1100048384,
        hidden_size=2048,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=4,
        max_context_length=2048,
    ),
}


class HuggingFaceMetadata:
    """
    Fetch model metadata without downloading repository files.

    This class never calls hf_hub_download().
    """

    def __init__(
        self,
        *,
        cache: MetadataCache | None = None,
        api: HfApi | None = None,
    ) -> None:
        self.cache = cache or MetadataCache()
        self.api = api or HfApi()

    def get(
        self,
        model_id: str,
        *,
        refresh: bool = False,
    ) -> ModelMetadata | None:

        model_id = model_id.strip()

        if not model_id:
            raise ValueError("model_id cannot be empty")

        cache_path = self.cache.model_path(model_id)

        if not refresh:
            cached = self.cache.get(cache_path)

            if cached is not None:
                metadata = self._from_dict(cached)

                if metadata is not None:
                    return metadata

        try:
            info = self.api.model_info(model_id)
        except Exception as error:
            print(f"Warning: failed to fetch metadata for {model_id}: {error}")
            return None

        metadata = self._build_metadata(
            info,
            model_id,
        )

        self.cache.put(
            cache_path,
            self._to_dict(metadata),
        )

        return metadata

    def _build_metadata(
        self,
        info: Any,
        model_id: str,
    ) -> ModelMetadata:

        config = getattr(info, "config", None)

        if not isinstance(config, dict):
            config = {}

        tags = getattr(info, "tags", None)

        if not isinstance(tags, (list, tuple)):
            tags = ()

        profile = self._find_profile(model_id)

        return ModelMetadata(
            model_id=model_id,
            author=_safe_string(getattr(info, "author", None)),
            downloads=_safe_int(getattr(info, "downloads", 0)),
            likes=_safe_int(getattr(info, "likes", 0)),
            tags=tuple(str(tag) for tag in tags),
            pipeline_tag=_safe_string(getattr(info, "pipeline_tag", None)),
            library_name=_safe_string(getattr(info, "library_name", None)),
            architecture=self._architecture(config),
            model_type=_safe_string(config.get("model_type")),
            parameter_count=(
                _safe_int_or_none(config.get("num_parameters"))
                or (
                    _safe_int_or_none(getattr(getattr(info, "safetensors", None), "total", None))
                    if hasattr(info, "safetensors")
                    else None
                )
                or (profile.parameter_count if profile else None)
                or (
                    int(
                        float(
                            re.search(
                                r"(?:^|[^\d])(\d+(?:\.\d+)?)\s*[bB](?:$|[^\d])",
                                model_id,
                            ).group(1)
                        )
                        * 1_000_000_000
                    )
                    if re.search(r"(?:^|[^\d])(\d+(?:\.\d+)?)\s*[bB](?:$|[^\d])", model_id)
                    else None
                )
            ),
            hidden_size=(
                _safe_int_or_none(config.get("hidden_size"))
                or (profile.hidden_size if profile else None)
            ),
            num_hidden_layers=(
                _safe_int_or_none(config.get("num_hidden_layers"))
                or (profile.num_hidden_layers if profile else None)
            ),
            num_attention_heads=(
                _safe_int_or_none(config.get("num_attention_heads"))
                or (profile.num_attention_heads if profile else None)
            ),
            num_key_value_heads=(
                _safe_int_or_none(config.get("num_key_value_heads"))
                or (profile.num_key_value_heads if profile else None)
                or _safe_int_or_none(config.get("num_attention_heads"))
                or (profile.num_attention_heads if profile else None)
            ),
            max_context_length=(
                self._context_length(config) or (profile.max_context_length if profile else None)
            ),
            license=self._license(info),
            quantization=self._quantization(model_id, tags, config),
            description="",
        )

    @staticmethod
    def _find_profile(
        model_id: str,
    ) -> ArchitectureProfile | None:

        normalized = model_id.lower().replace("_", "-")

        for key in sorted(KNOWN_ARCHITECTURES.keys(), key=len, reverse=True):
            if key in normalized:
                return KNOWN_ARCHITECTURES[key]

        return None

    @staticmethod
    def _architecture(
        config: dict[str, Any],
    ) -> str | None:

        architectures = config.get("architectures")

        if isinstance(architectures, list) and architectures:
            return str(architectures[0])

        if isinstance(architectures, str):
            return architectures

        return None

    @staticmethod
    def _context_length(
        config: dict[str, Any],
    ) -> int | None:

        for key in (
            "max_position_embeddings",
            "max_sequence_length",
            "max_seq_len",
            "model_max_length",
        ):
            value = _safe_int_or_none(config.get(key))

            if value is not None and value > 0:
                return value

        return None

    @staticmethod
    def _license(
        info: Any,
    ) -> str | None:

        card_data = getattr(info, "card_data", None)

        if isinstance(card_data, dict):
            return _safe_string(card_data.get("license"))

        return None

    @staticmethod
    def _quantization(
        model_id: str,
        tags: tuple[str, ...],
        config: dict[str, Any],
    ) -> str | None:

        quantization_config = config.get("quantization_config")

        if isinstance(quantization_config, dict):
            method = quantization_config.get("quant_method") or quantization_config.get(
                "quantization_method"
            )

            if method:
                return str(method).lower()

        text = (model_id + " " + " ".join(tags)).lower()

        for value in (
            "gguf",
            "awq",
            "gptq",
            "fp8",
            "int8",
            "int4",
            "4bit",
            "8bit",
            "nf4",
        ):
            if value in text:
                return value

        return None

    @staticmethod
    def _to_dict(
        metadata: ModelMetadata,
    ) -> dict[str, Any]:

        return {
            "model_id": metadata.model_id,
            "author": metadata.author,
            "downloads": metadata.downloads,
            "likes": metadata.likes,
            "tags": list(metadata.tags),
            "pipeline_tag": metadata.pipeline_tag,
            "library_name": metadata.library_name,
            "architecture": metadata.architecture,
            "model_type": metadata.model_type,
            "parameter_count": metadata.parameter_count,
            "hidden_size": metadata.hidden_size,
            "num_hidden_layers": metadata.num_hidden_layers,
            "num_attention_heads": metadata.num_attention_heads,
            "num_key_value_heads": metadata.num_key_value_heads,
            "max_context_length": metadata.max_context_length,
            "license": metadata.license,
            "quantization": metadata.quantization,
            "description": metadata.description,
        }

    @staticmethod
    def _from_dict(
        data: dict[str, Any],
    ) -> ModelMetadata | None:

        model_id = data.get("model_id")

        if not isinstance(model_id, str):
            return None

        tags = data.get("tags", [])

        if not isinstance(tags, list):
            tags = []

        meta = ModelMetadata(
            model_id=model_id,
            author=_safe_string(data.get("author")),
            downloads=_safe_int(data.get("downloads", 0)),
            likes=_safe_int(data.get("likes", 0)),
            tags=tuple(str(tag) for tag in tags),
            pipeline_tag=_safe_string(data.get("pipeline_tag")),
            library_name=_safe_string(data.get("library_name")),
            architecture=_safe_string(data.get("architecture")),
            model_type=_safe_string(data.get("model_type")),
            parameter_count=_safe_int_or_none(data.get("parameter_count")),
            hidden_size=_safe_int_or_none(data.get("hidden_size")),
            num_hidden_layers=_safe_int_or_none(data.get("num_hidden_layers")),
            num_attention_heads=_safe_int_or_none(data.get("num_attention_heads")),
            num_key_value_heads=_safe_int_or_none(data.get("num_key_value_heads")),
            max_context_length=_safe_int_or_none(data.get("max_context_length")),
            license=_safe_string(data.get("license")),
            quantization=_safe_string(data.get("quantization")),
            description=str(data.get("description", "")),
        )

        if meta.parameter_count is None or meta.hidden_size is None:
            profile = HuggingFaceMetadata._find_profile(model_id)
            if profile is not None:
                meta = replace(
                    meta,
                    parameter_count=meta.parameter_count or profile.parameter_count,
                    hidden_size=meta.hidden_size or profile.hidden_size,
                    num_hidden_layers=meta.num_hidden_layers or profile.num_hidden_layers,
                    num_attention_heads=meta.num_attention_heads or profile.num_attention_heads,
                    num_key_value_heads=meta.num_key_value_heads or profile.num_key_value_heads,
                    max_context_length=meta.max_context_length or profile.max_context_length,
                )

        return meta


def _safe_int(
    value: Any,
) -> int:

    if isinstance(value, bool):
        return 0

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_int_or_none(
    value: Any,
) -> int | None:

    if isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_string(
    value: Any,
) -> str | None:

    if value is None:
        return None

    text = str(value).strip()

    return text or None
