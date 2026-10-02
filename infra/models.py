import json
from dataclasses import dataclass
from enum import StrEnum
from math import prod

from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open

api = HfApi()


@dataclass
class ModelInfo:
    model_id: str
    author: str | None
    downloads: int | None
    likes: int | None
    tags: list[str]


@dataclass
class ModelConfig:
    model_type: str | None
    architecture: str | None
    hidden_size: int | None
    num_hidden_layers: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    max_position_embeddings: int | None


@dataclass
class ParameterInfo:
    count: int | None
    source: str


class ModelTask(StrEnum):
    CHAT = "chat"
    CODING = "coding"
    REASONING = "reasoning"
    RAG = "rag"
    SUMMARIZATION = "summarization"
    GENERAL = "general"


def inspect_model_files(model_id: str) -> None:

    files = api.list_repo_files(
        repo_id=model_id,
        repo_type="model",
    )
    for file in files:
        print(file)


def count_safetensor_parameters(model_id: str) -> ParameterInfo:
    model_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    parameter_count = 0
    with safe_open(model_path, framework="pt") as file:
        for tensor_name in file.keys():
            shape = file.get_slice(tensor_name).get_shape()
            parameter_count += prod(shape)

    return ParameterInfo(count=parameter_count, source="safetensors tensor shapes")


def load_model(model_id: str) -> ModelInfo:
    raw_info = api.model_info(model_id)

    return ModelInfo(
        model_id=model_id,
        author=raw_info.author,
        downloads=raw_info.downloads,
        likes=raw_info.likes,
        tags=raw_info.tags or [],
    )


def load_model_config(model_id: str) -> ModelConfig:
    config_path = hf_hub_download(
        repo_id=model_id,
        filename="config.json",
    )

    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)

    return ModelConfig(
        model_type=config.get("model_type"),
        architecture=config.get("architectures", [None])[0],
        hidden_size=config.get("hidden_size"),
        num_hidden_layers=config.get("num_hidden_layers"),
        num_attention_heads=config.get("num_attention_heads"),
        num_key_value_heads=config.get("num_key_value_heads"),
        max_position_embeddings=config.get("max_position_embeddings"),
    )
