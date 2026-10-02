from __future__ import annotations

from dataclasses import dataclass

from huggingface_hub import HfApi

from infra.cache import MetadataCache


@dataclass(frozen=True)
class DiscoveryRequest:
    use_case: str
    limit: int = 100


@dataclass(frozen=True)
class DiscoveredModel:
    model_id: str
    downloads: int
    likes: int
    tags: tuple[str, ...]
    pipeline_tag: str | None
    library_name: str | None


USE_CASE_SEARCHES: dict[str, tuple[str, ...]] = {
    "coding": (
        "code",
        "coder",
        "coding",
        "programming",
        "starcoder",
    ),
    "chat": (
        "chat",
        "instruct",
        "conversation",
        "assistant",
    ),
    "reasoning": (
        "deepseek-r1",
        "reasoning",
        "qwq",
        "thinking",
        "math",
    ),
    "rag": (
        "rag",
        "retrieval",
        "qa",
        "long-context",
        "qwen2.5",
        "llama-3.1",
    ),
    "summarization": (
        "summarization",
        "summarisation",
        "summary",
        "long-context",
    ),
    "general": (
        "instruct",
        "general",
        "text generation",
    ),
}


class HuggingFaceDiscovery:
    """
    Discover candidate text-generation models from Hugging Face.

    Discovery only retrieves lightweight Hub metadata.

    Model weights are never downloaded here.
    """

    def __init__(
        self,
        *,
        cache: MetadataCache | None = None,
        api: HfApi | None = None,
    ) -> None:
        self.cache = cache or MetadataCache()
        self.api = api or HfApi()

    def discover(
        self,
        request: DiscoveryRequest,
        *,
        refresh: bool = False,
    ) -> list[DiscoveredModel]:
        if request.limit <= 0:
            raise ValueError("limit must be greater than zero")

        use_case = request.use_case.strip().lower()

        searches = USE_CASE_SEARCHES.get(
            use_case,
            USE_CASE_SEARCHES["general"],
        )

        cache_key = f"{use_case}_{request.limit}"

        cache_path = self.cache.discovery_path(cache_key)

        if not refresh:
            cached = self.cache.get(cache_path)

            if cached is not None:
                models = self._deserialize(cached)

                if models:
                    return models

        discovered: list[DiscoveredModel] = []
        seen: set[str] = set()

        # We deliberately retrieve more candidates than the
        # final requested number. The recommender still needs
        # enough candidates after metadata and hardware
        # filtering.
        per_query_limit = max(
            20,
            request.limit,
        )

        target_count = max(
            request.limit,
            request.limit * 5,
        )

        for search in searches:
            if len(discovered) >= target_count:
                break

            try:
                results = self.api.list_models(
                    search=search,
                    pipeline_tag="text-generation",
                    sort="downloads",
                    limit=per_query_limit,
                )
            except Exception as error:
                print(f"Warning: Hugging Face search '{search}' failed: {error}")
                continue

            for item in results:
                model_id = getattr(
                    item,
                    "id",
                    None,
                )

                if not isinstance(
                    model_id,
                    str,
                ):
                    continue

                model_id = model_id.strip()

                if not model_id:
                    continue

                if model_id in seen:
                    continue

                seen.add(model_id)

                tags = getattr(
                    item,
                    "tags",
                    (),
                )

                if not isinstance(
                    tags,
                    (list, tuple),
                ):
                    tags = ()

                discovered.append(
                    DiscoveredModel(
                        model_id=model_id,
                        downloads=_safe_int(
                            getattr(
                                item,
                                "downloads",
                                0,
                            )
                        ),
                        likes=_safe_int(
                            getattr(
                                item,
                                "likes",
                                0,
                            )
                        ),
                        tags=tuple(str(tag) for tag in tags),
                        pipeline_tag=_safe_string(
                            getattr(
                                item,
                                "pipeline_tag",
                                None,
                            )
                        ),
                        library_name=_safe_string(
                            getattr(
                                item,
                                "library_name",
                                None,
                            )
                        ),
                    )
                )

                if len(discovered) >= target_count:
                    break

        discovered.sort(
            key=lambda model: (
                -model.downloads,
                -model.likes,
                model.model_id.lower(),
            )
        )

        discovered = discovered[:target_count]

        self.cache.put(
            cache_path,
            {
                "use_case": use_case,
                "limit": request.limit,
                "models": [self._serialize(model) for model in discovered],
            },
        )

        return discovered

    @staticmethod
    def _serialize(
        model: DiscoveredModel,
    ) -> dict[str, object]:
        return {
            "model_id": model.model_id,
            "downloads": model.downloads,
            "likes": model.likes,
            "tags": list(model.tags),
            "pipeline_tag": model.pipeline_tag,
            "library_name": model.library_name,
        }

    @staticmethod
    def _deserialize(
        data: dict[str, object],
    ) -> list[DiscoveredModel]:
        raw_models = data.get("models")

        if not isinstance(
            raw_models,
            list,
        ):
            return []

        result: list[DiscoveredModel] = []

        for raw in raw_models:
            if not isinstance(
                raw,
                dict,
            ):
                continue

            model_id = raw.get("model_id")

            if not isinstance(
                model_id,
                str,
            ):
                continue

            model_id = model_id.strip()

            if not model_id:
                continue

            tags = raw.get(
                "tags",
                [],
            )

            if not isinstance(
                tags,
                list,
            ):
                tags = []

            result.append(
                DiscoveredModel(
                    model_id=model_id,
                    downloads=_safe_int(
                        raw.get(
                            "downloads",
                            0,
                        )
                    ),
                    likes=_safe_int(
                        raw.get(
                            "likes",
                            0,
                        )
                    ),
                    tags=tuple(str(tag) for tag in tags),
                    pipeline_tag=_safe_string(raw.get("pipeline_tag")),
                    library_name=_safe_string(raw.get("library_name")),
                )
            )

        return result


def _safe_int(
    value: object,
) -> int:
    if isinstance(
        value,
        bool,
    ):
        return 0

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return 0


def _safe_string(
    value: object,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    return text or None
