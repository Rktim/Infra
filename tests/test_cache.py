from infra.cache import MetadataCache


def test_cache_creates_directories(tmp_path):
    cache = MetadataCache(root=tmp_path)

    assert cache.root == tmp_path
    assert cache.discovery_dir.exists()
    assert cache.models_dir.exists()


def test_put_and_get(tmp_path):
    cache = MetadataCache(root=tmp_path)

    path = cache.discovery_path("coding")

    data = {
        "use_case": "coding",
        "limit": 10,
        "models": ["model-a", "model-b"],
    }

    cache.put(path, data)

    assert path.exists()
    assert cache.get(path) == data


def test_get_missing_file_returns_none(tmp_path):
    cache = MetadataCache(root=tmp_path)

    path = cache.discovery_path("missing")

    assert cache.get(path) is None


def test_get_expired_cache_returns_none(tmp_path):
    cache = MetadataCache(
        root=tmp_path,
        ttl_seconds=0,
    )

    path = cache.discovery_path("coding")

    cache.put(
        path,
        {"models": ["model-a"]},
    )

    assert cache.get(path) is None


def test_get_invalid_json_returns_none(tmp_path):
    cache = MetadataCache(root=tmp_path)

    path = cache.discovery_path("coding")
    path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    assert cache.get(path) is None


def test_get_non_dictionary_json_returns_none(tmp_path):
    cache = MetadataCache(root=tmp_path)

    path = cache.discovery_path("coding")
    path.write_text(
        '["model-a", "model-b"]',
        encoding="utf-8",
    )

    assert cache.get(path) is None


def test_discovery_path_sanitizes_key(tmp_path):
    cache = MetadataCache(root=tmp_path)

    path = cache.discovery_path("coding/models test")

    assert path == (tmp_path / "discovery" / "coding_models_test.json")


def test_model_path_sanitizes_model_id(tmp_path):
    cache = MetadataCache(root=tmp_path)

    path = cache.model_path("Qwen/Test-Model")

    assert path == (tmp_path / "models" / "Qwen__Test-Model.json")


def test_put_creates_parent_directory(tmp_path):
    cache = MetadataCache(root=tmp_path)

    path = tmp_path / "nested" / "data.json"

    cache.put(
        path,
        {"value": 123},
    )

    assert path.exists()
    assert cache.get(path) == {"value": 123}


def test_clear_removes_json_cache_files(tmp_path):
    cache = MetadataCache(root=tmp_path)

    discovery = cache.discovery_path("coding")
    model = cache.model_path("Qwen/Test")

    cache.put(discovery, {"models": []})
    cache.put(model, {"model_id": "Qwen/Test"})

    assert discovery.exists()
    assert model.exists()

    cache.clear()

    assert not discovery.exists()
    assert not model.exists()


def test_clear_does_not_remove_directories(tmp_path):
    cache = MetadataCache(root=tmp_path)

    cache.clear()

    assert cache.discovery_dir.exists()
    assert cache.models_dir.exists()


def test_cache_supports_nested_data(tmp_path):
    cache = MetadataCache(root=tmp_path)

    path = cache.model_path("test/model")

    data = {
        "model_id": "test/model",
        "config": {
            "hidden_size": 1024,
            "layers": 12,
        },
        "tags": [
            "text-generation",
            "transformers",
        ],
    }

    cache.put(path, data)

    assert cache.get(path) == data
