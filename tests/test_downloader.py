from pathlib import Path

import pytest

from infra import downloader


def test_download_model_rejects_empty_model_id():
    with pytest.raises(
        ValueError,
        match="model_id cannot be empty",
    ):
        downloader.download_model("")


def test_download_model_rejects_whitespace_model_id():
    with pytest.raises(
        ValueError,
        match="model_id cannot be empty",
    ):
        downloader.download_model("   ")


def test_download_model_rejects_empty_patterns():
    with pytest.raises(
        ValueError,
        match="allow_patterns cannot be empty",
    ):
        downloader.download_model(
            "test/model",
            allow_patterns=[],
        )


def test_download_model_calls_snapshot_download(monkeypatch):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/tmp/test-model"

    monkeypatch.setattr(
        downloader,
        "snapshot_download",
        fake_snapshot_download,
    )

    result = downloader.download_model(
        "test/model",
    )

    assert result == Path("/tmp/test-model")
    assert captured["repo_id"] == "test/model"
    assert captured["repo_type"] == "model"
    assert captured["allow_patterns"] == downloader.MODEL_PATTERNS
    assert captured["tqdm_class"] is None


def test_download_model_uses_custom_patterns(monkeypatch):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/tmp/test-model"

    monkeypatch.setattr(
        downloader,
        "snapshot_download",
        fake_snapshot_download,
    )

    patterns = [
        "*.safetensors",
        "config.json",
    ]

    result = downloader.download_model(
        "test/model",
        allow_patterns=patterns,
    )

    assert result == Path("/tmp/test-model")
    assert captured["allow_patterns"] == patterns


def test_download_model_strips_model_id(monkeypatch):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/tmp/test-model"

    monkeypatch.setattr(
        downloader,
        "snapshot_download",
        fake_snapshot_download,
    )

    downloader.download_model(
        "  test/model  ",
    )

    assert captured["repo_id"] == "test/model"


def test_download_model_passes_progress_tqdm_class(
    monkeypatch,
):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/tmp/test-model"

    monkeypatch.setattr(
        downloader,
        "snapshot_download",
        fake_snapshot_download,
    )

    def callback(current, total):
        pass

    downloader.download_model(
        "test/model",
        progress_callback=callback,
    )

    assert captured["tqdm_class"] is not None


def test_progress_callback_is_called_when_progress_class_is_created(
    monkeypatch,
):
    captured = {}

    def fake_snapshot_download(**kwargs):
        tqdm_class = kwargs["tqdm_class"]

        progress = tqdm_class(total=100)

        captured["progress"] = progress
        return "/tmp/test-model"

    monkeypatch.setattr(
        downloader,
        "snapshot_download",
        fake_snapshot_download,
    )

    callback_values = []

    def callback(current, total):
        callback_values.append((current, total))

    downloader.download_model(
        "test/model",
        progress_callback=callback,
    )

    assert callback_values
    assert callback_values[0] == (0.0, 100.0)


def test_model_patterns_contain_required_files():
    patterns = downloader.MODEL_PATTERNS

    assert "*.safetensors" in patterns
    assert "*.safetensors.index.json" in patterns
    assert "config.json" in patterns
    assert "generation_config.json" in patterns
    assert "tokenizer.json" in patterns
    assert "tokenizer_config.json" in patterns
    assert "special_tokens_map.json" in patterns
    assert "tokenizer.model" in patterns
    assert "vocab.json" in patterns
    assert "merges.txt" in patterns
    assert "added_tokens.json" in patterns


def test_model_patterns_are_not_empty():
    assert downloader.MODEL_PATTERNS


def test_download_returns_path_object(monkeypatch):
    def fake_snapshot_download(**kwargs):
        return "/tmp/model"

    monkeypatch.setattr(
        downloader,
        "snapshot_download",
        fake_snapshot_download,
    )

    result = downloader.download_model("test/model")

    assert isinstance(result, Path)


def test_download_does_not_use_progress_class_without_callback(
    monkeypatch,
):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/tmp/model"

    monkeypatch.setattr(
        downloader,
        "snapshot_download",
        fake_snapshot_download,
    )

    downloader.download_model("test/model")

    assert captured["tqdm_class"] is None
