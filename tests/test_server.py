from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import infra.server as server_module
from infra.server import (
    ServerConfig,
    ServerState,
    ServerStatus,
    VLLMServer,
)


def make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return model_dir


def make_config(
    tmp_path: Path,
    **kwargs,
) -> ServerConfig:
    defaults = {
        "model_path": make_model_dir(tmp_path),
    }

    defaults.update(kwargs)

    return ServerConfig(**defaults)


def test_status_url():
    status = ServerStatus(
        state=ServerState.RUNNING,
        host="127.0.0.1",
        port=8000,
        pid=123,
    )

    assert status.url == "http://127.0.0.1:8000"
    assert status.api_url == "http://127.0.0.1:8000/v1"


def test_status_url_without_host():
    status = ServerStatus(
        state=ServerState.STOPPED,
        host=None,
        port=None,
        pid=None,
    )

    assert status.url is None
    assert status.api_url is None


def test_server_initial_status():
    server = VLLMServer()

    status = server.status

    assert status.state == ServerState.STOPPED
    assert status.host is None
    assert status.port is None
    assert status.pid is None


def test_is_running_initially_false():
    server = VLLMServer()

    assert server.is_running() is False


def test_save_and_load_state(tmp_path, monkeypatch):
    state_file = tmp_path / "server.json"
    server_dir = tmp_path / "server"

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_DIR",
        server_dir,
    )

    config = make_config(
        tmp_path,
        served_model_name="test-model",
        host="127.0.0.1",
        port=9000,
        dtype="float16",
        max_model_len=1024,
        gpu_memory_utilization=0.9,
        kv_cache_memory_bytes=None,
        max_num_seqs=4,
        enforce_eager=True,
        api_key="secret",
    )

    VLLMServer._save_state(
        config,
        12345,
    )

    assert state_file.exists()

    data = VLLMServer._load_state()

    assert data is not None
    assert data["pid"] == 12345
    assert data["model_path"] == str(config.model_path)
    assert data["served_model_name"] == "test-model"
    assert data["host"] == "127.0.0.1"
    assert data["port"] == 9000
    assert data["dtype"] == "float16"
    assert data["max_model_len"] == 1024
    assert data["gpu_memory_utilization"] == 0.9
    assert data["max_num_seqs"] == 4
    assert data["enforce_eager"] is True
    assert data["api_key"] == "secret"


def test_load_state_missing_file(tmp_path, monkeypatch):
    state_file = tmp_path / "missing.json"

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    assert VLLMServer._load_state() is None


def test_load_state_invalid_json(tmp_path, monkeypatch):
    state_file = tmp_path / "server.json"

    state_file.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    assert VLLMServer._load_state() is None


def test_load_state_non_dict(tmp_path, monkeypatch):
    state_file = tmp_path / "server.json"

    state_file.write_text(
        json.dumps(["not", "a", "dict"]),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    assert VLLMServer._load_state() is None


def test_clear_state(tmp_path, monkeypatch):
    state_file = tmp_path / "server.json"

    state_file.write_text(
        "{}",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    VLLMServer._clear_state()

    assert not state_file.exists()


def test_clear_state_when_missing(tmp_path, monkeypatch):
    state_file = tmp_path / "missing.json"

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    VLLMServer._clear_state()

    assert not state_file.exists()


def test_managed_status_without_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        tmp_path / "missing.json",
    )

    status = VLLMServer.managed_status()

    assert status.state == ServerState.STOPPED
    assert status.message == "No managed vLLM server."


def test_managed_status_invalid_pid(
    tmp_path,
    monkeypatch,
):
    state_file = tmp_path / "server.json"

    state_file.write_text(
        json.dumps(
            {
                "pid": "not-an-int",
                "host": "127.0.0.1",
                "port": 8000,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    status = VLLMServer.managed_status()

    assert status.state == ServerState.FAILED
    assert status.pid is None
    assert status.host == "127.0.0.1"
    assert status.port == 8000
    assert status.message == "Invalid PID in server state."
    assert not state_file.exists()


def test_managed_status_process_not_running(
    tmp_path,
    monkeypatch,
):
    state_file = tmp_path / "server.json"

    state_file.write_text(
        json.dumps(
            {
                "pid": 12345,
                "host": "127.0.0.1",
                "port": 8000,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    monkeypatch.setattr(
        server_module,
        "_process_exists",
        lambda pid: False,
    )

    status = VLLMServer.managed_status()

    assert status.state == ServerState.STOPPED
    assert status.pid == 12345
    assert status.message == "Server process is not running."
    assert not state_file.exists()


def test_managed_status_running(
    tmp_path,
    monkeypatch,
):
    state_file = tmp_path / "server.json"

    state_file.write_text(
        json.dumps(
            {
                "pid": 12345,
                "host": "127.0.0.1",
                "port": 8000,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    monkeypatch.setattr(
        server_module,
        "_process_exists",
        lambda pid: True,
    )

    status = VLLMServer.managed_status()

    assert status.state == ServerState.RUNNING
    assert status.pid == 12345
    assert status.host == "127.0.0.1"
    assert status.port == 8000
    assert status.url == "http://127.0.0.1:8000"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"host": ""},
        {"host": "   "},
        {"port": 0},
        {"port": 65536},
        {"dtype": ""},
        {"dtype": "   "},
        {"max_model_len": 0},
        {"max_model_len": -1},
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": -0.1},
        {"gpu_memory_utilization": 1.1},
        {"kv_cache_memory_bytes": 0},
        {"kv_cache_memory_bytes": -1},
        {"max_num_seqs": 0},
        {"max_num_seqs": -1},
    ],
)
def test_invalid_config(
    tmp_path,
    kwargs,
):
    server = VLLMServer()

    config = make_config(
        tmp_path,
        **kwargs,
    )

    with pytest.raises(ValueError):
        server._validate_config(
            config,
            timeout=10,
        )


def test_invalid_timeout(tmp_path):
    server = VLLMServer()
    config = make_config(tmp_path)

    with pytest.raises(
        ValueError,
        match="timeout",
    ):
        server._validate_config(
            config,
            timeout=0,
        )


def test_missing_model_path(tmp_path):
    server = VLLMServer()

    config = ServerConfig(
        model_path=tmp_path / "does-not-exist",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Model path does not exist",
    ):
        server._validate_config(
            config,
            timeout=10,
        )


def test_model_path_must_be_directory(tmp_path):
    server = VLLMServer()

    model_file = tmp_path / "model.bin"
    model_file.write_text("fake")

    config = ServerConfig(
        model_path=model_file,
    )

    with pytest.raises(
        NotADirectoryError,
        match="not a directory",
    ):
        server._validate_config(
            config,
            timeout=10,
        )


def test_valid_config(tmp_path):
    server = VLLMServer()

    config = make_config(
        tmp_path,
        host="127.0.0.1",
        port=8000,
        dtype="auto",
        max_model_len=1024,
        gpu_memory_utilization=0.9,
        max_num_seqs=4,
        enforce_eager=True,
    )

    server._validate_config(
        config,
        timeout=10,
    )


def test_build_command_minimal(
    tmp_path,
    monkeypatch,
):
    server = VLLMServer()

    monkeypatch.setattr(
        server_module.shutil,
        "which",
        lambda name: "/usr/bin/vllm",
    )

    config = make_config(tmp_path)

    command = server.build_command(config)

    assert command == [
        "/usr/bin/vllm",
        "serve",
        str(config.model_path),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        "0.85",
    ]


def test_build_command_all_options(
    tmp_path,
    monkeypatch,
):
    server = VLLMServer()

    monkeypatch.setattr(
        server_module.shutil,
        "which",
        lambda name: "/usr/bin/vllm",
    )

    config = make_config(
        tmp_path,
        served_model_name="my-model",
        host="0.0.0.0",
        port=9000,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=None,
        kv_cache_memory_bytes=123456,
        max_num_seqs=8,
        enforce_eager=True,
        api_key="secret",
    )

    command = server.build_command(config)

    assert command == [
        "/usr/bin/vllm",
        "serve",
        str(config.model_path),
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "--dtype",
        "float16",
        "--served-model-name",
        "my-model",
        "--max-model-len",
        "2048",
        "--kv-cache-memory-bytes",
        "123456",
        "--max-num-seqs",
        "8",
        "--enforce-eager",
        "--api-key",
        "secret",
    ]


def test_build_command_fails_without_vllm(
    tmp_path,
    monkeypatch,
):
    server = VLLMServer()

    monkeypatch.setattr(
        server_module.shutil,
        "which",
        lambda name: None,
    )

    config = make_config(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="Could not find `vllm`",
    ):
        server.build_command(config)


def test_headers_without_api_key(tmp_path):
    server = VLLMServer()

    server._config = make_config(tmp_path)

    assert server._headers() == {
        "Content-Type": "application/json",
    }


def test_headers_with_api_key(tmp_path):
    server = VLLMServer()

    server._config = make_config(
        tmp_path,
        api_key="secret",
    )

    headers = server._headers()

    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer secret"


def test_url(tmp_path):
    server = VLLMServer()

    server._config = make_config(
        tmp_path,
        host="127.0.0.1",
        port=9000,
    )

    assert server._url("/health") == ("http://127.0.0.1:9000/health")


def test_url_without_config():
    server = VLLMServer()

    with pytest.raises(
        RuntimeError,
        match="Server configuration",
    ):
        server._url("/health")


def test_default_model_name(tmp_path):
    server = VLLMServer()

    model_path = make_model_dir(tmp_path)

    server._config = ServerConfig(
        model_path=model_path,
    )

    assert server._default_model_name() == "model"


def test_default_model_name_ignores_served_name(tmp_path):
    server = VLLMServer()

    server._config = make_config(
        tmp_path,
        served_model_name="custom-name",
    )

    assert server._default_model_name() == "model"


def test_extract_chat_text():
    data = {
        "choices": [
            {
                "message": {
                    "content": "Hello world",
                }
            }
        ]
    }

    assert VLLMServer._extract_chat_text(data) == "Hello world"


def test_extract_chat_text_invalid_non_dict():
    with pytest.raises(
        RuntimeError,
        match="Unexpected vLLM API response",
    ):
        VLLMServer._extract_chat_text(None)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"choices": []},
        {"choices": "invalid"},
        {"choices": [None]},
        {"choices": [{}]},
        {
            "choices": [
                {
                    "message": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": 123,
                    }
                }
            ]
        },
    ],
)
def test_extract_chat_text_invalid_structure(data):
    with pytest.raises(
        RuntimeError,
        match="Unexpected vLLM chat response",
    ):
        VLLMServer._extract_chat_text(data)


def test_process_exists_for_current_process():
    assert server_module._process_exists(os.getpid())


def test_process_exists_for_missing_process():
    assert not server_module._process_exists(999999999)


def test_is_running_with_fake_process(tmp_path):
    server = VLLMServer()

    fake_process = SimpleNamespace(
        pid=123,
        poll=lambda: None,
    )

    server._process = fake_process
    server._config = make_config(tmp_path)

    assert server.is_running() is True


def test_is_running_when_process_exited(tmp_path):
    server = VLLMServer()

    fake_process = SimpleNamespace(
        pid=123,
        poll=lambda: 0,
    )

    server._process = fake_process
    server._config = make_config(tmp_path)

    assert server.is_running() is False


def test_require_running_when_stopped():
    server = VLLMServer()

    with pytest.raises(
        RuntimeError,
        match="server is not running",
    ):
        server._require_running()


def test_stop_without_process(
    tmp_path,
    monkeypatch,
):
    state_file = tmp_path / "server.json"

    state_file.write_text(
        "{}",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    server = VLLMServer()

    status = server.stop()

    assert status.state == ServerState.STOPPED
    assert status.message == ("vLLM server is not running.")
    assert not state_file.exists()


def test_terminate_process_when_none():
    server = VLLMServer()

    server._terminate_process()

    assert server._process is None


def test_start_rejects_existing_managed_server(
    tmp_path,
    monkeypatch,
):
    state_file = tmp_path / "server.json"

    state_file.write_text(
        json.dumps(
            {
                "pid": 123,
                "host": "127.0.0.1",
                "port": 8000,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    monkeypatch.setattr(
        server_module,
        "_process_exists",
        lambda pid: True,
    )

    server = VLLMServer()
    config = make_config(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="already running",
    ):
        server.start(
            config,
            timeout=1,
        )


def test_start_success(
    tmp_path,
    monkeypatch,
):
    state_file = tmp_path / "server.json"
    server_dir = tmp_path / "server"
    log_file = server_dir / "server.log"

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_DIR",
        server_dir,
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_LOG_FILE",
        log_file,
    )

    monkeypatch.setattr(
        server_module.shutil,
        "which",
        lambda name: "/usr/bin/vllm",
    )

    class FakeProcess:
        pid = 1234

        def poll(self):
            return None

    fake_process = FakeProcess()

    monkeypatch.setattr(
        server_module.subprocess,
        "Popen",
        lambda *args, **kwargs: fake_process,
    )

    server = VLLMServer()

    server._wait_until_ready = lambda timeout: None

    config = make_config(
        tmp_path,
        port=9000,
    )

    status = server.start(
        config,
        timeout=5,
    )

    assert status.state == ServerState.RUNNING
    assert status.pid == 1234
    assert status.host == "127.0.0.1"
    assert status.port == 9000
    assert status.message == ("vLLM server is ready.")

    assert state_file.exists()

    data = json.loads(
        state_file.read_text(
            encoding="utf-8",
        )
    )

    assert data["pid"] == 1234
    assert data["port"] == 9000


def test_start_missing_vllm_executable(
    tmp_path,
    monkeypatch,
):
    state_file = tmp_path / "server.json"
    server_dir = tmp_path / "server"
    log_file = server_dir / "server.log"

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_DIR",
        server_dir,
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_LOG_FILE",
        log_file,
    )

    monkeypatch.setattr(
        server_module.shutil,
        "which",
        lambda name: "/usr/bin/vllm",
    )

    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("vllm")

    monkeypatch.setattr(
        server_module.subprocess,
        "Popen",
        raise_missing,
    )

    server = VLLMServer()

    config = make_config(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="vLLM executable was not found",
    ):
        server.start(
            config,
            timeout=5,
        )

    assert server.status.state == ServerState.FAILED
    assert server._process is None


def test_wait_until_ready_requires_config():
    server = VLLMServer()

    server._process = SimpleNamespace(
        poll=lambda: None,
    )

    with pytest.raises(
        RuntimeError,
        match="configuration is unavailable",
    ):
        server._wait_until_ready(1)


def test_wait_until_ready_requires_process(
    tmp_path,
):
    server = VLLMServer()

    server._config = make_config(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="process has not been started",
    ):
        server._wait_until_ready(1)


def test_wait_until_ready_process_exits(
    tmp_path,
):
    server = VLLMServer()

    server._config = make_config(tmp_path)

    server._process = SimpleNamespace(
        poll=lambda: 1,
    )

    with pytest.raises(
        RuntimeError,
        match="exited during startup",
    ):
        server._wait_until_ready(1)


def test_wait_until_ready_health_success(
    tmp_path,
    monkeypatch,
):
    server = VLLMServer()

    server._config = make_config(tmp_path)

    server._process = SimpleNamespace(
        poll=lambda: None,
    )

    monkeypatch.setattr(
        server,
        "health",
        lambda timeout: True,
    )

    server._wait_until_ready(1)


def test_chat_requires_running():
    server = VLLMServer()

    with pytest.raises(
        RuntimeError,
        match="server is not running",
    ):
        server.chat(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )


def test_models_requires_running():
    server = VLLMServer()

    with pytest.raises(
        RuntimeError,
        match="server is not running",
    ):
        server.models()


def test_health_without_config():
    server = VLLMServer()

    assert server.health() is False


def test_managed_health_without_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        tmp_path / "missing.json",
    )

    assert VLLMServer.managed_health() is False


def test_stop_managed_without_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        tmp_path / "missing.json",
    )

    status = VLLMServer.stop_managed()

    assert status.state == ServerState.STOPPED
    assert status.message == ("No managed vLLM server.")


def test_stop_managed_invalid_pid(
    tmp_path,
    monkeypatch,
):
    state_file = tmp_path / "server.json"

    state_file.write_text(
        json.dumps(
            {
                "pid": "invalid",
                "host": "127.0.0.1",
                "port": 8000,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    status = VLLMServer.stop_managed()

    assert status.state == ServerState.FAILED
    assert status.message == "Invalid PID."
    assert not state_file.exists()


def test_stop_managed_already_stopped(
    tmp_path,
    monkeypatch,
):
    state_file = tmp_path / "server.json"

    state_file.write_text(
        json.dumps(
            {
                "pid": 12345,
                "host": "127.0.0.1",
                "port": 8000,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        server_module,
        "SERVER_STATE_FILE",
        state_file,
    )

    monkeypatch.setattr(
        server_module,
        "_process_exists",
        lambda pid: False,
    )

    status = VLLMServer.stop_managed()

    assert status.state == ServerState.STOPPED
    assert status.pid == 12345
    assert status.message == ("Server was already stopped.")
    assert not state_file.exists()
