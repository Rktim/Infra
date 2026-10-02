from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SERVER_DIR = Path.home() / ".cache" / "infra" / "server"

SERVER_STATE_FILE = SERVER_DIR / "server.json"

SERVER_LOG_FILE = SERVER_DIR / "server.log"


class ServerState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"


@dataclass(frozen=True)
class ServerConfig:
    """
    Configuration for a managed local vLLM server.
    """

    model_path: Path

    served_model_name: str | None = None

    host: str = "127.0.0.1"

    port: int = 8000

    dtype: str = "auto"

    max_model_len: int | None = None

    gpu_memory_utilization: float | None = 0.85

    kv_cache_memory_bytes: int | None = None

    max_num_seqs: int | None = None

    enforce_eager: bool = False

    api_key: str | None = None


@dataclass(frozen=True)
class ServerStatus:
    state: ServerState

    host: str | None

    port: int | None

    pid: int | None

    message: str | None = None

    @property
    def url(self) -> str | None:
        if self.host is None or self.port is None:
            return None

        return f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str | None:
        if self.url is None:
            return None

        return f"{self.url}/v1"


class VLLMServer:
    """
    Manage an external vLLM OpenAI-compatible server.

    This is intentionally separate from VLLMBackend.

    VLLMBackend:
        in-process model execution.

    VLLMServer:
        external `vllm serve` HTTP server.
    """

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None

        self._config: ServerConfig | None = None

        self._state = ServerState.STOPPED

        self._message: str | None = None

    @property
    def status(self) -> ServerStatus:
        return ServerStatus(
            state=self._state,
            host=(self._config.host if self._config is not None else None),
            port=(self._config.port if self._config is not None else None),
            pid=(self._process.pid if self._process is not None else None),
            message=self._message,
        )

    @classmethod
    def managed_status(cls) -> ServerStatus:
        """
        Inspect a server started by another CLI process.
        """

        data = cls._load_state()

        if data is None:
            return ServerStatus(
                state=ServerState.STOPPED,
                host=None,
                port=None,
                pid=None,
                message="No managed vLLM server.",
            )

        pid = data.get("pid")

        host = data.get("host")

        port = data.get("port")

        if not isinstance(pid, int):
            cls._clear_state()

            return ServerStatus(
                state=ServerState.FAILED,
                host=(host if isinstance(host, str) else None),
                port=(port if isinstance(port, int) else None),
                pid=None,
                message="Invalid PID in server state.",
            )

        if not _process_exists(pid):
            cls._clear_state()

            return ServerStatus(
                state=ServerState.STOPPED,
                host=(host if isinstance(host, str) else None),
                port=(port if isinstance(port, int) else None),
                pid=pid,
                message="Server process is not running.",
            )

        return ServerStatus(
            state=ServerState.RUNNING,
            host=(host if isinstance(host, str) else None),
            port=(port if isinstance(port, int) else None),
            pid=pid,
            message="Managed vLLM server is running.",
        )

    def start(
        self,
        config: ServerConfig,
        timeout: float = 180.0,
    ) -> ServerStatus:
        """
        Start vLLM and wait until /health responds.
        """

        managed = self.managed_status()

        if managed.state == ServerState.RUNNING:
            raise RuntimeError(f"vLLM server is already running (PID {managed.pid}, {managed.url})")

        self._validate_config(
            config,
            timeout,
        )

        self._config = config

        self._state = ServerState.STARTING

        self._message = "Starting vLLM server."

        command = self._build_command(config)

        environment = os.environ.copy()

        SERVER_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            log_file = SERVER_LOG_FILE.open(
                "a",
                encoding="utf-8",
            )

            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )

            log_file.close()

        except FileNotFoundError as error:
            self._state = ServerState.FAILED

            self._message = "vLLM executable was not found. Make sure vLLM is installed."

            self._process = None

            raise RuntimeError(self._message) from error

        except Exception as error:
            self._state = ServerState.FAILED

            self._message = str(error)

            self._process = None

            raise RuntimeError(f"Failed to start vLLM server: {error}") from error

        try:
            self._wait_until_ready(timeout)

        except Exception as error:
            self._terminate_process()

            self._process = None

            self._state = ServerState.FAILED

            self._message = str(error)

            raise RuntimeError(
                f"vLLM server failed to start: {error}\nSee log: {SERVER_LOG_FILE}"
            ) from error

        self._save_state(
            config,
            self._process.pid,
        )

        self._state = ServerState.RUNNING

        self._message = "vLLM server is ready."

        return self.status

    def stop(
        self,
        timeout: float = 30.0,
    ) -> ServerStatus:
        """
        Stop the locally managed process.
        """

        if self._process is None:
            self._clear_state()

            self._state = ServerState.STOPPED

            self._message = "vLLM server is not running."

            return self.status

        self._terminate_process(timeout)

        self._process = None

        self._clear_state()

        self._state = ServerState.STOPPED

        self._message = "vLLM server stopped."

        return self.status

    @classmethod
    def stop_managed(
        cls,
        timeout: float = 30.0,
    ) -> ServerStatus:
        """
        Stop a server started by another CLI process.
        """

        data = cls._load_state()

        if data is None:
            return ServerStatus(
                state=ServerState.STOPPED,
                host=None,
                port=None,
                pid=None,
                message=("No managed vLLM server."),
            )

        pid = data.get("pid")

        host = data.get("host")

        port = data.get("port")

        if not isinstance(pid, int):
            cls._clear_state()

            return ServerStatus(
                state=ServerState.FAILED,
                host=(host if isinstance(host, str) else None),
                port=(port if isinstance(port, int) else None),
                pid=None,
                message="Invalid PID.",
            )

        if not _process_exists(pid):
            cls._clear_state()

            return ServerStatus(
                state=ServerState.STOPPED,
                host=(host if isinstance(host, str) else None),
                port=(port if isinstance(port, int) else None),
                pid=pid,
                message=("Server was already stopped."),
            )

        try:
            os.killpg(
                os.getpgid(pid),
                signal.SIGTERM,
            )

        except ProcessLookupError:
            pass

        except PermissionError as error:
            raise RuntimeError(f"Permission denied stopping PID {pid}.") from error

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if not _process_exists(pid):
                cls._clear_state()

                return ServerStatus(
                    state=ServerState.STOPPED,
                    host=(host if isinstance(host, str) else None),
                    port=(port if isinstance(port, int) else None),
                    pid=pid,
                    message=("vLLM server stopped."),
                )

            time.sleep(0.25)

        # Force kill if graceful shutdown failed.
        try:
            os.killpg(
                os.getpgid(pid),
                signal.SIGKILL,
            )

        except ProcessLookupError:
            pass

        cls._clear_state()

        return ServerStatus(
            state=ServerState.STOPPED,
            host=(host if isinstance(host, str) else None),
            port=(port if isinstance(port, int) else None),
            pid=pid,
            message=("vLLM server force-stopped."),
        )

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 1.0,
        timeout: float = 60.0,
    ) -> str:
        """
        Send an OpenAI-compatible chat completion request.
        """

        self._require_running()

        if not messages:
            raise ValueError("messages cannot be empty")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        if not 0 < top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        model_name = (
            self._config.served_model_name
            if self._config and self._config.served_model_name
            else self._default_model_name()
        )

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        request = Request(
            self._url("/v1/chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=timeout,
            ) as response:
                body = response.read()

                data = json.loads(body.decode("utf-8"))

        except HTTPError as error:
            body = error.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(f"vLLM API returned HTTP {error.code}: {body}") from error

        except URLError as error:
            raise RuntimeError(f"Could not reach vLLM API: {error}") from error

        except TimeoutError as error:
            raise RuntimeError("vLLM API request timed out") from error

        return self._extract_chat_text(data)

    def models(
        self,
        timeout: float = 10.0,
    ) -> list[str]:
        """
        Return models exposed by vLLM.
        """

        self._require_running()

        request = Request(
            self._url("/v1/models"),
            headers=self._headers(),
            method="GET",
        )

        try:
            with urlopen(
                request,
                timeout=timeout,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))

        except HTTPError as error:
            body = error.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(f"vLLM API returned HTTP {error.code}: {body}") from error

        except URLError as error:
            raise RuntimeError(f"Could not reach vLLM API: {error}") from error

        raw_models = data.get(
            "data",
            [],
        )

        if not isinstance(
            raw_models,
            list,
        ):
            raise RuntimeError(f"Unexpected /v1/models response: {data}")

        result: list[str] = []

        for model in raw_models:
            if not isinstance(
                model,
                dict,
            ):
                continue

            model_id = model.get("id")

            if isinstance(
                model_id,
                str,
            ):
                result.append(model_id)

        return result

    def health(
        self,
        timeout: float = 5.0,
    ) -> bool:
        """
        Check the vLLM health endpoint.
        """

        if self._config is None:
            return False

        request = Request(
            self._url("/health"),
            headers=self._headers(),
            method="GET",
        )

        try:
            with urlopen(
                request,
                timeout=timeout,
            ) as response:
                return 200 <= response.status < 300

        except (
            HTTPError,
            URLError,
            TimeoutError,
        ):
            return False

    @classmethod
    def managed_health(
        cls,
        timeout: float = 5.0,
    ) -> bool:
        """
        Check the persisted managed server.
        """

        data = cls._load_state()

        if data is None:
            return False

        host = data.get("host")

        port = data.get("port")

        if not isinstance(host, str):
            return False

        if not isinstance(port, int):
            return False

        request = Request(
            f"http://{host}:{port}/health",
            method="GET",
        )

        try:
            with urlopen(
                request,
                timeout=timeout,
            ) as response:
                return 200 <= response.status < 300

        except (
            HTTPError,
            URLError,
            TimeoutError,
        ):
            return False

    def build_command(
        self,
        config: ServerConfig,
    ) -> list[str]:
        """
        Public command builder.
        """

        return self._build_command(config)

    def _build_command(
        self,
        config: ServerConfig,
    ) -> list[str]:

        vllm_executable = shutil.which("vllm")

        if vllm_executable is None:
            raise RuntimeError("Could not find `vllm` executable on PATH.")

        command = [
            vllm_executable,
            "serve",
            str(config.model_path),
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--dtype",
            config.dtype,
        ]

        if config.served_model_name:
            command.extend(
                [
                    "--served-model-name",
                    config.served_model_name,
                ]
            )

        if config.max_model_len is not None:
            command.extend(
                [
                    "--max-model-len",
                    str(config.max_model_len),
                ]
            )

        if config.kv_cache_memory_bytes is not None:
            command.extend(
                [
                    "--kv-cache-memory-bytes",
                    str(config.kv_cache_memory_bytes),
                ]
            )

        elif config.gpu_memory_utilization is not None:
            command.extend(
                [
                    "--gpu-memory-utilization",
                    str(config.gpu_memory_utilization),
                ]
            )

        if config.max_num_seqs is not None:
            command.extend(
                [
                    "--max-num-seqs",
                    str(config.max_num_seqs),
                ]
            )

        if config.enforce_eager:
            command.append("--enforce-eager")

        if config.api_key:
            command.extend(
                [
                    "--api-key",
                    config.api_key,
                ]
            )

        return command

    def _validate_config(
        self,
        config: ServerConfig,
        timeout: float,
    ) -> None:

        if not config.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {config.model_path}")

        if not config.model_path.is_dir():
            raise NotADirectoryError(f"Model path is not a directory: {config.model_path}")

        if not config.host.strip():
            raise ValueError("host cannot be empty")

        if not (0 < config.port < 65536):
            raise ValueError("port must be between 1 and 65535")

        if not config.dtype.strip():
            raise ValueError("dtype cannot be empty")

        if config.max_model_len is not None and config.max_model_len <= 0:
            raise ValueError("max_model_len must be greater than zero")

        if config.gpu_memory_utilization is not None and not (
            0 < config.gpu_memory_utilization <= 1
        ):
            raise ValueError("gpu_memory_utilization must be between 0 and 1")

        if config.kv_cache_memory_bytes is not None and config.kv_cache_memory_bytes <= 0:
            raise ValueError("kv_cache_memory_bytes must be greater than zero")

        if config.max_num_seqs is not None and config.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be greater than zero")

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

    def _wait_until_ready(
        self,
        timeout: float,
    ) -> None:

        if self._config is None:
            raise RuntimeError("Server configuration is unavailable")

        if self._process is None:
            raise RuntimeError("Server process has not been started")

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            return_code = self._process.poll()

            if return_code is not None:
                raise RuntimeError(f"vLLM process exited during startup with code {return_code}")

            if self.health(timeout=2.0):
                return

            time.sleep(1.0)

        raise TimeoutError(f"vLLM server did not become ready within {timeout:.1f} seconds")

    def _terminate_process(
        self,
        timeout: float = 30.0,
    ) -> None:

        process = self._process

        if process is None:
            return

        if process.poll() is not None:
            return

        try:
            os.killpg(
                os.getpgid(process.pid),
                signal.SIGTERM,
            )

        except (
            ProcessLookupError,
            PermissionError,
        ):
            process.terminate()

        try:
            process.wait(timeout=timeout)

        except subprocess.TimeoutExpired:
            try:
                os.killpg(
                    os.getpgid(process.pid),
                    signal.SIGKILL,
                )

            except (
                ProcessLookupError,
                PermissionError,
            ):
                process.kill()

            process.wait()

    @staticmethod
    def _save_state(
        config: ServerConfig,
        pid: int,
    ) -> None:

        SERVER_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "pid": pid,
            "model_path": str(config.model_path),
            "served_model_name": (config.served_model_name),
            "host": config.host,
            "port": config.port,
            "dtype": config.dtype,
            "max_model_len": (config.max_model_len),
            "gpu_memory_utilization": (config.gpu_memory_utilization),
            "kv_cache_memory_bytes": (config.kv_cache_memory_bytes),
            "max_num_seqs": (config.max_num_seqs),
            "enforce_eager": (config.enforce_eager),
            "api_key": config.api_key,
        }

        temporary = SERVER_STATE_FILE.with_suffix(".json.tmp")

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
            )

        temporary.replace(SERVER_STATE_FILE)

    @staticmethod
    def _load_state() -> dict[str, object] | None:

        if not SERVER_STATE_FILE.exists():
            return None

        try:
            with SERVER_STATE_FILE.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except (
            OSError,
            json.JSONDecodeError,
        ):
            return None

        if not isinstance(
            data,
            dict,
        ):
            return None

        return data

    @staticmethod
    def _clear_state() -> None:

        try:
            SERVER_STATE_FILE.unlink()

        except FileNotFoundError:
            pass

        except OSError:
            pass

    def _require_running(self) -> None:

        if not self.is_running():
            raise RuntimeError("vLLM server is not running")

        if self._config is None:
            raise RuntimeError("Server configuration is unavailable")

    def _default_model_name(self) -> str:

        if self._config is None:
            raise RuntimeError("Server configuration is unavailable")

        return self._config.model_path.name

    def _url(
        self,
        path: str,
    ) -> str:

        if self._config is None:
            raise RuntimeError("Server configuration is unavailable")

        return f"http://{self._config.host}:{self._config.port}{path}"

    def _headers(self) -> dict[str, str]:

        headers = {"Content-Type": ("application/json")}

        if self._config is not None and self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        return headers

    @staticmethod
    def _extract_chat_text(
        data: object,
    ) -> str:

        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(f"Unexpected vLLM API response: {data}")

        try:
            choices = data["choices"]

            if not isinstance(
                choices,
                list,
            ):
                raise TypeError

            first = choices[0]

            if not isinstance(
                first,
                dict,
            ):
                raise TypeError

            message = first["message"]

            if not isinstance(
                message,
                dict,
            ):
                raise TypeError

            content = message["content"]

            if not isinstance(
                content,
                str,
            ):
                raise TypeError

            return content

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            raise RuntimeError(f"Unexpected vLLM chat response: {data}") from error


def _process_exists(
    pid: int,
) -> bool:
    try:
        os.kill(
            pid,
            0,
        )

    except ProcessLookupError:
        return False

    except PermissionError:
        return True

    return True
