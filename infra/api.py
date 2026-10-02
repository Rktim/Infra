from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class APIMessage:
    role: str
    content: str


@dataclass(frozen=True)
class APIResponse:
    text: str
    model: str
    request_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None


class infraAPIError(RuntimeError):
    """Raised when the local model API returns an error."""


class infraClient:
    """
    Thin client for an OpenAI-compatible vLLM server.

    infra does not implement the generation protocol itself.
    vLLM remains responsible for inference and the HTTP API.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def models(self) -> list[dict[str, object]]:
        data = self._request("GET", "/models")
        models = data.get("data", [])

        if not isinstance(models, list):
            raise infraAPIError("Invalid /models response.")

        return [model for model in models if isinstance(model, dict)]

    def health(self) -> bool:
        health_url = self.base_url.removesuffix("/v1") + "/health"

        try:
            request = Request(
                health_url,
                method="GET",
                headers=self._headers(),
            )

            with urlopen(request, timeout=self.timeout) as response:
                return 200 <= response.status < 300

        except (HTTPError, URLError, TimeoutError, OSError):
            return False

    def chat(
        self,
        messages: list[APIMessage | dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> APIResponse:
        if not messages:
            raise ValueError("messages cannot be empty")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        if not 0 < top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

        if model is None:
            available_models = self.models()

            if not available_models:
                raise infraAPIError("No model is available on the server.")

            model_value = available_models[0].get("id")

            if not isinstance(model_value, str):
                raise infraAPIError("Server returned an invalid model ID.")

            model = model_value

        normalized_messages: list[dict[str, str]] = []

        for message in messages:
            if isinstance(message, APIMessage):
                normalized_messages.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )
            else:
                role = message.get("role")
                content = message.get("content")

                if not role or content is None:
                    raise ValueError("Each message requires role and content.")

                normalized_messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        payload = {
            "model": model,
            "messages": normalized_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        data = self._request(
            "POST",
            "/chat/completions",
            payload,
        )

        choices = data.get("choices")

        if not isinstance(choices, list) or not choices:
            raise infraAPIError("Server returned no completion choices.")

        choice = choices[0]

        if not isinstance(choice, dict):
            raise infraAPIError("Server returned an invalid completion.")

        message = choice.get("message")

        if not isinstance(message, dict):
            raise infraAPIError("Server returned an invalid message.")

        text = message.get("content", "")

        if not isinstance(text, str):
            text = str(text)

        usage = data.get("usage")

        if not isinstance(usage, dict):
            usage = {}

        return APIResponse(
            text=text,
            model=str(data.get("model", model)),
            request_id=(str(data["id"]) if data.get("id") is not None else None),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            total_tokens=_optional_int(usage.get("total_tokens")),
            finish_reason=(
                str(choice["finish_reason"]) if choice.get("finish_reason") is not None else None
            ),
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        url = f"{self.base_url}{path}"

        body = None

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        request = Request(
            url,
            data=body,
            method=method,
            headers={
                **self._headers(),
                "Content-Type": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()

        except HTTPError as error:
            detail = error.read().decode(
                "utf-8",
                errors="replace",
            )

            raise infraAPIError(f"HTTP {error.code}: {detail}") from error

        except (URLError, TimeoutError, OSError) as error:
            raise infraAPIError(f"Could not connect to infra API: {error}") from error

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise infraAPIError("Server returned invalid JSON.") from error

        if not isinstance(data, dict):
            raise infraAPIError("Server returned a non-object JSON response.")

        if "error" in data:
            raise infraAPIError(str(data["error"]))

        return data

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers


def _optional_int(value: object) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None
