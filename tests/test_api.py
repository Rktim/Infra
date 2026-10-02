import pytest

from infra.api import (
    APIMessage,
    infraAPIError,
    infraClient,
)


def test_models():
    client = infraClient()

    client._request = lambda method, path, payload=None: {
        "data": [
            {
                "id": "Qwen/Test-1.5B",
                "object": "model",
            }
        ]
    }

    result = client.models()

    assert len(result) == 1
    assert result[0]["id"] == "Qwen/Test-1.5B"


def test_models_rejects_invalid_response():
    client = infraClient()

    client._request = lambda method, path, payload=None: {"data": {}}

    with pytest.raises(
        infraAPIError,
        match="Invalid /models response",
    ):
        client.models()


def test_chat_with_api_message():
    client = infraClient()

    captured = {}

    def fake_request(method, path, payload=None):
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload

        return {
            "id": "chatcmpl-test",
            "model": "Qwen/Test-1.5B",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    client._request = fake_request

    result = client.chat(
        [
            APIMessage(
                role="user",
                content="Hello",
            )
        ],
        model="Qwen/Test-1.5B",
    )

    assert result.text == "Hello!"
    assert result.model == "Qwen/Test-1.5B"
    assert result.request_id == "chatcmpl-test"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15
    assert result.finish_reason == "stop"

    assert captured["method"] == "POST"
    assert captured["path"] == "/chat/completions"
    assert captured["payload"]["messages"] == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]


def test_chat_with_dictionary_message():
    client = infraClient()

    client._request = lambda method, path, payload=None: {
        "id": "test",
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "content": "response",
                },
                "finish_reason": "stop",
            }
        ],
    }

    result = client.chat(
        [
            {
                "role": "user",
                "content": "test",
            }
        ],
        model="test-model",
    )

    assert result.text == "response"


def test_chat_automatically_selects_model():
    client = infraClient()

    calls = []

    def fake_request(method, path, payload=None):
        calls.append((method, path, payload))

        if path == "/models":
            return {"data": [{"id": "auto-selected-model"}]}

        return {
            "id": "test",
            "model": "auto-selected-model",
            "choices": [
                {
                    "message": {"content": "OK"},
                    "finish_reason": "stop",
                }
            ],
        }

    client._request = fake_request

    result = client.chat(
        [
            APIMessage(
                role="user",
                content="test",
            )
        ]
    )

    assert result.model == "auto-selected-model"
    assert result.text == "OK"

    assert calls[0][1] == "/models"
    assert calls[1][1] == "/chat/completions"


def test_chat_rejects_empty_messages():
    client = infraClient()

    with pytest.raises(
        ValueError,
        match="messages",
    ):
        client.chat([])


def test_chat_rejects_invalid_max_tokens():
    client = infraClient()

    with pytest.raises(
        ValueError,
        match="max_tokens",
    ):
        client.chat(
            [
                APIMessage(
                    role="user",
                    content="test",
                )
            ],
            model="test-model",
            max_tokens=0,
        )


def test_chat_rejects_negative_temperature():
    client = infraClient()

    with pytest.raises(
        ValueError,
        match="temperature",
    ):
        client.chat(
            [
                APIMessage(
                    role="user",
                    content="test",
                )
            ],
            model="test-model",
            temperature=-1,
        )


def test_chat_rejects_invalid_top_p():
    client = infraClient()

    with pytest.raises(
        ValueError,
        match="top_p",
    ):
        client.chat(
            [
                APIMessage(
                    role="user",
                    content="test",
                )
            ],
            model="test-model",
            top_p=0,
        )

    with pytest.raises(
        ValueError,
        match="top_p",
    ):
        client.chat(
            [
                APIMessage(
                    role="user",
                    content="test",
                )
            ],
            model="test-model",
            top_p=1.1,
        )


def test_chat_rejects_invalid_message():
    client = infraClient()

    with pytest.raises(
        ValueError,
        match="role and content",
    ):
        client.chat(
            [
                {
                    "role": "user",
                }
            ],
            model="test-model",
        )


def test_chat_rejects_no_available_models():
    client = infraClient()

    client._request = lambda method, path, payload=None: {"data": []}

    with pytest.raises(
        infraAPIError,
        match="No model is available",
    ):
        client.chat(
            [
                APIMessage(
                    role="user",
                    content="test",
                )
            ]
        )


def test_chat_rejects_missing_choices():
    client = infraClient()

    client._request = lambda method, path, payload=None: {}

    with pytest.raises(
        infraAPIError,
        match="no completion choices",
    ):
        client.chat(
            [
                APIMessage(
                    role="user",
                    content="test",
                )
            ],
            model="test-model",
        )


def test_chat_handles_missing_usage():
    client = infraClient()

    client._request = lambda method, path, payload=None: {
        "model": "test",
        "choices": [
            {
                "message": {
                    "content": "hello",
                },
                "finish_reason": "stop",
            }
        ],
    }

    result = client.chat(
        [
            APIMessage(
                role="user",
                content="test",
            )
        ],
        model="test",
    )

    assert result.text == "hello"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens is None


def test_api_key_header():
    client = infraClient(api_key="secret")

    headers = client._headers()

    assert headers["Accept"] == "application/json"
    assert headers["Authorization"] == "Bearer secret"


def test_without_api_key():
    client = infraClient()

    headers = client._headers()

    assert "Authorization" not in headers
