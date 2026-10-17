from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import (
    Container,
    Grid,
    Horizontal,
    Vertical,
    VerticalScroll,
)
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    LoadingIndicator,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from infra.api import (
    APIMessage,
    infraClient,
)
from infra.downloader import download_model
from infra.hardware import detect_hardware
from infra.recommender import (
    RecommendationRequest,
    recommend_models,
)
from infra.server import (
    ServerConfig,
    ServerState,
    VLLMServer,
)
from infra.suggestor import suggest_use_case

DEFAULT_USE_CASE = "coding"

DEFAULT_CONTEXT_LENGTH = 1024

DEFAULT_MAX_MODELS = 5

DEFAULT_GPU_UTILIZATION = 0.85

DEFAULT_API_CLIENT = "python"

DEFAULT_API_URL = "http://127.0.0.1:8000/v1"

DEFAULT_MODEL_PLACEHOLDER = "your-model-id"


USE_CASES = [
    ("Coding", "coding"),
    ("Chat", "chat"),
    ("Reasoning", "reasoning"),
    ("RAG", "rag"),
    ("Summarization", "summarization"),
    ("General", "general"),
]


API_CLIENTS = [
    ("OpenAI Python (Standard)", "python"),
    ("OpenAI Python (Streaming)", "python-stream"),
    ("OpenAI cURL (HTTP API)", "curl"),
]


API_SNIPPETS: dict[str, tuple[str, str]] = {
    "python": (
        ".py",
        r"""# pip install openai
from openai import OpenAI

# Connect to local vLLM OpenAI-compatible server (no API key required)
client = OpenAI(
    base_url="__BASE_URL__",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="__MODEL__",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=512,
)

print(response.choices[0].message.content)
""",
    ),
    "python-stream": (
        ".py",
        r"""# pip install openai
from openai import OpenAI

# Connect to local vLLM OpenAI-compatible server with streaming
client = OpenAI(
    base_url="__BASE_URL__",
    api_key="EMPTY",
)

stream = client.chat.completions.create(
    model="__MODEL__",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=512,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
""",
    ),
    "curl": (
        ".sh",
        r"""# Test local OpenAI Chat Completions endpoint
curl __BASE_URL__/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "__MODEL__",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
""",
    ),
}


_DOWNLOADED_CACHE: list[tuple[str, Path]] | None = None
_DOWNLOADED_CACHE_TIME: float = 0.0


def clear_downloaded_models_cache() -> None:
    global _DOWNLOADED_CACHE
    _DOWNLOADED_CACHE = None


def get_downloaded_models(force: bool = False) -> list[tuple[str, Path]]:
    """
    Discover locally downloaded Hugging Face models with fast in-memory caching.
    """
    global _DOWNLOADED_CACHE, _DOWNLOADED_CACHE_TIME
    now = time.time()
    if not force and _DOWNLOADED_CACHE is not None and (now - _DOWNLOADED_CACHE_TIME) < 15.0:
        return _DOWNLOADED_CACHE

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.is_dir():
        _DOWNLOADED_CACHE = []
        _DOWNLOADED_CACHE_TIME = now
        return []

    models: list[tuple[str, Path]] = []

    try:
        for repo_dir in sorted(cache_dir.glob("models--*")):
            if not repo_dir.is_dir():
                continue

            snapshots_dir = repo_dir / "snapshots"
            if not snapshots_dir.is_dir():
                continue

            snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]

            if not snapshots:
                continue

            snapshot_path = max(
                snapshots,
                key=lambda path: path.stat().st_mtime,
            )

            directory_name = repo_dir.name.removeprefix("models--")
            parts = directory_name.split("--")
            if len(parts) < 2:
                continue

            model_id = parts[0] + "/" + "--".join(parts[1:])
            models.append((model_id, snapshot_path))
    except Exception:
        pass

    models.sort(key=lambda item: item[0].lower())
    _DOWNLOADED_CACHE = models
    _DOWNLOADED_CACHE_TIME = now
    return models


def get_downloaded_model_path(
    model_id: str,
) -> Path | None:
    """
    Return the latest local Hugging Face snapshot for a specific model.
    Uses cached models for instant 0ms lookup.
    """
    for mid, path in get_downloaded_models():
        if mid == model_id:
            return path

    # Fallback to direct path query if not in cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_dir / (
        "models--"
        + model_id.replace(
            "/",
            "--",
        )
    )
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    try:
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            return max(
                snapshots,
                key=lambda path: path.stat().st_mtime,
            )
    except Exception:
        pass

    return None


class ConfirmScreen(ModalScreen[bool]):
    """
    Small confirmation dialog.
    """

    CSS = """
    ConfirmScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.75);
    }

    #dialog {
        width: 62;
        height: auto;
        padding: 1 2;
        border: round #38bdf8;
        background: #161f2e;
    }

    #dialog-title {
        height: 1;
        color: #38bdf8;
        text-style: bold;
        margin-bottom: 1;
    }

    #dialog-message {
        height: auto;
        max-height: 8;
        color: #e2e8f0;
        margin-bottom: 1;
    }

    #dialog-actions {
        height: 3;
        align: right middle;
    }

    #dialog-actions Button {
        width: 16;
        height: 3;
        margin-left: 1;
    }
    """

    def __init__(
        self,
        title: str,
        message: str,
    ) -> None:

        super().__init__()

        self.dialog_title = title

        self.dialog_message = message

    def compose(
        self,
    ) -> ComposeResult:

        with Vertical(
            id="dialog",
        ):
            yield Static(
                self.dialog_title,
                id="dialog-title",
            )

            yield Static(
                self.dialog_message,
                id="dialog-message",
            )

            with Horizontal(
                id="dialog-actions",
            ):
                yield Button(
                    "Cancel",
                    id="cancel-button",
                )

                yield Button(
                    "Confirm",
                    id="confirm-button",
                    variant="primary",
                )

    def on_button_pressed(
        self,
        event: Button.Pressed,
    ) -> None:

        if event.button.id == "confirm-button":
            self.dismiss(True)

        else:
            self.dismiss(False)


GEAR_LEVER_PULL_FRAMES = [
    # Frame 0: Forward / Rest
    r"""[bold #fbbf24]                  .-----.
                 /  / \  \
                |  <   >  |
                 \  \ /  /
                  '-----'[/]
[#94a3b8]                    ||
                    ||
                   //
                  //[/]
[#64748b]             .--------.
            /    / / /|
           /    /_/ / |
          +--------+  /
          |        | /
          +--------+[/]""",
    # Frame 1: Pulling (Midway)
    r"""
[bold #fbbf24]              .-----.
             /  / \  \
            |  <   >  |
             \  \ /  /
              '-----'[/]
[#94a3b8]               //
              //
             //
            //[/]
[#64748b]             .--------.
            /   / /  /|
           /   /_/  / |
          +--------+  /
          |        | /
          +--------+[/]""",
    # Frame 2: Pulled Back
    r"""

[bold #fbbf24]          .-----.
         /  / \  \
        |  <   >  |
         \  \ /  /
          '-----'[/]
[#94a3b8]           \\
            \\
             \\
              \\ [/]
[#64748b]             .--------.
            / / /    /|
           / /_/    / |
          +--------+  /
          |        | /
          +--------+[/]""",
    # Frame 3: Locked / Engaged
    r"""

[bold #fbbf24]          .-----.
         /  / \  \
        |  <   >  |[/]   [bold #10b981]* ENGAGED *[/]
[bold #fbbf24]         \  \ /  /
          '-----'[/]
[#94a3b8]           \\
            \\
             \\
              \\ [/]
[#64748b]             .--------.
            / / /    /|
           / /_/    / |
          +--------+  /
          |        | /
          +--------+[/]""",
]


class GearShiftSplashScreen(ModalScreen[None]):
    """
    Startup splash screen featuring the isometric gear lever pulling animation.
    Auto-dismisses once engaged, or immediately on any key or click.
    """

    CSS = """
    GearShiftSplashScreen {
        align: center middle;
        background: rgba(11, 15, 23, 0.88);
    }

    #splash-dialog {
        width: 50;
        height: auto;
        padding: 1 2;
        border: round #fbbf24;
        background: #0f172a;
    }

    #splash-title {
        text-align: center;
        color: #fbbf24;
        text-style: bold;
    }

    #splash-subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 1;
    }

    #splash-frame {
        height: 15;
        color: #e2e8f0;
    }

    #splash-footer {
        text-align: center;
        color: #64748b;
        margin-top: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.frame_idx = 0
        self._timer: Any | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="splash-dialog"):
            yield Static("INFRA", id="splash-title")
            yield Static("Hardware-Aware Local Runtime", id="splash-subtitle")
            yield Static(GEAR_LEVER_PULL_FRAMES[0], id="splash-frame")
            yield Static("[ Press any key or click to skip ]", id="splash-footer")

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.18, self._next_frame)

    def _next_frame(self) -> None:
        self.frame_idx += 1
        if self.frame_idx < len(GEAR_LEVER_PULL_FRAMES):
            try:
                self.query_one("#splash-frame", Static).update(
                    GEAR_LEVER_PULL_FRAMES[self.frame_idx]
                )
            except Exception:
                pass
        else:
            if self._timer:
                self._timer.stop()
                self._timer = None
            self.set_timer(0.2, self._dismiss_splash)

    def _dismiss_splash(self) -> None:
        if self.is_mounted:
            self.dismiss()

    def on_key(self, event: Any) -> None:
        event.prevent_default()
        if self._timer:
            self._timer.stop()
            self._timer = None
        self.dismiss()

    def on_click(self, event: Any) -> None:
        if self._timer:
            self._timer.stop()
            self._timer = None
        self.dismiss()


class InfraApp(App):
    """
    infra Terminal User Interface.
    """

    TITLE = "infra"

    SUB_TITLE = "Hardware-Aware Local Model Infrastructure"

    CSS = """

    $ink: #0b0f17;
    $panel: #131d2a;
    $field: #0d1520;
    $rule: #22344a;
    $rule-soft: #172433;
    $body: #e2e8f0;
    $muted: #8899ac;
    $cyan: #38bdf8;
    $jade: #10b981;
    $amber: #f59e0b;
    $rust: #f43f5e;

    Screen {
        background: $ink;
        color: $body;
    }

    Header {
        background: #131d2a;
        color: $cyan;
        text-style: bold;
    }

    Footer {
        background: #131d2a;
        color: $muted;
    }

    FooterKey {
        background: #131d2a;
        color: $muted;
    }

    /* Tabs */
    TabbedContent {
        height: 1fr;
    }

    ContentSwitcher {
        height: 1fr;
    }

    Tabs {
        height: 2;
        background: #131d2a;
    }

    Tab {
        padding: 0 3;
        color: $muted;
    }

    Tab.-active {
        color: $cyan;
        text-style: bold;
    }

    Underline > .underline--bar {
        color: $cyan;
        background: $rule-soft;
    }

    TabPane {
        height: 1fr;
        padding: 0;
        background: $ink;
    }

    .tab-body {
        height: 1fr;
        padding: 0 2 1 2;
        background: $ink;
        overflow-y: auto;
    }

    /* Common */
    .section-title {
        height: 1;
        padding: 0 1;
        margin: 1 0 0 0;
        color: $cyan;
        text-style: bold;
    }

    Static {
        color: $body;
    }

    Input {
        background: $field;
        border: round $rule;
        color: $body;
    }

    Input:focus {
        border: round $cyan;
    }

    Select {
        background: $field;
        border: round $rule;
        color: $body;
    }

    Select:focus {
        border: round $cyan;
    }

    Button {
        min-width: 16;
        height: 3;
        margin-right: 1;
        background: #1a293b;
        color: #e2e8f0;
    }

    Button:hover {
        background: #273e59;
    }

    Button.-primary {
        background: #0284c7;
        color: #ffffff;
        text-style: bold;
    }

    Button.-primary:hover {
        background: #0ea5e9;
    }

    Button.-success {
        background: #059669;
        color: #ffffff;
        text-style: bold;
    }

    Button.-success:hover {
        background: #10b981;
    }

    Button.-error {
        background: #e11d48;
        color: #ffffff;
        text-style: bold;
    }

    Button.-error:hover {
        background: #f43f5e;
    }

    .panel {
        height: auto;
        padding: 1 2;
        border: round $rule;
        background: $panel;
    }

    .table-frame {
        height: 1fr;
        min-height: 7;
        border: round $rule;
        background: $panel;
    }

    DataTable {
        height: 1fr;
        border: none;
        background: $panel;
    }

    DataTable > .datatable--header {
        background: #1a293b;
        color: $cyan;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: #253952;
        color: #ffffff;
        text-style: bold;
    }

    DataTable > .datatable--odd-row {
        background: $panel;
    }

    DataTable > .datatable--even-row {
        background: #0e1622;
    }

    #hardware-info {
        min-height: 4;
    }

    #model-search-controls {
        width: 100%;
        height: auto;
        margin-top: 1;
    }

    #model-search-query {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }

    #model-search-grid {
        width: 100%;
        height: 3;
        grid-size: 3;
        grid-columns: 2fr 1fr 1fr;
        grid-gutter: 1;
    }

    #use-case {
        width: 100%;
        height: 3;
    }

    #context-length {
        width: 100%;
        height: 3;
    }

    #recommend-button {
        width: 100%;
        height: 3;
        margin: 0;
    }

    #recommendations-frame {
        height: 1fr;
        min-height: 8;
        margin-top: 1;
    }

    #models-status {
        height: auto;
        min-height: 2;
        padding: 0 1;
        color: $muted;
    }

    #selected-model {
        min-height: 5;
        margin-top: 1;
    }

    #model-actions {
        height: auto;
        margin-top: 1;
    }

    #recommend-loading {
        display: none;
        height: 1;
        color: $cyan;
        margin-left: 1;
    }

    #recommend-loading.visible {
        display: block;
    }

    #downloaded-models-frame {
        height: 12;
        min-height: 7;
        margin-top: 1;
    }

    #server-model-info {
        min-height: 5;
        margin-top: 1;
    }

    #server-config {
        width: 100%;
        height: 3;
        margin-top: 1;
        grid-size: 2;
        grid-columns: 18 18;
        grid-gutter: 1;
    }

    #server-context-length {
        width: 100%;
        height: 3;
    }

    #server-gpu-utilization {
        width: 100%;
        height: 3;
    }

    #server-status {
        min-height: 4;
        margin-top: 1;
    }

    #server-actions {
        width: 100%;
        height: 3;
        margin-top: 1;
        margin-bottom: 1;
    }

    #server-actions Button {
        width: 18;
        margin-right: 1;
    }

    #server-message {
        min-height: 5;
        color: $body;
    }

    #server-loading {
        display: none;
        height: 1;
        color: $cyan;
        margin-left: 1;
    }

    #server-loading.visible {
        display: block;
    }

    #api-server-info {
        min-height: 5;
    }

    #api-controls {
        width: 100%;
        height: 3;
        margin-top: 1;
        grid-size: 2;
        grid-columns: 36 18;
        grid-gutter: 1;
    }

    #api-client {
        width: 100%;
        height: 3;
    }

    #api-refresh-button {
        width: 18;
        height: 3;
        margin: 0;
    }

    #api-actions {
        width: 100%;
        height: 3;
        margin-top: 1;
    }

    #api-actions Button {
        width: 18;
        margin-right: 1;
    }

    #api-code-container {
        height: 1fr;
        min-height: 8;
        border: round $rule;
        background: $panel;
        padding: 1;
        margin-top: 1;
    }

    #api-code {
        height: auto;
        color: $body;
    }

    #api-status {
        height: auto;
        color: $muted;
        padding: 0 1;
    }

    """

    BINDINGS = [
        (
            "1",
            "show_models_tab",
            "Models [1]",
        ),
        (
            "f1",
            "show_models_tab",
            "Models [F1]",
        ),
        (
            "2",
            "show_server_tab",
            "Server [2]",
        ),
        (
            "f2",
            "show_server_tab",
            "Server [F2]",
        ),
        (
            "3",
            "show_api_tab",
            "API [3]",
        ),
        (
            "f3",
            "show_api_tab",
            "API [F3]",
        ),
        (
            "q",
            "quit",
            "Quit",
        ),
        (
            "r",
            "recommend",
            "Find Models",
        ),
        (
            "d",
            "download",
            "Download",
        ),
        (
            "s",
            "serve",
            "Serve",
        ),
        (
            "c",
            "copy_snippet",
            "Copy Code",
        ),
        (
            "a",
            "api_test",
            "Verify API",
        ),
    ]

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.hardware: Any | None = None

        self.recommendations: list[Any] = []

        self.selected_model: Any | None = None

        self.downloaded_model_path: Path | None = None

        self.downloaded_models: list[tuple[str, Path]] = []

        self.selected_downloaded_model_id: str | None = None

        self.selected_downloaded_model_path: Path | None = None

        self.server = VLLMServer()

        self.api_base_url = DEFAULT_API_URL

        self.api_model_id: str | None = None

        self.api_snippet = ""

    def compose(
        self,
    ) -> ComposeResult:

        yield Header(
            show_clock=True,
        )

        with TabbedContent(
            id="main-tabs",
        ):
            with TabPane(
                "Models",
                id="models-tab",
            ):
                with Vertical(
                    classes="tab-body",
                ):
                    yield Static(
                        "HARDWARE",
                        classes="section-title",
                    )

                    yield Static(
                        "Detecting hardware...",
                        id="hardware-info",
                        classes="panel",
                    )

                    yield Static(
                        "FIND MODELS",
                        classes="section-title",
                    )

                    with Vertical(
                        id="model-search-controls",
                    ):
                        yield Input(
                            placeholder="Describe your workload or query (e.g. 'Python async API', 'summarize PDFs')...",
                            id="model-search-query",
                        )

                        with Grid(
                            id="model-search-grid",
                        ):
                            yield Select(
                                USE_CASES,
                                value=DEFAULT_USE_CASE,
                                id="use-case",
                            )

                            yield Input(
                                value=str(DEFAULT_CONTEXT_LENGTH),
                                placeholder=("Context length"),
                                id="context-length",
                            )

                            yield Button(
                                "Find Models",
                                id="recommend-button",
                                variant="primary",
                            )

                        with Horizontal(
                            id="recommend-loading",
                        ):
                            yield LoadingIndicator()

                            yield Static("Searching...")

                    yield Static(
                        "RECOMMENDED MODELS",
                        classes="section-title",
                    )

                    with Container(
                        id="recommendations-frame",
                        classes="table-frame",
                    ):
                        yield DataTable(
                            id=("recommendations-table"),
                        )

                    yield Static(
                        "",
                        id="models-status",
                    )

                    yield Static(
                        "No model selected.\nSelect a model from the recommendation table.",
                        id="selected-model",
                        classes="panel",
                    )

                    with Horizontal(
                        id="model-actions",
                    ):
                        yield Button(
                            "Download",
                            id="download-button",
                            variant="success",
                            disabled=True,
                        )

                        yield Button(
                            "Serve Selected Model",
                            id="serve-button",
                            variant="primary",
                            disabled=True,
                        )

            with (
                TabPane(
                    "Server",
                    id="server-tab",
                ),
                Vertical(
                    classes="tab-body",
                ),
            ):
                yield Static(
                    "DOWNLOADED MODELS",
                    classes="section-title",
                )

                with Container(
                    id=("downloaded-models-frame"),
                    classes="table-frame",
                ):
                    yield DataTable(
                        id=("downloaded-models-table"),
                    )

                yield Static(
                    "No downloaded model selected.\nSelect a model from the table.",
                    id="server-model-info",
                    classes="panel",
                )

                yield Static(
                    "SERVER CONFIGURATION",
                    classes="section-title",
                )

                with Grid(
                    id="server-config",
                ):
                    yield Input(
                        value=str(DEFAULT_CONTEXT_LENGTH),
                        placeholder=("Context length"),
                        id=("server-context-length"),
                    )

                    yield Input(
                        value=str(DEFAULT_GPU_UTILIZATION),
                        placeholder=("GPU utilization"),
                        id=("server-gpu-utilization"),
                    )

                yield Static(
                    "SERVER STATUS",
                    classes="section-title",
                )

                yield Static(
                    "Checking server status...",
                    id="server-status",
                    classes="panel",
                )

                with Horizontal(
                    id="server-actions",
                ):
                    yield Button(
                        "Start Server",
                        id=("start-server-button"),
                        variant="success",
                    )

                    yield Button(
                        "Stop Server",
                        id=("stop-server-button"),
                        variant="error",
                    )

                    yield Button(
                        "Refresh",
                        id=("refresh-server-button"),
                    )

                with Horizontal(
                    id="server-loading",
                ):
                    yield LoadingIndicator()

                    yield Static("Working...")

                yield Static(
                    "ACTIVITY",
                    classes="section-title",
                )

                yield Static(
                    "Server idle.",
                    id="server-message",
                    classes="panel",
                )

            with (
                TabPane(
                    "API",
                    id="api-tab",
                ),
                Vertical(
                    classes="tab-body",
                ),
            ):
                yield Static(
                    "OPENAI-COMPATIBLE LOCAL API",
                    classes="section-title",
                )

                yield Static(
                    "Checking local server...",
                    id="api-server-info",
                    classes="panel",
                )

                yield Static(
                    "OPENAI CLIENT FORMAT",
                    classes="section-title",
                )

                with Grid(
                    id="api-controls",
                ):
                    yield Select(
                        API_CLIENTS,
                        value=DEFAULT_API_CLIENT,
                        id="api-client",
                    )

                    yield Button(
                        "Refresh",
                        id="api-refresh-button",
                        variant="primary",
                    )

                with Horizontal(
                    id="api-actions",
                ):
                    yield Button(
                        "Copy Code",
                        id="copy-api-button",
                    )

                    yield Button(
                        "Save File",
                        id="save-api-button",
                    )

                    yield Button(
                        "Test API",
                        id="verify-api-button",
                        variant="success",
                    )

                yield Static(
                    "",
                    id="api-status",
                )

                yield Static(
                    "READY-TO-USE CODE (NO API KEY REQUIRED)",
                    classes="section-title",
                )

                with VerticalScroll(
                    id="api-code-container",
                ):
                    yield Static(
                        "Start a server to generate client code.",
                        id="api-code",
                    )

        yield Footer()

    def on_mount(
        self,
    ) -> None:

        self.setup_tables()

        self.detect_local_hardware()

        self.refresh_downloaded_models()

        self.refresh_server_status()

        self.update_api_info()

        self.push_screen(GearShiftSplashScreen())

    def setup_tables(
        self,
    ) -> None:

        recommendations = self.query_one(
            "#recommendations-table",
            DataTable,
        )

        recommendations.add_columns(
            "Model",
            "Parameters",
            "Runtime",
            "Status",
            "Score",
        )

        recommendations.cursor_type = "row"

        downloaded = self.query_one(
            "#downloaded-models-table",
            DataTable,
        )

        downloaded.add_columns(
            "Model",
            "Status",
        )

        downloaded.cursor_type = "row"

    @work(
        thread=True,
    )
    def detect_local_hardware(
        self,
    ) -> None:

        try:
            hardware = detect_hardware()

            self.call_from_thread(
                self.hardware_ready,
                hardware,
            )

        except Exception as error:
            self.call_from_thread(
                self.query_one(
                    "#hardware-info",
                    Static,
                ).update,
                f"Hardware detection failed:\n{error}",
            )

    def hardware_ready(
        self,
        hardware: Any,
    ) -> None:

        self.hardware = hardware

        lines: list[str] = []

        gpus = getattr(
            hardware,
            "gpus",
            [],
        )

        if gpus:
            gpu = gpus[0]
            gpu_name = getattr(gpu, "name", "Unknown GPU")
            vram = getattr(gpu, "total_vram_bytes", None)
            vram_text = f"{vram / (1024**3):.2f} GiB" if vram is not None else "Unknown"

            lines.append(
                f"[bold #38bdf8][GPU][/bold #38bdf8] [bold white]{gpu_name}[/bold white]   "
                f"[dim]│[/dim]   [bold #38bdf8]VRAM:[/bold #38bdf8] [bold #10b981]{vram_text}[/bold #10b981]"
            )
        else:
            lines.append(
                "[bold #f59e0b][GPU][/bold #f59e0b] [dim]No dedicated GPU detected (CPU mode)[/dim]"
            )

        cpu = getattr(hardware, "cpu", None)
        cpu_name = getattr(cpu, "name", "CPU") if cpu else "CPU"

        memory = getattr(hardware, "memory", None)
        if memory is not None:
            total = getattr(memory, "total_bytes", None)
            avail = getattr(memory, "available_bytes", None)
            if total is not None:
                ram_str = f"{total / (1024**3):.1f} GiB"
                avail_str = f" (avail: {avail / (1024**3):.1f} GiB)" if avail is not None else ""
                lines.append(
                    f"[bold #38bdf8][RAM][/bold #38bdf8] [white]{ram_str}{avail_str}[/white]   "
                    f"[dim]│[/dim]   [bold #38bdf8][CPU][/bold #38bdf8] [white]{cpu_name}[/white]"
                )

        self.query_one(
            "#hardware-info",
            Static,
        ).update("\n".join(lines))

    @work(thread=True)
    def refresh_downloaded_models(
        self,
    ) -> None:
        """
        Scan the Hugging Face cache in background and populate the Server tab.
        """
        models = get_downloaded_models(force=True)
        self.call_from_thread(self._populate_downloaded_models, models)

    def _populate_downloaded_models(
        self,
        models: list[tuple[str, Path]],
    ) -> None:
        self.downloaded_models = models
        table = self.query_one(
            "#downloaded-models-table",
            DataTable,
        )
        table.clear()

        if not self.downloaded_models:
            self.selected_downloaded_model_id = None
            self.selected_downloaded_model_path = None
            self.query_one(
                "#server-model-info",
                Static,
            ).update(
                "[dim]No downloaded Hugging Face models found.\n\n"
                "Download a model from the Models tab first.[/dim]"
            )
            return

        selected_index: int | None = None

        for index, (model_id, _model_path) in enumerate(self.downloaded_models):
            status = "[bold #10b981]READY[/bold #10b981]"
            table.add_row(
                model_id,
                status,
                key=str(index),
            )

            if self.selected_downloaded_model_id == model_id:
                selected_index = index

        if selected_index is not None:
            table.move_cursor(row=selected_index)
        else:
            self.query_one(
                "#server-model-info",
                Static,
            ).update(
                f"[bold #38bdf8]Found {len(self.downloaded_models)} downloaded model(s).[/bold #38bdf8]\n\n"
                "[dim]Select one from the table above to configure and serve.[/dim]"
            )

    def select_downloaded_model(
        self,
        row_index: int,
    ) -> None:

        if row_index < 0 or row_index >= len(self.downloaded_models):
            return

        model_id, model_path = self.downloaded_models[row_index]

        self.selected_downloaded_model_id = model_id

        self.selected_downloaded_model_path = model_path

        self.downloaded_model_path = model_path

        self.query_one(
            "#server-model-info",
            Static,
        ).update(f"SELECTED DOWNLOADED MODEL\n{model_id}\n\nSnapshot:\n{model_path}")

        self.set_server_message(f"Selected downloaded model:\n{model_id}")

    def action_show_models_tab(self) -> None:
        try:
            self.query_one("#main-tabs", TabbedContent).active = "models-tab"
        except Exception:
            pass

    def action_show_server_tab(self) -> None:
        try:
            self.query_one("#main-tabs", TabbedContent).active = "server-tab"
        except Exception:
            pass

    def action_show_api_tab(self) -> None:
        try:
            self.query_one("#main-tabs", TabbedContent).active = "api-tab"
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in {"model-search-query", "context-length"}:
            self.start_recommendation()
        elif event.input.id in {"server-context-length", "server-gpu-utilization"}:
            self.start_server()

    def action_recommend(
        self,
    ) -> None:

        self.start_recommendation()

    def start_recommendation(
        self,
    ) -> None:

        if self.hardware is None:
            self.set_models_status("Hardware is not ready yet.")
            return

        try:
            query = ""
            try:
                query = self.query_one(
                    "#model-search-query",
                    Input,
                ).value.strip()
            except Exception:
                pass

            use_case_value = self.query_one(
                "#use-case",
                Select,
            ).value

            if not isinstance(
                use_case_value,
                str,
            ):
                use_case_value = DEFAULT_USE_CASE

            context_value = self.query_one(
                "#context-length",
                Input,
            ).value

            context_length = int(context_value)

            if context_length <= 0:
                raise ValueError("Context length must be greater than zero.")

        except Exception as error:
            self.set_models_status(f"Invalid input: {error}")
            return

        self.set_recommend_loading(True)

        status_msg = (
            f"Analyzing query '{query}'..."
            if query
            else f"Searching Hugging Face ({use_case_value.upper()})..."
        )
        self.set_models_status(status_msg)

        self.run_recommendation(
            query,
            use_case_value,
            context_length,
        )

    def _sync_use_case_selection(
        self,
        task_val: str,
    ) -> None:
        try:
            self.query_one(
                "#use-case",
                Select,
            ).value = task_val
        except Exception:
            pass

    @work(
        thread=True,
    )
    def run_recommendation(
        self,
        query: str,
        task: str,
        context_length: int,
    ) -> None:

        try:
            detected_task = task
            if query:
                suggestion = suggest_use_case(query)
                if suggestion and suggestion.task:
                    detected_task = suggestion.task.value
                    self.call_from_thread(
                        self._sync_use_case_selection,
                        detected_task,
                    )
                    self.call_from_thread(
                        self.set_models_status,
                        f"Workload: {detected_task.upper()} ({suggestion.reason}). Searching models...",
                    )

            request = RecommendationRequest(
                task=detected_task,
                context_length=context_length,
                max_models=DEFAULT_MAX_MODELS,
                gpu_memory_utilization=DEFAULT_GPU_UTILIZATION,
            )

            results = recommend_models(
                self.hardware,
                request,
            )

            self.call_from_thread(
                self.update_recommendations,
                results,
            )

        except Exception as error:
            self.call_from_thread(
                self.set_models_status,
                f"Recommendation failed:\n{error}",
            )

        finally:
            self.call_from_thread(
                self.set_recommend_loading,
                False,
            )

    def update_recommendations(
        self,
        results: list[Any],
    ) -> None:

        self.recommendations = results

        self.selected_model = None

        table = self.query_one(
            "#recommendations-table",
            DataTable,
        )

        table.clear()

        self.query_one(
            "#download-button",
            Button,
        ).disabled = True

        self.query_one(
            "#serve-button",
            Button,
        ).disabled = True

        self.query_one(
            "#selected-model",
            Static,
        ).update("No model selected.\nSelect a model from the recommendation table.")

        if not results:
            self.set_models_status("No compatible models found.")

            return

        for result in results:
            table.add_row(
                self._get_model_id(result),
                self._get_parameters(result),
                self._get_runtime(result),
                self._get_status(result),
                self._get_score(result),
            )

        self.set_models_status(
            f"Found {len(results)} compatible model(s). Select one from the table."
        )

        if results:
            table.move_cursor(row=0)
            self._select_recommendation_row(0)

    def set_recommend_loading(
        self,
        loading: bool,
    ) -> None:

        widget = self.query_one(
            "#recommend-loading",
            Horizontal,
        )

        if loading:
            widget.add_class("visible")

        else:
            widget.remove_class("visible")

        self.query_one(
            "#recommend-button",
            Button,
        ).disabled = loading

    def _select_recommendation_row(
        self,
        row_index: int,
    ) -> None:
        if row_index < 0 or row_index >= len(self.recommendations):
            return

        self.selected_model = self.recommendations[row_index]
        model_id = self._get_model_id(self.selected_model)
        local_path = get_downloaded_model_path(model_id)
        self.downloaded_model_path = local_path

        dl_status = "[bold #10b981][CACHED][/bold #10b981]" if local_path else "[dim][REMOTE][/dim]"

        opt = getattr(self.selected_model, "optimization_params", None)
        opt_text = ""
        if opt:
            src = "Auto-Tuned"
            exec_str = "Eager (Safe)" if opt.enforce_eager else "CUDA Graphs (Fast)"
            opt_text = (
                f"\n  [bold #38bdf8][*] Optimal Settings ({src}):[/bold #38bdf8] "
                f"Context: [bold]{opt.max_model_len:,}[/bold] tokens [dim]│[/dim] "
                f"Util: [bold]{opt.gpu_memory_utilization:.2f}[/bold] [dim]│[/dim] "
                f"Mode: [bold]{exec_str}[/bold]"
            )
            if opt.tuning_rationale:
                opt_text += f"\n  [dim italic]-> {opt.tuning_rationale}[/dim italic]"

            try:
                self.query_one("#server-context-length", Input).value = str(opt.max_model_len)
                self.query_one(
                    "#server-gpu-utilization", Input
                ).value = f"{opt.gpu_memory_utilization:.2f}"
            except Exception:
                pass

        self.query_one(
            "#selected-model",
            Static,
        ).update(
            f"[bold #38bdf8]:: {model_id}[/bold #38bdf8]   {dl_status}\n\n"
            f"  [bold]Parameters:[/bold] {self._get_parameters(self.selected_model)}   "
            f"[dim]│[/dim]   [bold]Runtime Memory:[/bold] {self._get_runtime(self.selected_model)}   "
            f"[dim]│[/dim]   [bold]Status:[/bold] {self._get_status(self.selected_model)}"
            f"{opt_text}"
        )

        self.query_one(
            "#download-button",
            Button,
        ).disabled = False

        self.query_one(
            "#serve-button",
            Button,
        ).disabled = False

        self.set_models_status(f"Selected: {model_id}")

    def on_data_table_row_selected(
        self,
        event: DataTable.RowSelected,
    ) -> None:

        table_id = event.data_table.id

        if table_id == "recommendations-table":
            self._select_recommendation_row(event.data_table.cursor_row)

        elif table_id == "downloaded-models-table":
            self.select_downloaded_model(event.data_table.cursor_row)

    def on_data_table_row_highlighted(
        self,
        event: DataTable.RowHighlighted,
    ) -> None:

        if event.cursor_row < 0:
            return

        table_id = event.data_table.id

        if table_id == "recommendations-table":
            self._select_recommendation_row(event.cursor_row)

        elif table_id == "downloaded-models-table":
            self.select_downloaded_model(event.cursor_row)

    def action_download(
        self,
    ) -> None:

        self.start_download()

    def start_download(
        self,
    ) -> None:

        if self.selected_model is None:
            self.set_models_status("Select a model first.")

            return

        model_id = self._get_model_id(self.selected_model)

        self.push_screen(
            ConfirmScreen(
                "Download Model",
                f"Download this model?\n\n{model_id}",
            ),
            callback=(self.confirm_download),
        )

    def confirm_download(
        self,
        confirmed: bool,
    ) -> None:

        if confirmed:
            self.download_selected_model()

    @work(
        thread=True,
    )
    def download_selected_model(
        self,
    ) -> None:

        try:
            if self.selected_model is None:
                raise RuntimeError("No model selected.")

            model_id = self._get_model_id(self.selected_model)

            self.call_from_thread(
                self.set_models_status,
                f"Downloading:\n{model_id}",
            )

            path = download_model(model_id)

            self.call_from_thread(
                self.download_completed,
                model_id,
                path,
            )

        except Exception as error:
            self.call_from_thread(
                self.set_models_status,
                f"Download failed:\n{error}",
            )

    def download_completed(
        self,
        model_id: str,
        path: Path,
    ) -> None:

        self.downloaded_model_path = path

        self.set_models_status(f"Download complete.\n{model_id}")

        # Make the newly downloaded model the active Server-tab
        # selection before refreshing the table.
        self.selected_downloaded_model_id = model_id
        self.selected_downloaded_model_path = path

        self.refresh_downloaded_models()

        for index, (downloaded_id, _downloaded_path) in enumerate(self.downloaded_models):
            if downloaded_id == model_id:
                self.select_downloaded_model(index)
                break

        self.update_api_info()

    def action_serve(
        self,
    ) -> None:

        self.start_server()

    def start_server(
        self,
    ) -> None:

        if (
            self.selected_downloaded_model_id is not None
            and self.selected_downloaded_model_path is not None
        ):
            model_id = self.selected_downloaded_model_id

        elif self.selected_model is not None:
            model_id = self._get_model_id(self.selected_model)

            model_path = get_downloaded_model_path(model_id)

            if model_path is None:
                self.set_server_message("Selected model is not downloaded yet.")

                return

            self.selected_downloaded_model_id = model_id

            self.selected_downloaded_model_path = model_path

        else:
            self.set_server_message("Select a downloaded model first.")

            return

        self.push_screen(
            ConfirmScreen(
                "Start vLLM Server",
                f"Start the local OpenAI-compatible server?\n\nModel: {model_id}",
            ),
            callback=(self.confirm_start_server),
        )

    def confirm_start_server(
        self,
        confirmed: bool,
    ) -> None:

        if confirmed:
            self.start_selected_server()

    @work(
        thread=True,
    )
    def start_selected_server(
        self,
    ) -> None:

        try:
            if self.selected_downloaded_model_id is None:
                raise RuntimeError("No downloaded model selected.")

            if self.selected_downloaded_model_path is None:
                raise RuntimeError("No local model path selected.")

            model_id = self.selected_downloaded_model_id

            model_path = self.selected_downloaded_model_path

            context_length = self._get_server_context_length()

            gpu_utilization = self._get_gpu_utilization()

            self.call_from_thread(
                self.set_server_loading,
                True,
            )

            self.call_from_thread(
                self.set_server_message,
                "Starting vLLM server...\n"
                f"Model: {model_id}\n"
                f"Context: {context_length}\n"
                f"GPU utilization: "
                f"{gpu_utilization}",
            )

            enforce_eager = True
            max_num_seqs = 4
            if hasattr(self, "selected_model") and self.selected_model:
                opt = getattr(self.selected_model, "optimization_params", None)
                if opt:
                    enforce_eager = opt.enforce_eager
                    max_num_seqs = opt.max_num_seqs
            elif self.hardware and self.hardware.gpus:
                vram = self.hardware.gpus[0].total_vram_bytes or 0
                if vram >= 20 * 1024**3:
                    enforce_eager = False
                    max_num_seqs = 8

            config = ServerConfig(
                model_path=model_path,
                served_model_name=model_id,
                host="127.0.0.1",
                port=8000,
                dtype="auto",
                max_model_len=context_length,
                gpu_memory_utilization=gpu_utilization,
                max_num_seqs=max_num_seqs,
                enforce_eager=enforce_eager,
            )

            status = self.server.start(config)

            self.call_from_thread(
                self.server_started,
                status,
                model_id,
            )

        except Exception as error:
            self.call_from_thread(
                self.set_server_message,
                f"Server start failed:\n{error}",
            )

            self.call_from_thread(
                self.refresh_server_status,
            )

        finally:
            self.call_from_thread(
                self.set_server_loading,
                False,
            )

    def server_started(
        self,
        status: Any,
        model_id: str,
    ) -> None:

        self.api_model_id = model_id

        self.update_server_status(status)

        self.set_server_message(
            "Server started successfully.\n\n"
            f"Model: {model_id}\n"
            f"API: "
            f"{getattr(status, 'api_url', DEFAULT_API_URL)}"
        )

        self.update_api_info()

    def stop_server(
        self,
    ) -> None:

        self.push_screen(
            ConfirmScreen(
                "Stop Server",
                "Stop the managed vLLM server?",
            ),
            callback=(self.confirm_stop_server),
        )

    def confirm_stop_server(
        self,
        confirmed: bool,
    ) -> None:

        if confirmed:
            self.stop_selected_server()

    @work(
        thread=True,
    )
    def stop_selected_server(
        self,
    ) -> None:

        try:
            self.call_from_thread(
                self.set_server_loading,
                True,
            )

            self.call_from_thread(
                self.set_server_message,
                "Stopping server...",
            )

            status = self.server.stop()

            self.call_from_thread(
                self.server_stopped,
                status,
            )

        except Exception as error:
            self.call_from_thread(
                self.set_server_message,
                f"Stop failed:\n{error}",
            )

        finally:
            self.call_from_thread(
                self.set_server_loading,
                False,
            )

    def server_stopped(
        self,
        status: Any,
    ) -> None:

        self.update_server_status(status)

        self.set_server_message("Server stopped.")

        self.update_api_info()

    @work(thread=True)
    def refresh_server_status(
        self,
    ) -> None:
        try:
            status = VLLMServer.managed_status()
            self.call_from_thread(self.update_server_status, status)
        except Exception as error:
            self.call_from_thread(
                self.query_one("#server-status", Static).update,
                f"Could not check server:\n{error}",
            )

    def update_server_status(
        self,
        status: Any,
    ) -> None:

        state = getattr(
            status,
            "state",
            None,
        )

        state_text = state.value if state is not None else "unknown"

        host = getattr(
            status,
            "host",
            None,
        )

        port = getattr(
            status,
            "port",
            None,
        )

        pid = getattr(
            status,
            "pid",
            None,
        )

        message = getattr(
            status,
            "message",
            None,
        )

        if state_text.lower() == "running":
            state_badge = "[bold #10b981][ONLINE][/bold #10b981]"
        elif state_text.lower() == "starting":
            state_badge = "[bold #f59e0b][STARTING][/bold #f59e0b]"
        elif state_text.lower() == "failed":
            state_badge = "[bold #f43f5e][FAILED][/bold #f43f5e]"
        else:
            state_badge = "[dim][STOPPED][/dim]"

        lines = [f"[bold]Server Status:[/bold] {state_badge}"]
        if host and port:
            lines.append(
                f"[bold #38bdf8]Endpoint:[/bold #38bdf8] [white]http://{host}:{port}/v1[/white]"
            )
        if pid:
            lines.append(f"[bold #38bdf8]Process ID:[/bold #38bdf8] [white]{pid}[/white]")
        if message:
            lines.append(f"[dim]{message}[/dim]")

        self.query_one(
            "#server-status",
            Static,
        ).update("\n".join(lines))

        self.update_api_info()

    def set_server_loading(
        self,
        loading: bool,
    ) -> None:

        widget = self.query_one(
            "#server-loading",
            Horizontal,
        )

        if loading:
            widget.add_class("visible")

        else:
            widget.remove_class("visible")

        self.query_one(
            "#start-server-button",
            Button,
        ).disabled = loading

    def set_server_message(
        self,
        message: str,
    ) -> None:

        self.query_one(
            "#server-message",
            Static,
        ).update(message)

    def _get_server_context_length(
        self,
    ) -> int:

        value = self.query_one(
            "#server-context-length",
            Input,
        ).value

        context_length = int(value)

        if context_length <= 0:
            raise ValueError("Context length must be greater than zero.")

        return context_length

    def _get_gpu_utilization(
        self,
    ) -> float:

        value = self.query_one(
            "#server-gpu-utilization",
            Input,
        ).value

        utilization = float(value)

        if not (0 < utilization <= 1):
            raise ValueError("GPU utilization must be between 0 and 1.")

        return utilization

    def action_api_test(
        self,
    ) -> None:

        self.verify_api_endpoint()

    @work(thread=True)
    def update_api_info(
        self,
    ) -> None:
        try:
            status = VLLMServer.managed_status()
            running = getattr(status, "state", None) == ServerState.RUNNING
            api_url = getattr(status, "api_url", None) or DEFAULT_API_URL
            model_id = self._get_active_model_id()
            self.call_from_thread(self._apply_api_info, running, api_url, model_id)
        except Exception as error:
            self.call_from_thread(self._apply_api_info_error, str(error))

    def _apply_api_info(
        self,
        running: bool,
        api_url: str,
        model_id: str | None,
    ) -> None:
        if running:
            info = (
                "[bold #10b981][ONLINE] Local vLLM Server is Active & Healthy[/bold #10b981]\n\n"
                f"[bold #38bdf8]OpenAI Base URL:[/bold #38bdf8] [white]{api_url}[/white]\n"
                f"[bold #38bdf8]Active Model:[/bold #38bdf8]    [white]{model_id or DEFAULT_MODEL_PLACEHOLDER}[/white]\n"
                f"[bold #38bdf8]API Key:[/bold #38bdf8]         [dim]EMPTY (no key required for local access)[/dim]"
            )
        else:
            info = (
                "[dim][STOPPED] Local vLLM server is currently stopped.[/dim]\n"
                "[dim]Start a model from the Server tab or run `infra serve <model>` to activate this endpoint.[/dim]\n\n"
                f"[bold #38bdf8]Target Endpoint:[/bold #38bdf8] [white]{api_url}[/white]\n"
                f"[bold #38bdf8]Target Model:[/bold #38bdf8]    [white]{model_id or DEFAULT_MODEL_PLACEHOLDER}[/white]\n"
                f"[bold #38bdf8]API Key:[/bold #38bdf8]         [dim]EMPTY (no key required for local access)[/dim]"
            )

        try:
            self.query_one(
                "#api-server-info",
                Static,
            ).update(info)
        except Exception:
            pass

        self.api_base_url = api_url
        self.api_model_id = model_id
        self.update_api_code()

    def _apply_api_info_error(
        self,
        error: str,
    ) -> None:
        try:
            self.query_one(
                "#api-server-info",
                Static,
            ).update(f"API status unavailable:\n{error}")
        except Exception:
            pass

    def update_api_code(
        self,
    ) -> None:

        snippet = self._generate_client_code(
            self._selected_client(),
            self.api_base_url,
            self.api_model_id,
        )

        self.api_snippet = snippet

        try:
            self.query_one(
                "#api-code",
                Static,
            ).update(snippet)

        except Exception:
            pass

    def on_select_changed(
        self,
        event: Select.Changed,
    ) -> None:

        if event.select.id == "api-client":
            self.update_api_code()

    def action_copy_snippet(
        self,
    ) -> None:

        self.copy_api_snippet()

    def copy_api_snippet(
        self,
    ) -> None:

        if not self.api_snippet:
            self.set_api_status("Nothing to copy.")

            return

        try:
            self.copy_to_clipboard(self.api_snippet)

            self.set_api_status("Code copied to clipboard.")

        except Exception:
            self.set_api_status("Clipboard unavailable.")

    def save_api_snippet(
        self,
    ) -> None:

        if not self.api_snippet:
            self.set_api_status("Nothing to save.")

            return

        extension = API_SNIPPETS.get(
            self._selected_client(),
            API_SNIPPETS[DEFAULT_API_CLIENT],
        )[0]

        path = Path.cwd() / f"infra_client{extension}"

        try:
            path.write_text(
                self.api_snippet + "\n",
                encoding="utf-8",
            )

        except OSError as error:
            self.set_api_status(f"Could not save file:\n{error}")

            return

        self.set_api_status(f"Saved to:\n{path}")

    def _api_base_url(
        self,
        status: Any | None = None,
    ) -> str:

        if status is None:
            return DEFAULT_API_URL

        return (
            getattr(
                status,
                "api_url",
                None,
            )
            or DEFAULT_API_URL
        )

    def _selected_client(
        self,
    ) -> str:

        try:
            value = self.query_one(
                "#api-client",
                Select,
            ).value

        except Exception:
            return DEFAULT_API_CLIENT

        if (
            isinstance(
                value,
                str,
            )
            and value in API_SNIPPETS
        ):
            return value

        return DEFAULT_API_CLIENT

    def _generate_client_code(
        self,
        client_key: str,
        url: str,
        model_id: str | None,
    ) -> str:

        template = API_SNIPPETS.get(
            client_key,
            API_SNIPPETS[DEFAULT_API_CLIENT],
        )[1]

        model = model_id or DEFAULT_MODEL_PLACEHOLDER

        return (
            template.replace(
                "__BASE_URL__",
                url,
            )
            .replace(
                "__MODEL__",
                model,
            )
            .rstrip()
        )

    def _get_active_model_id(
        self,
    ) -> str | None:
        if self.api_model_id:
            return self.api_model_id

        if self.selected_downloaded_model_id is not None:
            return self.selected_downloaded_model_id

        try:
            status = VLLMServer.managed_status()

            if status.state != ServerState.RUNNING:
                raise RuntimeError("Server not running.")

            client = infraClient(
                getattr(
                    status,
                    "api_url",
                    None,
                )
                or DEFAULT_API_URL,
                timeout=0.5,
            )

            models = client.models()

            if models:
                first = models[0]

                model_id = first.get("id")

                if isinstance(
                    model_id,
                    str,
                ):
                    return model_id

        except Exception:
            pass

        if self.selected_downloaded_model_id is not None:
            return self.selected_downloaded_model_id

        if self.selected_model is not None:
            return self._get_model_id(self.selected_model)

        return None

    @work(
        thread=True,
    )
    def verify_api_endpoint(
        self,
    ) -> None:

        try:
            status = VLLMServer.managed_status()

            if status.state != ServerState.RUNNING:
                raise RuntimeError("vLLM server is not running.")

            api_url = (
                getattr(
                    status,
                    "api_url",
                    None,
                )
                or DEFAULT_API_URL
            )

            self.call_from_thread(
                self.set_api_status,
                "Contacting local API...",
            )

            client = infraClient(
                api_url,
                timeout=60,
            )

            models = client.models()

            response = client.chat(
                [
                    APIMessage(
                        role="user",
                        content=("Reply with one word: ready"),
                    )
                ],
                max_tokens=16,
                temperature=0.0,
            )

            reply = response.text.strip()

            self.call_from_thread(
                self.set_api_status,
                f"API verified successfully.\nModels: {len(models)}\nResponse: {reply}",
            )

        except Exception as error:
            self.call_from_thread(
                self.set_api_status,
                f"API verification failed:\n{error}",
            )

    def set_api_status(
        self,
        message: str,
    ) -> None:

        self.query_one(
            "#api-status",
            Static,
        ).update(message)

    def on_button_pressed(
        self,
        event: Button.Pressed,
    ) -> None:

        button_id = event.button.id

        if button_id == "recommend-button":
            self.start_recommendation()

        elif button_id == "download-button":
            self.start_download()

        elif (button_id == "serve-button") or (button_id == "start-server-button"):
            self.start_server()

        elif button_id == "stop-server-button":
            self.stop_server()

        elif button_id == "refresh-server-button":
            self.refresh_downloaded_models()

            self.refresh_server_status()

            self.set_server_message("Downloaded models and server status refreshed.")

        elif button_id == "api-refresh-button":
            self.update_api_info()

        elif button_id == "copy-api-button":
            self.copy_api_snippet()

        elif button_id == "save-api-button":
            self.save_api_snippet()

        elif button_id == "verify-api-button":
            self.verify_api_endpoint()

    def set_models_status(
        self,
        message: str,
    ) -> None:

        self.query_one(
            "#models-status",
            Static,
        ).update(message)

    def _get_metadata(
        self,
        result: Any,
    ) -> Any:

        metadata = getattr(
            result,
            "metadata",
            None,
        )

        if metadata is not None:
            return metadata

        return result

    def _get_model_id(
        self,
        result: Any,
    ) -> str:

        metadata = self._get_metadata(result)

        value = getattr(
            metadata,
            "model_id",
            None,
        )

        if value:
            return str(value)

        if isinstance(
            metadata,
            dict,
        ):
            value = metadata.get("model_id")

            if value:
                return str(value)

        return "Unknown model"

    def _get_parameters(
        self,
        result: Any,
    ) -> str:

        metadata = self._get_metadata(result)

        value = getattr(
            metadata,
            "parameter_count",
            None,
        )

        if value is None:
            value = getattr(
                result,
                "parameter_count",
                None,
            )

        if value is None:
            return "Unknown"

        try:
            value = float(value)

            if value >= 1_000_000_000:
                return f"{value / 1_000_000_000:.2f}B"

            if value >= 1_000_000:
                return f"{value / 1_000_000:.2f}M"

            if value >= 1_000:
                return f"{value / 1_000:.2f}K"

            return str(int(value))

        except (
            ValueError,
            TypeError,
        ):
            return "Unknown"

    def _get_runtime(
        self,
        result: Any,
    ) -> str:

        runtime_estimate = getattr(
            result,
            "runtime_estimate",
            None,
        )

        value = None

        if runtime_estimate is not None:
            value = getattr(
                runtime_estimate,
                "deployment_memory_bytes",
                None,
            )

            if value is None:
                value = getattr(
                    runtime_estimate,
                    "total_memory_bytes",
                    None,
                )

        if value is None:
            static_estimate = getattr(
                result,
                "static_estimate",
                None,
            )

            if static_estimate is not None:
                value = getattr(
                    static_estimate,
                    "total_memory_bytes",
                    None,
                )

        if value is None:
            return "Unknown"

        try:
            gib = float(value) / (1024**3)

            return f"{gib:.2f} GiB"

        except (
            ValueError,
            TypeError,
        ):
            return "Unknown"

    def _get_status(
        self,
        result: Any,
    ) -> str:
        status_str = "UNKNOWN"
        runtime = getattr(
            result,
            "runtime_compatibility",
            None,
        )

        if runtime is not None:
            fits = getattr(
                runtime,
                "fits",
                None,
            )
            if fits is True:
                status_str = "[OK] FITS"
            elif fits is False:
                status_str = "[WARN] TIGHT"
        else:
            value = getattr(
                result,
                "status",
                None,
            )
            if value is not None:
                status_str = (
                    str(value)
                    .replace("✓", "[OK]")
                    .replace("⚠", "[WARN]")
                    .replace("✗", "[FAIL]")
                    .strip()
                )

        # Style with Rich tags
        if "FITS" in status_str and "NOT" not in status_str:
            return f"[bold #10b981]{status_str}[/bold #10b981]"
        elif "TIGHT" in status_str:
            return f"[bold #f59e0b]{status_str}[/bold #f59e0b]"
        elif "DOES NOT FIT" in status_str or "FAIL" in status_str:
            return f"[bold #f43f5e]{status_str}[/bold #f43f5e]"

        return status_str

    def _get_score(
        self,
        result: Any,
    ) -> str:

        value = getattr(
            result,
            "score",
            None,
        )

        if value is None:
            return "-"

        try:
            return f"{float(value):.2f}"

        except (
            ValueError,
            TypeError,
        ):
            return str(value)


def main() -> None:
    """
    Launch the infra TUI.
    """

    InfraApp().run()


if __name__ == "__main__":
    main()
