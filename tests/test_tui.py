from __future__ import annotations

import pytest
from textual.containers import Horizontal
from textual.widgets import Button

from infra.tui import (
    GEAR_LEVER_PULL_FRAMES,
    GearShiftSplashScreen,
    InfraApp,
)


def test_gear_lever_pull_frames_integrity() -> None:
    assert len(GEAR_LEVER_PULL_FRAMES) == 4

    for frame in GEAR_LEVER_PULL_FRAMES:
        assert ".-----." in frame
        assert ".--------." in frame


@pytest.mark.asyncio
async def test_gear_shift_splash_screen_skip() -> None:
    app = InfraApp()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        # Splash modal screen is on top
        assert len(app.screen_stack) == 2
        assert isinstance(app.screen_stack[-1], GearShiftSplashScreen)

        # Press any key to skip splash immediately
        await pilot.press("space")
        await pilot.pause(0.1)
        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_gear_shift_splash_screen_auto_dismiss() -> None:
    app = InfraApp()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert len(app.screen_stack) == 2

        # Wait for full lever pull sequence to tick and auto-dismiss
        await pilot.pause(1.2)
        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_recommend_loading_toggle() -> None:
    app = InfraApp()
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause(0.1)

        app.set_recommend_loading(True)
        assert app.query_one("#recommend-loading", Horizontal).has_class("visible")
        assert app.query_one("#recommend-button", Button).disabled is True

        app.set_recommend_loading(False)
        assert not app.query_one("#recommend-loading", Horizontal).has_class("visible")
        assert app.query_one("#recommend-button", Button).disabled is False


@pytest.mark.asyncio
async def test_server_loading_toggle() -> None:
    app = InfraApp()
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause(0.1)

        app.set_server_loading(True)
        assert app.query_one("#server-loading", Horizontal).has_class("visible")
        assert app.query_one("#start-server-button", Button).disabled is True

        app.set_server_loading(False)
        assert not app.query_one("#server-loading", Horizontal).has_class("visible")
        assert app.query_one("#start-server-button", Button).disabled is False
