"""TUI should relay subagent lifecycle events even when tool_progress is off."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_subagent_off")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()
        mod._child_mirrors.clear()
        mod._active_child_runs.clear()


@pytest.fixture()
def emits(server, monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda sid: False)
    return captured


def test_subagent_start_emitted_when_tool_progress_off(server, emits):
    server._on_tool_progress(
        "parent-sid",
        "subagent.start",
        None,
        "research docs",
        None,
        goal="research docs",
        model="kimi-k2.6:cloud",
        task_count=1,
        task_index=0,
    )

    assert [(e, s) for e, s, _ in emits] == [("subagent.start", "parent-sid")]
    payload = emits[0][2]
    assert payload["goal"] == "research docs"
    assert payload["model"] == "kimi-k2.6:cloud"


def test_tool_started_suppressed_when_tool_progress_off(server, emits):
    server._on_tool_progress(
        "parent-sid",
        "tool.started",
        "terminal",
        "pwd",
        {"command": "pwd"},
    )

    assert emits == []
