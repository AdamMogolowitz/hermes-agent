"""CLI subagent.start progress when delegate spinner is inactive."""

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cli_mod = None


def _make_cli():
    global _cli_mod
    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "off"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as mod

        mod = importlib.reload(mod)
        _cli_mod = mod
        with patch.object(mod, "get_tool_definitions", return_value=[]), patch.dict(
            mod.__dict__, {"CLI_CONFIG": clean_config}
        ):
            return mod.HermesCLI()


def test_subagent_start_prints_without_delegate_spinner():
    cli = _make_cli()
    cli.agent = MagicMock(_delegate_spinner=None)

    with patch.object(_cli_mod, "_cprint") as mock_print:
        cli._on_tool_progress(
            "subagent.start",
            None,
            "Summarize docs",
            None,
            model="kimi-k2.6:cloud",
            goal="Summarize docs",
        )

    mock_print.assert_called_once()
    line = mock_print.call_args[0][0]
    assert "Summarize docs" in line
    assert "kimi-k2.6:cloud" in line


def test_subagent_start_skipped_when_delegate_spinner_active():
    cli = _make_cli()
    cli.agent = MagicMock(_delegate_spinner=MagicMock())

    with patch.object(_cli_mod, "_cprint") as mock_print:
        cli._on_tool_progress(
            "subagent.start",
            None,
            "Summarize docs",
            None,
            model="kimi-k2.6:cloud",
            goal="Summarize docs",
        )

    mock_print.assert_not_called()
