from __future__ import annotations

import pytest

from gateway.config import Platform
from gateway.delegated_notifications import (
    build_delegated_task_start_message,
    should_emit_delegated_task_start,
)
from gateway.platform_utils import (
    GATEWAY_RAW_TEXT_PLATFORMS,
    gateway_surface_passes_raw_text,
)


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        (Platform.LOCAL, False),
        (Platform.API_SERVER, False),
        (Platform.WEBHOOK, False),
        (Platform.MSGRAPH_WEBHOOK, False),
        (Platform.TELEGRAM, True),
        (Platform.DISCORD, True),
        (Platform.BLUEBUBBLES, True),
        (Platform.SIGNAL, True),
        ("", True),
        ("unknown_plugin_platform", True),
        (Platform("irc"), True),
    ],
)
def test_should_emit_delegated_task_start(platform, expected):
    assert should_emit_delegated_task_start(platform) is expected


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        (Platform.LOCAL, True),
        (Platform.API_SERVER, True),
        (Platform.WEBHOOK, True),
        (Platform.TELEGRAM, False),
        ("", False),
    ],
)
def test_gateway_surface_passes_raw_text(platform, expected):
    assert gateway_surface_passes_raw_text(platform) is expected


@pytest.mark.parametrize("platform", sorted(GATEWAY_RAW_TEXT_PLATFORMS))
def test_raw_text_platforms_never_emit_delegated_start(platform):
    assert should_emit_delegated_task_start(platform) is False


def test_build_delegated_task_start_message_helper():
    msg = build_delegated_task_start_message(
        "  my goal  ",
        " kimi-k2.6:cloud ",
    )
    assert msg is not None
    assert "🚀 **Task delegated**" in msg
    assert "kimi-k2.6:cloud" in msg
    assert "Configured model" in msg
    assert "• Goal: my goal" in msg


def test_build_delegated_task_start_message_omits_empty_goal():
    msg = build_delegated_task_start_message("", "kimi-k2.6:cloud")
    assert msg is not None
    assert "• Goal:" not in msg
    assert "kimi-k2.6:cloud" in msg


def test_build_delegated_task_start_message_returns_none_when_both_empty():
    assert build_delegated_task_start_message("", "") is None
    assert build_delegated_task_start_message("   ", "  ") is None


def test_build_delegated_task_start_message_truncates_long_goal_and_model():
    long_goal = "g" * 100
    long_model = "m" * 50
    msg = build_delegated_task_start_message(long_goal, long_model)
    assert msg is not None
    assert "g" * 55 + "..." in msg
    assert "m" * 35 + "..." in msg
    assert long_goal not in msg
    assert long_model not in msg


def test_build_delegated_task_start_message_batch_prefix():
    msg = build_delegated_task_start_message(
        "goal a",
        "model-a",
        task_index=1,
        task_count=3,
    )
    assert msg is not None
    assert "🚀 **[2] Task delegated**" in msg


def test_build_delegated_task_start_message_i18n():
    msg = build_delegated_task_start_message(
        "mein ziel",
        "kimi-k2.6:cloud",
        lang="de",
    )
    assert msg is not None
    assert "Aufgabe delegiert" in msg
    assert "Ziel: mein ziel" in msg
