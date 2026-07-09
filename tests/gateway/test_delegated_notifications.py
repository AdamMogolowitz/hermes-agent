from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.delegated_notifications import deliver_delegated_start_notice
from gateway.delegated_notifications import (
    build_delegated_task_start_message,
    should_emit_delegated_task_start,
)
from gateway.platform_utils import (
    GATEWAY_RAW_TEXT_PLATFORMS,
    gateway_surface_passes_raw_text,
)
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
from gateway.session_state import AGENT_PENDING_SENTINEL


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


class _StubAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.DISCORD):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent: list[dict] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="delegated-1")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class _StubRunner:
    def __init__(self, adapter: _StubAdapter):
        self.adapters = {adapter.platform: adapter}
        self._adapter = adapter
        self._running_agents: dict = {}
        self._session_run_generation: dict = {}

    def _is_session_run_current(self, session_key: str, generation: int) -> bool:
        return self._session_run_generation.get(session_key) == generation

    def _adapter_for_source(self, source: SessionSource):
        return self._adapter

    def _thread_metadata_for_source(self, source: SessionSource, reply_to):
        return {"thread_id": source.thread_id} if source.thread_id else None


def _discord_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="-1001",
        chat_type="group",
        thread_id="thread-1",
    )


@pytest.mark.asyncio
async def test_deliver_delegated_start_notice_sends_on_success():
    adapter = _StubAdapter()
    runner = _StubRunner(adapter)
    source = _discord_source()
    session_key = "agent:main:discord:group:-1001:thread-1"
    runner._session_run_generation[session_key] = 1

    await deliver_delegated_start_notice(
        runner,
        source,
        "🚀 **Task delegated**\n• Goal: test",
        session_key=session_key,
        run_generation=1,
    )

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["content"].startswith("🚀")
    assert adapter.sent[0]["chat_id"] == "-1001"


@pytest.mark.asyncio
async def test_deliver_delegated_start_notice_drops_stale_generation():
    adapter = _StubAdapter()
    runner = _StubRunner(adapter)
    source = _discord_source()
    session_key = "agent:main:discord:group:-1001:thread-1"
    runner._session_run_generation[session_key] = 2

    await deliver_delegated_start_notice(
        runner,
        source,
        "🚀 **Task delegated**",
        session_key=session_key,
        run_generation=1,
    )

    assert adapter.sent == []


@pytest.mark.asyncio
async def test_deliver_delegated_start_notice_drops_when_live_agent_interrupted():
    adapter = _StubAdapter()
    runner = _StubRunner(adapter)
    source = _discord_source()
    interrupted = SimpleNamespace(is_interrupted=True)

    await deliver_delegated_start_notice(
        runner,
        source,
        "🚀 **Task delegated**",
        live_agent_getter=lambda: interrupted,
    )

    assert adapter.sent == []


@pytest.mark.asyncio
async def test_deliver_delegated_start_notice_drops_when_running_agent_interrupted():
    adapter = _StubAdapter()
    runner = _StubRunner(adapter)
    source = _discord_source()
    session_key = "agent:main:discord:group:-1001:thread-1"
    runner._running_agents[session_key] = SimpleNamespace(is_interrupted=True)
    runner._session_run_generation[session_key] = 1

    await deliver_delegated_start_notice(
        runner,
        source,
        "🚀 **Task delegated**",
        session_key=session_key,
        run_generation=1,
    )

    assert adapter.sent == []


@pytest.mark.asyncio
async def test_deliver_delegated_start_notice_drops_without_run_generation():
    adapter = _StubAdapter()
    runner = _StubRunner(adapter)
    source = _discord_source()
    session_key = "agent:main:discord:group:-1001:thread-1"

    await deliver_delegated_start_notice(
        runner,
        source,
        "🚀 **Task delegated**",
        session_key=session_key,
        run_generation=None,
    )

    assert adapter.sent == []


@pytest.mark.asyncio
async def test_deliver_delegated_start_notice_drops_when_pending_sentinel_and_getter_interrupted():
    """track_agent race: slot is PENDING but agent_holder already interrupted."""
    adapter = _StubAdapter()
    runner = _StubRunner(adapter)
    source = _discord_source()
    session_key = "agent:main:discord:group:-1001:thread-1"
    runner._running_agents[session_key] = AGENT_PENDING_SENTINEL
    runner._session_run_generation[session_key] = 1
    interrupted = SimpleNamespace(is_interrupted=True)

    await deliver_delegated_start_notice(
        runner,
        source,
        "🚀 **Task delegated**",
        session_key=session_key,
        run_generation=1,
        live_agent_getter=lambda: interrupted,
    )

    assert adapter.sent == []
