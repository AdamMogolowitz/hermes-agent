from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class ProgressCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.typing: list[dict] = []

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
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(self, chat_id, message_id, content, *, finalize: bool = False, metadata=None) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": {"stopped": True}})

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_runner(adapter: BasePlatformAdapter):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


class DelegationStartAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        import time

        cb = self.tool_progress_callback
        if cb is not None:
            cb(
                "subagent.start",
                None,
                "Generate marketing copy",
                None,
                model="kimi-k2.6:cloud",
                task_index=0,
                task_count=1,
                goal="Generate marketing copy",
            )
            # Give the gateway's background progress-sender task a chance
            # to drain the queue and render the delegated-start bubble.
            time.sleep(0.35)
            # Must be suppressed because display.tool_progress=off.
            cb("tool.started", "terminal", "pwd", {})

        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


def _setup_gateway(monkeypatch, tmp_path, *, platform: Platform, agent_cls, config_data):
    import yaml

    if config_data:
        (tmp_path / "config.yaml").write_text(yaml.dump(config_data), encoding="utf-8")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=platform)
    runner = _make_runner(adapter)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=platform,
        chat_id="-1001",
        chat_type="group",
        thread_id="thread-1",
    )
    session_key = f"agent:main:{platform.value}:{source.chat_type}:{source.chat_id}:{source.thread_id}"
    return runner, adapter, source, session_key


@pytest.mark.asyncio
async def test_discord_delegated_start_emitted_when_tool_progress_off(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    runner, adapter, source, session_key = _setup_gateway(
        monkeypatch,
        tmp_path,
        platform=Platform.DISCORD,
        agent_cls=DelegationStartAgent,
        config_data={"display": {"tool_progress": "off", "thinking_progress": False}},
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-delegation-start",
        session_key=session_key,
    )

    assert result.get("final_response") == "done"

    delegated_contents = {
        str(c.get("content", ""))
        for c in (adapter.sent + adapter.edits)
        if "Task delegated" in str(c.get("content", ""))
    }
    assert len(delegated_contents) == 1

    content = next(iter(delegated_contents))
    assert "Generate marketing copy" in content
    assert "kimi-k2.6:cloud" in content

    # Tool progress must be suppressed with display.tool_progress=off.
    blob = "\n".join([str(c["content"]) for c in (adapter.sent + adapter.edits)])
    assert "pwd" not in blob


def test_discord_message_formatter_helper(monkeypatch, tmp_path):
    gateway_run = importlib.import_module("gateway.run")
    msg = gateway_run._build_discord_delegated_task_start_message(
        "  my goal  ",
        " kimi-k2.6:cloud ",
    )
    assert "🚀 **Task delegated**" in msg
    assert "kimi-k2.6:cloud" in msg
    assert "• Goal: my goal" in msg

