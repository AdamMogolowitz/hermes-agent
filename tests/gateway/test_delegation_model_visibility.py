from __future__ import annotations

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.runner_registry import set_gateway_runner
from gateway.session import SessionSource
from tests.gateway.progress_fixtures import (
    NonEditProgressCaptureAdapter,
    ProgressCaptureAdapter,
    make_progress_runner,
)


class DelegationStartAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
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
            # Must be suppressed because display.tool_progress=off.
            cb("tool.started", "terminal", "pwd", {})

        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class InterruptedDelegationStartAgent(DelegationStartAgent):
    """Simulates /stop during an in-flight turn before subagent.start delivers."""

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.is_interrupted = True
        return super().run_conversation(message, conversation_history, task_id)


class BatchDelegationStartAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is not None:
            cb(
                "subagent.start",
                None,
                "First delegated goal",
                None,
                model="model-a",
                task_index=0,
                task_count=2,
                goal="First delegated goal",
            )
            cb(
                "subagent.start",
                None,
                "Second delegated goal",
                None,
                model="model-b",
                task_index=1,
                task_count=2,
                goal="Second delegated goal",
            )

        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


def _setup_gateway(
    monkeypatch,
    tmp_path,
    *,
    platform: Platform,
    agent_cls,
    config_data,
    adapter: BasePlatformAdapter | None = None,
):
    import yaml

    if config_data:
        (tmp_path / "config.yaml").write_text(yaml.dump(config_data), encoding="utf-8")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = adapter or ProgressCaptureAdapter(platform=platform)
    runner = make_progress_runner(adapter)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    set_gateway_runner(lambda: runner)

    source = SessionSource(
        platform=platform,
        chat_id="-1001",
        chat_type="group",
        thread_id="thread-1",
    )
    session_key = f"agent:main:{platform.value}:{source.chat_type}:{source.chat_id}:{source.thread_id}"
    return runner, adapter, source, session_key


def _delegated_sent_contents(adapter: BasePlatformAdapter) -> list[str]:
    return [
        str(c.get("content", ""))
        for c in adapter.sent
        if "Task delegated" in str(c.get("content", ""))
    ]


@pytest.mark.asyncio
async def test_discord_delegated_start_emitted_when_tool_progress_off(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    runner, adapter, source, session_key = _setup_gateway(
        monkeypatch,
        tmp_path,
        platform=Platform.DISCORD,
        agent_cls=DelegationStartAgent,
        config_data={
            "display": {
                "tool_progress": "off",
                "thinking_progress": False,
                "delegated_start_notifications": True,
            }
        },
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

    delegated_sent = _delegated_sent_contents(adapter)
    assert len(delegated_sent) == 1

    content = delegated_sent[0]
    assert "Generate marketing copy" in content
    assert "kimi-k2.6:cloud" in content

    delegated_edits = [
        str(c.get("content", ""))
        for c in adapter.edits
        if "Task delegated" in str(c.get("content", ""))
    ]
    assert delegated_edits == []

    blob = "\n".join([str(c["content"]) for c in (adapter.sent + adapter.edits)])
    assert "pwd" not in blob


@pytest.mark.asyncio
async def test_delegated_start_emitted_when_tool_progress_log(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "log")

    runner, adapter, source, session_key = _setup_gateway(
        monkeypatch,
        tmp_path,
        platform=Platform.DISCORD,
        agent_cls=DelegationStartAgent,
        config_data={
            "display": {
                "tool_progress": "log",
                "thinking_progress": False,
                "delegated_start_notifications": True,
            }
        },
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-delegation-log",
        session_key=session_key,
    )

    assert result.get("final_response") == "done"

    delegated_sent = _delegated_sent_contents(adapter)
    assert len(delegated_sent) == 1
    assert "Generate marketing copy" in delegated_sent[0]
    assert "kimi-k2.6:cloud" in delegated_sent[0]

    blob = "\n".join([str(c["content"]) for c in (adapter.sent + adapter.edits)])
    assert "pwd" not in blob

    log_path = tmp_path / "logs" / "tool_calls.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "terminal:" in log_text
    assert "pwd" in log_text


@pytest.mark.asyncio
async def test_delegated_start_suppressed_when_notifications_disabled(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    runner, adapter, source, session_key = _setup_gateway(
        monkeypatch,
        tmp_path,
        platform=Platform.DISCORD,
        agent_cls=DelegationStartAgent,
        config_data={
            "display": {
                "tool_progress": "off",
                "thinking_progress": False,
                "delegated_start_notifications": False,
            }
        },
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

    blob = "\n".join([str(c["content"]) for c in (adapter.sent + adapter.edits)])
    assert "Task delegated" not in blob
    assert "pwd" not in blob


@pytest.mark.asyncio
async def test_delegated_start_suppressed_by_platform_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    runner, adapter, source, session_key = _setup_gateway(
        monkeypatch,
        tmp_path,
        platform=Platform.DISCORD,
        agent_cls=DelegationStartAgent,
        config_data={
            "display": {
                "tool_progress": "off",
                "thinking_progress": False,
                "delegated_start_notifications": True,
                "platforms": {
                    "discord": {
                        "delegated_start_notifications": False,
                    }
                },
            }
        },
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-delegation-platform-off",
        session_key=session_key,
    )

    assert result.get("final_response") == "done"

    blob = "\n".join([str(c["content"]) for c in (adapter.sent + adapter.edits)])
    assert "Task delegated" not in blob
    assert "pwd" not in blob


@pytest.mark.asyncio
async def test_batch_delegation_emits_indexed_start_bubbles(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    runner, adapter, source, session_key = _setup_gateway(
        monkeypatch,
        tmp_path,
        platform=Platform.DISCORD,
        agent_cls=BatchDelegationStartAgent,
        config_data={
            "display": {
                "tool_progress": "off",
                "thinking_progress": False,
                "delegated_start_notifications": True,
            }
        },
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-delegation-batch",
        session_key=session_key,
    )

    assert result.get("final_response") == "done"

    delegated_sent = _delegated_sent_contents(adapter)
    assert len(delegated_sent) == 2

    indexed = [c for c in delegated_sent if "[1]" in c or "[2]" in c]
    assert len(indexed) == 2
    assert any("First delegated goal" in c and "model-a" in c for c in delegated_sent)
    assert any("Second delegated goal" in c and "model-b" in c for c in delegated_sent)


@pytest.mark.asyncio
async def test_delegated_start_sent_on_non_edit_adapter_when_tool_progress_off(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    adapter = NonEditProgressCaptureAdapter(platform=Platform.BLUEBUBBLES)

    runner, adapter, source, session_key = _setup_gateway(
        monkeypatch,
        tmp_path,
        platform=Platform.BLUEBUBBLES,
        adapter=adapter,
        agent_cls=DelegationStartAgent,
        config_data={
            "display": {
                "tool_progress": "off",
                "thinking_progress": False,
                "delegated_start_notifications": True,
            }
        },
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

    delegated_sent = _delegated_sent_contents(adapter)
    assert len(delegated_sent) == 1

    content = delegated_sent[0]
    assert "Generate marketing copy" in content
    assert "kimi-k2.6:cloud" in content

    blob = "\n".join([str(c["content"]) for c in adapter.sent])
    assert "pwd" not in blob


class BackgroundDelegationStartAgent:
    """Fires subagent.start after a delay, simulating background delegation."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        import threading
        import time

        cb = self.tool_progress_callback

        def _delayed_start():
            time.sleep(0.45)
            if cb is not None:
                cb(
                    "subagent.start",
                    None,
                    "Background research task",
                    None,
                    model="kimi-k2.6:cloud",
                    task_index=0,
                    task_count=1,
                    goal="Background research task",
                )

        if cb is not None:
            threading.Thread(target=_delayed_start, daemon=True).start()

        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


@pytest.mark.asyncio
async def test_delegated_start_durable_after_turn_ends(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    runner, adapter, source, session_key = _setup_gateway(
        monkeypatch,
        tmp_path,
        platform=Platform.DISCORD,
        agent_cls=BackgroundDelegationStartAgent,
        config_data={
            "display": {
                "tool_progress": "off",
                "thinking_progress": False,
                "delegated_start_notifications": True,
            }
        },
    )
    runner._gateway_loop = asyncio.get_running_loop()

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-delegation-bg",
        session_key=session_key,
    )

    assert result.get("final_response") == "done"

    deadline = time.time() + 2.0
    delegated_contents = set()
    while time.time() < deadline:
        delegated_contents = {
            str(c.get("content", ""))
            for c in (adapter.sent + adapter.edits)
            if "Task delegated" in str(c.get("content", ""))
        }
        if delegated_contents:
            break
        await asyncio.sleep(0.05)

    assert len(delegated_contents) == 1
    content = next(iter(delegated_contents))
    assert "Background research task" in content
    assert "kimi-k2.6:cloud" in content


@pytest.mark.asyncio
async def test_delegated_start_suppressed_when_session_interrupted(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    runner, adapter, source, session_key = _setup_gateway(
        monkeypatch,
        tmp_path,
        platform=Platform.DISCORD,
        agent_cls=InterruptedDelegationStartAgent,
        config_data={
            "display": {
                "tool_progress": "off",
                "thinking_progress": False,
                "delegated_start_notifications": True,
            }
        },
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-delegation-interrupted",
        session_key=session_key,
    )

    assert result.get("final_response") == "done"

    delegated_sent = _delegated_sent_contents(adapter)
    assert delegated_sent == []


@pytest.mark.asyncio
async def test_delegated_start_suppressed_when_session_generation_stale(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    runner, adapter, source, session_key = _setup_gateway(
        monkeypatch,
        tmp_path,
        platform=Platform.DISCORD,
        agent_cls=BackgroundDelegationStartAgent,
        config_data={
            "display": {
                "tool_progress": "off",
                "thinking_progress": False,
                "delegated_start_notifications": True,
            }
        },
    )
    runner._gateway_loop = asyncio.get_running_loop()
    run_generation = runner._begin_session_run_generation(session_key)

    async def _invalidate_generation():
        await asyncio.sleep(0.2)
        runner._begin_session_run_generation(session_key)

    invalidate_task = asyncio.create_task(_invalidate_generation())

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-delegation-stale-gen",
        session_key=session_key,
        run_generation=run_generation,
    )

    await invalidate_task

    assert result.get("final_response") == "done"

    deadline = time.time() + 2.0
    delegated_contents = set()
    while time.time() < deadline:
        delegated_contents = {
            str(c.get("content", ""))
            for c in (adapter.sent + adapter.edits)
            if "Task delegated" in str(c.get("content", ""))
        }
        if delegated_contents:
            break
        await asyncio.sleep(0.05)

    assert delegated_contents == set()
