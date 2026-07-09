"""Delegated subagent start notification helpers for the gateway."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent.async_utils import safe_schedule_threadsafe
from agent.delegation_display import (
    DELEGATED_GOAL_PREVIEW_MAX,
    DELEGATED_MODEL_PREVIEW_MAX,
    truncate_delegated_label,
)
from agent.i18n import t
from gateway.display_config import resolve_display_setting
from gateway.platform_utils import (
    GATEWAY_RAW_TEXT_PLATFORMS,
    gateway_surface_passes_raw_text,
    non_conversational_metadata,
    platform_uses_thread_reply,
)
from gateway.runner_registry import get_gateway_runner
from utils import is_truthy_value

if TYPE_CHECKING:
    from gateway.session import SessionSource

logger = logging.getLogger(__name__)


def should_emit_delegated_task_start(platform: Any) -> bool:
    """True for chat platforms that should receive delegated-start bubbles."""
    return not gateway_surface_passes_raw_text(platform)


def build_delegated_task_start_message(
    goal: str,
    model: str,
    *,
    task_index: int = 0,
    task_count: int = 1,
    lang: str | None = None,
) -> str:
    """Format a one-shot delegated task start notification.

    Cache-safe + side-effect free: used by the gateway progress callback.
    """

    def _tr(key: str, **kwargs) -> str:
        if lang:
            return t(key, lang=lang, **kwargs)
        return t(key, **kwargs)

    _goal = truncate_delegated_label(goal, DELEGATED_GOAL_PREVIEW_MAX)
    _model = truncate_delegated_label(model, DELEGATED_MODEL_PREVIEW_MAX)

    if int(task_count) > 1:
        header = _tr(
            "gateway.delegated.start_header_indexed",
            index=int(task_index) + 1,
        )
    else:
        header = _tr("gateway.delegated.start_header")

    lines = [header]
    if _model:
        lines.append(_tr("gateway.delegated.label_model", model=_model))
    if _goal:
        lines.append(_tr("gateway.delegated.label_goal", goal=_goal))
    return "\n".join(lines)


def delegated_start_enabled(user_config: dict, platform_key: str) -> bool:
    """True when delegated-subagent start notifications are enabled."""
    value = resolve_display_setting(
        user_config,
        platform_key,
        "delegated_start_notifications",
        fallback=True,
    )
    return is_truthy_value(value, default=True)


def schedule_delegated_start_notice(
    *,
    source: SessionSource,
    message: str,
    reply_to: str | None = None,
    session_key: str | None = None,
    run_generation: int | None = None,
) -> None:
    """Schedule a one-shot delegated-start bubble on the gateway loop.

    Used when ``subagent.start`` fires after the parent turn ends (e.g.
    ``delegate_task(background=true)``) and the per-turn progress sender
    task has already been cancelled.
    """
    runner = get_gateway_runner()
    if runner is None:
        return
    loop = getattr(runner, "_gateway_loop", None)
    if loop is None or loop.is_closed():
        logger.warning(
            "delegated start notice dropped: gateway loop unavailable "
            "(platform=%s)",
            getattr(getattr(source, "platform", None), "value", source.platform),
        )
        return
    safe_schedule_threadsafe(
        deliver_delegated_start_notice(
            runner,
            source,
            message,
            reply_to=reply_to,
            session_key=session_key,
            run_generation=run_generation,
        ),
        loop,
        logger=logger,
        log_message="delegated start notice scheduling error",
    )


async def deliver_delegated_start_notice(
    runner: Any,
    source: SessionSource,
    message: str,
    *,
    reply_to: str | None = None,
    session_key: str | None = None,
    run_generation: int | None = None,
) -> None:
    """Send a delegated-subagent start bubble outside the per-turn progress sender."""
    if not should_emit_delegated_task_start(source.platform):
        return
    if (
        session_key
        and run_generation is not None
        and not runner._is_session_run_current(session_key, run_generation)
    ):
        logger.debug(
            "delegated start notice dropped: stale session generation "
            "(session_key=%s generation=%s)",
            session_key,
            run_generation,
        )
        return
    adapter = runner._adapter_for_source(source)
    if not adapter or not source.chat_id:
        return
    thread_meta = runner._thread_metadata_for_source(source, reply_to)
    metadata = non_conversational_metadata(thread_meta, platform=source.platform)
    progress_reply_to = (
        reply_to
        if platform_uses_thread_reply(source.platform)
        and source.thread_id
        and reply_to
        else None
    )
    try:
        await adapter.send(
            chat_id=source.chat_id,
            content=message,
            reply_to=progress_reply_to,
            metadata=metadata,
        )
    except Exception as exc:
        logger.debug("delegated start delivery failed: %s", exc)
