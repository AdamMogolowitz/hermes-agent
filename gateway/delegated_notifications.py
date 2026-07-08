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
from gateway.config import Platform
from utils import is_truthy_value

if TYPE_CHECKING:
    from gateway.session import SessionSource

logger = logging.getLogger(__name__)

# Surfaces that consume gateway text programmatically (CLI/TUI local
# diagnostics, API JSON, webhook payloads) and must keep raw text.
GATEWAY_RAW_TEXT_PLATFORMS = frozenset(
    {"local", "api_server", "webhook", "msgraph_webhook"}
)


def gateway_platform_value(platform: Any) -> str:
    """Return a normalized gateway platform value for enums or raw strings."""
    return str(getattr(platform, "value", platform) or "").strip().lower()


def gateway_surface_passes_raw_text(platform: Any) -> bool:
    """True only for programmatic/local surfaces that must keep raw text."""
    return gateway_platform_value(platform) in GATEWAY_RAW_TEXT_PLATFORMS


def should_emit_delegated_task_start(platform: Any) -> bool:
    """True for chat platforms that should receive delegated-start bubbles."""
    return not gateway_surface_passes_raw_text(platform)


def _non_conversational_metadata(
    metadata: dict[str, Any] | None = None,
    *,
    platform: Any = None,
) -> dict[str, Any] | None:
    """Mark Discord lifecycle/status sends without changing other platforms."""
    if gateway_platform_value(platform) != "discord":
        return metadata
    merged = dict(metadata or {})
    merged["non_conversational"] = True
    return merged


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
    from gateway.display_config import resolve_display_setting

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
) -> None:
    """Schedule a one-shot delegated-start bubble on the gateway loop.

    Used when ``subagent.start`` fires after the parent turn ends (e.g.
    ``delegate_task(background=true)``) and the per-turn progress sender
    task has already been cancelled.
    """
    from gateway.run import _gateway_runner_ref

    runner = _gateway_runner_ref()
    if runner is None:
        return
    loop = getattr(runner, "_gateway_loop", None)
    if loop is None or loop.is_closed():
        return
    safe_schedule_threadsafe(
        deliver_delegated_start_notice(runner, source, message, reply_to=reply_to),
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
) -> None:
    """Send a delegated-subagent start bubble outside the per-turn progress sender."""
    if not should_emit_delegated_task_start(source.platform):
        return
    adapter = runner._adapter_for_source(source)
    if not adapter or not source.chat_id:
        return
    thread_meta = runner._thread_metadata_for_source(source, reply_to)
    metadata = _non_conversational_metadata(thread_meta, platform=source.platform)
    progress_reply_to = (
        reply_to
        if source.platform in (Platform.FEISHU, Platform.MATTERMOST)
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
