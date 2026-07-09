"""Shared gateway platform normalization and metadata helpers."""

from __future__ import annotations

from typing import Any

# Surfaces that consume gateway text programmatically (CLI/TUI local
# diagnostics, API JSON, webhook payloads) and must keep raw text.
GATEWAY_RAW_TEXT_PLATFORMS = frozenset(
    {"local", "api_server", "webhook", "msgraph_webhook"}
)

_THREAD_REPLY_PLATFORMS = frozenset({"feishu", "mattermost"})


def gateway_platform_value(platform: Any) -> str:
    """Return a normalized gateway platform value for enums or raw strings."""
    return str(getattr(platform, "value", platform) or "").strip().lower()


def gateway_surface_passes_raw_text(platform: Any) -> bool:
    """True only for programmatic/local surfaces that must keep raw text."""
    return gateway_platform_value(platform) in GATEWAY_RAW_TEXT_PLATFORMS


def platform_uses_thread_reply(platform: Any) -> bool:
    """True when progress/delegation replies should use thread reply_to."""
    return gateway_platform_value(platform) in _THREAD_REPLY_PLATFORMS


def non_conversational_metadata(
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
