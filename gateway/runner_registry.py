"""Process-local gateway runner lookup for modules that must not import run.py."""

from __future__ import annotations

from typing import Any, Callable

_runner_getter: Callable[[], Any] = lambda: None


def set_gateway_runner(getter: Callable[[], Any]) -> None:
    """Register the callable that returns the live GatewayRunner (or None)."""
    global _runner_getter
    _runner_getter = getter


def get_gateway_runner() -> Any:
    """Return the live GatewayRunner for this process, or None."""
    return _runner_getter()
