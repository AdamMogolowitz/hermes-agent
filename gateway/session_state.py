"""Shared gateway session-slot constants (no run.py import)."""

# Placed into _running_agents immediately when a session starts processing,
# before any await — prevents a second message from bypassing the "already
# running" guard during the async gap before agent creation.
AGENT_PENDING_SENTINEL = object()
