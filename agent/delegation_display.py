"""Shared truncation constants for delegated-subagent start UI."""

DELEGATED_GOAL_PREVIEW_MAX = 55
DELEGATED_MODEL_PREVIEW_MAX = 35


def truncate_delegated_label(text: str, max_len: int) -> str:
    text = (text or "").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text
