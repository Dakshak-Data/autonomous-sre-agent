"""
Token Helper: Exact BPE token counting via tiktoken + Smart Head/Tail Log Trimmer.
"""

from __future__ import annotations

import json
from typing import Any

import tiktoken

_ENCODING = None
_ENCODING_LOADED = False


def _get_encoding():
    global _ENCODING, _ENCODING_LOADED
    if not _ENCODING_LOADED:
        _ENCODING_LOADED = True
        try:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODING = None
    return _ENCODING


def count_text_tokens(text: str) -> int:
    """Count tokens in a plain string (falls back to ~4 chars/token if offline)."""
    if not text:
        return 0
    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def count_message_tokens(msg: dict[str, Any]) -> int:
    """Count tokens in a single OpenAI-format chat message dictionary."""
    tokens = 4  # Message wrapper overhead
    if msg.get("content"):
        tokens += count_text_tokens(str(msg["content"]))
    if msg.get("tool_calls"):
        tokens += count_text_tokens(json.dumps(msg["tool_calls"]))
    return tokens


def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Count total tokens across an entire conversation history."""
    return sum(count_message_tokens(m) for m in messages)


def truncate_head_tail(raw_text: str, max_tokens: int) -> tuple[str, int]:
    """
    Smart Head + Tail truncation for huge production logs.
    Keeps first 40% (headers) + last 60% (actual crash stack traces!).

    Returns:
        (trimmed_text, omitted_tokens_count)
    """
    orig_tokens = count_text_tokens(raw_text)
    if orig_tokens <= max_tokens:
        return raw_text, 0

    chars_per_token = max(1.0, len(raw_text) / orig_tokens)
    allowed_chars = int(max_tokens * chars_per_token)
    head_chars = int(allowed_chars * 0.40)
    tail_chars = int(allowed_chars * 0.60)
    omitted_tokens = orig_tokens - max_tokens

    trimmed = (
        f"{raw_text[:head_chars]}\n"
        f"\n... [TOKEN HELPER: ~{omitted_tokens} middle tokens trimmed to protect budget] ...\n\n"
        f"{raw_text[-tail_chars:]}"
    )
    return trimmed, omitted_tokens
