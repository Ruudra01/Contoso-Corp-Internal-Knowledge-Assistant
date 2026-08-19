"""Token counting.

The chunker's size budget has to match what the embedding model actually
charges, so counts come from tiktoken's `cl100k_base` — the encoding used by the
`text-embedding-3-*` family. If tiktoken is unavailable the estimator degrades to
a word-based approximation rather than failing the run.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.logging import get_logger

logger = get_logger(__name__)

_ENCODING_NAME = "cl100k_base"
# Empirical ratio of tokens to whitespace-separated words for English prose,
# used only when tiktoken cannot be loaded.
_TOKENS_PER_WORD = 1.35


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding(_ENCODING_NAME)
    except Exception as exc:
        logger.warning(
            "tiktoken unavailable, falling back to word-count estimate",
            extra={"error": str(exc)},
        )
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if (encoder := _encoder()) is not None:
        return len(encoder.encode(text, disallowed_special=()))
    return max(1, int(len(text.split()) * _TOKENS_PER_WORD))


def truncate_to_tokens(text: str, limit: int) -> str:
    """Trim text to `limit` tokens. Used to keep a single oversized paragraph
    inside the embedding model's request ceiling."""
    if limit <= 0 or not text:
        return ""
    if (encoder := _encoder()) is not None:
        tokens = encoder.encode(text, disallowed_special=())
        if len(tokens) <= limit:
            return text
        return encoder.decode(tokens[:limit])
    words = text.split()
    keep = max(1, int(limit / _TOKENS_PER_WORD))
    return " ".join(words[:keep])
