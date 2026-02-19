"""Token counting and estimation utilities for the Anthropic API.

Provides heuristic-based token estimation for text, images, and PDFs,
as well as a wrapper around the Anthropic ``count_tokens`` API endpoint.

Image token estimation follows the formula documented by Anthropic:
    ``tokens = ceil((width * height) / 750)``
with automatic downscaling when the long edge exceeds 1 568 px or the
total pixel count exceeds ~1.2 MP.

PDF token estimation renders each page at ~2 000 tokens (midpoint of
Anthropic's documented 1 500-3 000 range).  Page count is obtained via
PyMuPDF.
"""

from __future__ import annotations

import base64
import io
import json
import math
import warnings
from typing import Any, Optional

import fitz  # pymupdf
from PIL import Image

from ..logging import get_logger

logger = get_logger(__name__)

# ── Model context-window limits ────────────────────────────────────────
# Set at ~80% of each model's context window to leave room for output.

MODEL_TOKEN_LIMITS: dict[str, int] = {
    "claude-sonnet-4-5": 160_000,
    "claude-opus-4": 160_000,
    "claude-3-5-sonnet": 160_000,
    "claude-3-opus": 160_000,
    "claude-3-sonnet": 160_000,
    "claude-3-haiku": 160_000,
    "claude-3-5-haiku": 160_000,
    "default": 160_000,
}


def get_model_token_limit(model: str) -> int:
    """Return the token-budget threshold for *model*.

    Checks for an exact match first, then falls back to substring matching
    (e.g. ``"claude-sonnet-4-5-20250514"`` matches ``"claude-sonnet-4-5"``).
    """
    if model in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model]

    for model_key, limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model.lower():
            return limit

    return MODEL_TOKEN_LIMITS["default"]


# ── Image token heuristic ──────────────────────────────────────────────
# https://docs.anthropic.com/en/docs/build-with-claude/vision

MAX_LONG_EDGE = 1568
MAX_IMAGE_TOKENS = 1600
TOKEN_DIVISOR = 750


def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate tokens for an image of *width* x *height* pixels.

    Simulates Anthropic's auto-resize logic before applying the formula
    ``ceil(w * h / 750)``.
    """
    long_edge = max(width, height)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        width = int(width * scale)
        height = int(height * scale)

    max_pixels = MAX_IMAGE_TOKENS * TOKEN_DIVISOR
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width = int(width * scale)
        height = int(height * scale)

    return math.ceil((width * height) / TOKEN_DIVISOR)


def _image_tokens_from_b64(data: str) -> int:
    """Decode a base64 image and return its estimated token count."""
    raw = base64.b64decode(data)
    img = Image.open(io.BytesIO(raw))
    w, h = img.size
    return estimate_image_tokens(w, h)


# ── PDF token heuristic ────────────────────────────────────────────────
# https://docs.anthropic.com/en/docs/build-with-claude/pdf-support

TOKENS_PER_PDF_PAGE = 2000


def estimate_pdf_tokens(pdf_bytes: bytes) -> int:
    """Estimate tokens for a PDF given its raw bytes.

    Each page costs approximately 1 500-3 000 tokens; we use the
    midpoint of 2 000.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = doc.page_count
    doc.close()
    return page_count * TOKENS_PER_PDF_PAGE


def _pdf_tokens_from_b64(data: str) -> int:
    """Decode a base64 PDF and return its estimated token count."""
    raw = base64.b64decode(data)
    return estimate_pdf_tokens(raw)


# ── Binary-aware message stripping ─────────────────────────────────────

def _extract_binary_tokens_and_strip(obj: Any) -> tuple[Any, int]:
    """Walk *obj*, strip base64 payloads, and return estimated binary tokens.

    For every ``source`` dict with ``"type": "base64"`` and a ``data``
    field, the ``data`` is replaced with a short placeholder and the
    binary tokens are accumulated based on the ``media_type``:

    * ``image/*``  — decoded with Pillow, dimension-aware heuristic.
    * ``application/pdf`` — decoded with PyMuPDF, page-count heuristic.
    * anything else — falls back to ``MAX_IMAGE_TOKENS`` (1 600).
    """
    if isinstance(obj, list):
        stripped: list[Any] = []
        total = 0
        for item in obj:
            s, t = _extract_binary_tokens_and_strip(item)
            stripped.append(s)
            total += t
        return stripped, total

    if isinstance(obj, dict):
        source = obj.get("source")
        if (
            isinstance(source, dict)
            and source.get("type") == "base64"
            and "data" in source
        ):
            media_type: str = source.get("media_type", "") or obj.get("media_type", "")
            data_str: str = source["data"]
            tokens = _tokens_for_binary(media_type, data_str)

            new_source = {k: v for k, v in source.items() if k != "data"}
            new_source["data"] = "[binary]"
            return {**obj, "source": new_source}, tokens

        rebuilt: dict[str, Any] = {}
        total = 0
        for k, v in obj.items():
            s, t = _extract_binary_tokens_and_strip(v)
            rebuilt[k] = s
            total += t
        return rebuilt, total

    return obj, 0


def _tokens_for_binary(media_type: str, data_b64: str) -> int:
    """Return estimated tokens for a single base64-encoded binary block."""
    try:
        if media_type.startswith("image/"):
            return _image_tokens_from_b64(data_b64)
        if media_type == "application/pdf":
            return _pdf_tokens_from_b64(data_b64)
    except Exception:
        logger.debug("Failed to decode binary for token estimation", media_type=media_type, exc_info=True)
    return MAX_IMAGE_TOKENS


# ── Public estimation functions ─────────────────────────────────────────

def estimate_tokens(messages: list[dict]) -> int:
    """Estimate the token count of *messages* using a character-based heuristic.

    Text content uses ~4 characters per token.  Base64 image and PDF
    payloads are excluded from the character count and instead contribute
    dimension-aware (images) or page-count-aware (PDFs) estimates.
    """
    stripped, binary_tokens = _extract_binary_tokens_and_strip(messages)
    text_chars = len(json.dumps(stripped))
    return (text_chars // 4) + binary_tokens


def estimate_tokens_heuristic(
    *,
    messages: Optional[list[dict[str, Any]]] = None,
    system: Optional[str] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    thinking: Optional[dict[str, Any]] = None,
    betas: Optional[list[str]] = None,
    container: Optional[str] = None,
) -> int:
    """Heuristically estimate token count for a request payload.

    Accepts the same keyword arguments as the agent's ``_estimate_tokens``
    so it can be used as a drop-in replacement.  Text is estimated at
    ~4 characters per token.
    """
    text_parts: list[str] = []

    if system:
        text_parts.append(system)

    if tools:
        try:
            text_parts.append(json.dumps(tools, separators=(",", ":")))
        except TypeError:
            text_parts.append(str(tools))

    if messages:
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and "text" in block:
                        text_parts.append(str(block["text"]))
                    else:
                        try:
                            text_parts.append(
                                json.dumps(block, separators=(",", ":"), ensure_ascii=False)
                            )
                        except TypeError:
                            text_parts.append(str(block))
            elif isinstance(content, dict):
                try:
                    text_parts.append(
                        json.dumps(content, separators=(",", ":"), ensure_ascii=False)
                    )
                except TypeError:
                    text_parts.append(str(content))

    full_text = " ".join(text_parts)
    return max(0, len(full_text) // 4)


# ── Token counting via API ──────────────────────────────────────────────

ALLOWED_SERVER_TOOL_TYPES = {
    "bash_20250124",
    "custom",
    "text_editor_20250124",
    "text_editor_20250429",
    "text_editor_20250728",
    "web_search_20250305",
}


def filter_messages_for_token_count(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove content types unsupported by the ``count_tokens`` endpoint.

    The endpoint does not support URL-based document sources (especially
    PDFs).  This function strips such blocks to prevent 400 errors.
    """
    filtered: list[dict[str, Any]] = []

    for msg in messages:
        content = msg.get("content")

        if not isinstance(content, list):
            filtered.append(msg)
            continue

        filtered_content: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered_content.append(block)
                continue

            if block.get("type") == "document":
                source = block.get("source", {})
                if isinstance(source, dict) and source.get("type") == "url":
                    continue

            filtered_content.append(block)

        if filtered_content:
            filtered.append({"role": msg.get("role"), "content": filtered_content})

    return filtered


async def count_tokens_api(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    system: Optional[str] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    thinking: Optional[dict[str, Any]] = None,
    betas: Optional[list[str]] = None,
    container: Optional[str] = None,
) -> Optional[int]:
    """Best-effort token counting via the Anthropic ``count_tokens`` API.

    .. deprecated::
        This function is deprecated.  Prefer :func:`estimate_tokens_heuristic`.

    Returns the ``input_tokens`` value from the API, or ``None`` on failure.
    """
    warnings.warn(
        "count_tokens_api is deprecated and will be removed in a future version. "
        "Use estimate_tokens_heuristic instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    filtered_tools: list[dict[str, Any]] = []
    if tools:
        for tool in tools:
            tool_type = tool.get("type")
            if tool_type is None or tool_type in ALLOWED_SERVER_TOOL_TYPES:
                filtered_tools.append(tool)

    filtered_messages = filter_messages_for_token_count(messages)

    params: dict[str, Any] = {
        "model": model,
        "messages": filtered_messages,
    }
    if system:
        params["system"] = system
    if filtered_tools:
        params["tools"] = filtered_tools
    if thinking:
        params["thinking"] = thinking

    if betas:
        params["extra_headers"] = {"anthropic-beta": ",".join(betas)}

    logger.debug("Anthropic count_tokens params: %s", params)
    try:
        response = await client.messages.count_tokens(**params)
        return getattr(response, "input_tokens", None)
    except Exception:
        logger.warning("Token count API call failed", exc_info=True)
        return None
