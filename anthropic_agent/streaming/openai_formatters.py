"""OpenAI streaming formatters.

This module provides formatters for OpenAI Chat Completions streaming responses,
producing output compatible with the Anthropic XML format used by the agent system.

The formatters handle OpenAI's streaming delta pattern:
- `chunk.choices[0].delta.content` for text
- `chunk.choices[0].delta.tool_calls` for tool calls
- `chunk.choices[0].finish_reason` for completion status
"""

import asyncio
import json
import html
from dataclasses import dataclass
from typing import Any, Literal, Callable, Awaitable, Optional

# Type alias for OpenAI formatter names
OpenAIFormatterType = Literal["xml", "raw"]


def escape_xml_attr(value: str) -> str:
    """Escape a string for safe use as an XML attribute value."""
    return html.escape(str(value), quote=True)


async def stream_xml_to_aqueue(chunk: Any, queue: asyncio.Queue) -> None:
    """Send a chunk to an async queue with double-escaping for XML format.

    Escaping strategy:
    1. First escapes backslashes: \\ → \\\\
    2. Then escapes newlines: newline → \\n

    This ensures the frontend can distinguish:
    - \\n (two chars) → actual newline
    - \\\\n (three chars) → literal backslash-n
    """
    if isinstance(chunk, str):
        chunk = chunk.replace('\\', '\\\\')  # First: escape backslashes
        chunk = chunk.replace('\n', '\\n')   # Then: escape newlines
    await queue.put(chunk)


async def stream_to_aqueue(chunk: Any, queue: asyncio.Queue) -> None:
    """Send a chunk to an async queue with simple SSE-safe escaping.

    Only escapes actual newlines to prevent breaking SSE framing.
    Used by raw_formatter where JSON already has its own escaping.
    """
    if isinstance(chunk, str):
        chunk = chunk.replace('\n', '\\n')
    await queue.put(chunk)


@dataclass
class OpenAIStreamResult:
    """Result of an OpenAI streaming call.

    Attributes:
        assistant_message: Canonical assistant message dict with role and content blocks.
        finish_reason: The finish reason from OpenAI (stop, tool_calls, length, etc.).
        usage: Usage information if available from final chunk.
        model: Model identifier from the response.
    """
    assistant_message: dict[str, Any]
    finish_reason: Optional[str]
    usage: Optional[dict[str, Any]]
    model: Optional[str]


async def openai_xml_formatter(
    stream: Any,
    queue: asyncio.Queue,
    stream_tool_results: bool = True,
) -> OpenAIStreamResult:
    """Format OpenAI streaming response with custom XML tags.

    This formatter wraps different content types in custom XML tags matching
    the Anthropic format:
    - Text: <content-block-text>...</content-block-text>
    - Tool calls: <content-block-tool_call id="..." name="..." arguments="..."></content-block-tool_call>
    - Errors: <content-block-error><![CDATA[...]]></content-block-error>

    Args:
        stream: OpenAI async stream (AsyncIterator of ChatCompletionChunk).
        queue: Async queue to send formatted output chunks.
        stream_tool_results: Whether to stream tool results (kept for API compatibility).

    Returns:
        OpenAIStreamResult with the accumulated message and metadata.
    """
    text_open = False
    collected_text: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: Optional[str] = None
    final_usage: Any = None
    model: Optional[str] = None

    try:
        async for chunk in stream:
            # Capture model from first chunk
            if model is None and hasattr(chunk, 'model'):
                model = chunk.model

            # Usage-only chunk (when stream_options.include_usage=True)
            if getattr(chunk, "usage", None) is not None:
                final_usage = chunk.usage

            if not getattr(chunk, "choices", None):
                continue

            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            # Text delta
            delta_text = getattr(delta, "content", None)
            if delta_text:
                collected_text.append(delta_text)
                if not text_open:
                    await stream_xml_to_aqueue("<content-block-text>", queue)
                    text_open = True
                await stream_xml_to_aqueue(delta_text, queue)

            # Tool call deltas
            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                for tc in delta_tool_calls:
                    idx = getattr(tc, "index", None)
                    if idx is None:
                        idx = 0
                    entry = tool_calls.setdefault(int(idx), {"id": None, "name": None, "arguments": ""})
                    if getattr(tc, "id", None):
                        entry["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            entry["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            entry["arguments"] += fn.arguments

    except Exception as e:
        # Stream error
        err = escape_xml_attr(str(e))
        await stream_xml_to_aqueue(
            f"<content-block-error><![CDATA[{err}]]></content-block-error>",
            queue
        )
        raise
    finally:
        # Close text block if open
        if text_open:
            await stream_xml_to_aqueue("</content-block-text>", queue)

    # Emit tool calls after text block closes
    tool_use_blocks: list[dict[str, Any]] = []
    for _, tc in sorted(tool_calls.items(), key=lambda kv: kv[0]):
        tc_id = tc.get("id")
        tc_name = tc.get("name")
        tc_args_raw = tc.get("arguments") or "{}"

        # Stream tool call tag
        args_attr = escape_xml_attr(tc_args_raw)
        id_attr = escape_xml_attr(str(tc_id))
        name_attr = escape_xml_attr(str(tc_name))
        await stream_xml_to_aqueue(
            f'<content-block-tool_call id="{id_attr}" name="{name_attr}" arguments="{args_attr}"></content-block-tool_call>',
            queue
        )

        # Build canonical tool_use block
        try:
            parsed_args = json.loads(tc_args_raw) if isinstance(tc_args_raw, str) else tc_args_raw
        except Exception:
            parsed_args = {"_raw": tc_args_raw}

        tool_use_blocks.append({
            "type": "tool_use",
            "id": tc_id,
            "name": tc_name,
            "input": parsed_args,
        })

    # Build canonical assistant message
    blocks: list[dict[str, Any]] = []
    full_text = "".join(collected_text)
    if full_text:
        blocks.append({"type": "text", "text": full_text})
    blocks.extend(tool_use_blocks)

    assistant_message: dict[str, Any] = {"role": "assistant", "content": blocks}

    # Normalize usage
    usage: Optional[dict[str, Any]] = None
    if final_usage is not None:
        usage = _normalize_usage(final_usage)

    return OpenAIStreamResult(
        assistant_message=assistant_message,
        finish_reason=finish_reason,
        usage=usage,
        model=model,
    )


async def openai_raw_formatter(
    stream: Any,
    queue: asyncio.Queue,
    stream_tool_results: bool = True,
) -> OpenAIStreamResult:
    """Format OpenAI streaming response with raw/minimal formatting.

    This formatter streams raw OpenAI chunks as JSON with minimal processing.

    Args:
        stream: OpenAI async stream (AsyncIterator of ChatCompletionChunk).
        queue: Async queue to send formatted output chunks.
        stream_tool_results: Whether to stream tool results (kept for API compatibility).

    Returns:
        OpenAIStreamResult with the accumulated message and metadata.
    """
    collected_text: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: Optional[str] = None
    final_usage: Any = None
    model: Optional[str] = None

    async for chunk in stream:
        # Capture model
        if model is None and hasattr(chunk, 'model'):
            model = chunk.model

        # Usage-only chunk
        if getattr(chunk, "usage", None) is not None:
            final_usage = chunk.usage

        if not getattr(chunk, "choices", None):
            # Stream non-choice chunks as raw JSON
            try:
                chunk_json = chunk.model_dump_json() if hasattr(chunk, 'model_dump_json') else json.dumps(chunk, default=str)
                await stream_to_aqueue(chunk_json, queue)
            except Exception:
                pass
            continue

        choice = chunk.choices[0]
        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue

        # Collect text
        delta_text = getattr(delta, "content", None)
        if delta_text:
            collected_text.append(delta_text)
            await stream_to_aqueue(delta_text, queue)

        # Collect tool calls
        delta_tool_calls = getattr(delta, "tool_calls", None)
        if delta_tool_calls:
            for tc in delta_tool_calls:
                idx = getattr(tc, "index", None) or 0
                entry = tool_calls.setdefault(int(idx), {"id": None, "name": None, "arguments": ""})
                if getattr(tc, "id", None):
                    entry["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        entry["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        entry["arguments"] += fn.arguments

    # Build tool use blocks
    tool_use_blocks: list[dict[str, Any]] = []
    for _, tc in sorted(tool_calls.items(), key=lambda kv: kv[0]):
        tc_args_raw = tc.get("arguments") or "{}"
        try:
            parsed_args = json.loads(tc_args_raw) if isinstance(tc_args_raw, str) else tc_args_raw
        except Exception:
            parsed_args = {"_raw": tc_args_raw}
        tool_use_blocks.append({
            "type": "tool_use",
            "id": tc.get("id"),
            "name": tc.get("name"),
            "input": parsed_args,
        })

        # Stream tool call info
        await stream_to_aqueue(f"\n[tool_call] id={tc.get('id')} name={tc.get('name')} args={tc_args_raw}\n", queue)

    # Build canonical assistant message
    blocks: list[dict[str, Any]] = []
    full_text = "".join(collected_text)
    if full_text:
        blocks.append({"type": "text", "text": full_text})
    blocks.extend(tool_use_blocks)

    assistant_message: dict[str, Any] = {"role": "assistant", "content": blocks}

    # Normalize usage
    usage: Optional[dict[str, Any]] = None
    if final_usage is not None:
        usage = _normalize_usage(final_usage)

    return OpenAIStreamResult(
        assistant_message=assistant_message,
        finish_reason=finish_reason,
        usage=usage,
        model=model,
    )


def _normalize_usage(usage_obj: Any) -> dict[str, Any]:
    """Normalize OpenAI usage fields to the agent's common shape."""
    if usage_obj is None:
        return {}

    def _get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    prompt_tokens = _get(usage_obj, "prompt_tokens")
    completion_tokens = _get(usage_obj, "completion_tokens")
    total_tokens = _get(usage_obj, "total_tokens")

    out: dict[str, Any] = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw": usage_obj,
    }

    # prompt_tokens_details.cached_tokens
    ptd = _get(usage_obj, "prompt_tokens_details")
    if ptd is not None:
        cached = _get(ptd, "cached_tokens")
        if cached is not None:
            out["cache_read_input_tokens"] = cached

    # completion_tokens_details.reasoning_tokens
    ctd = _get(usage_obj, "completion_tokens_details")
    if ctd is not None:
        reasoning = _get(ctd, "reasoning_tokens")
        if reasoning is not None:
            out["reasoning_tokens"] = reasoning

    return out


# Formatter registry
OPENAI_FORMATTERS: dict[str, Callable[[Any, asyncio.Queue, bool], Awaitable[OpenAIStreamResult]]] = {
    "xml": openai_xml_formatter,
    "raw": openai_raw_formatter,
}


def get_openai_formatter(name: OpenAIFormatterType) -> Callable[[Any, asyncio.Queue, bool], Awaitable[OpenAIStreamResult]]:
    """Get an OpenAI formatter function by name.

    Args:
        name: Formatter name ("xml" or "raw")

    Returns:
        The formatter function

    Raises:
        ValueError: If formatter name is not recognized
    """
    if name not in OPENAI_FORMATTERS:
        raise ValueError(
            f"Unknown OpenAI formatter '{name}'. Available formatters: {list(OPENAI_FORMATTERS.keys())}"
        )
    return OPENAI_FORMATTERS[name]
