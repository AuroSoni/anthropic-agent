"""Anthropic message formatter — translates canonical types to/from Anthropic wire format.

Implements the ``MessageFormatter`` ABC from ``agent_base.core.messages``.
This is a pure translator: no HTTP calls, no retries, no side effects.

Responsibilities:
    - ``format_messages()`` — canonical ``Message`` list → Anthropic API request dict
    - ``parse_response()`` — Anthropic ``BetaMessage`` → canonical ``Message``
    - ``format_tool_schemas()`` — canonical tool schema dicts → Anthropic format (pass-through)
    - Cache control injection (``_apply_cache_control``)
"""
from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from agent_base.core.messages import MessageFormatter, Message, Usage
from agent_base.core.types import (
    ContentBlock,
    ContentBlockType,
    Role,
    TextContent,
    ThinkingContent,
    ImageContent,
    DocumentContent,
    AttachmentContent,
    ToolUseContent,
    ServerToolUseContent,
    MCPToolUseContent,
    ToolResultContent,
    ServerToolResultContent,
    MCPToolResultContent,
    ErrorContent,
)
from agent_base.logging import get_logger

if TYPE_CHECKING:
    from agent_base.providers.anthropic.anthropic_agent import AnthropicLLMConfig

logger = get_logger(__name__)

# Cache control constants
MAX_CACHE_BLOCKS = 4
MIN_CACHE_TOKENS_SONNET = 1024
MIN_CACHE_TOKENS_HAIKU = 2048
DEFAULT_MAX_TOKENS = 16384


# ---------------------------------------------------------------------------
# Content block conversion: canonical → Anthropic wire format
# ---------------------------------------------------------------------------


def _content_block_to_anthropic(block: ContentBlock) -> dict[str, Any] | None:
    """Convert a single canonical ContentBlock to Anthropic wire-format dict.

    Returns ``None`` for blocks that have no Anthropic representation
    (e.g. citations, which are not sent to the API).
    """
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}

    if isinstance(block, ThinkingContent):
        d: dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
        if block.signature:
            d["signature"] = block.signature
        return d

    if isinstance(block, ImageContent):
        return {
            "type": "image",
            "source": {
                "type": block.source_type or "base64",
                "media_type": block.media_type,
                "data": block.data,
            },
        }

    if isinstance(block, DocumentContent):
        return {
            "type": "document",
            "source": {
                "type": block.source_type or "base64",
                "media_type": block.media_type,
                "data": block.data,
            },
        }

    if isinstance(block, ToolUseContent):
        return {
            "type": "tool_use",
            "id": block.tool_id,
            "name": block.tool_name,
            "input": block.tool_input,
        }

    if isinstance(block, ServerToolUseContent):
        return {
            "type": "server_tool_use",
            "id": block.tool_id,
            "name": block.tool_name,
            "input": block.tool_input,
        }

    if isinstance(block, MCPToolUseContent):
        return {
            "type": "tool_use",
            "id": block.tool_id,
            "name": block.tool_name,
            "input": block.tool_input,
        }

    if isinstance(block, (ToolResultContent, ServerToolResultContent, MCPToolResultContent)):
        content: list[dict[str, Any]] = []
        if isinstance(block.tool_result, str):
            if block.tool_result:
                content.append({"type": "text", "text": block.tool_result})
        elif isinstance(block.tool_result, list):
            for inner in block.tool_result:
                converted = _content_block_to_anthropic(inner)
                if converted:
                    content.append(converted)
        d = {
            "type": "tool_result",
            "tool_use_id": block.tool_id,
            "content": content,
        }
        if block.is_error:
            d["is_error"] = True
        return d

    if isinstance(block, ErrorContent):
        return {"type": "text", "text": f"Error: {block.error_message}"}

    # Citations and other blocks are not sent to the Anthropic API.
    return None


def _format_message(msg: Message) -> dict[str, Any]:
    """Convert a canonical Message to Anthropic wire-format dict."""
    content_blocks: list[dict[str, Any]] = []

    # Tool result messages need special handling — each tool result is
    # a top-level entry in the content list, not nested.
    has_tool_results = any(
        isinstance(b, (ToolResultContent, ServerToolResultContent, MCPToolResultContent))
        for b in msg.content
    )

    for block in msg.content:
        converted = _content_block_to_anthropic(block)
        if converted is not None:
            content_blocks.append(converted)

    return {
        "role": msg.role.value,
        "content": content_blocks,
    }


# ---------------------------------------------------------------------------
# Response parsing: Anthropic wire format → canonical
# ---------------------------------------------------------------------------


def _parse_content_block(raw_block: Any) -> ContentBlock | None:
    """Convert an Anthropic response content block to a canonical ContentBlock."""
    block_type = getattr(raw_block, "type", None)

    if block_type == "text":
        return TextContent(text=getattr(raw_block, "text", ""))

    if block_type == "thinking":
        return ThinkingContent(
            thinking=getattr(raw_block, "thinking", ""),
            signature=getattr(raw_block, "signature", None),
        )

    if block_type == "tool_use":
        return ToolUseContent(
            tool_name=getattr(raw_block, "name", ""),
            tool_id=getattr(raw_block, "id", ""),
            tool_input=getattr(raw_block, "input", {}),
            raw=raw_block,
        )

    if block_type == "server_tool_use":
        return ServerToolUseContent(
            tool_name=getattr(raw_block, "name", ""),
            tool_id=getattr(raw_block, "id", ""),
            tool_input=getattr(raw_block, "input", {}),
            raw=raw_block,
        )

    if block_type and block_type.endswith("_tool_result"):
        content = getattr(raw_block, "content", "")
        return ServerToolResultContent(
            tool_name=block_type,
            tool_id=getattr(raw_block, "tool_use_id", "") or getattr(raw_block, "id", "unknown"),
            tool_result=content if isinstance(content, str) else str(content),
            raw=raw_block,
        )

    logger.debug("unknown_anthropic_block_type", block_type=block_type)
    return None


# ---------------------------------------------------------------------------
# Cache control
# ---------------------------------------------------------------------------


def _apply_cache_control(
    messages: list[dict[str, Any]],
    system: str | None,
    model: str,
    enable: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | str | None]:
    """Apply Anthropic cache_control to message content blocks.

    Anthropic limits cache_control to 4 blocks maximum. This function
    applies caching in priority order:
    1. System prompt (if large enough)
    2. Document/image blocks
    3. Large text blocks (sorted by size descending)
    4. Recent message blocks (fallback)

    Args:
        messages: Wire-format message dicts.
        system: System prompt string.
        model: Model identifier (for min token threshold).
        enable: Whether cache control is enabled.

    Returns:
        Tuple of (processed_messages, processed_system).
    """
    if not enable:
        return messages, system

    min_tokens = (
        MIN_CACHE_TOKENS_HAIKU
        if "haiku" in model.lower()
        else MIN_CACHE_TOKENS_SONNET
    )
    remaining_slots = MAX_CACHE_BLOCKS
    supported_types = {"text", "image", "document"}
    blocks_to_cache: list[tuple[int, int]] = []

    # Priority 1: System prompt
    processed_system: list[dict[str, Any]] | str | None = system
    if system and remaining_slots > 0:
        system_tokens = len(system) // 4
        if system_tokens >= min_tokens:
            processed_system = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
            remaining_slots -= 1

    # Priority 2: Document/image blocks
    doc_image_blocks: list[tuple[int, int]] = []
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block_idx, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("document", "image"):
                doc_image_blocks.append((msg_idx, block_idx))

    for loc in doc_image_blocks:
        if remaining_slots <= 0:
            break
        blocks_to_cache.append(loc)
        remaining_slots -= 1

    # Priority 3: Large text blocks (sorted by size descending)
    if remaining_slots > 0:
        large_text_blocks: list[tuple[int, int, int]] = []
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block_idx, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and "text" in block:
                    text_len = len(block["text"])
                    if text_len // 4 >= min_tokens:
                        if (msg_idx, block_idx) not in blocks_to_cache:
                            large_text_blocks.append((msg_idx, block_idx, text_len))

        large_text_blocks.sort(key=lambda x: x[2], reverse=True)
        for msg_idx, block_idx, _ in large_text_blocks:
            if remaining_slots <= 0:
                break
            blocks_to_cache.append((msg_idx, block_idx))
            remaining_slots -= 1

    # Priority 4: Recent message blocks (fallback)
    if remaining_slots > 0:
        for msg_idx in range(len(messages) - 1, -1, -1):
            if remaining_slots <= 0:
                break
            msg = messages[msg_idx]
            role = msg.get("role")
            content = msg.get("content", [])
            if role not in ("user", "assistant") or not isinstance(content, list):
                continue
            for block_idx in range(len(content) - 1, -1, -1):
                if remaining_slots <= 0:
                    break
                block = content[block_idx]
                if not isinstance(block, dict):
                    continue
                if block.get("type") in supported_types:
                    if (msg_idx, block_idx) not in blocks_to_cache:
                        blocks_to_cache.append((msg_idx, block_idx))
                        remaining_slots -= 1

    # Build result with cache_control injected
    blocks_to_cache_set = set(blocks_to_cache)
    result: list[dict[str, Any]] = []
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content", [])
        if not isinstance(content, list):
            result.append(msg)
            continue
        new_msg = {"role": msg.get("role"), "content": []}
        for block_idx, block in enumerate(content):
            new_block = dict(block) if isinstance(block, dict) else block
            if (msg_idx, block_idx) in blocks_to_cache_set:
                new_block["cache_control"] = {"type": "ephemeral"}
            new_msg["content"].append(new_block)
        result.append(new_msg)

    return result, processed_system


# ---------------------------------------------------------------------------
# AnthropicMessageFormatter
# ---------------------------------------------------------------------------


class AnthropicMessageFormatter(MessageFormatter):
    """Translates canonical Messages to/from Anthropic API wire format.

    Handles:
    - Content block conversion (text, thinking, images, documents, tool use/result)
    - Request dict assembly (model, max_tokens, system, thinking, tools, betas, container)
    - Cache control injection (up to 4 blocks per Anthropic limits)
    - Response parsing (BetaMessage → canonical Message)
    - Tool schema formatting (pass-through for Anthropic)
    """

    def format_messages(
        self, messages: List[Message], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a complete Anthropic API request dict.

        Args:
            messages: Canonical message history.
            params: Request parameters dict with keys:
                - ``system_prompt`` (str | None): System prompt text.
                - ``llm_config`` (AnthropicLLMConfig): Provider-specific config.
                - ``model`` (str): Model identifier.
                - ``tool_schemas`` (list[dict]): Already-formatted tool schemas.
                - ``enable_cache_control`` (bool): Whether to apply cache control.

        Returns:
            Dict ready for ``client.beta.messages.create(**result)`` or
            ``client.beta.messages.stream(**result)``.
        """
        system_prompt: str | None = params.get("system_prompt")
        llm_config = params.get("llm_config")
        model: str = params.get("model", "")
        tool_schemas: list[dict[str, Any]] = params.get("tool_schemas", [])
        enable_cache_control: bool = params.get("enable_cache_control", True)

        # Convert canonical Messages to wire-format dicts
        wire_messages = [_format_message(msg) for msg in messages]

        # Apply cache control
        wire_messages, processed_system = _apply_cache_control(
            wire_messages, system_prompt, model, enable_cache_control
        )

        # Build request dict
        max_tokens = DEFAULT_MAX_TOKENS
        if llm_config and hasattr(llm_config, "max_tokens") and llm_config.max_tokens:
            max_tokens = llm_config.max_tokens

        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": wire_messages,
        }

        if processed_system:
            request_params["system"] = processed_system

        # Extended thinking
        if llm_config and hasattr(llm_config, "thinking_tokens"):
            thinking_tokens = llm_config.thinking_tokens
            if thinking_tokens and thinking_tokens > 0:
                request_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_tokens,
                }

        # Tools: client tools + server tools
        combined_tools: list[dict[str, Any]] = []
        if tool_schemas:
            combined_tools.extend(tool_schemas)
        if llm_config and hasattr(llm_config, "server_tools") and llm_config.server_tools:
            combined_tools.extend(llm_config.server_tools)
        if combined_tools:
            request_params["tools"] = combined_tools

        # Beta headers
        if llm_config and hasattr(llm_config, "beta_headers") and llm_config.beta_headers:
            request_params["betas"] = llm_config.beta_headers

        # Container (supports Skills + multi-turn container ID)
        container: dict[str, Any] = {}
        if llm_config and hasattr(llm_config, "container_id") and llm_config.container_id:
            container["id"] = llm_config.container_id
        if llm_config and hasattr(llm_config, "skills") and llm_config.skills:
            container["skills"] = llm_config.skills
        if container:
            request_params["container"] = container

        return request_params

    def parse_response(self, raw_response: Any) -> Message:
        """Convert an Anthropic BetaMessage to a canonical Message.

        Args:
            raw_response: Anthropic ``BetaMessage`` object.

        Returns:
            Canonical ``Message`` with content blocks, usage, stop_reason.
        """
        content_blocks: list[ContentBlock] = []
        for raw_block in getattr(raw_response, "content", []):
            parsed = _parse_content_block(raw_block)
            if parsed is not None:
                content_blocks.append(parsed)

        # Parse usage
        raw_usage = getattr(raw_response, "usage", None)
        usage = None
        if raw_usage:
            usage = Usage(
                input_tokens=getattr(raw_usage, "input_tokens", 0),
                output_tokens=getattr(raw_usage, "output_tokens", 0),
                cache_write_tokens=getattr(raw_usage, "cache_creation_input_tokens", None),
                cache_read_tokens=getattr(raw_usage, "cache_read_input_tokens", None),
                raw_usage={
                    k: v
                    for k, v in (raw_usage.model_dump() if hasattr(raw_usage, "model_dump") else {}).items()
                    if v is not None
                },
            )

        return Message(
            role=Role.ASSISTANT,
            content=content_blocks,
            stop_reason=getattr(raw_response, "stop_reason", None),
            usage=usage,
            provider="anthropic",
            model=getattr(raw_response, "model", ""),
        )

    def format_tool_schemas(
        self, schemas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert canonical tool schemas to Anthropic format.

        For Anthropic, the canonical format (``name``, ``description``,
        ``input_schema``) is the same as the wire format. Pass-through.

        Args:
            schemas: Canonical tool schema dicts.

        Returns:
            Same schemas (Anthropic format is identical).
        """
        return schemas
