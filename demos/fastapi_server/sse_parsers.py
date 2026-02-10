"""
SSE Stream Parsers for XML and Raw formatter outputs.

These parsers convert SSE log files into structured data for testing and debugging.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Any


# Type definitions for content block types
ContentBlockType = Literal[
    "meta_init",
    "thinking",
    "text",
    "tool_call",
    "server_tool_call",
    "server_tool_result",
    "tool_result",
    "error",
    "meta_files",
    "embedded_tag",
]


@dataclass
class ContentBlock:
    """Represents a parsed content block from the stream."""
    
    type: ContentBlockType
    content: str | dict | None
    id: str | None = None
    name: str | None = None  # For tool calls
    tag_name: str | None = None  # For embedded tags
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedSSEStream:
    """Result of parsing an SSE stream."""
    
    format: Literal["xml", "raw", "json"]
    blocks: list[ContentBlock]
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_content: str = ""


def parse_xml_stream(
    sse_content: str,
    embedded_tags: list[str] | None = None,
) -> ParsedSSEStream:
    """
    Parse XML-formatted SSE stream content.
    
    Args:
        sse_content: Raw SSE stream content (from log file)
        embedded_tags: Optional list of tag names to extract from text content
                       (e.g., ["chart", "code"])
    
    Returns:
        ParsedSSEStream with extracted content blocks
    """
    blocks: list[ContentBlock] = []
    metadata: dict[str, Any] = {}
    
    # Extract data lines and combine content
    lines = sse_content.strip().split("\n")
    data_chunks: list[str] = []
    
    for line in lines:
        if line.startswith("data: "):
            data_chunks.append(line[6:])  # Remove "data: " prefix
    
    # Join all data chunks to reconstruct the full stream
    full_content = "".join(data_chunks)
    
    # Parse meta_init if present
    meta_init_match = re.search(
        r'<meta_init data="([^"]*)"></meta_init>',
        full_content,
    )
    if meta_init_match:
        meta_data_escaped = meta_init_match.group(1)
        # Decode HTML entities
        meta_data_str = _decode_html_entities(meta_data_escaped)
        try:
            meta_dict = json.loads(meta_data_str)
            metadata = meta_dict
            blocks.append(ContentBlock(
                type="meta_init",
                content=meta_dict,
            ))
        except json.JSONDecodeError:
            blocks.append(ContentBlock(
                type="meta_init",
                content=meta_data_str,
            ))
    
    # Parse content blocks
    # Pattern for content-block-* tags
    content_block_pattern = re.compile(
        r'<content-block-(\w+)([^>]*)>(.*?)</content-block-\1>',
        re.DOTALL,
    )
    
    for match in content_block_pattern.finditer(full_content):
        block_type = match.group(1)
        attributes_str = match.group(2)
        content = match.group(3)
        
        # Parse attributes
        attributes = _parse_attributes(attributes_str)
        
        # Map block type to ContentBlockType
        type_mapping: dict[str, ContentBlockType] = {
            "thinking": "thinking",
            "text": "text",
            "tool_call": "tool_call",
            "server_tool_call": "server_tool_call",
            "server_tool_result": "server_tool_result",
            "tool_result": "tool_result",
            "error": "error",
        }
        
        mapped_type = type_mapping.get(block_type, "text")
        
        # Handle different block types
        if mapped_type == "server_tool_call":
            # Parse CDATA or direct content
            parsed_content = _extract_cdata_or_content(content)
            blocks.append(ContentBlock(
                type=mapped_type,
                content=parsed_content,
                id=attributes.get("id"),
                name=attributes.get("name"),
                attributes=attributes,
            ))
        elif mapped_type == "server_tool_result":
            # Parse CDATA content
            parsed_content = _extract_cdata_or_content(content)
            blocks.append(ContentBlock(
                type=mapped_type,
                content=parsed_content,
                id=attributes.get("id"),
                name=attributes.get("name"),
                attributes=attributes,
            ))
        elif mapped_type == "text":
            # If embedded_tags specified, split text by those tags
            if embedded_tags:
                text_blocks = _split_text_by_embedded_tags(content, embedded_tags)
                blocks.extend(text_blocks)
            else:
                blocks.append(ContentBlock(
                    type=mapped_type,
                    content=content,
                    attributes=attributes,
                ))
        else:
            blocks.append(ContentBlock(
                type=mapped_type,
                content=content,
                attributes=attributes,
            ))
    
    return ParsedSSEStream(
        format="xml",
        blocks=blocks,
        metadata=metadata,
        raw_content=sse_content,
    )


def _split_text_by_embedded_tags(
    text: str,
    embedded_tags: list[str],
) -> list[ContentBlock]:
    """Split text content by embedded tags, creating separate blocks."""
    blocks: list[ContentBlock] = []
    
    # Build pattern to match any of the embedded tags
    tags_pattern = "|".join(re.escape(tag) for tag in embedded_tags)
    pattern = re.compile(
        rf'<({tags_pattern})([^>]*)>(.*?)</\1>',
        re.DOTALL,
    )
    
    last_end = 0
    for match in pattern.finditer(text):
        # Add text before the tag
        if match.start() > last_end:
            before_text = text[last_end:match.start()]
            if before_text.strip():
                blocks.append(ContentBlock(
                    type="text",
                    content=before_text,
                ))
        
        # Add the embedded tag as a separate block
        tag_name = match.group(1)
        tag_attrs = match.group(2)
        tag_content = match.group(3)
        
        blocks.append(ContentBlock(
            type="embedded_tag",
            content=tag_content,
            tag_name=tag_name,
            attributes=_parse_attributes(tag_attrs),
        ))
        
        last_end = match.end()
    
    # Add remaining text after last tag
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining.strip():
            blocks.append(ContentBlock(
                type="text",
                content=remaining,
            ))
    
    # If no embedded tags found, return text as single block
    if not blocks and text.strip():
        blocks.append(ContentBlock(
            type="text",
            content=text,
        ))
    
    return blocks


def _decode_html_entities(text: str) -> str:
    """Decode HTML entities in text."""
    replacements = {
        "&quot;": '"',
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&#x27;": "'",
        "&#39;": "'",
        "&apos;": "'",
    }
    for entity, char in replacements.items():
        text = text.replace(entity, char)
    return text


def _parse_attributes(attr_str: str) -> dict[str, Any]:
    """Parse XML attributes from a string."""
    attributes: dict[str, Any] = {}
    if not attr_str:
        return attributes
    
    # Pattern for attribute="value" or attribute='{json}'
    attr_pattern = re.compile(r'(\w+)="([^"]*)"')
    for match in attr_pattern.finditer(attr_str):
        key = match.group(1)
        value = _decode_html_entities(match.group(2))
        
        # Try to parse as JSON if it looks like JSON
        if value.startswith("{") or value.startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        
        attributes[key] = value
    
    return attributes


def _extract_cdata_or_content(content: str) -> str | list | dict:
    """Extract content from CDATA section or return as-is."""
    # Check for CDATA
    cdata_match = re.search(r'<!\[CDATA\[(.*?)\]\]>', content, re.DOTALL)
    if cdata_match:
        cdata_content = cdata_match.group(1)
        # Try to parse as JSON (tool results are often JSON)
        try:
            return json.loads(cdata_content)
        except json.JSONDecodeError:
            return cdata_content
    return content


def parse_raw_stream(sse_content: str) -> ParsedSSEStream:
    """
    Parse raw/JSON-formatted SSE stream content.
    
    Args:
        sse_content: Raw SSE stream content (from log file)
    
    Returns:
        ParsedSSEStream with extracted content blocks
    """
    blocks: list[ContentBlock] = []
    metadata: dict[str, Any] = {}
    
    lines = sse_content.strip().split("\n")
    
    # Track accumulated content per block index
    block_contents: dict[int, dict[str, Any]] = {}
    
    for line in lines:
        if not line.startswith("data: "):
            continue
        
        data = line[6:]  # Remove "data: " prefix
        
        # Check for meta_init (XML tag in raw format)
        meta_init_match = re.search(
            r'<meta_init data="([^"]*)"></meta_init>',
            data,
        )
        if meta_init_match:
            meta_data_str = _decode_html_entities(meta_init_match.group(1))
            try:
                metadata = json.loads(meta_data_str)
                blocks.append(ContentBlock(
                    type="meta_init",
                    content=metadata,
                ))
            except json.JSONDecodeError:
                blocks.append(ContentBlock(
                    type="meta_init",
                    content=meta_data_str,
                ))
            continue
        
        # Try to parse as JSON
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        
        event_type = event.get("type", "")
        
        # Handle different event types
        if event_type == "message_start":
            # Extract message metadata
            message = event.get("message", {})
            metadata.update({
                "message_id": message.get("id"),
                "model": message.get("model"),
                "usage": message.get("usage"),
            })
        
        elif event_type == "content_block_start":
            index = event.get("index", 0)
            content_block = event.get("content_block", {})
            block_type = content_block.get("type", "")
            
            block_contents[index] = {
                "type": block_type,
                "content": "",
                "id": content_block.get("id"),
                "name": content_block.get("name"),
                "attributes": content_block,
            }
        
        elif event_type == "content_block_delta":
            index = event.get("index", 0)
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")
            
            if index in block_contents:
                if delta_type == "thinking_delta":
                    block_contents[index]["content"] += delta.get("thinking", "")
                elif delta_type == "text_delta":
                    block_contents[index]["content"] += delta.get("text", "")
                elif delta_type == "input_json_delta":
                    block_contents[index]["content"] += delta.get("partial_json", "")
        
        elif event_type == "content_block_stop":
            index = event.get("index", 0)
            if index in block_contents:
                block_data = block_contents[index]
                block_type = block_data["type"]
                
                # Map raw types to ContentBlockType
                type_mapping: dict[str, ContentBlockType] = {
                    "thinking": "thinking",
                    "text": "text",
                    "tool_use": "tool_call",
                    "server_tool_use": "server_tool_call",
                    "tool_result": "tool_result",
                    "web_search_tool_result": "server_tool_result",
                }
                
                mapped_type = type_mapping.get(block_type, "text")
                
                # Try to parse content as JSON for tool calls
                content = block_data["content"]
                if mapped_type in ("tool_call", "server_tool_call"):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass
                
                # Check for completed content in content_block
                stop_content = event.get("content_block", {})
                if stop_content.get("thinking"):
                    content = stop_content.get("thinking")
                if stop_content.get("input"):
                    content = stop_content.get("input")
                
                blocks.append(ContentBlock(
                    type=mapped_type,
                    content=content,
                    id=block_data.get("id"),
                    name=block_data.get("name"),
                    attributes=block_data.get("attributes", {}),
                ))
    
    return ParsedSSEStream(
        format="raw",
        blocks=blocks,
        metadata=metadata,
        raw_content=sse_content,
    )


# ============================================================================
# Utility Functions
# ============================================================================

def detect_format(sse_content: str) -> Literal["xml", "raw", "json"]:
    """Auto-detect the format of SSE content."""
    # Check first few data lines
    lines = sse_content.strip().split("\n")
    for line in lines[:10]:
        if line.startswith("data: "):
            data = line[6:]
            # XML format uses content-block-* tags
            if "<content-block-" in data:
                return "xml"
            # JSON envelope format: has "agent" and "final" fields
            if data.startswith("{") and '"agent"' in data and '"final"' in data:
                return "json"
            # Raw format uses JSON events (Anthropic event types)
            if data.startswith("{") and '"type"' in data:
                return "raw"

    # Check for meta_init format indicator
    if '"format": "xml"' in sse_content or '"format": \\"xml\\"' in sse_content:
        return "xml"
    if '"format": "json"' in sse_content or '"format": \\"json\\"' in sse_content:
        return "json"
    if '"format": "raw"' in sse_content or '"format": \\"raw\\"' in sse_content:
        return "raw"

    return "xml"  # Default to xml


def parse_sse_stream(
    sse_content: str,
    embedded_tags: list[str] | None = None,
) -> ParsedSSEStream:
    """
    Parse SSE stream content, auto-detecting format.
    
    Args:
        sse_content: Raw SSE stream content
        embedded_tags: Optional list of embedded tags for XML format
    
    Returns:
        ParsedSSEStream with extracted content blocks
    """
    fmt = detect_format(sse_content)
    if fmt == "xml":
        return parse_xml_stream(sse_content, embedded_tags)
    if fmt == "json":
        return parse_json_stream(sse_content)
    return parse_raw_stream(sse_content)


def get_text_content(parsed: ParsedSSEStream) -> str:
    """Extract merged text content from all text blocks."""
    text_parts: list[str] = []
    for block in parsed.blocks:
        if block.type == "text" and block.content:
            text_parts.append(str(block.content))
        elif block.type == "embedded_tag" and block.content:
            text_parts.append(str(block.content))
    return "".join(text_parts)


def get_tool_calls(parsed: ParsedSSEStream) -> list[ContentBlock]:
    """Extract tool call blocks from parsed stream."""
    return [
        block for block in parsed.blocks
        if block.type in ("tool_call", "server_tool_call")
    ]


def parse_json_stream(sse_content: str) -> ParsedSSEStream:
    """Parse JSON-envelope-formatted SSE stream content.

    Each ``data:`` line is a self-contained JSON object with fields
    ``type``, ``agent``, ``final``, ``delta`` and optional extra fields.
    Messages arrive in SSE order; block boundaries are determined by the
    ``final`` flag.  When ``final`` is ``true``, the current block for
    that ``(agent, type)`` is closed and the next message of the same
    type opens a new block.

    Args:
        sse_content: Raw SSE stream content (from a log file).

    Returns:
        ParsedSSEStream with extracted content blocks.
    """
    blocks: list[ContentBlock] = []
    metadata: dict[str, Any] = {}

    # Accumulator keyed by a monotonic string key → {type, delta, extras}
    accum: dict[str, dict[str, Any]] = {}
    # Track insertion order so blocks appear in stream order
    order: list[str] = []
    # Current open (non-final) block key per (agent, type)
    current_block_key: dict[str, str] = {}
    block_counter = 0

    lines = sse_content.strip().split("\n")
    for line in lines:
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        try:
            msg = json.loads(payload)
        except json.JSONDecodeError:
            continue

        agent = msg.get("agent", "")
        msg_type = msg.get("type", "")
        is_final = msg.get("final", False)
        agent_type_key = f"{agent}:{msg_type}"

        key = current_block_key.get(agent_type_key)

        # If there is no open block for this (agent, type), create one
        if key is None:
            key = f"blk_{block_counter}"
            block_counter += 1
            current_block_key[agent_type_key] = key
            accum[key] = {
                "type": msg_type,
                "delta": "",
                "extras": {},
            }
            order.append(key)

        accum[key]["delta"] += msg.get("delta", "")

        # Capture extra fields (id, name, citation fields, etc.)
        for extra_key in ("id", "name", "citation_type",
                          "document_index", "document_title",
                          "start_char_index", "end_char_index",
                          "start_page_number", "end_page_number",
                          "url", "title", "src", "media_type"):
            if extra_key in msg:
                accum[key]["extras"][extra_key] = msg[extra_key]

        # When final, close the tracker so the next message of the
        # same type opens a fresh block.
        if is_final:
            current_block_key.pop(agent_type_key, None)

    # Convert accumulated data into ContentBlock list
    for key in order:
        entry = accum[key]
        msg_type = entry["type"]
        delta = entry["delta"]
        extras = entry["extras"]

        if msg_type == "meta_init":
            try:
                meta_dict = json.loads(delta)
                metadata = meta_dict
                blocks.append(ContentBlock(type="meta_init", content=meta_dict))
            except json.JSONDecodeError:
                blocks.append(ContentBlock(type="meta_init", content=delta))
        elif msg_type == "meta_final":
            try:
                blocks.append(ContentBlock(type="meta_init", content=json.loads(delta)))
            except json.JSONDecodeError:
                pass
        elif msg_type in ("thinking", "text", "error"):
            mapped: ContentBlockType = msg_type  # type: ignore[assignment]
            blocks.append(ContentBlock(type=mapped, content=delta))
        elif msg_type in ("tool_call", "server_tool_call"):
            mapped = msg_type  # type: ignore[assignment]
            blocks.append(ContentBlock(
                type=mapped,
                content=delta,
                id=extras.get("id"),
                name=extras.get("name"),
                attributes=extras,
            ))
        elif msg_type in ("tool_result", "server_tool_result"):
            mapped = msg_type  # type: ignore[assignment]
            blocks.append(ContentBlock(
                type=mapped,
                content=delta,
                id=extras.get("id"),
                name=extras.get("name"),
                attributes=extras,
            ))
        elif msg_type == "citation":
            blocks.append(ContentBlock(
                type="text",
                content=delta,
                attributes=extras,
            ))
        elif msg_type == "awaiting_frontend_tools":
            try:
                blocks.append(ContentBlock(type="tool_call", content=json.loads(delta)))
            except json.JSONDecodeError:
                blocks.append(ContentBlock(type="tool_call", content=delta))
        elif msg_type == "meta_files":
            blocks.append(ContentBlock(type="meta_files", content=delta))
        else:
            blocks.append(ContentBlock(type="text", content=delta))

    return ParsedSSEStream(
        format="json",
        blocks=blocks,
        metadata=metadata,
        raw_content=sse_content,
    )


def get_blocks_by_type(
    parsed: ParsedSSEStream,
    block_type: ContentBlockType,
) -> list[ContentBlock]:
    """Filter blocks by type."""
    return [block for block in parsed.blocks if block.type == block_type]


def get_embedded_tags(
    parsed: ParsedSSEStream,
    tag_name: str | None = None,
) -> list[ContentBlock]:
    """Get embedded tag blocks, optionally filtered by tag name."""
    blocks = [block for block in parsed.blocks if block.type == "embedded_tag"]
    if tag_name:
        blocks = [block for block in blocks if block.tag_name == tag_name]
    return blocks

