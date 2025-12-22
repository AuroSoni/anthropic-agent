"""Anthropic streaming formatters.

This module provides stream formatting functions for Anthropic's streaming API.
Formatters process streaming events and output them in different formats
(XML tags, raw JSON, etc.) to an async queue.
"""

import asyncio
from typing import Any, Literal, Callable, Awaitable
import json
import html
from anthropic.lib.streaming._beta_messages import BetaAsyncMessageStream
from anthropic.types.beta import BetaMessage

# Type alias for formatter names
FormatterType = Literal["xml", "raw"]


def escape_xml_attr(value: str) -> str:
    """Escape a string for safe use as an XML attribute value.
    
    Uses html.escape to convert special characters to XML entities.
    
    Args:
        value: The string to escape
        
    Returns:
        The escaped string safe for use in XML attributes
    """
    return html.escape(str(value), quote=True)


async def stream_to_aqueue(chunk: Any, queue: asyncio.Queue) -> None:
    """Send a chunk to an async queue.
    
    Simple helper function to put items onto an async queue,
    used for streaming output to consumers.
    
    Args:
        chunk: The data chunk to send (can be any type)
        queue: The async queue to put the chunk into
    """
    await queue.put(chunk)


async def xml_formatter(stream: BetaAsyncMessageStream, queue: asyncio.Queue) -> BetaMessage:
    """Format Anthropic streaming response with custom XML tags.
    
    This formatter wraps different content types in custom XML tags:
    - Thinking: <content-block-thinking>...</content-block-thinking>
    - Text: <content-block-text>...</content-block-text>
    - Tool calls: <content-block-tool_call id="..." name="..." arguments="...">
    - Tool results: <content-block-tool_result id="..." name="..."><![CDATA[...]]>
    - Errors: <content-block-error><![CDATA[...]]></content-block-error>
    
    Args:
        stream: Anthropic streaming context manager
        queue: Async queue to send formatted output chunks
        
    Returns:
        The final accumulated message from stream.get_final_message()
    """
    # Track content blocks by index
    content_blocks: dict[int, dict[str, Any]] = {}
    
    async for event in stream:
        event_type = event.type
        
        # Message-level events
        if event_type in ("message_start", "message_delta", "message_stop", "ping"):
            continue
        
        elif event_type == "error":
            error_data = getattr(event, 'error', event)
            error_content = json.dumps(error_data, default=str)
            await stream_to_aqueue(
                f'<content-block-error><![CDATA[{error_content}]]></content-block-error>',
                queue
            )
            continue
        
        # Content block start
        elif event_type == "content_block_start":
            idx = event.index
            block_type = event.content_block.type
            content_blocks[idx] = {
                "type": block_type,
                "is_open": False,
                "tool_data": None,
                "signature": None
            }
            
            if block_type == "thinking":
                await stream_to_aqueue("<content-block-thinking>", queue)
                content_blocks[idx]["is_open"] = True
            
            elif block_type == "text":
                await stream_to_aqueue("<content-block-text>", queue)
                content_blocks[idx]["is_open"] = True
            
            elif block_type == "tool_use":
                content_blocks[idx]["tool_data"] = {
                    "id": getattr(event.content_block, "id", ""),
                    "name": getattr(event.content_block, "name", ""),
                    "input": {}
                }
            
            elif block_type == "server_tool_use":
                content_blocks[idx]["tool_data"] = {
                    "id": getattr(event.content_block, "id", ""),
                    "name": getattr(event.content_block, "name", ""),
                    "input": {}
                }
            
            elif block_type.endswith("_tool_result"):
                content_blocks[idx]["tool_data"] = {
                    "tool_use_id": getattr(event.content_block, "tool_use_id", ""),
                    "content": getattr(event.content_block, "content", None)
                }
        
        # Content block deltas
        elif event_type == "content_block_delta":
            idx = event.index
            delta = event.delta
            block_info = content_blocks.get(idx)
            
            if not hasattr(delta, 'type'):
                continue
            
            delta_type = delta.type
            
            if delta_type == "text_delta":
                text_content = getattr(delta, 'text', '')
                if text_content:
                    await stream_to_aqueue(text_content, queue)
            
            elif delta_type == "thinking_delta":
                thinking_content = getattr(delta, 'thinking', '')
                if thinking_content:
                    await stream_to_aqueue(thinking_content, queue)
            
            elif delta_type == "signature_delta":
                if block_info:
                    block_info["signature"] = getattr(delta, 'signature', '')
            
            elif delta_type == "input_json_delta":
                if block_info and block_info["tool_data"]:
                    partial_json = getattr(delta, 'partial_json', '')
                    if "input_json" not in block_info["tool_data"]:
                        block_info["tool_data"]["input_json"] = ""
                    block_info["tool_data"]["input_json"] += partial_json
        
        # Content block stop
        elif event_type == "content_block_stop":
            idx = event.index
            block_info = content_blocks.get(idx)
            
            if not block_info:
                continue
            
            block_type = block_info["type"]
            
            # Close text/thinking blocks
            if block_info["is_open"]:
                if block_type == "thinking":
                    await stream_to_aqueue("</content-block-thinking>", queue)
                elif block_type == "text":
                    await stream_to_aqueue("</content-block-text>", queue)
                block_info["is_open"] = False
            
            # Stream complete tool call
            elif block_type == "tool_use" and block_info["tool_data"]:
                tool_data = block_info["tool_data"]
                try:
                    if "input_json" in tool_data:
                        tool_data["input"] = json.loads(tool_data["input_json"])
                        del tool_data["input_json"]
                except json.JSONDecodeError:
                    if "input_json" in tool_data:
                        tool_data["input"] = tool_data["input_json"]
                        del tool_data["input_json"]
                
                tool_id = escape_xml_attr(tool_data["id"])
                tool_name = escape_xml_attr(tool_data["name"])
                arguments = escape_xml_attr(json.dumps(tool_data["input"]))
                await stream_to_aqueue(
                    f'<content-block-tool_call id="{tool_id}" name="{tool_name}" arguments="{arguments}"></content-block-tool_call>',
                    queue
                )
            
            # Stream server tool call
            elif block_type == "server_tool_use" and block_info["tool_data"]:
                tool_data = block_info["tool_data"]
                try:
                    if "input_json" in tool_data:
                        tool_data["input"] = json.loads(tool_data["input_json"])
                        del tool_data["input_json"]
                except json.JSONDecodeError:
                    if "input_json" in tool_data:
                        tool_data["input"] = tool_data["input_json"]
                        del tool_data["input_json"]
                
                tool_id = escape_xml_attr(tool_data["id"])
                tool_name = escape_xml_attr(tool_data["name"])
                arguments = escape_xml_attr(json.dumps(tool_data["input"]))
                await stream_to_aqueue(
                    f'<content-block-tool_call id="{tool_id}" name="{tool_name}" arguments="{arguments}"></content-block-tool_call>',
                    queue
                )
            
            # Stream server tool result
            elif block_type.endswith("_tool_result") and block_info["tool_data"]:
                tool_data = block_info["tool_data"]
                tool_use_id = escape_xml_attr(tool_data["tool_use_id"])
                result_name = escape_xml_attr(block_type)
                
                content = tool_data.get("content")
                if content is None:
                    content_str = ""
                elif isinstance(content, str):
                    content_str = content
                else:
                    content_str = json.dumps(content, default=str)
                
                await stream_to_aqueue(
                    f'<content-block-tool_result id="{tool_use_id}" name="{result_name}"><![CDATA[{content_str}]]></content-block-tool_result>',
                    queue
                )
    
    # Safety: close any open blocks
    for idx, block_info in content_blocks.items():
        if block_info.get("is_open"):
            if block_info["type"] == "thinking":
                await stream_to_aqueue("</content-block-thinking>", queue)
            elif block_info["type"] == "text":
                await stream_to_aqueue("</content-block-text>", queue)
            block_info["is_open"] = False
    
    return await stream.get_final_message()


async def raw_formatter(stream: BetaAsyncMessageStream, queue: asyncio.Queue) -> BetaMessage:
    """Format Anthropic streaming response with raw JSON events.
    
    Streams events as JSON strings with minimal processing.
    
    Args:
        stream: Anthropic streaming context manager
        queue: Async queue to send formatted output chunks
        
    Returns:
        The final accumulated message from stream.get_final_message()
    """
    async for event in stream:
        if event.type in ['message_start', 'message_delta', 'message_stop', 
                          'content_block_start', 'content_block_delta', 'content_block_stop']:
            event_str = event.model_dump_json(warnings=False)
            await stream_to_aqueue(event_str, queue)

    return await stream.get_final_message()


# Formatter registry
FORMATTERS: dict[str, Callable[[BetaAsyncMessageStream, asyncio.Queue], Awaitable[BetaMessage]]] = {
    "xml": xml_formatter,
    "raw": raw_formatter,
}


def get_formatter(name: FormatterType) -> Callable[[BetaAsyncMessageStream, asyncio.Queue], Awaitable[BetaMessage]]:
    """Get a formatter function by name.
    
    Args:
        name: Formatter name ("xml" or "raw")
        
    Returns:
        The formatter function
        
    Raises:
        ValueError: If formatter name is not recognized
    """
    if name not in FORMATTERS:
        raise ValueError(
            f"Unknown formatter '{name}'. Available formatters: {list(FORMATTERS.keys())}"
        )
    return FORMATTERS[name]

