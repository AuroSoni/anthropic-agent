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
    - Tool calls: <content-block-tool_call id="..." name="..." arguments="..."></content-block-tool_call>
    - Tool results: <content-block-tool_result id="..." name="..."><![CDATA[...]]></content-block-tool_result>
    - Errors: <content-block-error><![CDATA[...]]></content-block-error>
    
    Supported event types (per Anthropic spec):
    - message_start, message_delta, message_stop (handled gracefully)
    - content_block_start, content_block_delta, content_block_stop
    - ping (ignored)
    - error (propagated)
    
    Supported delta types:
    - text_delta (streamed as received)
    - thinking_delta (streamed as received)
    - signature_delta (captured but not streamed)
    - input_json_delta (buffered until complete)
    
    Supported content block types:
    - text
    - thinking
    - tool_use (client tools)
    - server_tool_use (server-side tools)
    - *_tool_result (any server tool result type)
    
    Args:
        stream: Anthropic streaming context manager
        queue: Async queue to send formatted output chunks
        
    Returns:
        The final accumulated message from stream.get_final_message()
        
    Note:
        - Content blocks are tracked by index to properly open/close tags
        - Tool calls are buffered and streamed as attribute-based XML when complete
        - Tool results use CDATA sections for safe content wrapping
        - We process raw delta events, not simplified helper events to avoid duplication
    """
    # Track content blocks by index
    content_blocks = {}  # index -> {"type": str, "is_open": bool, "tool_data": dict, "signature": str}
    
    # Iterate through streaming events with async for to yield control to event loop
    async for event in stream:
        event_type = event.type
        
        # Message-level events (handle gracefully per Anthropic versioning policy)
        if event_type == "message_start":
            # Message initialization - contains empty message object
            continue
        
        elif event_type == "message_delta":
            # Top-level message changes (usage stats, stop_reason)
            # Not streamed to user, but available in final message
            continue
        
        elif event_type == "message_stop":
            # End of stream
            continue
        
        elif event_type == "ping":
            # Keep-alive ping - ignore
            continue
        
        elif event_type == "error":
            # Error event - propagate to user with CDATA wrapping
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
            
            # Open appropriate tag based on content type
            if block_type == "thinking":
                await stream_to_aqueue("<content-block-thinking>", queue)
                content_blocks[idx]["is_open"] = True
            
            elif block_type == "text":
                await stream_to_aqueue("<content-block-text>", queue)
                content_blocks[idx]["is_open"] = True
            
            elif block_type == "tool_use":
                # Start buffering client tool call data
                content_blocks[idx]["tool_data"] = {
                    "id": getattr(event.content_block, "id", ""),
                    "name": getattr(event.content_block, "name", ""),
                    "input": {}
                }
            
            elif block_type == "server_tool_use":
                # Start buffering server tool call data (e.g., web_search, code_execution)
                content_blocks[idx]["tool_data"] = {
                    "id": getattr(event.content_block, "id", ""),
                    "name": getattr(event.content_block, "name", ""),
                    "input": {}
                }
            
            elif block_type.endswith("_tool_result"):
                # Handle all server tool result types (*_tool_result)
                # e.g., web_search_tool_result, bash_code_execution_tool_result, 
                # text_editor_code_execution_tool_result, etc.
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
            
            # Text delta - stream immediately
            if delta_type == "text_delta":
                text_content = getattr(delta, 'text', '')
                if text_content:
                    await stream_to_aqueue(text_content, queue)
            
            # Thinking delta - stream immediately
            elif delta_type == "thinking_delta":
                thinking_content = getattr(delta, 'thinking', '')
                if thinking_content:
                    await stream_to_aqueue(thinking_content, queue)
            
            # Signature delta - buffer (sent before content_block_stop for thinking)
            elif delta_type == "signature_delta":
                if block_info:
                    block_info["signature"] = getattr(delta, 'signature', '')
            
            # Input JSON delta - buffer for tool_use/server_tool_use
            elif delta_type == "input_json_delta":
                if block_info and block_info["tool_data"]:
                    partial_json = getattr(delta, 'partial_json', '')
                    if "input_json" not in block_info["tool_data"]:
                        block_info["tool_data"]["input_json"] = ""
                    block_info["tool_data"]["input_json"] += partial_json
        
        # NOTE: We skip simplified event types (text, thinking, input_json, citation, signature)
        # because we already process them via content_block_delta events above.
        # Processing both would cause duplication in the output.
        
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
            
            # Stream complete tool call (client tools)
            elif block_type == "tool_use" and block_info["tool_data"]:
                tool_data = block_info["tool_data"]
                try:
                    # Parse the accumulated input JSON
                    if "input_json" in tool_data:
                        tool_data["input"] = json.loads(tool_data["input_json"])
                        del tool_data["input_json"]
                except json.JSONDecodeError:
                    # If parsing fails, keep as string
                    if "input_json" in tool_data:
                        tool_data["input"] = tool_data["input_json"]
                        del tool_data["input_json"]
                
                # Format as attribute-based XML
                tool_id = escape_xml_attr(tool_data["id"])
                tool_name = escape_xml_attr(tool_data["name"])
                arguments = escape_xml_attr(json.dumps(tool_data["input"]))
                await stream_to_aqueue(
                    f'<content-block-tool_call id="{tool_id}" name="{tool_name}" arguments="{arguments}"></content-block-tool_call>',
                    queue
                )
            
            # Stream complete server tool call (e.g., web_search, code_execution)
            elif block_type == "server_tool_use" and block_info["tool_data"]:
                tool_data = block_info["tool_data"]
                try:
                    # Parse the accumulated input JSON
                    if "input_json" in tool_data:
                        tool_data["input"] = json.loads(tool_data["input_json"])
                        del tool_data["input_json"]
                except json.JSONDecodeError:
                    if "input_json" in tool_data:
                        tool_data["input"] = tool_data["input_json"]
                        del tool_data["input_json"]
                
                # Format as attribute-based XML (same tag as client tools)
                tool_id = escape_xml_attr(tool_data["id"])
                tool_name = escape_xml_attr(tool_data["name"])
                arguments = escape_xml_attr(json.dumps(tool_data["input"]))
                await stream_to_aqueue(
                    f'<content-block-tool_call id="{tool_id}" name="{tool_name}" arguments="{arguments}"></content-block-tool_call>',
                    queue
                )
            
            # Stream server tool result (handles all *_tool_result types)
            elif block_type.endswith("_tool_result") and block_info["tool_data"]:
                tool_data = block_info["tool_data"]
                tool_use_id = escape_xml_attr(tool_data["tool_use_id"])
                result_name = escape_xml_attr(block_type)  # Use block type as name
                
                # Serialize content to string for CDATA
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
        
        # Handle unknown event types gracefully (per Anthropic versioning policy)
        else:
            # Unknown event type - log but don't fail
            pass
    
    # Safety: close any open blocks at the end
    for idx, block_info in content_blocks.items():
        if block_info.get("is_open"):
            if block_info["type"] == "thinking":
                await stream_to_aqueue("</content-block-thinking>", queue)
            elif block_info["type"] == "text":
                await stream_to_aqueue("</content-block-text>", queue)
            block_info["is_open"] = False
    
    # Get and return the final accumulated message (async method)
    return await stream.get_final_message()


async def raw_formatter(stream: BetaAsyncMessageStream, queue: asyncio.Queue) -> BetaMessage:
    """Format Anthropic streaming response with raw/minimal formatting.
    
    This is a placeholder formatter that streams content with minimal processing.
    Implement this formatter based on your specific needs.
    
    Args:
        stream: Anthropic streaming context manager
        queue: Async queue to send formatted output chunks
        
    Returns:
        The final accumulated message from stream.get_final_message()
    """
    
    async for event in stream:
        # event_str = json.dumps(event, default=str)
        if event.type in ['message_start', 'message_delta', 'message_stop', 'content_block_start', 'content_block_delta', 'content_block_stop']:
            event_str = event.model_dump_json(warnings=False)
            await stream_to_aqueue(event_str, queue)

    # Return final message
    return await stream.get_final_message()


# Formatter registry mapping string names to formatter functions
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

