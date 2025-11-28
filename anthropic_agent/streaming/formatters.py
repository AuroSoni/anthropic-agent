import asyncio
from typing import Any, Literal, Callable, Awaitable
import json
from anthropic.lib.streaming._beta_messages import BetaAsyncMessageStream
from anthropic.types.beta import BetaMessage

# Type alias for formatter names
FormatterType = Literal["xml", "raw"]


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
    - Thinking content wrapped in <content-block:thinking>...</content-block:thinking> tags
    - Text content wrapped in <content-block:text>...</content-block:text> tags
    - Tool calls formatted as <content-block:tool_call>...</content-block:tool_call> with complete JSON
    - Server tool calls formatted as <content-block:server_tool_call>...</content-block:server_tool_call>
    - Server tool results formatted as <content-block:server_tool_result>...</content-block:server_tool_result>
    - File IDs (from code execution) wrapped in <content-block:meta_files>...</content-block:meta_files> at end
    
    Supported event types (per Anthropic spec):
    - message_start, message_delta, message_stop (handled gracefully)
    - content_block_start, content_block_delta, content_block_stop
    - ping (ignored)
    - error (propagated)
    
    Supported delta types:
    - text_delta (streamed as received)
    - thinking_delta (streamed as received)
    - signature_delta (captured but not streamed separately)
    - input_json_delta (buffered until complete)
    
    Supported content block types:
    - text
    - thinking
    - tool_use (client tools)
    - server_tool_use (server-side tools like web_search)
    - web_search_tool_result
    - bash_code_execution_tool_result (file IDs extracted)
    
    Args:
        stream: Anthropic streaming context manager
        queue: Async queue to send formatted output chunks
        
    Returns:
        The final accumulated message from stream.get_final_message()
        
    Note:
        - Content blocks are tracked by index to properly open/close tags
        - Tool calls are buffered and only streamed when complete
        - We process raw delta events, not simplified helper events (text, thinking, etc.)
          to avoid duplication
        - If the agent creates files during code execution, their file IDs are extracted
          and streamed as a final meta tag with JSON format: {"file_ids": ["id1", "id2", ...]}
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
            # Error event - propagate to user
            error_data = getattr(event, 'error', event)
            await stream_to_aqueue(
                f'\n<content-block:error>{json.dumps(error_data, default=str)}</content-block:error>\n',
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
                await stream_to_aqueue("<content-block:thinking>", queue)
                content_blocks[idx]["is_open"] = True
            
            elif block_type == "text":
                await stream_to_aqueue("<content-block:text>", queue)
                content_blocks[idx]["is_open"] = True
            
            elif block_type == "tool_use":
                # Start buffering client tool call data
                content_blocks[idx]["tool_data"] = {
                    "type": "tool_use",
                    "id": getattr(event.content_block, "id", ""),
                    "name": getattr(event.content_block, "name", ""),
                    "input": {}
                }
            
            elif block_type == "server_tool_use":
                # Start buffering server tool call data (e.g., web_search)
                content_blocks[idx]["tool_data"] = {
                    "type": "server_tool_use",
                    "id": getattr(event.content_block, "id", ""),
                    "name": getattr(event.content_block, "name", ""),
                    "input": {}
                }
            
            elif block_type == "web_search_tool_result":
                # Server tool result - buffer the content
                content_blocks[idx]["tool_data"] = {
                    "type": "web_search_tool_result",
                    "tool_use_id": getattr(event.content_block, "tool_use_id", ""),
                    "content": getattr(event.content_block, "content", [])
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
                if block_type in ("thinking", "text"):
                    await stream_to_aqueue("\n", queue)
                
                if block_type == "thinking":
                    await stream_to_aqueue("</content-block:thinking>", queue)
                elif block_type == "text":
                    await stream_to_aqueue("</content-block:text>", queue)
                block_info["is_open"] = False
            
            # Stream complete tool call (client tools)
            elif block_type == "tool_use" and block_info["tool_data"]:
                tool_data = block_info["tool_data"]
                try:
                    # Parse the accumulated input JSON
                    if "input_json" in tool_data:
                        tool_data["input"] = json.loads(tool_data["input_json"])
                        del tool_data["input_json"]
                except json.JSONDecodeError as e:
                    # If parsing fails, keep as string
                    if "input_json" in tool_data:
                        tool_data["input"] = tool_data["input_json"]
                        del tool_data["input_json"]
                        tool_data["parse_error"] = str(e)
                
                # Format and stream
                tool_json = json.dumps(tool_data, indent=2)
                await stream_to_aqueue(
                    f'<content-block:tool_call>\n{tool_json}\n</content-block:tool_call>',
                    queue
                )
            
            # Stream complete server tool call (e.g., web_search)
            elif block_type == "server_tool_use" and block_info["tool_data"]:
                tool_data = block_info["tool_data"]
                try:
                    # Parse the accumulated input JSON
                    if "input_json" in tool_data:
                        tool_data["input"] = json.loads(tool_data["input_json"])
                        del tool_data["input_json"]
                except json.JSONDecodeError as e:
                    if "input_json" in tool_data:
                        tool_data["input"] = tool_data["input_json"]
                        del tool_data["input_json"]
                        tool_data["parse_error"] = str(e)
                
                # Format and stream as server tool
                tool_json = json.dumps(tool_data, indent=2)
                await stream_to_aqueue(
                    f'<content-block:server_tool_call>\n{tool_json}\n</content-block:server_tool_call>',
                    queue
                )
            
            # Stream server tool result
            elif block_type == "web_search_tool_result" and block_info["tool_data"]:
                result_json = json.dumps(block_info["tool_data"], indent=2, default=str)
                await stream_to_aqueue(
                    f'<content-block:server_tool_result>\n{result_json}\n</content-block:server_tool_result>',
                    queue
                )
        
        # Handle unknown event types gracefully (per Anthropic versioning policy)
        else:
            # Unknown event type - log but don't fail
            pass
    
    # Safety: close any open blocks at the end
    for idx, block_info in content_blocks.items():
        if block_info.get("is_open"):
            # Add newline before closing tag
            await stream_to_aqueue("\n", queue)
            
            if block_info["type"] == "thinking":
                await stream_to_aqueue("</content-block:thinking>", queue)
            elif block_info["type"] == "text":
                await stream_to_aqueue("</content-block:text>", queue)
            block_info["is_open"] = False
    
    # Get and return the final accumulated message (async method)
    final_message = await stream.get_final_message()
    
    # Extract file IDs from bash_code_execution_tool_result blocks
    file_ids = []
    for item in final_message.content:
        # Check if this is a bash code execution tool result
        if hasattr(item, 'type') and item.type == 'bash_code_execution_tool_result':
            # Access the content attribute
            if hasattr(item, 'content'):
                content_item = item.content
                # Check if it's a bash_code_execution_result
                if hasattr(content_item, 'type') and content_item.type == 'bash_code_execution_result':
                    # Access the content array
                    if hasattr(content_item, 'content') and isinstance(content_item.content, list):
                        for file_obj in content_item.content:
                            # Extract file_id if present
                            if hasattr(file_obj, 'file_id'):
                                file_ids.append(file_obj.file_id)
    
    # Stream file IDs meta tag if any files were created
    if file_ids:
        file_meta = json.dumps({"file_ids": file_ids}, indent=2)
        await stream_to_aqueue(
            f'\n<content-block:meta_files>\n{file_meta}\n</content-block:meta_files>',
            queue
        )
    
    return final_message


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
            event_str = event.model_dump_json()
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

