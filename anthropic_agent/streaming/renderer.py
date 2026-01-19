import asyncio
from typing import Callable, Awaitable
from anthropic.lib.streaming._beta_messages import BetaAsyncMessageStream
from anthropic.types.beta import BetaMessage
from .formatters import xml_formatter


async def render_stream(
    stream: BetaAsyncMessageStream,
    queue: asyncio.Queue,
    formatter: Callable[..., Awaitable[BetaMessage]] = xml_formatter,
    stream_tool_results: bool = True
) -> BetaMessage:
    """Render Anthropic streaming response with configurable formatting.
    
    This function delegates the actual formatting logic to a formatter function,
    allowing different formatting strategies (XML, raw, custom, etc.).
    
    Args:
        stream: Anthropic async message stream (from inside async with context)
        queue: Async queue to send formatted output chunks
        formatter: Async function that formats the stream (default: xml_formatter)
                  Signature: async def formatter(stream, queue, stream_tool_results) -> BetaMessage
        stream_tool_results: Whether to stream tool results to the queue (default: True).
            When False, server tool results are not streamed to reduce output volume.
        
    Returns:
        The final accumulated message from stream.get_final_message()
        
    Example:
        >>> from anthropic_agent.src.streaming.formatters import xml_formatter, raw_formatter
        >>> 
        >>> # Use default XML formatter
        >>> with client.beta.messages.stream(...) as stream:
        >>>     message = await render_stream(stream, queue)
        >>> 
        >>> # Use raw formatter
        >>> with client.beta.messages.stream(...) as stream:
        >>>     message = await render_stream(stream, queue, formatter=raw_formatter)
    
    Note:
        - Formatters receive the stream and queue, and must return the final message
        - The default xml_formatter provides the original XML-based formatting
        - Custom formatters can be provided for different output formats
    """
    return await formatter(stream, queue, stream_tool_results=stream_tool_results)
