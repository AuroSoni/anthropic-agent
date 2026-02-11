import asyncio
from typing import Callable, Awaitable
from anthropic.lib.streaming._beta_messages import BetaAsyncMessageStream
from anthropic.types.beta import BetaMessage
from .formatters import json_formatter


async def render_stream(
    stream: BetaAsyncMessageStream,
    queue: asyncio.Queue,
    formatter: Callable[..., Awaitable[BetaMessage]] = json_formatter,
    stream_tool_results: bool = True,
    agent_uuid: str = "",
) -> BetaMessage:
    """Render Anthropic streaming response with configurable formatting.

    This function delegates the actual formatting logic to a formatter function,
    allowing different formatting strategies (JSON envelope, XML, raw, etc.).

    Args:
        stream: Anthropic async message stream (from inside async with context)
        queue: Async queue to send formatted output chunks
        formatter: Async function that formats the stream (default: json_formatter).
            The "xml" and "raw" formatters are deprecated and will be removed in a
            future release.
        stream_tool_results: Whether to stream tool results to the queue (default: True).
            When False, server tool results are not streamed to reduce output volume.
        agent_uuid: UUID of the emitting agent (used by json_formatter).

    Returns:
        The final accumulated message from stream.get_final_message()

    Example:
        >>> from anthropic_agent.streaming.formatters import json_formatter
        >>>
        >>> # Use default JSON envelope formatter
        >>> with client.beta.messages.stream(...) as stream:
        >>>     message = await render_stream(stream, queue)

    Note:
        - Formatters receive the stream and queue, and must return the final message
        - The default json_formatter emits self-contained JSON envelopes over SSE
        - The json_formatter additionally accepts agent_uuid
        - The xml_formatter and raw_formatter are deprecated
    """
    # Build kwargs – only pass agent_uuid when it is set
    # so that xml_formatter and raw_formatter (which don't accept it) still work.
    kwargs: dict = {"stream_tool_results": stream_tool_results}
    if agent_uuid:
        kwargs["agent_uuid"] = agent_uuid
    return await formatter(stream, queue, **kwargs)
