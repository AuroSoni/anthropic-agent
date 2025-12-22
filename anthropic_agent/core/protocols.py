"""Core protocols for LLM provider abstractions.

This module defines the core protocols that all LLM providers must implement.
Using Protocol enables structural subtyping (duck typing with type safety),
allowing existing classes to satisfy protocols without inheritance changes.

These protocols are provider-agnostic and do not import any provider SDK.
"""

import asyncio
from typing import Protocol, Any, AsyncIterator, runtime_checkable


@runtime_checkable
class Usage(Protocol):
    """Protocol for token usage information."""
    
    input_tokens: int
    output_tokens: int


@runtime_checkable
class LLMResponse(Protocol):
    """Protocol for LLM response objects.
    
    This protocol defines the minimal interface that all provider-specific
    response objects must satisfy. Provider implementations can have
    additional attributes beyond these.
    """
    
    content: list[dict]
    stop_reason: str
    usage: Usage
    model: str


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM client implementations.
    
    This protocol defines the interface that all LLM provider clients must
    implement. It provides a common abstraction for streaming completions
    and token counting across different providers (Anthropic, OpenAI, etc.).
    
    Provider-specific options should be passed via **kwargs.
    
    Example:
        class AnthropicClient:
            async def stream(
                self,
                messages: list[dict],
                model: str,
                system: str | None = None,
                tools: list[dict] | None = None,
                max_tokens: int = 2048,
                **kwargs
            ) -> AsyncIterator[Any]:
                # Anthropic-specific implementation
                ...
    """
    
    async def stream(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Stream a completion from the LLM.
        
        Args:
            messages: List of message dictionaries in provider-agnostic format.
                Each message should have 'role' and 'content' keys.
            model: Model identifier string (e.g., "claude-sonnet-4-5", "gpt-4o")
            system: Optional system prompt/instructions
            tools: Optional list of tool schemas for function calling
            max_tokens: Maximum tokens in the response
            **kwargs: Provider-specific options (e.g., thinking_tokens, temperature)
            
        Yields:
            Provider-specific streaming events. The exact format depends on
            the provider implementation.
        """
        ...
    
    async def count_tokens(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> int | None:
        """Count tokens for the given messages.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier string
            system: Optional system prompt
            tools: Optional list of tool schemas
            
        Returns:
            Token count as integer, or None if counting is not supported
            or fails.
        """
        ...


class StreamFormatter(Protocol):
    """Protocol for stream formatters.
    
    Stream formatters transform provider-specific streaming events into
    a consistent output format (e.g., XML tags, raw JSON) while sending
    chunks to an async queue for real-time consumption.
    
    Example:
        class XMLFormatter:
            async def format(
                self,
                stream: AsyncIterator[Any],
                queue: asyncio.Queue
            ) -> LLMResponse:
                async for event in stream:
                    # Format event and send to queue
                    await queue.put(formatted_chunk)
                return final_response
    """
    
    async def format(
        self,
        stream: AsyncIterator[Any],
        queue: asyncio.Queue
    ) -> LLMResponse:
        """Format streaming events and return final response.
        
        Args:
            stream: Async iterator of provider-specific streaming events
            queue: Async queue to send formatted output chunks
            
        Returns:
            The final accumulated response object
        """
        ...

