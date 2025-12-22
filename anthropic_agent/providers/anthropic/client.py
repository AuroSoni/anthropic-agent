"""Anthropic LLM client implementation.

This module provides the AnthropicClient class that implements the LLMClient
protocol for streaming completions from Anthropic's Claude models.
"""

import asyncio
import random
import logging
from typing import Any, AsyncIterator, Optional

import anthropic
from anthropic.types.beta import BetaMessage

from .streaming import get_formatter, FormatterType

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Anthropic LLM client implementing the LLMClient protocol.
    
    This client wraps the Anthropic Python SDK to provide streaming completions
    from Claude models. It handles retry logic with exponential backoff for
    transient failures.
    
    Attributes:
        client: The underlying Anthropic async client
        
    Example:
        >>> client = AnthropicClient()
        >>> async for event in client.stream(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     model="claude-sonnet-4-5",
        ... ):
        ...     print(event)
    """
    
    # Anthropic-specific error types that are retryable
    RETRYABLE_ERRORS = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.InternalServerError,
    )
    
    # Anthropic-specific error types that should NOT be retried
    NON_RETRYABLE_ERRORS = (
        anthropic.BadRequestError,
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.UnprocessableEntityError,
    )
    
    def __init__(self, api_key: str | None = None):
        """Initialize the Anthropic client.
        
        Args:
            api_key: Optional API key. If not provided, uses ANTHROPIC_API_KEY
                environment variable.
        """
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def stream(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream a completion from Anthropic's API.
        
        This method yields raw streaming events from the Anthropic API.
        Use with a StreamFormatter to process the events into a usable format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier (e.g., "claude-sonnet-4-5")
            system: Optional system prompt
            tools: Optional list of tool schemas in Anthropic format
            max_tokens: Maximum tokens in response (default: 2048)
            **kwargs: Additional Anthropic-specific options:
                - thinking: Extended thinking configuration
                - betas: Beta feature headers
                - container: Container ID for multi-turn
                - temperature, top_p, top_k, stop_sequences, etc.
        
        Yields:
            Anthropic streaming events (BetaMessageStreamEvent)
            
        Note:
            This method does NOT handle retries. Use stream_with_retry for
            automatic retry logic, or wrap calls in your own retry handling.
        """
        # Build request parameters
        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        
        if system:
            request_params["system"] = system
        
        if tools:
            request_params["tools"] = tools
        
        # Extract Anthropic-specific options from kwargs
        if "thinking" in kwargs:
            request_params["thinking"] = kwargs.pop("thinking")
        
        if "betas" in kwargs:
            request_params["betas"] = kwargs.pop("betas")
        
        if "container" in kwargs:
            request_params["container"] = kwargs.pop("container")
        
        # Pass through remaining kwargs (temperature, top_p, etc.)
        request_params.update(kwargs)
        
        logger.debug("Anthropic stream request params: %s", request_params)
        
        async with self.client.beta.messages.stream(**request_params) as stream:
            async for event in stream:
                yield event
            # Yield the final message as a special marker
            final_message = await stream.get_final_message()
            yield final_message
    
    async def stream_with_retry(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        queue: Optional[asyncio.Queue] = None,
        formatter: FormatterType = "xml",
        max_retries: int = 5,
        base_delay: float = 5.0,
        **kwargs: Any,
    ) -> BetaMessage:
        """Stream completion with automatic retry on transient failures.
        
        This method wraps stream() with exponential backoff retry logic for
        transient failures (rate limits, network issues, server errors).
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            system: Optional system prompt
            tools: Optional list of tool schemas
            max_tokens: Maximum tokens in response
            queue: Optional async queue for formatted output chunks
            formatter: Formatter to use ("xml" or "raw")
            max_retries: Maximum retry attempts (default: 5)
            base_delay: Base delay in seconds for backoff (default: 5.0)
            **kwargs: Additional Anthropic-specific options
            
        Returns:
            The final accumulated BetaMessage
            
        Raises:
            anthropic.BadRequestError: Invalid request (not retried)
            anthropic.AuthenticationError: Auth failure (not retried)
            Exception: Re-raises after all retries exhausted
        """
        # Build request parameters
        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        
        if system:
            request_params["system"] = system
        
        if tools:
            request_params["tools"] = tools
        
        # Extract Anthropic-specific options
        if "thinking" in kwargs:
            request_params["thinking"] = kwargs.pop("thinking")
        if "betas" in kwargs:
            request_params["betas"] = kwargs.pop("betas")
        if "container" in kwargs:
            request_params["container"] = kwargs.pop("container")
        
        request_params.update(kwargs)
        
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Attempting Anthropic stream call (attempt {attempt + 1}/{max_retries})"
                )
                
                async with self.client.beta.messages.stream(**request_params) as stream:
                    if queue:
                        formatter_fn = get_formatter(formatter)
                        accumulated_message = await formatter_fn(stream, queue)
                    else:
                        async for _ in stream:
                            pass
                        accumulated_message = await stream.get_final_message()
                
                logger.info(
                    f"Anthropic stream completed successfully "
                    f"(stop_reason: {accumulated_message.stop_reason})"
                )
                return accumulated_message
                
            except self.RETRYABLE_ERRORS as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}: {type(e).__name__}: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"All {max_retries} attempts failed. Final error: {type(e).__name__}: {e}"
                    )
                    raise
                    
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"Server error (5xx) on attempt {attempt + 1}: {e.status_code}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e.status_code}")
                        raise
                else:
                    logger.error(f"Non-retryable API error: {e.status_code} - {e}")
                    raise
                    
            except self.NON_RETRYABLE_ERRORS as e:
                logger.error(f"Non-retryable error: {type(e).__name__}: {e}")
                raise
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise
        
        # Should not reach here
        raise RuntimeError("stream_with_retry exhausted retries")
    
    async def count_tokens(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> int | None:
        """Count tokens for the given messages using Anthropic's API.
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            system: Optional system prompt
            tools: Optional list of tool schemas
            
        Returns:
            Token count as integer, or None if counting fails
        """
        # Filter tools to only those supported by count_tokens API
        allowed_server_tool_types = {
            "bash_20250124",
            "custom",
            "text_editor_20250124",
            "text_editor_20250429",
            "text_editor_20250728",
            "web_search_20250305",
        }
        
        filtered_tools: list[dict] = []
        if tools:
            for tool in tools:
                tool_type = tool.get("type")
                if tool_type is None or tool_type in allowed_server_tool_types:
                    filtered_tools.append(tool)
        
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        
        if system:
            params["system"] = system
        
        if filtered_tools:
            params["tools"] = filtered_tools
        
        try:
            response = await self.client.messages.count_tokens(**params)
            return getattr(response, "input_tokens", None)
        except Exception as e:
            logger.warning(f"Token count API call failed: {e}")
            return None
    
    async def download_file(self, file_id: str) -> tuple[Any, bytes]:
        """Download a file from Anthropic's Files API.
        
        Args:
            file_id: Anthropic file identifier
            
        Returns:
            Tuple of (file_metadata, file_content_bytes)
            
        Raises:
            Exception: If download fails
        """
        try:
            response = await self.client.beta.files.download(file_id)
            file_content = await response.read()
            file_metadata = await self.client.beta.files.retrieve_metadata(file_id)
            return file_metadata, file_content
        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}", exc_info=True)
            raise

