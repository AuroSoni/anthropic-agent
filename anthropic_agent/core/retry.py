import asyncio
import anthropic
import random
from typing import Optional, Callable, TypeVar, Awaitable, Any
from anthropic.types.beta import BetaMessage
from ..streaming import render_stream, get_formatter, FormatterType
from ..logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

async def anthropic_stream_with_backoff(
    client: anthropic.AsyncAnthropic,
    request_params: dict,
    queue: Optional[asyncio.Queue] = None,
    max_retries: int = 5,
    base_delay: float = 5.0,
    formatter: FormatterType = "xml",
    stream_tool_results: bool = True,
) -> BetaMessage:
    """Execute Anthropic streaming with exponential backoff for transient failures.
    
    Handles transient failures (rate limits, network issues, server errors) with
    exponential backoff and jitter. Retries the entire stream on failure to ensure
    consistent state.
    
    Retryable errors:
    - RateLimitError: API rate limits exceeded
    - APIConnectionError: Network connectivity issues
    - APITimeoutError: Request timeout
    - InternalServerError: Server errors (5xx)
    - APIStatusError with 5xx status codes
    
    Non-retryable errors (will raise immediately):
    - BadRequestError (400): Invalid request parameters
    - AuthenticationError (401): Invalid API key
    - PermissionDeniedError (403): Insufficient permissions
    - NotFoundError (404): Resource not found
    - UnprocessableEntityError (422): Validation errors
    
    Args:
        client: Anthropic client instance
        request_params: Parameters dict for client.beta.messages.stream(**params)
        queue: Optional async queue for streaming formatted output
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Base delay in seconds for exponential backoff (default: 5.0)
        formatter: Formatter to use for stream output ("xml" or "raw", default: "xml")
        stream_tool_results: Whether to stream tool results to the queue (default: True).
            When False, server tool results are not streamed to reduce output volume.
        
    Returns:
        The final accumulated message from stream.get_final_message()
        
    Raises:
        anthropic.BadRequestError: Invalid request (not retried)
        anthropic.AuthenticationError: Auth failure (not retried)
        anthropic.PermissionDeniedError: Permission denied (not retried)
        anthropic.NotFoundError: Resource not found (not retried)
        anthropic.UnprocessableEntityError: Validation error (not retried)
        Exception: Re-raises the final exception after all retries exhausted
        
    Example:
        >>> message = await anthropic_stream_with_backoff(
        ...     client=client,
        ...     request_params={"model": "claude-sonnet-4-5", "messages": [...]},
        ...     queue=output_queue,
        ...     max_retries=3,
        ...     base_delay=2.0
        ... )
        
    Note:
        The delay between retries follows the formula:
        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
        
        For base_delay=5.0:
        - Attempt 1: ~5 seconds
        - Attempt 2: ~10 seconds
        - Attempt 3: ~20 seconds
        - Attempt 4: ~40 seconds
        - Attempt 5: ~80 seconds
    """
    for attempt in range(max_retries):
        try:
            logger.info("Attempting Anthropic stream call", attempt=attempt + 1, max_retries=max_retries)
            logger.debug("Anthropic stream call parameters", request_params=request_params)
            # Execute the streaming call with async context manager
            async with client.beta.messages.stream(**request_params) as stream:
                if queue:
                    # Use render_stream for formatted output with specified formatter
                    formatter_fn = get_formatter(formatter)
                    accumulated_message = await render_stream(
                        stream, queue, formatter=formatter_fn, stream_tool_results=stream_tool_results
                    )
                else:
                    # Simple iteration without queue
                    async for event in stream:
                        pass  # Just consume events
                    accumulated_message = await stream.get_final_message()
            
            logger.info("Anthropic stream completed", stop_reason=accumulated_message.stop_reason)
            return accumulated_message
            
        except (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        ) as e:
            # Retryable errors
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff + jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning("Retryable error, will retry", attempt=attempt + 1, error_type=type(e).__name__, delay=f"{delay:.2f}s")
                await asyncio.sleep(delay)
                continue
            else:
                # Exhausted all retries
                logger.error("All retry attempts failed", max_retries=max_retries, error_type=type(e).__name__)
                raise
                
        except anthropic.APIStatusError as e:
            # Check if it's a 5xx error (retryable) or 4xx (not retryable)
            if e.status_code >= 500:
                # Server error - retryable
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning("Server error (5xx), will retry", attempt=attempt + 1, status_code=e.status_code, delay=f"{delay:.2f}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error("All retry attempts failed with server error", max_retries=max_retries, status_code=e.status_code)
                    raise
            else:
                # Client error (4xx) - don't retry
                logger.error("Non-retryable API status error", status_code=e.status_code)
                raise
                
        except (
            anthropic.BadRequestError,
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.UnprocessableEntityError,
        ) as e:
            # Non-retryable errors - fail immediately
            logger.error("Non-retryable error", error_type=type(e).__name__)
            raise
            
        except Exception as e:
            # Unknown error - retry with caution
            logger.error("Unexpected error", attempt=attempt + 1, error_type=type(e).__name__)
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning("Retrying unexpected error", delay=f"{delay:.2f}s")
                await asyncio.sleep(delay)
                continue
            else:
                logger.error("All retry attempts failed", max_retries=max_retries)
                raise


def retry_with_backoff(
    max_retries: int,
    base_delay: float,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator factory for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of attempts (including the first).
        base_delay: Base delay in seconds for exponential backoff.

    Behavior:
        - Retries on any exception up to max_retries.
        - Delay between retries follows: delay = base_delay * (2 ** attempt).
        - Logs a warning for intermediate failures and an error when retries are exhausted.
        - Re-raises the last exception after all retries are used.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    last_exc = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning("Retryable error, will retry", func=func.__name__, attempt=attempt + 1, max_retries=max_retries, delay=f"{delay:.2f}s")
                        await asyncio.sleep(delay)
                    else:
                        logger.error("Function failed after all retries", func=func.__name__, max_retries=max_retries, exc_info=True)
                        raise last_exc

            # This line should be unreachable
            raise RuntimeError(f"retry_with_backoff: exhausted retries for {func.__name__}")  # pragma: no cover

        return wrapper

    return decorator
