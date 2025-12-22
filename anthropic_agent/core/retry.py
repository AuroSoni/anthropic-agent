"""Retry utilities for async operations.

This module provides retry decorators and functions for handling transient
failures with exponential backoff. It includes both provider-agnostic utilities
and Anthropic-specific retry logic.
"""

import asyncio
import anthropic
import random
import logging
from typing import Optional, Callable, TypeVar, Awaitable, Any
from anthropic.types.beta import BetaMessage
from ..streaming import render_stream, get_formatter, FormatterType

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Provider-Agnostic Retry Utilities
# =============================================================================

def async_retry(
    max_retries: int = 5,
    base_delay: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    non_retryable_exceptions: tuple[type[Exception], ...] = (),
    jitter: bool = True,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Generic async retry decorator with exponential backoff.
    
    A configurable decorator for retrying async functions with exponential
    backoff. Allows fine-grained control over which exceptions trigger retries.
    
    Args:
        max_retries: Maximum number of attempts (including the first). Default: 5
        base_delay: Base delay in seconds for exponential backoff. Default: 1.0
        retryable_exceptions: Tuple of exception types that should trigger retries.
            Default: (Exception,) - retries on any exception.
        non_retryable_exceptions: Tuple of exception types that should NOT be retried,
            even if they match retryable_exceptions. These take precedence.
            Default: () - no exceptions are explicitly non-retryable.
        jitter: Whether to add random jitter (0-1 seconds) to delay. Default: True
    
    Behavior:
        - Exceptions matching non_retryable_exceptions are raised immediately.
        - Exceptions matching retryable_exceptions trigger a retry (up to max_retries).
        - Other exceptions are raised immediately.
        - Delay follows: delay = base_delay * (2 ** attempt) [+ random(0,1) if jitter]
        - Logs warnings for retries, errors when exhausted.
    
    Example:
        >>> # Retry on any exception
        >>> @async_retry(max_retries=3, base_delay=1.0)
        >>> async def fetch_data():
        >>>     ...
        
        >>> # Retry only on specific exceptions
        >>> @async_retry(
        ...     max_retries=5,
        ...     retryable_exceptions=(TimeoutError, ConnectionError),
        ...     non_retryable_exceptions=(ValueError, KeyError),
        ... )
        >>> async def api_call():
        >>>     ...
    
    Returns:
        Decorator function that wraps async functions with retry logic.
    """
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Exception | None = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except non_retryable_exceptions as e:
                    # Non-retryable exceptions are raised immediately
                    logger.error(
                        f"Non-retryable error in {func.__name__}: {type(e).__name__}: {e}"
                    )
                    raise
                except retryable_exceptions as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        if jitter:
                            delay += random.uniform(0, 1)
                        logger.warning(
                            f"Retryable error in {func.__name__} "
                            f"(attempt {attempt + 1}/{max_retries}): "
                            f"{type(e).__name__}: {e}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}",
                            exc_info=True,
                        )
                        raise last_exc
                except Exception as e:
                    # Unknown exceptions are not retried by default
                    logger.error(
                        f"Unexpected error in {func.__name__}: {type(e).__name__}: {e}"
                    )
                    raise
            
            # This line should be unreachable
            raise RuntimeError(
                f"async_retry: exhausted retries for {func.__name__}"
            )  # pragma: no cover
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


# =============================================================================
# Anthropic-Specific Retry Utilities
# =============================================================================

async def anthropic_stream_with_backoff(
    client: anthropic.AsyncAnthropic,
    request_params: dict,
    queue: Optional[asyncio.Queue] = None,
    max_retries: int = 5,
    base_delay: float = 5.0,
    formatter: FormatterType = "xml",
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
            logger.info(
                f"Attempting Anthropic stream call (attempt {attempt + 1}/{max_retries})"
            )
            
            logger.debug("Anthropic request params: %s", request_params)
            # Execute the streaming call with async context manager
            async with client.beta.messages.stream(**request_params) as stream:
                if queue:
                    # Use render_stream for formatted output with specified formatter
                    formatter_fn = get_formatter(formatter)
                    accumulated_message = await render_stream(stream, queue, formatter=formatter_fn)
                else:
                    # Simple iteration without queue
                    async for event in stream:
                        pass  # Just consume events
                    accumulated_message = await stream.get_final_message()
            
            logger.info(
                f"Anthropic stream completed successfully "
                f"(stop_reason: {accumulated_message.stop_reason})"
            )
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
                logger.warning(
                    f"Retryable error on attempt {attempt + 1}: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                await asyncio.sleep(delay)
                continue
            else:
                # Exhausted all retries
                logger.error(
                    f"All {max_retries} attempts failed with retryable error. "
                    f"Final error: {type(e).__name__}: {e}"
                )
                raise
                
        except anthropic.APIStatusError as e:
            # Check if it's a 5xx error (retryable) or 4xx (not retryable)
            if e.status_code >= 500:
                # Server error - retryable
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Server error (5xx) on attempt {attempt + 1}: {e.status_code} - {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"All {max_retries} attempts failed with server error. "
                        f"Final error: {e.status_code} - {e}"
                    )
                    raise
            else:
                # Client error (4xx) - don't retry
                logger.error(
                    f"Non-retryable API status error: {e.status_code} - {e}"
                )
                raise
                
        except (
            anthropic.BadRequestError,
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.UnprocessableEntityError,
        ) as e:
            # Non-retryable errors - fail immediately
            logger.error(
                f"Non-retryable error: {type(e).__name__}: {e}"
            )
            raise
            
        except Exception as e:
            # Unknown error - retry with caution
            logger.error(
                f"Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}"
            )
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Retrying unexpected error in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
                continue
            else:
                logger.error(f"All {max_retries} attempts failed. Final error: {e}")
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
                        logger.warning(
                            f"Retryable error in {func.__name__} "
                            f"(attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}",
                            exc_info=True,
                        )
                        raise last_exc

            # This line should be unreachable
            raise RuntimeError(f"retry_with_backoff: exhausted retries for {func.__name__}")  # pragma: no cover

        return wrapper

    return decorator
