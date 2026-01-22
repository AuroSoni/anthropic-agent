"""OpenAI streaming with exponential backoff retry logic.

This module provides retry functionality for OpenAI Chat Completions streaming,
similar to the Anthropic retry module (`retry.py`).
"""

import asyncio
import random
import logging
from typing import Optional, Any

from ..streaming.openai_formatters import (
    OpenAIFormatterType,
    OpenAIStreamResult,
    get_openai_formatter,
)

logger = logging.getLogger(__name__)


async def openai_stream_with_backoff(
    client: Any,
    request_params: dict[str, Any],
    queue: Optional[asyncio.Queue] = None,
    max_retries: int = 5,
    base_delay: float = 5.0,
    formatter: OpenAIFormatterType = "xml",
    stream_tool_results: bool = True,
) -> OpenAIStreamResult:
    """Execute OpenAI streaming with exponential backoff for transient failures.

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

    Args:
        client: OpenAI AsyncOpenAI client instance.
        request_params: Parameters dict for client.chat.completions.create(**params).
        queue: Optional async queue for streaming formatted output.
        max_retries: Maximum number of retry attempts (default: 5).
        base_delay: Base delay in seconds for exponential backoff (default: 5.0).
        formatter: Formatter to use for stream output ("xml" or "raw", default: "xml").
        stream_tool_results: Whether to stream tool results to the queue (default: True).

    Returns:
        OpenAIStreamResult with accumulated message and metadata.

    Raises:
        openai.BadRequestError: Invalid request (not retried)
        openai.AuthenticationError: Auth failure (not retried)
        openai.PermissionDeniedError: Permission denied (not retried)
        openai.NotFoundError: Resource not found (not retried)
        Exception: Re-raises the final exception after all retries exhausted

    Example:
        >>> result = await openai_stream_with_backoff(
        ...     client=client,
        ...     request_params={"model": "gpt-4o", "messages": [...]},
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
    # Import OpenAI errors lazily to avoid import errors if openai is not installed
    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "OpenAI SDK is not installed. Install with `pip install openai`."
        ) from e

    # Prepare request params for streaming
    streaming_params = dict(request_params)
    streaming_params["stream"] = True
    streaming_params.setdefault("stream_options", {"include_usage": True})

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Attempting OpenAI stream call (attempt {attempt + 1}/{max_retries})"
            )
            logger.debug("OpenAI request params: %s", streaming_params)

            # Execute the streaming call
            stream = await client.chat.completions.create(**streaming_params)

            if queue:
                # Use formatter for formatted output
                formatter_fn = get_openai_formatter(formatter)
                result = await formatter_fn(stream, queue, stream_tool_results)
            else:
                # Simple iteration without queue - still need to accumulate result
                formatter_fn = get_openai_formatter(formatter)
                # Create a dummy queue to capture the result
                dummy_queue: asyncio.Queue[Any] = asyncio.Queue()
                result = await formatter_fn(stream, dummy_queue, stream_tool_results)

            logger.info(
                f"OpenAI stream completed successfully "
                f"(finish_reason: {result.finish_reason})"
            )
            return result

        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ) as e:
            # Retryable errors
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
                    f"All {max_retries} attempts failed with retryable error. "
                    f"Final error: {type(e).__name__}: {e}"
                )
                raise

        except openai.APIStatusError as e:
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
            openai.BadRequestError,
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
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

    # This should be unreachable
    raise RuntimeError("openai_stream_with_backoff: exhausted retries unexpectedly")
