"""Provider ABC — abstracts the raw LLM API call.

The Agent orchestrates the loop; the Provider owns the transport
(HTTP call, streaming, retries, authentication). The Provider holds a
``MessageFormatter`` for request/response translation.

This module MUST NOT import any provider SDK (anthropic, openai, etc.).
Provider-specific implementations live in ``agent_base.providers.<name>/``.

Separation of concerns:
    - **Provider** → authentication, request building (via MessageFormatter),
      retry/backoff, response parsing, stream event translation.
    - **Agent** → orchestration loop, step counting, tool dispatch, compaction,
      memory, relay handling.
    - **MessageFormatter** → pure canonical ↔ wire-format translation.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_base.core.config import LLMConfig
    from agent_base.core.messages import Message, MessageFormatter
    from agent_base.streaming.base import StreamFormatter
    from agent_base.tools.types import ToolSchema


class Provider(ABC):
    """Abstract base class for LLM providers.

    Concrete implementations (``AnthropicProvider``, ``OpenAIProvider``, etc.)
    handle authentication, request formation via their ``MessageFormatter``,
    retry/backoff, and response parsing.

    The Provider is injected into the Agent and called by the agent loop.
    """

    formatter: MessageFormatter

    @abstractmethod
    async def generate(
        self,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[ToolSchema],
        llm_config: LLMConfig,
        model: str,
        max_retries: int,
        base_delay: float,
        agent_uuid: str = "",
    ) -> Message:
        """Execute a non-streaming LLM API call.

        Args:
            system_prompt: System prompt text (or ``None``).
            messages: Canonical message history.
            tool_schemas: Canonical tool schemas from ``ToolRegistry``.
            llm_config: Provider-specific LLM configuration (e.g.,
                ``AnthropicLLMConfig``).
            model: Model identifier string.
            max_retries: Maximum retry attempts for transient failures.
            base_delay: Base delay in seconds for exponential backoff.
            agent_uuid: Agent session UUID for logging/tracking.

        Returns:
            Parsed canonical ``Message`` with content blocks, usage,
            and stop_reason.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        system_prompt: str | None,
        messages: list[Message],
        tool_schemas: list[ToolSchema],
        llm_config: LLMConfig,
        model: str,
        max_retries: int,
        base_delay: float,
        queue: asyncio.Queue,
        stream_formatter: StreamFormatter,
        stream_tool_results: bool = True,
        agent_uuid: str = "",
    ) -> Message:
        """Execute a streaming LLM API call.

        Translates provider-native stream events into canonical
        ``StreamDelta`` objects, formats them via
        ``stream_formatter.format_delta()``, and puts them on the queue
        for the transport layer.

        Returns the complete ``Message`` after the stream finishes.

        Args:
            system_prompt: System prompt text (or ``None``).
            messages: Canonical message history.
            tool_schemas: Canonical tool schemas.
            llm_config: Provider-specific LLM configuration.
            model: Model identifier string.
            max_retries: Maximum retry attempts.
            base_delay: Base delay for exponential backoff.
            queue: Async queue for serialized stream output.
            stream_formatter: Formatter for serializing ``StreamDelta``
                objects to wire format.
            stream_tool_results: Whether to stream server tool results.
            agent_uuid: Agent session UUID.

        Returns:
            Complete canonical ``Message`` after stream finishes.
        """
        ...
