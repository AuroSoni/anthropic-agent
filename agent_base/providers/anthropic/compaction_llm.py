"""Anthropic implementation of the CompactionLLM protocol.

Satisfies the ``CompactionLLM`` protocol from ``agent_base.compaction.base``
by using the Anthropic API to generate conversation summaries.

Usage::

    from agent_base.providers.anthropic import AnthropicCompactionLLM
    from agent_base.compaction import SummarizingCompactor

    llm = AnthropicCompactionLLM(model="claude-sonnet-4-5")
    compactor = SummarizingCompactor(llm=llm, threshold=160_000)
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import anthropic

from agent_base.logging import get_logger

if TYPE_CHECKING:
    from agent_base.core.messages import Message

logger = get_logger(__name__)

DEFAULT_COMPACTION_MODEL = "claude-sonnet-4-5"


class AnthropicCompactionLLM:
    """CompactionLLM implementation using Anthropic's API.

    Uses ``AnthropicMessageFormatter`` to convert canonical ``Message``
    objects into Anthropic wire format, then calls the API to produce
    a summary.

    Args:
        client: Anthropic async client. If ``None``, creates one.
        formatter: Message formatter. If ``None``, creates a default one.
        model: Model to use for summarization calls.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        formatter: Any | None = None,
        model: str = DEFAULT_COMPACTION_MODEL,
    ) -> None:
        self.client = client or anthropic.AsyncAnthropic()
        if formatter is None:
            from .formatters import AnthropicMessageFormatter
            formatter = AnthropicMessageFormatter()
        self.formatter = formatter
        self.model = model

    async def generate_summary(
        self,
        messages: list[Message],
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a summary of the given messages using Anthropic.

        Appends the summarization ``prompt`` as a final user message,
        builds a minimal Anthropic API request, and extracts the text
        from the response.

        Args:
            messages: The messages to summarize.
            prompt: The summarization prompt / instructions.
            **kwargs: Optional overrides (``model``, ``max_tokens``).

        Returns:
            The summary text.
        """
        from agent_base.core.messages import Message

        # Add the summarization prompt as a final user message
        all_messages = list(messages) + [Message.user(prompt)]

        model = kwargs.get("model", self.model)
        max_tokens = kwargs.get("max_tokens", 4096)

        # Build minimal request params using formatter for block conversion only
        wire_messages = [
            {
                "role": msg.role.value,
                "content": self.formatter.format_blocks_to_wire(msg.content),
            }
            for msg in all_messages
        ]
        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": wire_messages,
            "system": "You are a conversation summarizer. Produce concise, structured summaries.",
        }

        logger.debug(
            "compaction_llm_call",
            model=model,
            num_messages=len(all_messages),
        )

        response = await self.client.messages.create(**request_params)

        # Extract text from response
        for block in response.content:
            if hasattr(block, "text"):
                return block.text

        return ""
