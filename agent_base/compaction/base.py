"""Abstract base class for context compaction strategies.

Compactors manage **within-session** context window pressure. They operate
on canonical ``Message`` objects and are completely provider-agnostic.

Compactors are stateless — they receive the full ``AgentConfig`` and return
a ``(did_compact, compacted_messages)`` tuple. The caller replaces
``agent_config.context_messages`` when ``did_compact`` is ``True``.

See also ``agent_base.memory`` for the cross-session knowledge layer,
which operates at run boundaries and is independent of compaction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_base.core.config import AgentConfig
    from agent_base.core.messages import Message

# Type alias for compactor names
CompactorType = Literal["summarizing", "none"]


# ---------------------------------------------------------------------------
# CompactionLLM protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CompactionLLM(Protocol):
    """Protocol for the LLM backend used by compactors that need to
    generate summaries or other LLM-derived content.

    Any provider can satisfy this by implementing ``generate_summary``.
    The protocol is intentionally narrow — it captures only the capability
    a compactor needs, keeping the compaction module provider-agnostic.

    Example Anthropic implementation::

        class AnthropicCompactionLLM:
            def __init__(self, client, model, formatter):
                ...
            async def generate_summary(self, messages, prompt, **kwargs):
                formatted = self.formatter.format_messages(messages, {})
                response = await self.client.messages.create(...)
                return response.content[0].text
    """

    async def generate_summary(
        self,
        messages: list["Message"],
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a summary of the given messages.

        Args:
            messages: The messages to summarize.
            prompt: The summarization prompt / instructions.
            **kwargs: Provider-specific options (model override, etc.).

        Returns:
            The summary text.
        """
        ...


# ---------------------------------------------------------------------------
# Compactor ABC
# ---------------------------------------------------------------------------


class Compactor(ABC):
    """Abstract base class for context compaction strategies.

    Compactors take an ``AgentConfig`` (which carries ``context_messages``,
    ``model``, and token tracking) and decide whether compaction is needed.
    If so, they return a shortened message list.

    All concrete compactors must inherit from this class and implement
    ``apply_compaction()``.
    """

    @abstractmethod
    async def apply_compaction(
        self, agent_config: AgentConfig
    ) -> tuple[bool, list[Message]]:
        """Apply compaction strategy to the agent's context messages.

        Args:
            agent_config: The agent's current configuration. The compactor
                reads ``context_messages``, ``model``, and
                ``last_known_input_tokens`` from here.

        Returns:
            Tuple of (did_compact, compacted_messages).
            - did_compact: True if compaction was applied.
            - compacted_messages: The (possibly shortened) message list.
              When did_compact is False, this is the original list unchanged.
        """
        ...
