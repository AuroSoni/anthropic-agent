"""Abstract base class for memory store implementations.

Memory stores manage persistent **cross-session** knowledge. They operate
at run boundaries only: ``retrieve()`` at the start of a run to inject
relevant prior knowledge, and ``update()`` at the end to extract and
persist new learnings for future runs.

Memory stores never participate in context compaction. Compaction (shrinking
the live message list to fit the model's token budget) is handled entirely
by the ``Compactor`` in ``agent_base.compaction``. The two systems are
independent: a compactor manages *within-session* context size, while a
memory store manages *across-session* knowledge.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_base.core.messages import Message
    from agent_base.core.types import ContentBlock

# Type alias for memory store names
MemoryStoreType = Literal["none"]


class MemoryStore(ABC):
    """Abstract base class for memory store implementations.

    Memory stores manage persistent cross-session knowledge that can be
    injected into agent conversations. They operate at run boundaries:
    ``retrieve()`` at the start of a run, ``update()`` at the end.

    All concrete memory stores must inherit from this class and implement
    both ``retrieve()`` and ``update()``.
    """

    @abstractmethod
    async def retrieve(
        self,
        user_message: Message,
        messages: list[Message],
        **kwargs: Any,
    ) -> list[ContentBlock]:
        """Retrieve relevant memories to inject into the prompt.

        Called once per ``agent.run()`` before the agent loop begins.
        Returns content blocks that the caller appends to the user
        message's content list.

        Args:
            user_message: The current user message.
            messages: Current ``context_messages`` list.
            **kwargs: Additional context (e.g., model, tools).

        Returns:
            List of ``ContentBlock`` instances to inject into the prompt.
            Empty list means no memories to inject.
        """
        ...

    @abstractmethod
    async def update(
        self,
        messages: list[Message],
        conversation_history: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update memory store with learnings from the completed run.

        Called after ``agent.run()`` completes successfully. The memory
        store can extract facts, entities, or summaries to persist for
        future retrieval.

        Args:
            messages: Compacted ``context_messages`` (what was sent to the LLM).
            conversation_history: Full uncompacted conversation history.
            **kwargs: Additional context (e.g., model, tools).

        Returns:
            Metadata dict describing what was stored/updated.
        """
        ...
