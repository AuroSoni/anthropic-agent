"""Concrete compactor implementations.

``NoOpCompactor`` — pass-through, no compaction.
``SummarizingCompactor`` — LLM-based summarization of older history (provider-agnostic).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

from agent_base.logging import get_logger
from .base import Compactor, CompactionLLM

if TYPE_CHECKING:
    from agent_base.core.config import AgentConfig
    from agent_base.core.messages import Message

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# NoOpCompactor
# ---------------------------------------------------------------------------


class NoOpCompactor(Compactor):
    """No-operation compactor that returns messages unchanged.

    Useful for disabling compaction or as a baseline comparison.
    """

    async def apply_compaction(
        self, agent_config: AgentConfig
    ) -> tuple[bool, list[Message]]:
        """Return context_messages unchanged."""
        return False, agent_config.context_messages


# ---------------------------------------------------------------------------
# SummarizingCompactor
# ---------------------------------------------------------------------------

_DEFAULT_SUMMARY_PROMPT = """\
Summarize the conversation history above into a concise structured summary \
preserving the following categories.  Omit any category that has no relevant content.

**User Intent** — Original request and any refinements.
**Completed Work** — Actions performed, identifiers, values produced.
**Errors & Corrections** — What failed and how it was corrected.
**Active Work** — What was in progress, partial results.
**Pending Tasks** — Remaining items.
**Key References** — IDs, file paths, URLs, configuration values.

Rules:
- Be concise; prefer bullet points over prose.
- Weight recent content more heavily than old content.
- Omit pleasantries, filler, and meta-commentary.
- Wrap your entire output in <session_summary> tags.
"""


class SummarizingCompactor(Compactor):
    """Compactor that uses an LLM to summarize older conversation history.

    When the estimated token count exceeds the threshold, older messages are
    sent to the LLM for summarization.  The resulting summary replaces the
    older messages while recent turns are preserved verbatim.

    This compactor is **provider-agnostic**.  It accepts a ``CompactionLLM``
    instance that any provider can implement.  The compactor calls
    ``llm.generate_summary()`` to produce the summary text.

    Args:
        llm: A ``CompactionLLM`` instance that the provider supplies.
        threshold: Token count threshold to trigger compaction.  If ``None``,
            defaults to 160 000 (≈80 % of a 200k context window).
        keep_recent_turns: Number of recent turns to preserve verbatim.
        token_estimator: Optional callable ``(list[Message]) -> int``.
            If not provided, uses a simple character-based heuristic.
        summary_prompt: Custom summary prompt.  Defaults to built-in prompt.
    """

    def __init__(
        self,
        llm: CompactionLLM,
        threshold: int | None = None,
        keep_recent_turns: int = 5,
        token_estimator: Callable[[list[Message]], int] | None = None,
        summary_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.threshold = threshold
        self.keep_recent_turns = keep_recent_turns
        self.token_estimator = token_estimator or _default_token_estimator
        self.summary_prompt = summary_prompt or _DEFAULT_SUMMARY_PROMPT

    async def apply_compaction(
        self, agent_config: AgentConfig
    ) -> tuple[bool, list[Message]]:
        """Summarize older messages when over threshold."""
        from agent_base.core.messages import Message
        from agent_base.core.types import TextContent

        messages = agent_config.context_messages
        threshold = self.threshold if self.threshold is not None else 160_000
        estimated_tokens = self.token_estimator(messages)

        if estimated_tokens <= threshold:
            logger.debug(
                "compaction_skipped",
                reason="below_threshold",
                estimated_tokens=estimated_tokens,
                threshold=threshold,
            )
            return False, messages

        if len(messages) <= 1:
            return False, messages

        # Split at turn boundary, preserving recent turns verbatim.
        to_summarize, to_keep = _split_at_turn_boundary(
            messages, self.keep_recent_turns
        )

        if not to_summarize:
            logger.debug("compaction_skipped", reason="nothing_to_summarize")
            return False, messages

        # Call the provider-supplied LLM to generate the summary.
        summary_text = await self.llm.generate_summary(to_summarize, self.summary_prompt)

        if not summary_text.startswith("<session_summary>"):
            summary_text = f"<session_summary>\n{summary_text}\n</session_summary>"

        # Create summary message as a canonical user Message.
        summary_message = Message.user([TextContent(text=summary_text)])

        compacted = [summary_message] + to_keep

        logger.info(
            "compaction_applied",
            messages_summarized=len(to_summarize),
            messages_kept=len(to_keep),
            timestamp=datetime.now().isoformat(),
        )

        return True, compacted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_at_turn_boundary(
    messages: list[Message], keep_count: int
) -> tuple[list[Message], list[Message]]:
    """Split messages into (to_summarize, to_keep) at a turn boundary.

    A "turn" is an assistant message and any immediately following user
    message that carries tool results.  Walks backward counting
    ``keep_count`` turns, then splits.
    """
    from agent_base.core.types import Role, ToolResultBase

    if keep_count <= 0 or len(messages) <= 1:
        return messages, []

    turns_found = 0
    split_idx = len(messages)

    i = len(messages) - 1
    while i >= 1 and turns_found < keep_count:
        msg = messages[i]
        if msg.role == Role.ASSISTANT:
            turns_found += 1
            split_idx = i
        elif msg.role == Role.USER:
            # Check if this user message contains tool results.
            has_tool_result = any(
                isinstance(block, ToolResultBase) for block in msg.content
            )
            if not has_tool_result:
                turns_found += 1
                split_idx = i
        i -= 1

    return messages[:split_idx], messages[split_idx:]


def _default_token_estimator(messages: list[Message]) -> int:
    """Simple character-based token estimator.

    Serializes each message to its dict form and estimates at ~4 characters
    per token.  This is intentionally rough — providers can supply a more
    accurate estimator via the ``token_estimator`` parameter.
    """
    total_chars = 0
    for msg in messages:
        try:
            total_chars += len(json.dumps(msg.to_dict()))
        except (TypeError, ValueError):
            total_chars += 500  # fallback estimate per message
    return total_chars // 4
