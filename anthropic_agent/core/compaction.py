"""Context compaction strategies for managing message history size.

Compactors are the agent's **within-session context window managers**. Their
sole responsibility is keeping the live message list (``self.messages``) under
the model's token budget so the agent can continue operating without hitting
context limits.

A compactor is self-sufficient: it reads the current message chain, decides
whether compaction is needed, and returns a (possibly shortened) message list
together with metadata describing what it did. Compactors never interact with
the memory store — they are independent systems that operate at different
points in the agent lifecycle.

Available strategies (via ``CompactorType``):

* ``SlidingWindowCompactor`` — progressive multi-phase compaction (default).
* ``SummarizingCompactor`` — LLM-based summarization of older history.
* ``ToolResultRemovalCompactor`` — strips / replaces old tool results.
* ``NoOpCompactor`` — pass-through, compaction disabled.

See also ``anthropic_agent.memory.stores`` for the cross-session knowledge
layer, which operates at run boundaries and is independent of compaction.
"""

from __future__ import annotations

import json
import copy
from abc import ABC, abstractmethod
from typing import Literal, Any
from datetime import datetime

import anthropic

from ..logging import get_logger
from .token_counting import estimate_tokens, get_model_token_limit

logger = get_logger(__name__)

# Type alias for compactor names
CompactorType = Literal["sliding_window", "tool_result_removal", "summarizing", "none"]


class Compactor(ABC):
    """Abstract base class for context compaction strategies.
    
    Compactors take a list of messages and apply a strategy to reduce
    their size while preserving important context. Each compactor manages
    its own threshold and decision logic for when to compact.
    
    All concrete compactors must inherit from this class and implement
    the ``compact()`` method.
    """
    
    @abstractmethod
    async def compact(
        self, 
        messages: list[dict], 
        model: str,
        estimated_tokens: int | None = None,
        enable_cache_control: bool = False,
    ) -> tuple[list[dict], dict[str, Any]]:
        """Apply compaction strategy to messages.
        
        Args:
            messages: List of message dictionaries to compact
            model: Model name being used (for model-specific decisions)
            estimated_tokens: Optional estimated token count from previous API response.
                If provided, this is used instead of the heuristic estimation.
            enable_cache_control: Whether to apply cache_control markers to compacted
                messages. Used by compactors that produce new content (e.g. summaries).
            
        Returns:
            Tuple of (compacted_messages, metadata)
            - compacted_messages: New list with compaction applied (or unchanged if no compaction needed)
            - metadata: Dict with information about what was compacted
        """
        ...


class NoOpCompactor(Compactor):
    """No-operation compactor that returns messages unchanged.
    
    Useful for disabling compaction or as a baseline comparison.
    """
    
    def __init__(self, threshold: int | None = None):
        """Initialize the no-op compactor.
        
        Args:
            threshold: Ignored for NoOpCompactor (accepted for interface consistency)
        """
        self.threshold = threshold
    
    async def compact(self, messages: list[dict], model: str, estimated_tokens: int | None = None, enable_cache_control: bool = False) -> tuple[list[dict], dict[str, Any]]:
        """Return messages unchanged."""
        return messages, {
            "compaction_applied": False,
            "messages_removed": 0,
            "tool_results_modified": 0,
            "estimated_tokens_saved": 0
        }


class ToolResultRemovalCompactor(Compactor):
    """Compactor that removes old tool results and assistant turns.
    
    Strategy:
    1. Check if token count exceeds threshold
    2. Phase 1: Replace old tool result content with placeholder text
    3. Phase 2: If still needed, remove entire old assistant turns (with tool calls + results)
    
    Always preserves:
    - First user message (to maintain original intent)
    - Recent messages (works backwards from oldest)
    
    The compactor manages its own threshold and decides when to compact based on
    the current message token count and model being used.
    """
    
    PLACEHOLDER_TEXT = "[Tool result removed during compaction]"
    
    def __init__(self, threshold: int | None = None, aggressive: bool = False):
        """Initialize the compactor.
        
        Args:
            threshold: Token count threshold to trigger compaction. If None, compaction is disabled.
            aggressive: If True, skips Phase 1 and goes straight to removing entire turns
        """
        self.threshold = threshold
        self.aggressive = aggressive
    
    async def compact(self, messages: list[dict], model: str, estimated_tokens: int | None = None, enable_cache_control: bool = False) -> tuple[list[dict], dict[str, Any]]:
        """Apply tool result removal compaction strategy.
        
        Args:
            messages: List of message dictionaries to compact
            model: Model name being used (for model-specific threshold decisions)
            estimated_tokens: Optional estimated token count from previous API response.
                If provided, this is used instead of the heuristic estimation.
            enable_cache_control: Ignored by this compactor.
            
        Returns:
            Tuple of (compacted_messages, metadata)
        """
        # Check if compaction is needed based on threshold
        # Use passed-in estimate if available, otherwise use heuristic
        original_tokens = estimated_tokens if estimated_tokens is not None else estimate_tokens(messages)
        
        if self.threshold is None or original_tokens <= self.threshold:
            # No compaction needed
            return messages, {
                "compaction_applied": False,
                "messages_removed": 0,
                "tool_results_modified": 0,
                "estimated_tokens_saved": 0,
                "reason": "below_threshold" if self.threshold else "threshold_not_set",
                "original_token_estimate": original_tokens,
                "threshold": self.threshold,
                "model": model
            }
        
        if len(messages) <= 1:
            # Nothing to compact if we only have 0-1 messages
            return messages, {
                "compaction_applied": False,
                "messages_removed": 0,
                "tool_results_modified": 0,
                "estimated_tokens_saved": 0,
                "reason": "insufficient_messages",
                "original_token_estimate": original_tokens,
                "threshold": self.threshold,
                "model": model
            }
        
        # Deep copy to avoid modifying original
        compacted = copy.deepcopy(messages)
        
        tool_results_modified = 0
        messages_removed = 0
        
        if not self.aggressive:
            # Phase 1: Replace tool result content with placeholders
            compacted, tool_results_modified = self._replace_tool_results(compacted)
        
        # Phase 2: Remove entire old turns if needed
        # For now, we'll apply Phase 2 if Phase 1 was skipped (aggressive mode)
        # or if we want to be more aggressive in the future
        if self.aggressive:
            compacted, messages_removed = self._remove_old_turns(compacted)
        
        final_tokens = estimate_tokens(compacted)
        tokens_saved = original_tokens - final_tokens
        
        return compacted, {
            "compaction_applied": True,
            "messages_removed": messages_removed,
            "tool_results_modified": tool_results_modified,
            "estimated_tokens_saved": tokens_saved,
            "original_token_estimate": original_tokens,
            "final_token_estimate": final_tokens,
            "threshold": self.threshold,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }
    
    def _replace_tool_results(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Replace tool result content with placeholders.
        
        Works backwards from oldest messages (after first user message).
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (modified_messages, count_of_modified_results)
        """
        tool_results_modified = 0
        
        # Skip first message (always preserve it)
        # Iterate through messages from index 1 onwards
        for i in range(1, len(messages)):
            msg = messages[i]
            
            # Look for user messages with tool results
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for content_block in msg["content"]:
                    if isinstance(content_block, dict) and content_block.get("type") == "tool_result":
                        # Replace the content with placeholder
                        if content_block.get("content") != self.PLACEHOLDER_TEXT:
                            content_block["content"] = self.PLACEHOLDER_TEXT
                            tool_results_modified += 1
        
        return messages, tool_results_modified
    
    def _remove_old_turns(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Remove entire old assistant turns with their tool calls and results.
        
        Works backwards, removing pairs of assistant messages (with tool_use)
        and their corresponding user messages (with tool_result).
        
        Always preserves first user message.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (modified_messages, count_of_removed_messages)
        """
        if len(messages) <= 2:
            # Keep at least first user message and one response
            return messages, 0
        
        messages_removed = 0
        filtered_messages = [messages[0]]  # Always keep first message
        
        # Find assistant/user pairs to remove from oldest onwards
        # Start from index 1, look for assistant messages with tool_use
        i = 1
        removed_indices = set()
        
        while i < len(messages) - 2:  # Keep at least the last 2 messages
            msg = messages[i]
            
            # Check if this is an assistant message with tool_use
            if msg.get("role") == "assistant":
                has_tool_use = False
                content = msg.get("content", [])
                if isinstance(content, list):
                    has_tool_use = any(
                        isinstance(block, dict) and block.get("type") == "tool_use"
                        for block in content
                    )
                
                # If has tool_use, check if next message is user with tool_result
                if has_tool_use and i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if next_msg.get("role") == "user":
                        next_content = next_msg.get("content", [])
                        if isinstance(next_content, list):
                            has_tool_result = any(
                                isinstance(block, dict) and block.get("type") == "tool_result"
                                for block in next_content
                            )
                            
                            if has_tool_result:
                                # Mark both for removal
                                removed_indices.add(i)
                                removed_indices.add(i + 1)
                                messages_removed += 2
                                i += 2  # Skip both messages
                                continue
            
            i += 1
        
        # Build filtered list (keep first message + non-removed messages)
        for i in range(1, len(messages)):
            if i not in removed_indices:
                filtered_messages.append(messages[i])
        
        return filtered_messages, messages_removed


class SlidingWindowCompactor(Compactor):
    """Progressive compactor that applies multiple strategies until under threshold.
    
    This is the recommended default compactor for production use. It applies
    increasingly aggressive compaction strategies in order:
    
    1. Remove thinking blocks from older assistant messages (keeps last one)
    2. Truncate long tool results to max_result_chars
    3. Replace old tool results with placeholders  
    4. Remove entire old assistant/user turn pairs (keeping recent N turns)
    
    The compactor is progressive - it stops as soon as the token count is
    under the threshold, preserving as much context as possible.
    
    Always preserves:
    - First user message (original intent)
    - Most recent messages (configurable via keep_recent_turns)
    - Thinking block on the LAST assistant message (required by Anthropic API)
    
    Note on thinking blocks:
        When extended thinking is enabled, the Anthropic API requires the final
        assistant message to start with a thinking block. Older thinking blocks
        can be safely removed - the API automatically ignores them and doesn't
        count them toward context usage. This compactor removes thinking blocks
        from all assistant messages EXCEPT the last one, which is both safe and
        more token-efficient.
    """
    
    PLACEHOLDER_TEXT = "[Content removed during compaction]"
    TRUNCATION_SUFFIX = "\n\n[... truncated ...]"
    
    def __init__(
        self,
        threshold: int | None = None,
        keep_recent_turns: int = 10,
        max_result_chars: int = 2000,
        remove_thinking: bool = True,
    ):
        """Initialize the sliding window compactor.
        
        Args:
            threshold: Token count threshold to trigger compaction. If None,
                uses model-specific defaults from MODEL_TOKEN_LIMITS.
            keep_recent_turns: Minimum number of recent turn pairs to preserve.
                A "turn" is an assistant message + its tool results (if any).
                Used for tool result truncation/replacement and turn removal phases.
            max_result_chars: Maximum characters to keep in tool results during
                truncation phase. Results longer than this are truncated.
            remove_thinking: Whether to remove thinking blocks from older assistant
                messages. When True (recommended), removes thinking blocks from all
                assistant messages EXCEPT the last one. The last assistant message
                always retains its thinking block as required by the Anthropic API
                when extended thinking is enabled with tool use. Older thinking
                blocks are safely removed - the API ignores them automatically.
        """
        self.threshold = threshold
        self.keep_recent_turns = keep_recent_turns
        self.max_result_chars = max_result_chars
        self.remove_thinking = remove_thinking
    
    def _get_threshold(self, model: str) -> int:
        """Get effective threshold, using model default if not explicitly set."""
        if self.threshold is not None:
            return self.threshold
        return get_model_token_limit(model)
    
    async def compact(
        self, 
        messages: list[dict], 
        model: str,
        estimated_tokens: int | None = None,
        enable_cache_control: bool = False,
    ) -> tuple[list[dict], dict[str, Any]]:
        """Apply progressive compaction until under threshold.
        
        Args:
            messages: List of message dictionaries to compact
            model: Model name being used
            estimated_tokens: Optional estimated token count from previous API response
            enable_cache_control: Ignored by this compactor.
            
        Returns:
            Tuple of (compacted_messages, metadata)
        """
        threshold = self._get_threshold(model)
        original_tokens = estimated_tokens if estimated_tokens is not None else estimate_tokens(messages)
        
        # Track what was done
        metadata: dict[str, Any] = {
            "compaction_applied": False,
            "original_token_estimate": original_tokens,
            "threshold": threshold,
            "model": model,
            "phases_applied": [],
            "thinking_blocks_removed": 0,
            "tool_results_truncated": 0,
            "tool_results_replaced": 0,
            "messages_removed": 0,
        }
        
        # Check if compaction is needed
        if original_tokens <= threshold:
            metadata["reason"] = "below_threshold"
            return messages, metadata
        
        if len(messages) <= 1:
            metadata["reason"] = "insufficient_messages"
            return messages, metadata
        
        # Deep copy to avoid modifying original
        compacted = copy.deepcopy(messages)
        current_tokens = original_tokens
        
        # Phase 1: Remove thinking blocks
        if self.remove_thinking:
            compacted, thinking_removed = self._remove_thinking_blocks(compacted)
            if thinking_removed > 0:
                metadata["phases_applied"].append("remove_thinking")
                metadata["thinking_blocks_removed"] = thinking_removed
                current_tokens = estimate_tokens(compacted)
                logger.info("Compaction phase 1: removed thinking blocks", removed=thinking_removed, tokens=current_tokens)
                
                if current_tokens <= threshold:
                    metadata["compaction_applied"] = True
                    metadata["final_token_estimate"] = current_tokens
                    metadata["estimated_tokens_saved"] = original_tokens - current_tokens
                    metadata["timestamp"] = datetime.now().isoformat()
                    return compacted, metadata
        
        # Phase 2: Truncate long tool results
        compacted, truncated_count = self._truncate_tool_results(compacted)
        if truncated_count > 0:
            metadata["phases_applied"].append("truncate_results")
            metadata["tool_results_truncated"] = truncated_count
            current_tokens = estimate_tokens(compacted)
            logger.info("Compaction phase 2: truncated tool results", truncated=truncated_count, tokens=current_tokens)
            
            if current_tokens <= threshold:
                metadata["compaction_applied"] = True
                metadata["final_token_estimate"] = current_tokens
                metadata["estimated_tokens_saved"] = original_tokens - current_tokens
                metadata["timestamp"] = datetime.now().isoformat()
                return compacted, metadata
        
        # Phase 3: Replace old tool results with placeholders
        compacted, replaced_count = self._replace_old_tool_results(compacted)
        if replaced_count > 0:
            metadata["phases_applied"].append("replace_results")
            metadata["tool_results_replaced"] = replaced_count
            current_tokens = estimate_tokens(compacted)
            logger.info("Compaction phase 3: replaced tool results", replaced=replaced_count, tokens=current_tokens)
            
            if current_tokens <= threshold:
                metadata["compaction_applied"] = True
                metadata["final_token_estimate"] = current_tokens
                metadata["estimated_tokens_saved"] = original_tokens - current_tokens
                metadata["timestamp"] = datetime.now().isoformat()
                return compacted, metadata
        
        # Phase 4: Remove old turns (sliding window)
        compacted, removed_count = self._remove_old_turns(compacted, threshold)
        if removed_count > 0:
            metadata["phases_applied"].append("remove_turns")
            metadata["messages_removed"] = removed_count
            current_tokens = estimate_tokens(compacted)
            logger.info("Compaction phase 4: removed messages", removed=removed_count, tokens=current_tokens)
        
        metadata["compaction_applied"] = True
        metadata["final_token_estimate"] = current_tokens
        metadata["estimated_tokens_saved"] = original_tokens - current_tokens
        metadata["timestamp"] = datetime.now().isoformat()
        
        # Warn if still over threshold after all phases
        if current_tokens > threshold:
            logger.warning("Compaction complete but still over threshold", tokens=current_tokens, threshold=threshold)
            metadata["warning"] = "still_over_threshold"
        
        return compacted, metadata
    
    def _remove_thinking_blocks(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Remove thinking blocks from all assistant messages EXCEPT the last one.
        
        When extended thinking is enabled in the Anthropic API, the LAST assistant
        message in the conversation history must retain its thinking block. This is
        required by the API when using tools with extended thinking - the final
        assistant turn must start with a thinking block.
        
        However, older thinking blocks can be safely removed. Per Anthropic's
        documentation:
        - "It is only strictly necessary to send back thinking blocks when using
          tools with extended thinking"
        - "You must include the complete unmodified block back to the API for
          the last assistant turn"
        - "The API automatically ignores thinking blocks from previous turns and
          they are not included when calculating context usage"
        
        This method removes thinking blocks from all assistant messages except the
        very last one, which is both safe and more token-efficient than preserving
        thinking in the last N messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (modified_messages, count_of_removed_blocks)
        """
        removed_count = 0
        
        # Find indices of assistant messages
        assistant_indices = [
            i for i, msg in enumerate(messages) 
            if msg.get("role") == "assistant"
        ]
        
        # Nothing to do if no assistant messages
        if not assistant_indices:
            return messages, 0
        
        # CRITICAL: Always preserve thinking block on the LAST assistant message.
        # The Anthropic API requires the final assistant turn to start with a
        # thinking block when extended thinking is enabled with tool use.
        # Older thinking blocks can be safely removed.
        last_assistant_idx = assistant_indices[-1]
        indices_to_process = [i for i in assistant_indices if i != last_assistant_idx]
        
        for i in indices_to_process:
            msg = messages[i]
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            
            # Filter out thinking blocks from old messages
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    removed_count += 1
                else:
                    new_content.append(block)
            
            # Only update if we actually removed something
            if len(new_content) < len(content):
                msg["content"] = new_content
        
        return messages, removed_count
    
    def _truncate_tool_results(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Truncate long tool result content to max_result_chars.
        
        Works on all tool results except the most recent ones.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (modified_messages, count_of_truncated_results)
        """
        truncated_count = 0
        
        # Find indices of user messages with tool results
        tool_result_indices = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in msg["content"]
                )
                if has_tool_result:
                    tool_result_indices.append(i)
        
        # Keep recent tool results intact
        indices_to_truncate = tool_result_indices[:-self.keep_recent_turns] if len(tool_result_indices) > self.keep_recent_turns else []
        
        for i in indices_to_truncate:
            msg = messages[i]
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, str) and len(content) > self.max_result_chars:
                        block["content"] = content[:self.max_result_chars] + self.TRUNCATION_SUFFIX
                        truncated_count += 1
                    elif isinstance(content, list):
                        # Handle content that's a list of blocks
                        for inner_block in content:
                            if isinstance(inner_block, dict) and inner_block.get("type") == "text":
                                text = inner_block.get("text", "")
                                if len(text) > self.max_result_chars:
                                    inner_block["text"] = text[:self.max_result_chars] + self.TRUNCATION_SUFFIX
                                    truncated_count += 1
        
        return messages, truncated_count
    
    def _replace_old_tool_results(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Replace old tool result content with placeholders.
        
        Keeps recent tool results intact based on keep_recent_turns.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (modified_messages, count_of_replaced_results)
        """
        replaced_count = 0
        
        # Find indices of user messages with tool results
        tool_result_indices = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in msg["content"]
                )
                if has_tool_result:
                    tool_result_indices.append(i)
        
        # Replace all but the most recent keep_recent_turns tool results
        indices_to_replace = tool_result_indices[:-self.keep_recent_turns] if len(tool_result_indices) > self.keep_recent_turns else []
        
        for i in indices_to_replace:
            msg = messages[i]
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if block.get("content") != self.PLACEHOLDER_TEXT:
                        block["content"] = self.PLACEHOLDER_TEXT
                        replaced_count += 1
        
        return messages, replaced_count
    
    def _remove_old_turns(self, messages: list[dict], threshold: int) -> tuple[list[dict], int]:
        """Remove old assistant/user turn pairs until under threshold.
        
        Preserves:
        - First user message (index 0)
        - Most recent keep_recent_turns pairs
        
        Args:
            messages: List of message dictionaries
            threshold: Target token threshold
            
        Returns:
            Tuple of (filtered_messages, count_of_removed_messages)
        """
        if len(messages) <= 2:
            return messages, 0
        
        # Always keep first message
        first_message = messages[0]
        remaining = messages[1:]
        
        # Calculate how many messages we need to keep for minimum recent turns
        # Each "turn" is roughly 2 messages (assistant + user with tool result)
        min_keep = self.keep_recent_turns * 2
        
        # Keep removing oldest messages until under threshold or at minimum
        removed_count = 0
        while len(remaining) > min_keep:
            current_messages = [first_message] + remaining
            current_tokens = estimate_tokens(current_messages)
            
            if current_tokens <= threshold:
                break
            
            # Remove the oldest message (first in remaining)
            remaining = remaining[1:]
            removed_count += 1
        
        return [first_message] + remaining, removed_count


_DEFAULT_SUMMARY_PROMPT = """\
You are a conversation summarizer. Given the conversation history below, \
produce a concise structured summary preserving the following categories. \
Omit any category that has no relevant content.

**User Intent** — Original request and any refinements.
**Completed Work** — Actions performed, identifiers, values produced.
**Errors & Corrections** — What failed and how the user corrected it (quote corrections verbatim).
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
    sent to the LLM for summarization. The resulting summary replaces the
    older messages, while recent turns are preserved verbatim. This follows
    Anthropic's recommended pattern for context window management.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        threshold: int | None = None,
        clear_at_least_tokens: int = 10_000,
        keep_recent_turns: int = 5,
        summary_model: str | None = None,
        summary_prompt: str | None = None,
    ):
        self.client = client
        self.threshold = threshold
        self.clear_at_least_tokens = clear_at_least_tokens
        self.keep_recent_turns = keep_recent_turns
        self.summary_model = summary_model
        self.summary_prompt = summary_prompt or _DEFAULT_SUMMARY_PROMPT

    def _get_threshold(self, model: str) -> int:
        if self.threshold is not None:
            return self.threshold
        return get_model_token_limit(model)

    @staticmethod
    def _split_at_turn_boundary(messages: list[dict], keep_count: int) -> tuple[list[dict], list[dict]]:
        """Split messages into (to_summarize, to_keep) at a turn boundary.

        A "turn" here is a contiguous group of messages starting with an
        assistant message and including any immediately following user message
        that carries tool results. We walk backwards from the end of
        ``messages`` counting ``keep_count`` such turns, then split.
        """
        if keep_count <= 0 or len(messages) <= 1:
            return messages, []

        turns_found = 0
        split_idx = len(messages)

        i = len(messages) - 1
        while i >= 1 and turns_found < keep_count:
            msg = messages[i]
            if msg.get("role") == "assistant":
                turns_found += 1
                split_idx = i
            elif msg.get("role") == "user":
                content = msg.get("content", [])
                has_tool_result = isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                )
                if not has_tool_result:
                    turns_found += 1
                    split_idx = i
            i -= 1

        to_summarize = messages[:split_idx]
        to_keep = messages[split_idx:]
        return to_summarize, to_keep

    async def compact(
        self,
        messages: list[dict],
        model: str,
        estimated_tokens: int | None = None,
        enable_cache_control: bool = False,
    ) -> tuple[list[dict], dict[str, Any]]:
        threshold = self._get_threshold(model)
        original_tokens = estimated_tokens if estimated_tokens is not None else estimate_tokens(messages)

        if original_tokens <= threshold:
            return messages, {
                "compaction_applied": False,
                "reason": "below_threshold",
                "original_token_estimate": original_tokens,
                "threshold": threshold,
            }

        if len(messages) <= 1:
            return messages, {
                "compaction_applied": False,
                "reason": "insufficient_messages",
                "original_token_estimate": original_tokens,
                "threshold": threshold,
            }

        keep_turns = self.keep_recent_turns
        to_summarize, to_keep = self._split_at_turn_boundary(messages, keep_turns)

        # Iteratively move the split point until we would free enough tokens
        summary_overhead_estimate = 1_500  # rough token budget for the summary itself
        while keep_turns > 1:
            kept_tokens = estimate_tokens(to_keep) + summary_overhead_estimate
            freed = original_tokens - kept_tokens
            if freed >= self.clear_at_least_tokens:
                break
            keep_turns -= 1
            to_summarize, to_keep = self._split_at_turn_boundary(messages, keep_turns)

        if not to_summarize:
            return messages, {
                "compaction_applied": False,
                "reason": "nothing_to_summarize",
                "original_token_estimate": original_tokens,
                "threshold": threshold,
            }

        summary_model_used = self.summary_model or model

        summary_response = await self.client.messages.create(
            model=summary_model_used,
            max_tokens=4096,
            system=self.summary_prompt,
            messages=to_summarize,
        )
        summary_text = summary_response.content[0].text

        if not summary_text.startswith("<session_summary>"):
            summary_text = f"<session_summary>\n{summary_text}\n</session_summary>"

        summary_content_block: dict[str, Any] = {"type": "text", "text": summary_text}
        if enable_cache_control:
            summary_content_block["cache_control"] = {"type": "ephemeral"}

        summary_message: dict[str, Any] = {
            "role": "user",
            "content": [summary_content_block],
        }

        compacted = [summary_message] + to_keep
        final_tokens = estimate_tokens(compacted)

        return compacted, {
            "compaction_applied": True,
            "original_token_estimate": original_tokens,
            "final_token_estimate": final_tokens,
            "estimated_tokens_saved": original_tokens - final_tokens,
            "messages_summarized": len(to_summarize),
            "messages_kept": len(to_keep),
            "summary_model_used": summary_model_used,
            "threshold": threshold,
            "timestamp": datetime.now().isoformat(),
        }


# Compactor registry mapping string names to compactor classes
COMPACTORS: dict[str, type[Compactor]] = {
    "sliding_window": SlidingWindowCompactor,
    "tool_result_removal": ToolResultRemovalCompactor,
    "summarizing": SummarizingCompactor,
    "none": NoOpCompactor,
}

# Default compactor for production use
DEFAULT_COMPACTOR: CompactorType = "sliding_window"


def get_compactor(name: CompactorType, threshold: int | None = None, **kwargs) -> Compactor:
    """Get a compactor instance by name.
    
    Args:
        name: Compactor name ("sliding_window", "tool_result_removal",
            "summarizing", or "none")
        threshold: Token count threshold for compaction. The compactor uses this to decide
            when to compact. If None, model-specific defaults are used (for sliding_window
            and summarizing) or compaction is disabled (for other compactors).
        **kwargs: Additional arguments to pass to the compactor constructor.
            For SummarizingCompactor a ``client`` (anthropic.AsyncAnthropic)
            kwarg is required.
        
    Returns:
        An instance of the requested compactor
        
    Raises:
        ValueError: If compactor name is not recognized or required kwargs missing.
    """
    if name not in COMPACTORS:
        raise ValueError(
            f"Unknown compactor '{name}'. Available compactors: {list(COMPACTORS.keys())}"
        )
    
    if name == "summarizing" and "client" not in kwargs:
        raise ValueError(
            "SummarizingCompactor requires a 'client' (anthropic.AsyncAnthropic) kwarg."
        )
    
    compactor_class = COMPACTORS[name]
    return compactor_class(threshold=threshold, **kwargs)


def get_default_compactor(threshold: int | None = None) -> Compactor:
    """Get the default compactor instance for production use.
    
    This returns a SlidingWindowCompactor with sensible defaults:
    - Uses model-specific token thresholds (or provided threshold)
    - Keeps 10 recent turns for tool results and turn removal
    - Truncates tool results > 2000 chars
    - Removes thinking blocks from older messages (preserves last assistant's)
    
    Note: Thinking blocks are removed from all assistant messages except the last
    one, which is required by the Anthropic API when extended thinking is enabled.
    
    Args:
        threshold: Optional override for token threshold. If None, uses
            model-specific defaults (~160k for Claude models).
            
    Returns:
        A configured SlidingWindowCompactor instance
    """
    return SlidingWindowCompactor(
        threshold=threshold,
        keep_recent_turns=10,
        max_result_chars=2000,
        remove_thinking=True,
    )

