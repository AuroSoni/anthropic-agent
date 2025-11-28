"""Context compaction strategies for managing message history size.

This module provides pluggable compaction strategies that can reduce the size
of conversation history by removing or summarizing old messages, tool results,
and assistant turns.
"""

import json
import copy
from typing import Protocol, Literal, Any
from datetime import datetime


# Type alias for compactor names
CompactorType = Literal["tool_result_removal", "none"]


def estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count of messages using simple heuristic.
    
    Uses ~4 characters per token as a rough approximation.
    This is a simple heuristic and may not be accurate for all content types.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Estimated token count
    """
    total_chars = 0
    message_json = json.dumps(messages)
    total_chars = len(message_json)
    return total_chars // 4


class Compactor(Protocol):
    """Protocol for context compaction strategies.
    
    Compactors take a list of messages and apply a strategy to reduce
    their size while preserving important context. Each compactor manages
    its own threshold and decision logic for when to compact.
    """
    
    def compact(
        self, 
        messages: list[dict], 
        model: str,
        estimated_tokens: int | None = None
    ) -> tuple[list[dict], dict[str, Any]]:
        """Apply compaction strategy to messages.
        
        Args:
            messages: List of message dictionaries to compact
            model: Model name being used (for model-specific decisions)
            estimated_tokens: Optional estimated token count from previous API response.
                If provided, this is used instead of the heuristic estimation.
            
        Returns:
            Tuple of (compacted_messages, metadata)
            - compacted_messages: New list with compaction applied (or unchanged if no compaction needed)
            - metadata: Dict with information about what was compacted
        """
        ...


class NoOpCompactor:
    """No-operation compactor that returns messages unchanged.
    
    Useful for disabling compaction or as a baseline comparison.
    """
    
    def __init__(self, threshold: int | None = None):
        """Initialize the no-op compactor.
        
        Args:
            threshold: Ignored for NoOpCompactor (accepted for interface consistency)
        """
        self.threshold = threshold
    
    def compact(self, messages: list[dict], model: str, estimated_tokens: int | None = None) -> tuple[list[dict], dict[str, Any]]:
        """Return messages unchanged.
        
        Args:
            messages: List of message dictionaries
            model: Model name (ignored)
            estimated_tokens: Estimated token count (ignored)
            
        Returns:
            Tuple of (messages, empty_metadata)
        """
        return messages, {
            "compaction_applied": False,
            "messages_removed": 0,
            "tool_results_modified": 0,
            "estimated_tokens_saved": 0
        }


class ToolResultRemovalCompactor:
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
    
    def compact(self, messages: list[dict], model: str, estimated_tokens: int | None = None) -> tuple[list[dict], dict[str, Any]]:
        """Apply tool result removal compaction strategy.
        
        Args:
            messages: List of message dictionaries to compact
            model: Model name being used (for model-specific threshold decisions)
            estimated_tokens: Optional estimated token count from previous API response.
                If provided, this is used instead of the heuristic estimation.
            
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


# Compactor registry mapping string names to compactor classes
COMPACTORS: dict[str, type[Compactor]] = {
    "tool_result_removal": ToolResultRemovalCompactor,
    "none": NoOpCompactor,
}


def get_compactor(name: CompactorType, threshold: int | None = None, **kwargs) -> Compactor:
    """Get a compactor instance by name.
    
    Args:
        name: Compactor name ("tool_result_removal" or "none")
        threshold: Token count threshold for compaction. The compactor uses this to decide
            when to compact. If None, the compactor may not perform any compaction.
        **kwargs: Additional arguments to pass to the compactor constructor
            (e.g., aggressive=True for ToolResultRemovalCompactor)
        
    Returns:
        An instance of the requested compactor
        
    Raises:
        ValueError: If compactor name is not recognized
    """
    if name not in COMPACTORS:
        raise ValueError(
            f"Unknown compactor '{name}'. Available compactors: {list(COMPACTORS.keys())}"
        )
    
    compactor_class = COMPACTORS[name]
    return compactor_class(threshold=threshold, **kwargs)

