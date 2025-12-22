"""Type definitions for agent framework.

This module provides both provider-specific types (AgentResult with Anthropic types)
and provider-agnostic generic types (GenericUsage, GenericAgentResult) for multi-provider
support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING

from anthropic.types.beta import BetaMessage, BetaUsage

@dataclass
class AgentResult:
    """Result object returned from agent.run() containing the full agent execution context.
    
    Attributes:
        final_message: The last message from the assistant (BetaMessage object)
        conversation_history: Full conversation history including all messages (uncompacted)
        stop_reason: Why the model stopped generating ("end_turn", "tool_use", "max_tokens", etc.)
        model: The model used for this execution
        usage: Token usage statistics from the final message
        container_id: Container ID for multi-turn conversations (if applicable)
        total_steps: Number of agent steps taken (including tool use loops)
        agent_logs: List of log entries tracking compactions and other agent actions
        generated_files: List of metadata dicts for files generated during the run
    """
    final_message: BetaMessage
    conversation_history: list[dict]
    stop_reason: str
    model: str
    usage: BetaUsage
    container_id: Optional[str] = None
    total_steps: int = 1
    agent_logs: Optional[list[dict]] = None
    generated_files: Optional[list[dict]] = None
    final_answer: str = ""

    def __str__(self) -> str:
        """Return a JSON-formatted representation with all fields."""
        payload = {
            "final_message": self.final_message,
            "final_answer": self.final_answer,
            "conversation_history": self.conversation_history,
            "stop_reason": self.stop_reason,
            "model": self.model,
            "usage": self.usage,
            "container_id": self.container_id,
            "total_steps": self.total_steps,
            "agent_logs": self.agent_logs,
            "generated_files": self.generated_files,
        }
        return json.dumps(payload, indent=2, default=str)
    
    def to_generic(self) -> "GenericAgentResult":
        """Convert to provider-agnostic GenericAgentResult.
        
        Returns:
            GenericAgentResult with all fields converted to generic types.
        """
        return GenericAgentResult.from_anthropic(self)


# =============================================================================
# Provider-Agnostic Types (for multi-provider support)
# =============================================================================

@dataclass
class GenericUsage:
    """Provider-agnostic token usage information.
    
    This class provides a common representation of token usage across
    different LLM providers. Use the from_* class methods to convert
    from provider-specific usage objects.
    
    Attributes:
        input_tokens: Number of tokens in the input/prompt
        output_tokens: Number of tokens in the output/completion
        cache_creation_tokens: Tokens used to create cache entries (if supported)
        cache_read_tokens: Tokens read from cache (if supported)
    """
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    
    @classmethod
    def from_anthropic(cls, usage: BetaUsage) -> "GenericUsage":
        """Convert from Anthropic BetaUsage to GenericUsage.
        
        Args:
            usage: Anthropic BetaUsage object
            
        Returns:
            GenericUsage with values extracted from Anthropic usage.
        """
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_tokens=getattr(usage, 'cache_creation_input_tokens', 0) or 0,
            cache_read_tokens=getattr(usage, 'cache_read_input_tokens', 0) or 0,
        )
    
    @classmethod
    def from_openai(cls, usage: Any) -> "GenericUsage":
        """Convert from OpenAI usage to GenericUsage.
        
        Args:
            usage: OpenAI usage object (CompletionUsage or similar)
            
        Returns:
            GenericUsage with values extracted from OpenAI usage.
        """
        return cls(
            input_tokens=getattr(usage, 'prompt_tokens', 0) or 0,
            output_tokens=getattr(usage, 'completion_tokens', 0) or 0,
            cache_creation_tokens=0,  # OpenAI doesn't expose cache metrics
            cache_read_tokens=0,
        )
    
    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens
    
    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary representation."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class GenericAgentResult:
    """Provider-agnostic agent execution result.
    
    This class provides a common representation of agent execution results
    across different LLM providers. It contains all the information needed
    to understand what happened during an agent run.
    
    Use the from_* class methods to convert from provider-specific result
    objects.
    
    Attributes:
        final_message: The final assistant message as a dictionary
        final_answer: Extracted text answer from the final message
        conversation_history: Full list of messages in the conversation
        stop_reason: Why the model stopped ("end_turn", "tool_use", "max_tokens", etc.)
        model: Model identifier used for this execution
        usage: Token usage statistics
        total_steps: Number of agent steps taken
        agent_logs: List of log entries for debugging/monitoring
        generated_files: List of file metadata for any generated files
        provider_metadata: Provider-specific additional data
    """
    final_message: dict[str, Any]
    final_answer: str
    conversation_history: list[dict[str, Any]]
    stop_reason: str
    model: str
    usage: GenericUsage
    total_steps: int = 1
    agent_logs: Optional[list[dict[str, Any]]] = None
    generated_files: Optional[list[dict[str, Any]]] = None
    provider_metadata: Optional[dict[str, Any]] = None
    
    @classmethod
    def from_anthropic(cls, result: "AgentResult") -> "GenericAgentResult":
        """Convert from Anthropic AgentResult to GenericAgentResult.
        
        Args:
            result: Anthropic-specific AgentResult object
            
        Returns:
            GenericAgentResult with all fields converted to generic types.
        """
        # Convert BetaMessage to dict
        if hasattr(result.final_message, 'model_dump'):
            final_message_dict = result.final_message.model_dump(
                mode="json",
                exclude_unset=True,
                warnings=False
            )
        else:
            final_message_dict = {
                "role": getattr(result.final_message, 'role', 'assistant'),
                "content": getattr(result.final_message, 'content', []),
            }
        
        # Build provider metadata with Anthropic-specific fields
        provider_metadata = {
            "provider": "anthropic",
        }
        if result.container_id:
            provider_metadata["container_id"] = result.container_id
        
        return cls(
            final_message=final_message_dict,
            final_answer=result.final_answer,
            conversation_history=result.conversation_history,
            stop_reason=result.stop_reason,
            model=result.model,
            usage=GenericUsage.from_anthropic(result.usage),
            total_steps=result.total_steps,
            agent_logs=result.agent_logs,
            generated_files=result.generated_files,
            provider_metadata=provider_metadata,
        )
    
    def __str__(self) -> str:
        """Return a JSON-formatted representation with all fields."""
        payload = {
            "final_message": self.final_message,
            "final_answer": self.final_answer,
            "conversation_history": self.conversation_history,
            "stop_reason": self.stop_reason,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "total_steps": self.total_steps,
            "agent_logs": self.agent_logs,
            "generated_files": self.generated_files,
            "provider_metadata": self.provider_metadata,
        }
        return json.dumps(payload, indent=2, default=str)