"""Core agent components.

This module exports both provider-agnostic base components and
Anthropic-specific implementations.

Provider-agnostic exports:
    - BaseAgent: Abstract base agent orchestrator
    - LLMClient, LLMResponse, StreamFormatter, Usage: Core protocols
    - GenericUsage, GenericAgentResult: Provider-agnostic types
    - async_retry, retry_with_backoff: Generic retry utilities

Anthropic-specific exports:
    - AnthropicAgent: Anthropic implementation of BaseAgent
    - AgentResult: Legacy Anthropic-specific result type
    - anthropic_stream_with_backoff: Legacy streaming with retry
"""

# Provider-agnostic base agent
from .base_agent import BaseAgent

# Provider-agnostic protocols
from .protocols import (
    LLMClient,
    LLMResponse,
    StreamFormatter,
    Usage,
)

# Provider-agnostic types
from .types import (
    AgentResult,
    GenericUsage,
    GenericAgentResult,
)

# Anthropic-specific components (backward compatibility)
from .agent import AnthropicAgent
from .retry import anthropic_stream_with_backoff, async_retry, retry_with_backoff

# Compaction utilities (provider-agnostic)
from .compaction import (
    CompactorType,
    get_compactor,
    Compactor,
    NoOpCompactor,
    ToolResultRemovalCompactor,
    estimate_tokens,
)

__all__ = [
    # Provider-agnostic base agent
    'BaseAgent',
    # Provider-agnostic protocols
    'LLMClient',
    'LLMResponse',
    'StreamFormatter',
    'Usage',
    # Provider-agnostic types
    'GenericUsage',
    'GenericAgentResult',
    # Anthropic-specific (backward compatibility)
    'AgentResult',
    'AnthropicAgent',
    # Retry utilities
    'async_retry',
    'retry_with_backoff',
    'anthropic_stream_with_backoff',
    # Compaction (provider-agnostic)
    'CompactorType',
    'get_compactor',
    'Compactor',
    'NoOpCompactor',
    'ToolResultRemovalCompactor',
    'estimate_tokens',
]

