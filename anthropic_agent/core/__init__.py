"""Core agent components.

This module exports the provider-agnostic base agent and provider-specific
implementations:
- BaseAgent: Abstract base class for all agents
- AnthropicAgent: Anthropic Claude implementation
- OpenAIAgent: OpenAI GPT implementation
"""
from .types import AgentResult
from .base_agent import BaseAgent, LLMResponse
from .agent import AnthropicAgent
from .openai_agent import OpenAIAgent
from .retry import anthropic_stream_with_backoff
from .openai_retry import openai_stream_with_backoff
from .compaction import (
    CompactorType,
    get_compactor,
    get_default_compactor,
    get_model_token_limit,
    Compactor,
    NoOpCompactor,
    ToolResultRemovalCompactor,
    SlidingWindowCompactor,
    estimate_tokens,
)

__all__ = [
    # Types
    'AgentResult',
    'LLMResponse',
    # Base agent
    'BaseAgent',
    # Provider implementations
    'AnthropicAgent',
    'OpenAIAgent',
    # Retry utilities
    'anthropic_stream_with_backoff',
    'openai_stream_with_backoff',
    # Compaction
    'CompactorType',
    'get_compactor',
    'get_default_compactor',
    'get_model_token_limit',
    'Compactor',
    'NoOpCompactor',
    'ToolResultRemovalCompactor',
    'SlidingWindowCompactor',
    'estimate_tokens',
]

