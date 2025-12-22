"""
Multi-Provider Agent Library

A library for building AI agents with support for multiple LLM providers
including Anthropic Claude, with streaming, retry logic, and tool execution.

Main exports:
    - BaseAgent: Provider-agnostic base agent class
    - AnthropicAgent: Anthropic-specific agent (extends BaseAgent)
    - GenericAgentResult: Provider-agnostic result type
    - LLMClient: Protocol for implementing new providers

Provider-specific exports:
    - providers.anthropic: AnthropicClient, streaming formatters

Example (Anthropic - familiar API):
    >>> from anthropic_agent import AnthropicAgent
    >>> agent = AnthropicAgent(
    ...     system_prompt="You are a helpful assistant",
    ...     model="claude-sonnet-4-5"
    ... )
    >>> result = await agent.run("Hello!")

Example (Multi-provider pattern):
    >>> from anthropic_agent import BaseAgent
    >>> from anthropic_agent.providers.anthropic import AnthropicClient
    >>> 
    >>> client = AnthropicClient()
    >>> agent = MyCustomAgent(client=client, model="claude-sonnet-4-5")
"""

# Optional: Load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip loading .env file

# Core exports - Provider-agnostic
from .core import (
    BaseAgent,
    LLMClient,
    LLMResponse,
    StreamFormatter,
    Usage,
    GenericUsage,
    GenericAgentResult,
    async_retry,
    retry_with_backoff,
)

# Anthropic-specific exports (backward compatibility)
from .core import AgentResult, AnthropicAgent, anthropic_stream_with_backoff
from .streaming import render_stream, stream_to_aqueue, FormatterType

# Tools
from .tools import (
    ToolRegistry,
    tool,
    anthropic_to_openai,
    openai_to_anthropic,
    convert_schema,
    convert_schemas,
)

__version__ = "0.2.0"

__all__ = [
    # Provider-agnostic base
    'BaseAgent',
    'LLMClient',
    'LLMResponse',
    'StreamFormatter',
    'Usage',
    'GenericUsage',
    'GenericAgentResult',
    # Retry utilities
    'async_retry',
    'retry_with_backoff',
    # Anthropic-specific (backward compatibility)
    'AgentResult',
    'AnthropicAgent',
    'anthropic_stream_with_backoff',
    # Streaming
    'render_stream',
    'stream_to_aqueue',
    'FormatterType',
    # Tools
    'ToolRegistry',
    'tool',
    'anthropic_to_openai',
    'openai_to_anthropic',
    'convert_schema',
    'convert_schemas',
]

