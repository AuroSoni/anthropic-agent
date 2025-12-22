"""LLM Provider implementations.

This module contains provider-specific implementations of the LLMClient protocol.
Each provider is in its own submodule with its client, streaming formatters,
and type converters.

Available providers:
- anthropic: Anthropic Claude models (AnthropicClient)

Future providers (not yet implemented):
- openai: OpenAI GPT models
- gemini: Google Gemini models
- litellm: LiteLLM unified interface

Example:
    >>> from anthropic_agent.providers.anthropic import AnthropicClient
    >>> from anthropic_agent import BaseAgent
    >>> 
    >>> client = AnthropicClient()
    >>> agent = BaseAgent(client=client, model="claude-sonnet-4-5")
    >>> result = await agent.run("Hello!")
"""

from .anthropic import AnthropicClient

__all__ = [
    'AnthropicClient',
]

