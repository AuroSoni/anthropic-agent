"""Anthropic provider implementation.

This module provides the Anthropic-specific implementation of the LLMClient
protocol for use with Claude models.

Exports:
    - AnthropicClient: LLM client for Anthropic Claude API
    - xml_formatter, raw_formatter: Stream formatting functions
    - get_formatter: Get formatter by name
    - FormatterType: Type alias for formatter names
"""

from .client import AnthropicClient
from .streaming import (
    xml_formatter,
    raw_formatter,
    get_formatter,
    FormatterType,
    stream_to_aqueue,
)
from .types import (
    AnthropicUsage,
    AnthropicResponse,
    convert_to_generic_usage,
    convert_to_generic_result,
    extract_text_from_message,
    extract_file_ids_from_message,
)

__all__ = [
    # Client
    'AnthropicClient',
    # Streaming formatters
    'xml_formatter',
    'raw_formatter',
    'get_formatter',
    'FormatterType',
    'stream_to_aqueue',
    # Types
    'AnthropicUsage',
    'AnthropicResponse',
    'convert_to_generic_usage',
    'convert_to_generic_result',
    'extract_text_from_message',
    'extract_file_ids_from_message',
]

