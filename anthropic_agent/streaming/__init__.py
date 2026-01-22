"""Streaming utilities for LLM responses.

This module provides streaming formatters for different LLM providers:
- Anthropic: xml_formatter, raw_formatter (via formatters.py)
- OpenAI: openai_xml_formatter, openai_raw_formatter (via openai_formatters.py)
"""
from .renderer import render_stream
from .formatters import (
    xml_formatter,
    raw_formatter,
    stream_to_aqueue,
    FormatterType,
    get_formatter,
)
from .openai_formatters import (
    openai_xml_formatter,
    openai_raw_formatter,
    OpenAIFormatterType,
    OpenAIStreamResult,
    get_openai_formatter,
)

__all__ = [
    # Anthropic formatters
    'render_stream',
    'xml_formatter',
    'raw_formatter',
    'stream_to_aqueue',
    'FormatterType',
    'get_formatter',
    # OpenAI formatters
    'openai_xml_formatter',
    'openai_raw_formatter',
    'OpenAIFormatterType',
    'OpenAIStreamResult',
    'get_openai_formatter',
]

