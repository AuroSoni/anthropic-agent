"""Streaming utilities for Anthropic responses.

This module provides backward-compatible exports for streaming formatters.
The actual implementations now live in providers/anthropic/streaming.py,
but are re-exported here to maintain backward compatibility.

For new code, prefer importing directly from the provider:
    from anthropic_agent.providers.anthropic import xml_formatter, get_formatter
"""

from .renderer import render_stream

# Re-export from provider module for backward compatibility
from ..providers.anthropic.streaming import (
    xml_formatter,
    raw_formatter,
    stream_to_aqueue,
    FormatterType,
    get_formatter,
    FORMATTERS,
)

__all__ = [
    'render_stream',
    'xml_formatter',
    'raw_formatter',
    'stream_to_aqueue',
    'FormatterType',
    'get_formatter',
    'FORMATTERS',
]

