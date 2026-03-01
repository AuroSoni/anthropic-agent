"""Tool infrastructure — registry, decorator, envelope pattern, and configurable base."""

from .tool_types import ToolResultEnvelope, GenericErrorEnvelope, GenericTextEnvelope, ToolSchema
from .decorators import tool, ExecutorType
from .base import ConfigurableToolBase
from .registry import ToolRegistry, ToolCallInfo, ToolCallClassification, RegisteredTool
from .schema_utils import (
    generate_tool_schema,
    TypeHintParsingException,
    DocstringParsingException,
)

__all__ = [
    # Schema
    "ToolSchema",
    # Envelope types
    "ToolResultEnvelope",
    "GenericErrorEnvelope",
    "GenericTextEnvelope",
    # Decorator
    "tool",
    "ExecutorType",
    # Configurable base
    "ConfigurableToolBase",
    # Registry
    "ToolRegistry",
    "ToolCallInfo",
    "ToolCallClassification",
    "RegisteredTool",
    # Schema utilities
    "generate_tool_schema",
    "TypeHintParsingException",
    "DocstringParsingException",
]
