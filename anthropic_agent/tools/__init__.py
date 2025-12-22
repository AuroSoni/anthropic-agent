"""Tool execution utilities.

This module provides:
- ToolRegistry: Central registry for tool functions and schemas
- @tool decorator: Automatic schema generation from type hints
- Schema converters: Convert between Anthropic, OpenAI, and universal formats
"""

from .base import ToolExecutor, ToolRegistry
from .decorators import tool
from .schema_converters import (
    SchemaFormat,
    anthropic_to_openai,
    openai_to_anthropic,
    to_universal,
    from_universal,
    convert_schema,
    convert_schemas,
)
from .sample_tools import (
    SAMPLE_TOOL_SCHEMAS, 
    SAMPLE_TOOL_FUNCTIONS, 
    execute_tool,
    create_calculator_registry,
    get_tool_schemas,
)

__all__ = [
    # Core interfaces
    'ToolExecutor',
    'ToolRegistry',
    # Decorator
    'tool',
    # Schema converters
    'SchemaFormat',
    'anthropic_to_openai',
    'openai_to_anthropic',
    'to_universal',
    'from_universal',
    'convert_schema',
    'convert_schemas',
    # Sample tools and utilities
    'SAMPLE_TOOL_SCHEMAS', 
    'SAMPLE_TOOL_FUNCTIONS', 
    'execute_tool',
    'create_calculator_registry',
    'get_tool_schemas',
]

