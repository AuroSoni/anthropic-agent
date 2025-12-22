"""Schema converters for tool definitions across LLM providers.

This module provides bidirectional conversion between different tool schema formats:
- Anthropic format: Uses `input_schema` with JSON Schema
- OpenAI format: Uses `parameters` wrapped in `function` object
- Universal format: Provider-agnostic intermediate representation

The converters enable tools defined for one provider to be used with another,
making it easy to build provider-agnostic agent applications.
"""

from typing import Any, Literal


SchemaFormat = Literal["anthropic", "openai", "universal"]


def anthropic_to_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic tool schema to OpenAI function calling format.
    
    Anthropic format:
    ```
    {
        "name": "tool_name",
        "description": "Tool description",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    ```
    
    OpenAI format:
    ```
    {
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "Tool description",
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
    }
    ```
    
    Args:
        schema: Anthropic tool schema dictionary
        
    Returns:
        OpenAI function calling format dictionary
        
    Raises:
        ValueError: If required fields are missing from the schema
    """
    if "name" not in schema:
        raise ValueError("Anthropic schema missing required 'name' field")
    
    # Build OpenAI function format
    function_def: dict[str, Any] = {
        "name": schema["name"],
    }
    
    # Description is optional but recommended
    if "description" in schema:
        function_def["description"] = schema["description"]
    
    # Convert input_schema to parameters
    if "input_schema" in schema:
        function_def["parameters"] = schema["input_schema"]
    else:
        # Default to empty object schema if no input_schema
        function_def["parameters"] = {
            "type": "object",
            "properties": {},
        }
    
    return {
        "type": "function",
        "function": function_def,
    }


def openai_to_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI function calling format to Anthropic tool schema.
    
    OpenAI format:
    ```
    {
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "Tool description",
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
    }
    ```
    
    Anthropic format:
    ```
    {
        "name": "tool_name",
        "description": "Tool description",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    ```
    
    Args:
        schema: OpenAI function calling format dictionary
        
    Returns:
        Anthropic tool schema dictionary
        
    Raises:
        ValueError: If required fields are missing or format is invalid
    """
    # Handle both wrapped and unwrapped formats
    if "function" in schema:
        function_def = schema["function"]
    elif "name" in schema:
        # Already in unwrapped format (just the function definition)
        function_def = schema
    else:
        raise ValueError(
            "OpenAI schema must have 'function' key or be a function definition with 'name'"
        )
    
    if "name" not in function_def:
        raise ValueError("OpenAI function schema missing required 'name' field")
    
    # Build Anthropic format
    anthropic_schema: dict[str, Any] = {
        "name": function_def["name"],
    }
    
    # Description is optional but recommended
    if "description" in function_def:
        anthropic_schema["description"] = function_def["description"]
    
    # Convert parameters to input_schema
    if "parameters" in function_def:
        anthropic_schema["input_schema"] = function_def["parameters"]
    else:
        # Default to empty object schema if no parameters
        anthropic_schema["input_schema"] = {
            "type": "object",
            "properties": {},
        }
    
    return anthropic_schema


def to_universal(schema: dict[str, Any], source_format: SchemaFormat) -> dict[str, Any]:
    """Convert any provider format to universal intermediate representation.
    
    Universal format:
    ```
    {
        "name": "tool_name",
        "description": "Tool description",
        "parameters": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    ```
    
    This is similar to OpenAI's inner function format but without the wrapper.
    It serves as an intermediate representation for converting between formats.
    
    Args:
        schema: Tool schema in the specified source format
        source_format: The format of the input schema ("anthropic", "openai", or "universal")
        
    Returns:
        Universal format dictionary
        
    Raises:
        ValueError: If source_format is not recognized or schema is invalid
    """
    if source_format == "universal":
        # Already in universal format
        return schema.copy()
    
    if source_format == "anthropic":
        return {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema.get("input_schema", {"type": "object", "properties": {}}),
        }
    
    if source_format == "openai":
        # Extract from wrapper if present
        if "function" in schema:
            function_def = schema["function"]
        else:
            function_def = schema
        
        return {
            "name": function_def["name"],
            "description": function_def.get("description", ""),
            "parameters": function_def.get("parameters", {"type": "object", "properties": {}}),
        }
    
    raise ValueError(f"Unknown source format: {source_format}. Expected 'anthropic', 'openai', or 'universal'")


def from_universal(schema: dict[str, Any], target_format: SchemaFormat) -> dict[str, Any]:
    """Convert universal format to a specific provider format.
    
    Args:
        schema: Tool schema in universal format
        target_format: The target format ("anthropic", "openai", or "universal")
        
    Returns:
        Schema in the target format
        
    Raises:
        ValueError: If target_format is not recognized
    """
    if target_format == "universal":
        return schema.copy()
    
    if target_format == "anthropic":
        result: dict[str, Any] = {
            "name": schema["name"],
            "input_schema": schema.get("parameters", {"type": "object", "properties": {}}),
        }
        if schema.get("description"):
            result["description"] = schema["description"]
        return result
    
    if target_format == "openai":
        function_def: dict[str, Any] = {
            "name": schema["name"],
            "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
        }
        if schema.get("description"):
            function_def["description"] = schema["description"]
        return {
            "type": "function",
            "function": function_def,
        }
    
    raise ValueError(f"Unknown target format: {target_format}. Expected 'anthropic', 'openai', or 'universal'")


def convert_schema(
    schema: dict[str, Any],
    source_format: SchemaFormat,
    target_format: SchemaFormat,
) -> dict[str, Any]:
    """Convert a tool schema between any two formats.
    
    This is a convenience function that combines to_universal and from_universal.
    
    Args:
        schema: Tool schema in the source format
        source_format: The format of the input schema
        target_format: The desired output format
        
    Returns:
        Schema converted to the target format
        
    Example:
        >>> anthropic_schema = {"name": "add", "input_schema": {...}}
        >>> openai_schema = convert_schema(anthropic_schema, "anthropic", "openai")
    """
    if source_format == target_format:
        return schema.copy()
    
    universal = to_universal(schema, source_format)
    return from_universal(universal, target_format)


def convert_schemas(
    schemas: list[dict[str, Any]],
    source_format: SchemaFormat,
    target_format: SchemaFormat,
) -> list[dict[str, Any]]:
    """Convert a list of tool schemas between formats.
    
    Args:
        schemas: List of tool schemas in the source format
        source_format: The format of the input schemas
        target_format: The desired output format
        
    Returns:
        List of schemas converted to the target format
    """
    return [convert_schema(s, source_format, target_format) for s in schemas]

