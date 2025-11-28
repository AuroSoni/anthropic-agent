"""Decorators for automatic tool schema generation."""
from typing import Callable
from functools import wraps

from .type_hint_utils import (
    generate_anthropic_schema,
    TypeHintParsingException,
    DocstringParsingException,
)


def tool(func: Callable) -> Callable:
    """Decorator that generates and attaches Anthropic tool schema to a function.
    
    This decorator analyzes a function's type hints and docstring to automatically
    generate an Anthropic-compliant tool schema. The schema is attached to the
    function as the `__tool_schema__` attribute, allowing the function to be
    registered with a ToolRegistry.
    
    The decorator is non-invasive - it returns the original function unchanged,
    only adding the schema metadata.
    
    Requirements:
        - Function must have type hints for all parameters
        - Function should have a Google-style docstring with Args section
        - If docstring is missing or incomplete, fallback descriptions are used
    
    Args:
        func: The function to decorate
    
    Returns:
        The original function with __tool_schema__ attribute attached
    
    Raises:
        TypeHintParsingException: If type hints are missing or cannot be parsed
        DocstringParsingException: If there are critical docstring parsing errors
    
    Example:
        >>> @tool
        >>> def add(a: float, b: float) -> str:
        >>>     '''Add two numbers together and return the sum.
        >>>     
        >>>     Args:
        >>>         a: The first number to add
        >>>         b: The second number to add
        >>>     
        >>>     Returns:
        >>>         String representation of the sum
        >>>     '''
        >>>     return str(a + b)
        >>> 
        >>> # The function now has a __tool_schema__ attribute
        >>> print(add.__tool_schema__)
        {
            'name': 'add',
            'description': 'Add two numbers together and return the sum.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'a': {'type': 'number', 'description': 'The first number to add'},
                    'b': {'type': 'number', 'description': 'The second number to add'}
                },
                'required': ['a', 'b']
            }
        }
    """
    try:
        # Generate the Anthropic-compliant schema
        schema = generate_anthropic_schema(func)
        
        # Attach schema to the function as metadata
        func.__tool_schema__ = schema
        
        # Return the original function unchanged
        return func
        
    except (TypeHintParsingException, DocstringParsingException) as e:
        # Re-raise with more context
        raise type(e)(
            f"Failed to generate tool schema for function '{func.__name__}': {str(e)}"
        ) from e
    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(
            f"Unexpected error generating tool schema for function '{func.__name__}': {str(e)}"
        ) from e

