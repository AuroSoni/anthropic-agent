"""Decorators for automatic tool schema generation."""
from typing import Callable, Literal, overload

from .type_hint_utils import (
    generate_anthropic_schema,
    TypeHintParsingException,
    DocstringParsingException,
)

# Type alias for executor parameter
ExecutorType = Literal["backend", "frontend"]


@overload
def tool(func: Callable) -> Callable:
    """Simple decorator usage: @tool"""
    ...


@overload
def tool(
    func: None = None,
    *,
    executor: ExecutorType = "backend",
) -> Callable[[Callable], Callable]:
    """Parameterized decorator usage: @tool(executor="frontend")"""
    ...


def tool(
    func: Callable | None = None,
    *,
    executor: ExecutorType = "backend",
) -> Callable | Callable[[Callable], Callable]:
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
        func: The function to decorate (when used without parentheses)
        executor: Where the tool executes - "backend" (default) runs on server,
            "frontend" runs in the browser. Frontend tools are schema-only on
            the server; their actual execution happens client-side.
    
    Returns:
        The original function with __tool_schema__ and __tool_executor__ attributes
    
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
        
        >>> # Frontend tool example (runs in browser, not on server)
        >>> @tool(executor="frontend")
        >>> def user_confirm(message: str) -> str:
        >>>     '''Ask the user for yes/no confirmation.
        >>>     
        >>>     Args:
        >>>         message: The question to ask the user
        >>>     '''
        >>>     pass  # Actual execution happens in browser
        >>> 
        >>> print(add.__tool_executor__)  # "frontend"
    """
    def decorator(fn: Callable) -> Callable:
        try:
            # Generate the Anthropic-compliant schema
            schema = generate_anthropic_schema(fn)
            
            # Attach schema to the function as metadata
            fn.__tool_schema__ = schema
            
            # Attach executor type (backend or frontend)
            fn.__tool_executor__ = executor
            
            # Return the original function unchanged
            return fn
            
        except (TypeHintParsingException, DocstringParsingException) as e:
            # Re-raise with more context
            raise type(e)(
                f"Failed to generate tool schema for function '{fn.__name__}': {str(e)}"
            ) from e
        except Exception as e:
            # Catch any other unexpected errors
            raise RuntimeError(
                f"Unexpected error generating tool schema for function '{fn.__name__}': {str(e)}"
            ) from e
    
    # Handle both @tool and @tool(...) syntax
    if func is not None:
        # Called as @tool without parentheses
        return decorator(func)
    else:
        # Called as @tool(...) with parentheses
        return decorator

