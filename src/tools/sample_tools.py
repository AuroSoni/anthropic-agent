"""Sample tool functions demonstrating the @tool decorator usage."""
from .decorators import tool
from .base import ToolRegistry


@tool
def add(a: float, b: float) -> str:
    """Add two numbers together and return the sum.
    
    This function takes two numbers and returns their sum as a string.
    
    Args:
        a: The first number to add
        b: The second number to add
    
    Returns:
        String representation of the sum
    """
    result = a + b
    return str(result)


@tool
def subtract(a: float, b: float) -> str:
    """Subtract the second number from the first number.
    
    This function subtracts b from a and returns the result as a string.
    
    Args:
        a: The number to subtract from (minuend)
        b: The number to subtract (subtrahend)
    
    Returns:
        String representation of the difference
    """
    result = a - b
    return str(result)


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together and return the product.
    
    This function multiplies two numbers and returns their product as a string.
    
    Args:
        a: The first number to multiply (multiplicand)
        b: The second number to multiply (multiplier)
    
    Returns:
        String representation of the product
    """
    result = a * b
    return str(result)


@tool
def divide(a: float, b: float) -> str:
    """Divide the first number by the second number.
    
    This function divides a by b and returns the result as a string.
    Returns an error message if division by zero is attempted.
    
    Args:
        a: The dividend (number to be divided)
        b: The divisor (number to divide by)
    
    Returns:
        String representation of the quotient, or error message for division by zero
    """
    if b == 0:
        return "Error: Division by zero"
    result = a / b
    return str(result)


# Example: Create a registry and register all tools
def create_calculator_registry() -> ToolRegistry:
    """Create and return a ToolRegistry with calculator functions.
    
    This is a convenience function that demonstrates how to register
    multiple tools at once using the register_tools method.
    
    Returns:
        ToolRegistry instance with all calculator tools registered
    
    Example:
        >>> registry = create_calculator_registry()
        >>> schemas = registry.get_schemas()
        >>> result = registry.execute('add', {'a': 5, 'b': 3})
        >>> print(result)  # "8"
    """
    registry = ToolRegistry()
    registry.register_tools([add, subtract, multiply, divide])
    return registry


# For backward compatibility, provide a way to get schemas and execute tools
def get_tool_schemas() -> list[dict]:
    """Get all tool schemas for the sample calculator functions.
    
    Returns:
        List of Anthropic-compliant tool schemas
    """
    return [
        add.__tool_schema__,
        subtract.__tool_schema__,
        multiply.__tool_schema__,
        divide.__tool_schema__,
    ]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name with given input.
    
    This is a convenience function that creates a registry and executes
    the specified tool. For better performance with multiple calls,
    create a registry once and reuse it.
    
    Args:
        tool_name: Name of the tool to execute (add, subtract, multiply, divide)
        tool_input: Dictionary of input parameters
        
    Returns:
        String result from tool execution
    
    Example:
        >>> result = execute_tool('add', {'a': 10, 'b': 5})
        >>> print(result)  # "15"
    """
    registry = create_calculator_registry()
    return registry.execute(tool_name, tool_input)


# Export the tools and schemas for easy access
SAMPLE_TOOL_FUNCTIONS = [add, subtract, multiply, divide]
SAMPLE_TOOL_SCHEMAS = get_tool_schemas()
