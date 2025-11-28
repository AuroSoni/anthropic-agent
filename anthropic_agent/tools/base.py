"""Base interfaces and utilities for tool execution."""
from typing import Protocol, Dict, Any, Callable, Literal


class ToolExecutor(Protocol):
    """Protocol for tool executor implementations.
    
    A tool executor takes a tool name and input parameters,
    executes the corresponding function, and returns the result.
    """
    
    def __call__(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool by name with given input.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of input parameters
            
        Returns:
            String result from tool execution
        """
        ...


class ToolRegistry:
    """Registry for managing tool functions and their schemas.
    
    This class provides a centralized way to register tools and their schemas,
    and execute them by name.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools: Dict[str, Callable] = {}
        self.schemas: list[dict] = []
    
    def register(self, name: str, func: Callable, schema: dict) -> None:
        """Register a tool with its function and schema.
        
        Args:
            name: Name of the tool
            func: The function to execute
            schema: Anthropic-compliant tool schema
        """
        self.tools[name] = func
        self.schemas.append(schema)
    
    def register_tools(self, tools: list[Callable]) -> None:
        """Register multiple decorated functions at once.
        
        This method accepts a list of functions that have been decorated with
        the @tool decorator. Each function must have a __tool_schema__ attribute
        containing an Anthropic-compliant tool schema.
        
        Args:
            tools: List of decorated functions to register
        
        Raises:
            ValueError: If a function is missing the __tool_schema__ attribute
        
        Example:
            >>> registry = ToolRegistry()
            >>> @tool
            >>> def add(a: float, b: float) -> str:
            >>>     '''Add two numbers'''
            >>>     return str(a + b)
            >>> 
            >>> @tool
            >>> def subtract(a: float, b: float) -> str:
            >>>     '''Subtract two numbers'''
            >>>     return str(a - b)
            >>> 
            >>> registry.register_tools([add, subtract])
        """
        for func in tools:
            # Check if function has the __tool_schema__ attribute
            if not hasattr(func, '__tool_schema__'):
                raise ValueError(
                    f"Function '{func.__name__}' is missing __tool_schema__ attribute. "
                    f"Did you forget to apply the @tool decorator?"
                )
            
            # Extract schema and register
            schema = func.__tool_schema__
            name = schema['name']
            self.register(name, func, schema)
    
    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a registered tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of input parameters
            
        Returns:
            String result from tool execution
        """
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"
        
        try:
            tool_func = self.tools[tool_name]
            result = tool_func(**tool_input)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def get_schemas(self, schema_type: Literal["anthropic", "openai"] = "anthropic") -> list[dict]:
        """Get registered tool schemas in the requested format.
        
        Args:
            schema_type: Output format. 
                - ``anthropic`` returns the raw Anthropic schema dictionaries (default)
                - ``openai`` converts each schema into OpenAI's function-call payload
        
        Returns:
            List of schema dictionaries matching the requested format.
        """
        if schema_type == "anthropic":
            return self.schemas.copy()
        
        if schema_type == "openai":
            openai_payload = []
            for schema in self.schemas:
                openai_payload.append({
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema["description"],
                        "parameters": schema["input_schema"],
                    },
                })
            return openai_payload
        
        raise ValueError(f"Unsupported schema_type '{schema_type}'. Expected 'anthropic' or 'openai'.")

