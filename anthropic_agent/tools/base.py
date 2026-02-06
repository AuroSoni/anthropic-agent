"""Base interfaces and utilities for tool execution."""
from __future__ import annotations

import base64
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Dict, Any, Callable, Literal, Optional, TYPE_CHECKING

from .decorators import tool

if TYPE_CHECKING:
    from anthropic_agent.file_backends.base import FileStorageBackend

# Type alias for tool result content that can be sent to Anthropic API
ToolResultContent = str | list[dict[str, Any]]

@dataclass
class ImageBlock:
    """Image content block for tool results.
    
    Attributes:
        data: Raw image bytes
        media_type: MIME type of the image
    """
    data: bytes
    media_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"]

@dataclass
class ToolResult:
    """Wrapper for tool execution results supporting text and images.
    
    This class provides a type-safe way to return multimodal content from tools.
    For text-only results, use the `text()` class method. For results with images,
    use `with_image()` or construct directly with a list of content.
    
    Example:
        >>> # Text-only result
        >>> return ToolResult.text("Operation completed successfully")
        >>> 
        >>> # Result with image
        >>> screenshot = capture_screenshot()
        >>> return ToolResult.with_image("Here's the screenshot:", screenshot, "image/png")
    """
    content: str | list[str | ImageBlock]
    
    @classmethod
    def text(cls, text: str) -> "ToolResult":
        """Create a text-only result (most common case).
        
        Args:
            text: The text content to return
            
        Returns:
            ToolResult with text content
        """
        return cls(content=text)
    
    @classmethod
    def with_image(
        cls,
        text: str,
        image_data: bytes,
        media_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"],
    ) -> "ToolResult":
        """Create a result with text and one image.
        
        Args:
            text: Text description or context for the image
            image_data: Raw image bytes
            media_type: MIME type of the image
            
        Returns:
            ToolResult with text and image content
        """
        return cls(content=[text, ImageBlock(image_data, media_type)])
    
    async def to_api_format(
        self,
        file_backend: "FileStorageBackend | None" = None,
        agent_uuid: str | None = None,
    ) -> tuple[ToolResultContent, list[dict[str, Any]]]:
        """Convert to Anthropic API format.
        
        For text-only content, returns the string directly.
        For multimodal content, converts to list of content blocks with
        base64-encoded images for the API.
        
        Args:
            file_backend: Optional file backend for storing images (used for streaming references)
            agent_uuid: Agent UUID for file storage paths
            
        Returns:
            Tuple of (api_content, image_refs) where:
            - api_content: Content in Anthropic API format (str or list[dict])
            - image_refs: List of image reference dicts for streaming (empty for text-only)
        """
        # Text-only: return string directly
        if isinstance(self.content, str):
            return self.content, []
        
        # Multimodal: convert to content blocks
        api_blocks: list[dict[str, Any]] = []
        image_refs: list[dict[str, Any]] = []
        
        for item in self.content:
            if isinstance(item, str):
                api_blocks.append({"type": "text", "text": item})
            elif isinstance(item, ImageBlock):
                # Generate unique image ID
                image_id = f"img_{uuid.uuid4().hex[:12]}"
                
                # Always include base64 for API
                b64_data = base64.standard_b64encode(item.data).decode("utf-8")
                api_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": item.media_type,
                        "data": b64_data,
                    }
                })
                
                # Store image and build reference for streaming
                if file_backend is not None and agent_uuid is not None:
                    # Determine file extension from media type
                    ext = item.media_type.split("/")[1]
                    filename = f"{image_id}.{ext}"
                    
                    # Store image using file backend
                    metadata = await file_backend.store(
                        file_id=image_id,
                        filename=filename,
                        content=item.data,
                        agent_uuid=agent_uuid,
                    )
                    
                    # Build streaming reference based on backend type
                    if metadata.storage_backend == "s3":
                        # S3: use direct URL (presigned if needed)
                        src = metadata.storage_location or ""
                    else:
                        # Local: use API path
                        src = f"/agent/{agent_uuid}/images/{image_id}"
                    
                    image_refs.append({
                        "image_id": image_id,
                        "src": src,
                        "media_type": item.media_type,
                    })
        
        return api_blocks, image_refs

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
    
    async def execute(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        file_backend: "FileStorageBackend | None" = None,
        agent_uuid: str | None = None,
    ) -> tuple[ToolResultContent, list[dict[str, Any]]]:
        """Execute a registered tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of input parameters
            file_backend: Optional file backend for storing images from ToolResult
            agent_uuid: Agent UUID for file storage paths
            
        Returns:
            Tuple of (content, image_refs) where:
            - content: String or list of content blocks for Anthropic API
            - image_refs: List of image reference dicts for streaming (empty for text-only)
        """
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'", []
        
        try:
            tool_func = self.tools[tool_name]
            result = tool_func(**tool_input)
            
            # Handle ToolResult wrapper
            if isinstance(result, ToolResult):
                return await result.to_api_format(file_backend, agent_uuid)
            
            # Backward compatibility: raw string returns
            return result, []
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}", []
    
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

class ConfigurableToolBase(ABC):
    """Abstract base class for tools with templated docstrings.
    
    This class provides a pattern for creating tools where:
    - Docstrings use {placeholder} syntax that gets replaced with actual
      instance values at schema generation time
    - Library callers can provide custom docstring templates
    - Library callers can provide complete schema overrides for full control
    
    Subclasses should:
    1. Define DOCSTRING_TEMPLATE as a class attribute with {placeholder} syntax
    2. Override _get_template_context() to provide placeholder values
    3. Implement get_tool() and call self._apply_schema(func) on the inner function
    
    Example:
        >>> class MyTool(ConfigurableToolBase):
        ...     DOCSTRING_TEMPLATE = '''Do something with {max_items} items.
        ...     
        ...     Args:
        ...         input: The input to process.
        ...     '''
        ...     
        ...     def __init__(self, max_items: int = 10, **kwargs):
        ...         super().__init__(**kwargs)
        ...         self.max_items = max_items
        ...     
        ...     def _get_template_context(self) -> Dict[str, Any]:
        ...         return {"max_items": self.max_items}
        ...     
        ...     def get_tool(self) -> Callable:
        ...         def my_tool(input: str) -> str:
        ...             '''Placeholder'''
        ...             return input
        ...         return self._apply_schema(my_tool)
    """
    
    # Class-level docstring template with {placeholder} syntax
    # Subclasses should override this with their specific template
    DOCSTRING_TEMPLATE: str = ""
    
    def __init__(
        self,
        docstring_template: Optional[str] = None,
        schema_override: Optional[dict] = None,
    ):
        """Initialize the configurable tool base.
        
        Args:
            docstring_template: Optional custom docstring template with {placeholder}
                syntax. If provided, overrides the class-level DOCSTRING_TEMPLATE.
                Placeholders are replaced using values from _get_template_context().
            schema_override: Optional complete Anthropic tool schema dict. If provided,
                bypasses all docstring processing and uses this schema directly.
                Must include 'name', 'description', and 'input_schema' keys.
        """
        self._docstring_template = docstring_template
        self._schema_override = schema_override
    
    def _get_template_context(self) -> Dict[str, Any]:
        """Return a dict of {placeholder: value} for docstring template substitution.
        
        Override this method in subclasses to provide tool-specific placeholder values.
        Keys should match the {placeholder} names used in DOCSTRING_TEMPLATE.
        
        Returns:
            Dict mapping placeholder names to their values. Values are converted
            to strings during template rendering.
        
        Example:
            >>> def _get_template_context(self) -> Dict[str, Any]:
            ...     return {
            ...         "max_lines": self.max_lines,
            ...         "allowed_extensions_str": ", ".join(sorted(self.allowed_extensions)),
            ...     }
        """
        return {}
    
    def _render_docstring(self) -> str:
        """Render the docstring template by replacing {placeholders} with actual values.
        
        Uses the custom docstring_template if provided, otherwise falls back to
        the class-level DOCSTRING_TEMPLATE. Placeholders are replaced with values
        from _get_template_context().
        
        Returns:
            The rendered docstring with all placeholders replaced.
        
        Note:
            If a placeholder in the template is not found in the context,
            a warning is issued and the template is returned with that
            placeholder intact.
        """
        template = self._docstring_template or self.DOCSTRING_TEMPLATE
        
        if not template:
            return ""
        
        context = self._get_template_context()
        
        try:
            return template.format(**context)
        except KeyError as e:
            warnings.warn(
                f"{self.__class__.__name__}: Unknown docstring placeholder {e}. "
                f"Available placeholders: {list(context.keys())}",
                stacklevel=2
            )
            # Return template as-is if substitution fails
            return template
    
    def _apply_schema(self, func: Callable) -> Callable:
        """Apply schema to function - either override or generated from docstring.
        
        This method should be called in get_tool() on the inner function before
        returning it. It handles:
        1. Using schema_override directly if provided
        2. Otherwise, rendering the docstring template and applying @tool decorator
        
        Args:
            func: The inner tool function to apply schema to.
        
        Returns:
            The function with __tool_schema__ and __tool_executor__ attributes set.
        
        Example:
            >>> def get_tool(self) -> Callable:
            ...     def my_tool(input: str) -> str:
            ...         '''Placeholder'''
            ...         return input
            ...     return self._apply_schema(my_tool)
        """
        # Option 1: User provided complete schema - use it directly
        if self._schema_override is not None:
            func.__tool_schema__ = self._schema_override
            func.__tool_executor__ = "backend"
            return func
        
        # Option 2: Render templated docstring, then apply @tool decorator
        func.__doc__ = self._render_docstring()
        return tool(func)
    
    @abstractmethod
    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.
        
        Subclasses must implement this method. The implementation should:
        1. Define the inner tool function with proper type hints
        2. Call self._apply_schema(func) on the inner function
        3. Return the result
        
        Returns:
            A decorated function with __tool_schema__ attribute.
        """
        ...
