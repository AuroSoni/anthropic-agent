"""Base interfaces and utilities for tool execution."""
import base64
import uuid
from dataclasses import dataclass
from typing import Protocol, Dict, Any, Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic_agent.file_backends.backends import FileStorageBackend


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
    
    def to_api_format(
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
                    metadata = file_backend.store(
                        file_id=image_id,
                        filename=filename,
                        content=item.data,
                        agent_uuid=agent_uuid,
                    )
                    
                    # Build streaming reference based on backend type
                    storage_backend = metadata.get("storage_backend", "local")
                    if storage_backend == "s3":
                        # S3: use direct URL (presigned if needed)
                        src = metadata.get("storage_location", "")
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
    
    def execute(
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
                return result.to_api_format(file_backend, agent_uuid)
            
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

