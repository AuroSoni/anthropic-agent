"""Sandbox configuration for common tools.

This module provides ToolSandboxConfig, a dataclass that creates a configured
toolset with dynamically augmented schemas that include constraint information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Set


@dataclass
class ToolSandboxConfig:
    """Configuration for common tools sandbox.
    
    Creates a ready-to-use toolset with all common tools configured
    with the same base path, extension filters, and limits. Each tool's
    schema is augmented to include constraint information visible to the LLM.
    
    Example:
        >>> from pathlib import Path
        >>> config = ToolSandboxConfig(
        ...     base_path=Path("/workspace"),
        ...     allowed_extensions={".py", ".md", ".json"},
        ...     read_file_max_lines=200,
        ... )
        >>> tools = config.create_toolset()
        >>> agent = AnthropicAgent(tools=tools)
    """
    
    base_path: Path
    allowed_extensions: Set[str] = field(default_factory=lambda: {".md", ".mmd"})
    
    # Read file limits
    read_file_max_lines: int = 100
    read_file_streaming_threshold: int = 2 * 1024 * 1024  # 2 MB
    
    # Apply patch limits
    apply_patch_max_size: int = 1 * 1024 * 1024  # 1 MB
    apply_patch_max_file_size: int = 10 * 1024 * 1024  # 10 MB
    
    # Glob search limits
    glob_max_results: int = 50
    glob_summary_max_ext_groups: int = 3
    
    # Grep search limits
    grep_max_matches: int = 20
    grep_context_lines: int = 2
    
    # List dir limits
    list_dir_max_depth: int = 5
    list_dir_large_threshold: int = 50
    list_dir_show_files: int = 5
    list_dir_show_dirs: int = 5
    
    def __post_init__(self) -> None:
        """Ensure base_path is a Path object."""
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)
    
    def _augment_schema(self, tool_func: Callable, constraints: Dict[str, Any]) -> Callable:
        """Augment tool schema description with constraint information.
        
        Modifies the __tool_schema__ attached by the @tool decorator to include
        a "Constraints" section at the end of the description. This makes the
        operational limits visible to the LLM.
        
        Args:
            tool_func: A function decorated with @tool that has __tool_schema__.
            constraints: Dictionary of constraint name -> value to append.
            
        Returns:
            The same function with augmented schema description.
        """
        if not hasattr(tool_func, "__tool_schema__"):
            return tool_func
        
        schema = tool_func.__tool_schema__
        original_desc = schema.get("description", "")
        
        # Build constraint block
        constraint_lines: List[str] = []
        for key, value in constraints.items():
            constraint_lines.append(f"- {key}: {value}")
        
        constraint_block = "\n\nConstraints:\n" + "\n".join(constraint_lines)
        
        # Augment description
        schema["description"] = original_desc + constraint_block
        
        return tool_func
    
    def create_toolset(self) -> List[Callable]:
        """Create all common tools with configured limits and augmented schemas.
        
        Returns:
            List of tool functions ready to register with AnthropicAgent.
            Each tool's schema includes constraint information in its description.
        """
        from ..common_tools.read_file import ReadFileTool
        from ..common_tools.apply_patch import ApplyPatchTool
        from ..common_tools.glob_file_search import GlobFileSearchTool
        from ..common_tools.grep_search import GrepSearchTool
        from ..common_tools.list_dir import ListDirTool
        
        tools: List[Callable] = []
        extensions_str = ", ".join(sorted(self.allowed_extensions))
        
        # Read file tool
        read_tool = ReadFileTool(
            base_path=self.base_path,
            max_lines=self.read_file_max_lines,
            streaming_threshold_bytes=self.read_file_streaming_threshold,
            allowed_extensions=self.allowed_extensions,
        ).get_tool()
        
        read_tool = self._augment_schema(read_tool, {
            "max_lines_per_read": self.read_file_max_lines,
            "allowed_extensions": extensions_str,
            "base_path": str(self.base_path),
        })
        tools.append(read_tool)
        
        # Apply patch tool
        apply_patch_tool = ApplyPatchTool(
            base_path=self.base_path,
            max_patch_size_bytes=self.apply_patch_max_size,
            max_file_size_bytes=self.apply_patch_max_file_size,
            allowed_extensions=self.allowed_extensions,
        ).get_tool()
        
        apply_patch_tool = self._augment_schema(apply_patch_tool, {
            "max_patch_size_mb": self.apply_patch_max_size // (1024 * 1024),
            "max_file_size_mb": self.apply_patch_max_file_size // (1024 * 1024),
            "allowed_extensions": extensions_str,
            "base_path": str(self.base_path),
        })
        tools.append(apply_patch_tool)
        
        # Glob file search tool
        glob_tool = GlobFileSearchTool(
            base_path=self.base_path,
            max_results=self.glob_max_results,
            summary_max_ext_groups=self.glob_summary_max_ext_groups,
            allowed_extensions=self.allowed_extensions,
        ).get_tool()
        
        glob_tool = self._augment_schema(glob_tool, {
            "max_results": self.glob_max_results,
            "allowed_extensions": extensions_str,
            "base_path": str(self.base_path),
        })
        tools.append(glob_tool)
        
        # Grep search tool
        grep_tool = GrepSearchTool(
            base_path=self.base_path,
            max_match_lines=self.grep_max_matches,
            context_lines=self.grep_context_lines,
            allowed_extensions=self.allowed_extensions,
        ).get_tool()
        
        grep_tool = self._augment_schema(grep_tool, {
            "max_match_lines": self.grep_max_matches,
            "context_lines": self.grep_context_lines,
            "allowed_extensions": extensions_str,
            "base_path": str(self.base_path),
        })
        tools.append(grep_tool)
        
        # List dir tool
        list_dir_tool = ListDirTool(
            base_path=self.base_path,
            max_depth=self.list_dir_max_depth,
            large_dir_threshold=self.list_dir_large_threshold,
            large_dir_show_files=self.list_dir_show_files,
            large_dir_show_dirs=self.list_dir_show_dirs,
            allowed_extensions=self.allowed_extensions,
        ).get_tool()
        
        list_dir_tool = self._augment_schema(list_dir_tool, {
            "max_depth": self.list_dir_max_depth,
            "allowed_extensions": extensions_str,
            "base_path": str(self.base_path),
        })
        tools.append(list_dir_tool)
        
        return tools
