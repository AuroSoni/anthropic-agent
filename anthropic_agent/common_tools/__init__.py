"""Common tools for agentic AI workflows.

This module provides configurable tools for file operations within sandboxed
directories. Each tool class can be instantiated with custom limits and
extension filters.

All tool classes inherit from ConfigurableToolBase, which provides:
- Templated docstrings with {placeholder} syntax for dynamic values
- Optional custom docstring templates via docstring_template parameter
- Optional complete schema override via schema_override parameter
"""
from __future__ import annotations

from typing import Set, Union

# ---------------------------------------------------------------------------
# Extension presets
# ---------------------------------------------------------------------------
EXTENSION_PRESETS: dict[str, set[str]] = {
    "docs": {".md", ".mmd", ".rst", ".txt"},
    "code": {
        ".py", ".pyi", ".pyx",
        ".js", ".jsx", ".mjs", ".cjs",
        ".ts", ".tsx",
        ".json", ".jsonc", ".json5",
        ".yaml", ".yml",
        ".toml",
    },
    "all_text": {
        # Documentation
        ".md", ".mmd", ".rst", ".txt",
        # Python
        ".py", ".pyi", ".pyx",
        # JavaScript/TypeScript
        ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
        # Data formats
        ".json", ".jsonc", ".json5", ".yaml", ".yml", ".toml",
        ".csv", ".tsv",
        # Web
        ".html", ".htm", ".xml", ".svg",
        ".css", ".scss", ".sass", ".less",
        # Shell
        ".sh", ".bash", ".zsh", ".fish",
        # Database/Query
        ".sql", ".graphql", ".gql",
        # Config
        ".ini", ".cfg", ".conf", ".config",
        ".env", ".env.local", ".env.example",
        ".gitignore", ".gitattributes", ".gitmodules",
        ".dockerignore", ".dockerfile",
        ".editorconfig", ".prettierrc", ".eslintrc",
        ".makefile", ".cmake",
        # Other languages
        ".r", ".R", ".rmd",
        ".java", ".kt", ".kts", ".scala",
        ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
        ".swift", ".m", ".mm",
        ".rb", ".rake", ".gemspec",
        ".php", ".pl", ".pm",
        ".lua", ".vim", ".el",
        ".tf", ".hcl",
        ".proto",
    },
}


def get_extensions(preset: Union[str, Set[str]]) -> set[str]:
    """Get a set of file extensions from a preset name or return the provided set.
    
    Args:
        preset: Either a preset name ("docs", "code", "all_text") or a set of extensions.
        
    Returns:
        A set of file extensions (with leading dots).
        
    Example:
        >>> get_extensions("docs")
        {'.md', '.mmd', '.rst', '.txt'}
        >>> get_extensions({".py", ".js"})
        {'.py', '.js'}
    """
    if isinstance(preset, str):
        return EXTENSION_PRESETS.get(preset, EXTENSION_PRESETS["docs"]).copy()
    return preset


# ---------------------------------------------------------------------------
# Tool class exports
# ---------------------------------------------------------------------------
from ..tools.base import ConfigurableToolBase
from .read_file import ReadFileTool
from .apply_patch import ApplyPatchTool
from .glob_file_search import GlobFileSearchTool
from .grep_search import GrepSearchTool
from .list_dir import ListDirTool
from .code_execution_tool import CodeExecutionTool

__all__ = [
    # Extension utilities
    "EXTENSION_PRESETS",
    "get_extensions",
    # Base class
    "ConfigurableToolBase",
    # Tool classes
    "ReadFileTool",
    "ApplyPatchTool",
    "GlobFileSearchTool",
    "GrepSearchTool",
    "ListDirTool",
    "CodeExecutionTool",
]
