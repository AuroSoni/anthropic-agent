"""ReadFileTool — async, sandbox-based file reading.

Migrated from anthropic_agent/common_tools/read_file.py.
All direct file I/O replaced with sandbox API calls.
"""
from __future__ import annotations

from posixpath import normpath as _posix_normpath
from typing import Any, Callable, Dict, Optional

from ..tools.base import ConfigurableToolBase


# ---------------------------------------------------------------------------
# Pure utility functions (no state needed)
# ---------------------------------------------------------------------------
def _format_header(start_line: int, end_line: int, total_lines: int, relative_posix_path: str) -> str:
    """Format the header line for read_file output."""
    return f"[lines {start_line}-{end_line} of {total_lines} in {relative_posix_path}]"


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class ReadFileTool(ConfigurableToolBase):
    """Configurable read_file tool with sandbox-based I/O and templated docstrings.

    All file access goes through the injected Sandbox instance.
    The sandbox handles path resolution and containment.

    Example:
        >>> read_tool = ReadFileTool(max_lines=500)
        >>> func = read_tool.get_tool()
        >>> registry.register_tools([func])
        >>> registry.attach_sandbox(sandbox)
    """

    DOCSTRING_TEMPLATE = """Read lines from a text file.

Use this tool to inspect file contents. Returns lines with a header showing
the range and total line count.

**Limits:**
- Max lines per read: {max_lines}
- Allowed extensions: {allowed_extensions_str}

Args:
    target_file: Path relative to workspace. Extension is optional - will
        auto-resolve to allowed extensions if omitted.
    start_line_one_indexed: 1-based line number to start from. Defaults to 1.
        Values < 1 are treated as 1.
    no_of_lines_to_read: Number of lines to return. Defaults to {max_lines}, max {max_lines}.
        Values > {max_lines} are clamped.

Returns:
    Header line followed by file content:
    ```
    [lines 1-50 of 200 in docs/guide.md]
    # Welcome to the Guide
    ...content...
    ```

    For pagination, check the header's total and call again with a higher start_line_one_indexed.

Example:
    read_file("docs/guide.md")  # First {max_lines} lines
    read_file("docs/guide", start_line_one_indexed=101)  # Next {max_lines} lines

**Error Recovery:**
- "Path does not exist" -> Use glob_file_search to find similar files
- "start_line_one_indexed > total lines" -> Check the file's total line count in header
- "Path is a directory" -> Use list_dir instead, or specify a file within the directory
"""

    def __init__(
        self,
        max_lines: int = 100,
        allowed_extensions: set[str] | None = None,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the ReadFileTool.

        Args:
            max_lines: Maximum number of lines to return per read. Defaults to 100.
            allowed_extensions: Set of allowed file extensions (with leading dot).
                               Defaults to {".md", ".mmd"} if None.
            docstring_template: Optional custom docstring template with {placeholder} syntax.
            schema_override: Optional complete Anthropic tool schema dict for full control.
        """
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.max_lines: int = max_lines
        self.allowed_extensions: set[str] = allowed_extensions or {".md", ".mmd"}

    def _get_template_context(self) -> Dict[str, Any]:
        """Return placeholder values for docstring template."""
        return {
            "max_lines": self.max_lines,
            "allowed_extensions_str": ", ".join(sorted(self.allowed_extensions)),
        }

    def get_tool(self) -> Callable:
        """Return a @tool decorated async function for use with an agent.

        Returns:
            A decorated read_file function that operates through the sandbox.
        """
        instance = self

        async def read_file(
            target_file: str,
            start_line_one_indexed: Optional[int] = None,
            no_of_lines_to_read: Optional[int] = None,
        ) -> str:
            """Placeholder docstring - replaced by template."""
            # Normalize parameters
            start_line = 1 if start_line_one_indexed is None else max(1, int(start_line_one_indexed))
            max_limit = instance.max_lines

            if no_of_lines_to_read is None:
                requested_limit = max_limit
            else:
                try:
                    requested_limit = int(no_of_lines_to_read)
                except Exception:
                    return f"ERROR: no_of_lines_to_read({no_of_lines_to_read}) is not an integer. Provide a positive integer."
                if requested_limit < 0:
                    return f"ERROR: no_of_lines_to_read({requested_limit}) cannot be negative. Use a positive integer or 0."
                if requested_limit > max_limit:
                    requested_limit = max_limit

            # Normalize the target path
            rel_path = _posix_normpath(str(target_file).replace("\\", "/"))

            # Check if path is a directory
            try:
                entries = await instance._sandbox.list_dir(rel_path)
                # If list_dir succeeds, it's a directory
                return f"Path is a directory: {rel_path}. Use list_dir to explore its contents, or specify a file within it."
            except (NotADirectoryError, FileNotFoundError, ValueError):
                pass

            # Try to resolve the file — check with allowed extensions
            resolved_path = await _resolve_allowed_target(instance, rel_path)

            if resolved_path is None:
                return f"Path does not exist: {target_file}. Use glob_file_search to find files matching a pattern."

            # Read the file content
            try:
                content = await instance._sandbox.read_file(resolved_path)
            except Exception as exc:
                return str(exc)

            all_lines = content.splitlines(keepends=True)
            total_lines = len(all_lines)

            # Handle empty file
            if total_lines == 0:
                header = _format_header(0, 0, 0, resolved_path)
                return header + "\n"

            if start_line > total_lines:
                return (
                    f"ERROR: start_line_one_indexed({start_line}) cannot be greater than total number of lines "
                    f"({total_lines}) in the file ({resolved_path}). Try start_line_one_indexed=1 to read from the beginning."
                )

            # Zero limit: return only header
            if requested_limit == 0:
                header = _format_header(0, 0, total_lines, resolved_path)
                return header + "\n"

            start_idx = start_line - 1
            end_idx_exclusive = min(total_lines, start_idx + requested_limit)
            shown_end_line = end_idx_exclusive

            slice_content = "".join(all_lines[start_idx:end_idx_exclusive])
            header = _format_header(start_line, shown_end_line, total_lines, resolved_path)
            return header + "\n" + slice_content

        func = self._apply_schema(read_file)
        func.__tool_instance__ = instance
        return func


async def _resolve_allowed_target(instance: ReadFileTool, raw_target: str) -> Optional[str]:
    """Resolve a target (with or without extension) to an existing allowed file.

    Returns the sandbox-relative path if found; otherwise None.
    """
    normalized = _posix_normpath(str(raw_target).replace("\\", "/"))

    # Check if the provided path has an allowed extension and exists
    filename = normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized
    has_allowed_ext = any(filename.lower().endswith(ext) for ext in instance.allowed_extensions)

    if has_allowed_ext:
        if await instance._sandbox.file_exists(normalized):
            return normalized

    # Strip any allowed extension and probe for allowed extensions
    stem = normalized
    for ext in instance.allowed_extensions:
        if filename.lower().endswith(ext):
            stem = normalized[:-len(ext)]
            break

    # Probe extensions in sorted order for determinism
    for ext in sorted(instance.allowed_extensions):
        candidate = stem + ext
        if await instance._sandbox.file_exists(candidate):
            return candidate

    return None
