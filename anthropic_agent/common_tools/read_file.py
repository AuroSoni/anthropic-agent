"""
### Spec for `read_file` in `anthropic_agent/common_tools/read_file.py`

- **Signature**
  - `def read_file(target_file: str, start_line_one_indexed: int | None = None, no_of_lines_to_read: int | None = None) -> str`

- **Purpose**
  - Read a text file located under the configured base path and return a slice of its lines with a header.

- **Parameters**
  - `target_file`: Path relative to workspace. Extension optional - auto-resolves to .md or .mmd.
  - `start_line_one_indexed`: 1-based line number to start from. Defaults to 1. Values < 1 treated as 1.
  - `no_of_lines_to_read`: Number of lines to return. Defaults to 100, max 100. Values > 100 clamped.

- **Limits**
  - Default/max lines per read: 100
  - Only `.md` and `.mmd` files are readable.
  - Reject paths that resolve outside `base_path` (e.g., via `..` or absolute paths):
    - `Base path escapes search root: {absolute_candidate_path}`
  - If the path does not exist (after resolution): `Path does not exist: {target_file}`
  - If the path is a directory: `Path is a directory: {target_file}`
  - All header paths are POSIX-style and relative to `base_path` (normalized, collapse `.` and `..`).

- **Encoding and line endings**
  - Opens files in UTF-8 with `errors="replace"` to tolerate any text-encoded file.
  - Line endings are read in text mode with universal-newline handling, resulting in `\n` in returned content.

- **Offsets and limits**
  - `offset` is 1-based; default is 1. Values < 1 are treated as 1.
  - `limit` default is 100. Any provided value is clamped to a maximum of 100.
  - If `limit < 0`: return `ERROR: limit({value}) cannot be negative.`
  - If `limit` is non-integer: return `ERROR: limit({raw_value}) is not an integer.`
  - If `offset > total_lines` and `total_lines > 0`: return exactly
    `ERROR: offset({offset}) cannot be greater than total number of lines ({total}) in the file ({relative_posix_path}).`
  - If `limit == 0` or `total_lines == 0` (empty file): return only the header line with `0-0` as the range and a trailing newline.

- **Buffered vs streaming modes**
  - If file size ≤ 2 MiB (configurable via `STREAMING_THRESHOLD_BYTES`): read whole file, then compute the slice.
  - Otherwise, use a streaming approach: iterate once to collect the requested slice while counting total lines.

- **Output format**
  - First line: `[lines X-Y of TOTAL in RELATIVE_POSIX_PATH]`
  - Followed by the requested slice of content.

- **Error Recovery Guidance**
  - "Path does not exist" -> Use glob_file_search to find similar files
  - "start_line_one_indexed > total lines" -> Check the file's total line count
  - "Path is a directory" -> Use list_dir instead
"""
from __future__ import annotations

from pathlib import Path
from posixpath import normpath as _posix_normpath
from typing import Any, Callable, Dict, List, Optional

from ..tools.base import ConfigurableToolBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STREAMING_THRESHOLD_BYTES: int = 2 * 1024 * 1024  # 2 MB
MAX_LIMIT: int = 100
ALLOWED_EXTS: set[str] = {".md", ".mmd"}


# ---------------------------------------------------------------------------
# Pure utility functions (no state needed)
# ---------------------------------------------------------------------------
def _is_within(child: Path, parent: Path) -> bool:
    """Check if a child path is within a parent directory."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _format_header(start_line: int, end_line: int, total_lines: int, relative_posix_path: str) -> str:
    """Format the header line for read_file output."""
    return f"[lines {start_line}-{end_line} of {total_lines} in {relative_posix_path}]"


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class ReadFileTool(ConfigurableToolBase):
    """Configurable read_file tool with a sandboxed base path and templated docstrings.
    
    This class encapsulates the read_file functionality, allowing configuration
    of the base path at instantiation time. The tool returned by get_tool()
    can be registered with an AnthropicAgent.
    
    The docstring uses {placeholder} syntax that gets replaced with actual
    configured values at schema generation time.
    
    Example:
        >>> # Default usage - docstring reflects actual config
        >>> read_file_tool = ReadFileTool(base_path="/path/to/workspace", max_lines=500)
        >>> agent = AnthropicAgent(tools=[read_file_tool.get_tool()])
        >>> # Docstring will say "Max lines per read: 500"
        
        >>> # Custom docstring template
        >>> read_file_tool = ReadFileTool(
        ...     base_path="/workspace",
        ...     docstring_template="Read files with {max_lines} line limit."
        ... )
        
        >>> # Complete schema override
        >>> read_file_tool = ReadFileTool(
        ...     base_path="/workspace",
        ...     schema_override={"name": "read_file", "description": "Custom", ...}
        ... )
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
        base_path: str | Path,
        max_lines: int = 100,
        streaming_threshold_bytes: int = 2 * 1024 * 1024,
        allowed_extensions: set[str] | None = None,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the ReadFileTool with a base path and configurable limits.
        
        Args:
            base_path: The root directory that read_file operates within.
                       All target_file paths will be relative to this.
            max_lines: Maximum number of lines to return per read. Defaults to 100.
            streaming_threshold_bytes: File size threshold for streaming mode. Defaults to 2MB.
            allowed_extensions: Set of allowed file extensions (with leading dot).
                               Defaults to {".md", ".mmd"} if None.
            docstring_template: Optional custom docstring template with {placeholder} syntax.
                               Available placeholders: {max_lines}, {allowed_extensions_str}.
            schema_override: Optional complete Anthropic tool schema dict for full control.
        """
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.search_root: Path = Path(base_path).resolve()
        self.max_lines: int = max_lines
        self.streaming_threshold: int = streaming_threshold_bytes
        self.allowed_extensions: set[str] = allowed_extensions or {".md", ".mmd"}
    
    def _get_template_context(self) -> Dict[str, Any]:
        """Return placeholder values for docstring template."""
        return {
            "max_lines": self.max_lines,
            "allowed_extensions_str": ", ".join(sorted(self.allowed_extensions)),
        }
    
    def _rel_posix_under_root(self, path: Path) -> str:
        """Convert a path to a POSIX-style path relative to search_root."""
        try:
            rel = path.resolve().relative_to(self.search_root.resolve())
            return _posix_normpath(rel.as_posix())
        except Exception:
            # Fallback to name; should not happen after _is_within check
            return _posix_normpath(path.name)
    
    def _resolve_allowed_target(self, raw_target: str) -> Optional[Path]:
        """Resolve a target (with or without extension) to an existing allowed file under search_root.

        The resolution first checks for a direct match with an allowed extension. If that fails,
        it strips any given extension and probes for allowed extensions in order.
        Returns the resolved Path if found; otherwise None.
        """
        # Normalize slashes but keep relative semantics
        normalized = _posix_normpath(str(raw_target).replace("\\", "/"))
        base_candidate = self.search_root / normalized

        # 1. First, check if the provided path is a direct match with an allowed extension
        # Check if the filename ends with one of the allowed extensions
        filename = base_candidate.name
        has_allowed_ext = any(filename.lower().endswith(ext) for ext in self.allowed_extensions)
        
        if has_allowed_ext:
            try:
                if base_candidate.exists() and base_candidate.is_file():
                    return base_candidate
            except Exception:
                # Continue to the next strategy if stat fails
                pass

        # 2. If no direct match, strip any allowed extension and probe for allowed extensions
        # Handle filenames with dots correctly by checking for actual extensions at the end
        stem_path = base_candidate
        
        # Remove any existing allowed extension from the end
        for ext in self.allowed_extensions:
            if filename.lower().endswith(ext):
                # Create stem by removing the extension from the full path
                stem_path = base_candidate.parent / filename[:-len(ext)]
                break
        
        # If no extension was removed, use the original path as stem
        # Probe extensions in sorted order for determinism
        for ext in sorted(self.allowed_extensions):
            candidate = Path(str(stem_path) + ext)
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate
            except Exception:
                continue
        return None
    
    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.
        
        Returns:
            A decorated read_file function that operates within the configured base_path.
            The docstring will reflect the actual configured limits.
        """
        # Capture self in closure for the inner function
        instance = self
        
        def read_file(
            target_file: str,
            start_line_one_indexed: Optional[int] = None,
            no_of_lines_to_read: Optional[int] = None
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

            # Security checks on the raw target (before resolution)
            raw_candidate_path = (instance.search_root / target_file)

            if not _is_within(raw_candidate_path, instance.search_root):
                return f"Base path escapes search root: {target_file}"

            # If the raw path is an existing directory, report as such
            if raw_candidate_path.exists() and raw_candidate_path.is_dir():
                return f"Path is a directory: {target_file}. Use list_dir to explore its contents, or specify a file within it."

            # Resolve to an allowed file (md/mmd) by stripping any extension
            candidate_path = instance._resolve_allowed_target(target_file)

            if candidate_path is None:
                return f"Path does not exist: {target_file}. Use glob_file_search to find files matching a pattern."

            rel_posix = instance._rel_posix_under_root(candidate_path)

            file_size_bytes = 0
            try:
                file_size_bytes = candidate_path.stat().st_size
            except Exception:
                # If we cannot stat the file size, fall back to streaming path
                file_size_bytes = instance.streaming_threshold + 1

            # Special handling for zero-limit: still compute total lines for header
            zero_limit = requested_limit == 0

            if file_size_bytes <= instance.streaming_threshold:
                # Read whole file
                try:
                    with candidate_path.open("r", encoding="utf-8", errors="replace") as fh:
                        all_lines: List[str] = fh.read().splitlines(keepends=True)
                except Exception as exc:
                    return str(exc)

                total_lines = len(all_lines)
                
                # Handle empty file: return header with 0-0 range
                if total_lines == 0:
                    header = _format_header(0, 0, 0, rel_posix)
                    return header + "\n"
                
                if start_line > total_lines:
                    return (
                        f"ERROR: start_line_one_indexed({start_line}) cannot be greater than total number of lines "
                        f"({total_lines}) in the file ({rel_posix}). Try start_line_one_indexed=1 to read from the beginning."
                    )

                if zero_limit:
                    header = _format_header(0, 0, total_lines, rel_posix)
                    return header + "\n"

                start_idx = start_line - 1
                end_idx_exclusive = min(total_lines, start_idx + requested_limit)
                shown_end_line = end_idx_exclusive  # since lines are 1-based

                slice_content = "".join(all_lines[start_idx:end_idx_exclusive])
                header = _format_header(start_line, shown_end_line, total_lines, rel_posix)
                return header + "\n" + slice_content

            # Streaming path for large files
            total_lines = 0
            collected: List[str] = []
            collect_from = start_line
            collect_to = start_line + requested_limit - 1 if not zero_limit else start_line - 1

            try:
                with candidate_path.open("r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        total_lines += 1
                        if zero_limit:
                            continue
                        if total_lines < collect_from:
                            continue
                        if total_lines > collect_to:
                            # We can continue counting to get total_lines accurately
                            # but skip collecting any further
                            continue
                        collected.append(line)
            except Exception as exc:
                return str(exc)

            # Handle empty file: return header with 0-0 range
            if total_lines == 0:
                header = _format_header(0, 0, 0, rel_posix)
                return header + "\n"

            if start_line > total_lines:
                return (
                    f"ERROR: start_line_one_indexed({start_line}) cannot be greater than total number of lines "
                    f"({total_lines}) in the file ({rel_posix}). Try start_line_one_indexed=1 to read from the beginning."
                )

            if zero_limit:
                header = _format_header(0, 0, total_lines, rel_posix)
                return header + "\n"

            shown_end_line = min(total_lines, collect_to)
            header = _format_header(start_line, shown_end_line, total_lines, rel_posix)
            return header + "\n" + "".join(collected)
        
        return self._apply_schema(read_file)
