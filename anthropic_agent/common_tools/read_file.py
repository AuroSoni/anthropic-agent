"""
### Spec for `read_file` in `src/tools/read_file.py`

- **Signature**
  - `def read_file(target_file: str, offset: int | None = None, limit: int | None = None) -> str`

- **Purpose**
  - Read a text file located under the configured base path and return a slice of its lines with a header.

- **Search root and path handling**
  - Uses configurable `base_path` provided at tool instantiation.
  - `target_file` is a path relative to `base_path`.
  - Accepts targets with or without an extension; internally strips any extension and resolves to an existing `.md` or `.mmd` file (in that order).
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
  - Followed by the exact requested slice of content (possibly empty if `limit == 0`).

- **Limits**
  - Effective limit (explicit or default) cannot exceed 100.
"""
from __future__ import annotations

from pathlib import Path
from posixpath import normpath as _posix_normpath
from typing import Callable, List, Optional

from ..tools.decorators import tool

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
class ReadFileTool:
    """Configurable read_file tool with a sandboxed base path.
    
    This class encapsulates the read_file functionality, allowing configuration
    of the base path at instantiation time. The tool returned by get_tool()
    can be registered with an AnthropicAgent.
    
    Example:
        >>> read_file_tool = ReadFileTool(base_path="/path/to/workspace")
        >>> agent = AnthropicAgent(tools=[read_file_tool.get_tool()])
    """
    
    def __init__(self, base_path: str | Path):
        """Initialize the ReadFileTool with a base path.
        
        Args:
            base_path: The root directory that read_file operates within.
                       All target_file paths will be relative to this.
        """
        self.search_root: Path = Path(base_path).resolve()
    
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
        it strips any given extension and probes for `.md` then `.mmd` in that order.
        Returns the resolved Path if found; otherwise None.
        """
        # Normalize slashes but keep relative semantics
        normalized = _posix_normpath(str(raw_target).replace("\\", "/"))
        base_candidate = self.search_root / normalized

        # 1. First, check if the provided path is a direct match with an allowed extension
        # Check if the filename ends with one of the allowed extensions
        filename = base_candidate.name
        has_allowed_ext = any(filename.lower().endswith(ext) for ext in ALLOWED_EXTS)
        
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
        for ext in ALLOWED_EXTS:
            if filename.lower().endswith(ext):
                # Create stem by removing the extension from the full path
                stem_path = base_candidate.parent / filename[:-len(ext)]
                break
        
        # If no extension was removed, use the original path as stem
        for ext in (".md", ".mmd"):
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
        """
        # Capture self in closure for the inner function
        instance = self
        
        @tool
        def read_file(target_file: str, offset: Optional[int] = None, limit: Optional[int] = None) -> str:
            """Read a UTF-8 text file under the configured base path and return a slice of its lines with a header.

            Paths must remain within the base path. Files up to STREAMING_THRESHOLD_BYTES are read whole; larger
            files are streamed. Line endings are normalized to '\\n'. Only `.md`/`.mmd` files are readable.
            The `target_file` may omit the extension; the function will resolve to `.md` or `.mmd`.

            Args:
                target_file: File path relative to the configured base path.
                offset: 1-based starting line number. Values < 1 are treated as 1.
                    Defaults to 1 when None.
                limit: Number of lines to return. Defaults to MAX_LIMIT when None.
                    Values > MAX_LIMIT are clamped; negative values yield an error; non-integers yield an error.

            Returns:
                A string starting with a header line:
                    "[lines X-Y of TOTAL in RELATIVE_POSIX_PATH]"
                    followed by the requested slice. If limit == 0, only the header and a trailing newline
                    are returned. On errors, returns a one-line message, e.g.:
                    - "Base path escapes search root: …"
                    - "Path does not exist: …"
                    - "Path is a directory: …"
                    - "ERROR: offset(…) cannot be greater than total number of lines (…)"
                    - "ERROR: limit(…) is not an integer." or "ERROR: limit(…) cannot be negative."
            """
            # Normalize parameters
            start_line = 1 if offset is None else max(1, int(offset))

            if limit is None:
                requested_limit = MAX_LIMIT
            else:
                try:
                    requested_limit = int(limit)
                except Exception:
                    return f"ERROR: limit({limit}) is not an integer."
                if requested_limit < 0:
                    return f"ERROR: limit({requested_limit}) cannot be negative."
                if requested_limit > MAX_LIMIT:
                    requested_limit = MAX_LIMIT

            # Security checks on the raw target (before resolution)
            raw_candidate_path = (instance.search_root / target_file)

            if not _is_within(raw_candidate_path, instance.search_root):
                return f"Base path escapes search root: {target_file}"

            # If the raw path is an existing directory, report as such
            if raw_candidate_path.exists() and raw_candidate_path.is_dir():
                return f"Path is a directory: {target_file}"

            # Resolve to an allowed file (md/mmd) by stripping any extension
            candidate_path = instance._resolve_allowed_target(target_file)

            if candidate_path is None:
                return f"Path does not exist: {target_file}"

            rel_posix = instance._rel_posix_under_root(candidate_path)

            file_size_bytes = 0
            try:
                file_size_bytes = candidate_path.stat().st_size
            except Exception:
                # If we cannot stat the file size, fall back to streaming path
                file_size_bytes = STREAMING_THRESHOLD_BYTES + 1

            # Special handling for zero-limit: still compute total lines for header
            zero_limit = requested_limit == 0

            if file_size_bytes <= STREAMING_THRESHOLD_BYTES:
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
                        f"ERROR: offset({start_line}) cannot be greater than total number of lines "
                        f"({total_lines}) in the file ({rel_posix})."
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
                    f"ERROR: offset({start_line}) cannot be greater than total number of lines "
                    f"({total_lines}) in the file ({rel_posix})."
                )

            if zero_limit:
                header = _format_header(0, 0, total_lines, rel_posix)
                return header + "\n"

            shown_end_line = min(total_lines, collect_to)
            header = _format_header(start_line, shown_end_line, total_lines, rel_posix)
            return header + "\n" + "".join(collected)
        
        return read_file
