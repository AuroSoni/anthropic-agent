"""
### Spec for `glob_file_search` in `anthropic_agent/common_tools/glob_file_search.py`

- **Signature**
  - `def glob_file_search(glob_pattern: str, target_directory: str | None = None) -> str`

- **Purpose**
  - Find files matching a glob pattern, sorted by modification time (newest first).

- **Search base and path handling**
  - Uses configurable `base_path` provided at tool instantiation.
  - Effective base directory:
    - If `target_directory` is provided: `base = (search_root / target_directory)`.
    - Else: `base = search_root`.
  - Reject if `base` resolves outside of `search_root` (e.g., via `..`); return a one-line error.
  - Reject if `base` does not exist or is not a directory; return a one-line error.
  - All output paths are POSIX-style and relative to `search_root` (not to `target_directory`).
  - Output paths are normalized (collapse `.` and `..`) without resolving symlinks.

- **Pattern normalization and matching**
  - If `glob_pattern` does not start with `**/`, automatically prepend `**/` for recursive search.
  - Perform recursive glob from `base`; include only files with `.md` or `.mmd` extensions matching the pattern.
  - Hidden files are included.

- **Sorting**
  - Sort matches by modification time (newest first) using `Path.stat().st_mtime`.
  - For ties, sort by case-insensitive relative path.

- **Output format**
  - Newline-separated list of relative paths (to `search_root`), POSIX-style, including file extensions.
  - Only `.md` and `.mmd` files are output.
  - If no matches: return `No matches found.`

- **Truncation and summary (like `list_dir`)**
  - Show up to the first `MAX_RESULTS` matches.
  - If more remain:
    - Append a bracketed file summary grouped by extension (no leading dot; use `noext` when no extension), with top `SUMMARY_MAX_EXT_GROUPS` groups and an `other` bucket for the remainder:
      - Example: `[12 more files of type py, 7 more files of type txt, 3 more files of other types]`
    - Append a directory summary if any remain: `[K more directories]`
  - Summaries reflect only the results beyond the first `MAX_RESULTS`.

- **Errors and robustness**
  - Permission errors during `stat()` or traversal are skipped; those paths are omitted.
  - Symlinks are returned as matched; output path generation does not resolve them.
  - Sorting uses symlink `stat()` where possible; broken symlinks are omitted.

"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from posixpath import normpath as _posix_normpath
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..tools.base import ConfigurableToolBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RESULTS: int = 50
SUMMARY_MAX_EXT_GROUPS: int = 3
ALLOWED_EXTS: set[str] = {".md", ".mmd"}


# ---------------------------------------------------------------------------
# Pure utility functions (no state needed)
# ---------------------------------------------------------------------------
def _normalize_pattern(glob_pattern: str) -> str:
    """Normalize a glob pattern to ensure recursive matching."""
    if glob_pattern.startswith("**/"):
        return glob_pattern
    return f"**/{glob_pattern}"


def _ext_label(path: Path) -> str:
    """Get the extension label for a path (without leading dot)."""
    suffix = path.suffix
    return suffix[1:] if suffix else "noext"


def _has_allowed_ext(path: Path, allowed_exts: set[str]) -> bool:
    """Check if a path has an allowed extension."""
    return path.suffix.lower() in allowed_exts


def _summarize_file_exts(paths: Iterable[Path], max_groups: Optional[int] = None) -> str:
    """Summarize remaining files by extension type."""
    counter: Counter[str] = Counter(_ext_label(p) for p in paths)
    if not counter:
        return ""
    groups = SUMMARY_MAX_EXT_GROUPS if max_groups is None else max_groups
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(items) <= groups:
        return ", ".join(f"{n} more files of type {ext}" for ext, n in items)
    top = items[:groups]
    rest = items[groups:]
    other_count = sum(n for _, n in rest)
    parts = [f"{n} more files of type {ext}" for ext, n in top]
    parts.append(f"{other_count} more files of other types")
    return ", ".join(parts)


def _safe_stat_mtime(path: Path) -> float | None:
    """Safely get modification time of a path, returning None on error."""
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def _is_within(child: Path, parent: Path) -> bool:
    """Check if a child path is within a parent directory."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class GlobFileSearchTool(ConfigurableToolBase):
    """Configurable glob_file_search tool with a sandboxed base path and templated docstrings.
    
    This class encapsulates the glob_file_search functionality, allowing configuration
    of the base path at instantiation time. The tool returned by get_tool()
    can be registered with an AnthropicAgent.
    
    The docstring uses {placeholder} syntax that gets replaced with actual
    configured values at schema generation time.
    
    Example:
        >>> # Default usage - docstring reflects actual config
        >>> glob_tool = GlobFileSearchTool(base_path="/path/to/workspace", max_results=100)
        >>> agent = AnthropicAgent(tools=[glob_tool.get_tool()])
        >>> # Docstring will say "Max results: 100"
        
        >>> # Custom docstring template
        >>> glob_tool = GlobFileSearchTool(
        ...     base_path="/workspace",
        ...     docstring_template="Find files with {max_results} result limit."
        ... )
    """
    
    DOCSTRING_TEMPLATE = """Find files matching a glob pattern, sorted by modification time (newest first).

Use this tool to discover files by name pattern. Searches recursively by default.

**Limits:**
- Max results: {max_results} (excess files are summarized by extension)
- Allowed extensions: {allowed_extensions_str}

Args:
    glob_pattern: Glob pattern to match. Auto-prepends "**/" for recursive search.
        Examples:
        - "*.md" -> finds all .md files recursively
        - "README*" -> finds all README files
        - "docs/*.md" -> finds .md files in any docs/ directory
        - "**/*.mmd" -> explicit recursive search
    target_directory: Optional subdirectory to search within. Defaults to workspace root.

Returns:
    Newline-separated file paths (newest first):
    ```
    docs/api/endpoints.md
    docs/getting-started.md
    README.md
    [12 more files of type md, 3 more files of type mmd]
    ```

**Error Recovery:**
- "No matches found" -> Try a broader pattern (e.g., "*.md" instead of "specific*.md")
- "Path does not exist" -> Check the directory path with list_dir first
- "Path is not a directory" -> Remove the file name, search in its parent directory
"""
    
    def __init__(
        self,
        base_path: str | Path,
        max_results: int = 50,
        summary_max_ext_groups: int = 3,
        allowed_extensions: set[str] | None = None,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the GlobFileSearchTool with a base path and configurable limits.
        
        Args:
            base_path: The root directory that glob_file_search operates within.
                       All searches are executed relative to this directory.
            max_results: Maximum number of results to return. Defaults to 50.
            summary_max_ext_groups: Maximum extension groups to show in summary. Defaults to 3.
            allowed_extensions: Set of allowed file extensions (with leading dot).
                               Defaults to {".md", ".mmd"} if None.
            docstring_template: Optional custom docstring template with {placeholder} syntax.
                               Available placeholders: {max_results}, {allowed_extensions_str}.
            schema_override: Optional complete Anthropic tool schema dict for full control.
        """
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.search_root: Path = Path(base_path).resolve()
        self.max_results: int = max_results
        self.summary_max_ext_groups: int = summary_max_ext_groups
        self.allowed_extensions: set[str] = allowed_extensions or {".md", ".mmd"}
    
    def _get_template_context(self) -> Dict[str, Any]:
        """Return placeholder values for docstring template."""
        return {
            "max_results": self.max_results,
            "allowed_extensions_str": ", ".join(sorted(self.allowed_extensions)),
        }
    
    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.
        
        Returns:
            A decorated glob_file_search function that operates within the configured base_path.
            The docstring will reflect the actual configured limits.
        """
        # Capture self in closure for the inner function
        instance = self
        
        def glob_file_search(glob_pattern: str, target_directory: str | None = None) -> str:
            """Placeholder docstring - replaced by template."""
            base: Path = instance.search_root if target_directory is None else (instance.search_root / target_directory)

            # Prepare a relative path string for error messages
            rel_arg = "." if target_directory is None else _posix_normpath(str(target_directory).replace("\\", "/"))

            if not _is_within(base, instance.search_root):
                return f"Base path escapes search root: {rel_arg}. Use paths relative to the workspace root without '..' components."
            if not base.exists():
                return f"Path does not exist: {rel_arg}. Use list_dir to explore available directories first."
            if not base.is_dir():
                return f"Path is not a directory: {rel_arg}. Remove the file name and search in its parent directory instead."

            pattern = _normalize_pattern(glob_pattern)

            # Collect matches and their mtimes, skipping entries we cannot stat
            matches_with_mtime: List[Tuple[Path, float]] = []
            for p in base.glob(pattern):
                mtime = _safe_stat_mtime(p)
                if mtime is None:
                    continue
                # Only include files (not directories) with allowed extensions
                try:
                    if p.is_dir():
                        continue
                except Exception:
                    pass
                if not _has_allowed_ext(p, instance.allowed_extensions):
                    continue
                matches_with_mtime.append((p, mtime))

            if not matches_with_mtime:
                return f"No matches found for pattern '{glob_pattern}'. Try a broader pattern (e.g., '*.md') or check the directory with list_dir."

            # Sort by mtime desc, then by path name (case-insensitive) for stability
            matches_with_mtime.sort(key=lambda item: (-item[1], item[0].as_posix().casefold()))

            # Prepare output paths relative to search_root without resolving symlinks
            rel_matches: List[Path] = []
            for p, _ in matches_with_mtime:
                try:
                    rel = p.relative_to(instance.search_root)
                except Exception:
                    # Fallback to path relative to base to avoid leaking absolute paths
                    try:
                        rel = p.relative_to(base)
                        rel = (base.relative_to(instance.search_root) / rel) if _is_within(base, instance.search_root) else rel
                    except Exception:
                        # As a last resort, skip this path
                        continue
                rel_matches.append(rel)

            if not rel_matches:
                return f"No matches found for pattern '{glob_pattern}'. Try a broader pattern (e.g., '*.md') or check the directory with list_dir."

            # Truncate and summarize
            shown_paths = rel_matches[:instance.max_results]
            remaining_paths = rel_matches[instance.max_results:]

            lines = [_posix_normpath(sp.as_posix()) for sp in shown_paths]

            if remaining_paths:
                # Split remaining into files and directories (by suffix presence is not enough; use filesystem info)
                abs_remaining = [instance.search_root / rp for rp in remaining_paths]
                remaining_files: List[Path] = []
                remaining_dirs: List[Path] = []
                for ap in abs_remaining:
                    try:
                        if ap.is_dir():
                            remaining_dirs.append(ap)
                        else:
                            remaining_files.append(ap)
                    except Exception:
                        # If we cannot determine, treat as file to avoid undercounting
                        remaining_files.append(ap)

                file_summary = _summarize_file_exts(remaining_files, instance.summary_max_ext_groups)
                if file_summary:
                    lines.append(f"[{file_summary}]")

                if remaining_dirs:
                    lines.append(f"[{len(remaining_dirs)} more directories]")

            return "\n".join(lines)
        
        return self._apply_schema(glob_file_search)
