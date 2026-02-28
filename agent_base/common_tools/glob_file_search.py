"""GlobFileSearchTool — async, sandbox-based file search.

Migrated from anthropic_agent/common_tools/glob_file_search.py.
Path.glob() replaced with sandbox.exec("find ...") commands.
"""
from __future__ import annotations

import re
import shlex
from collections import Counter
from posixpath import normpath as _posix_normpath
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..tools.base import ConfigurableToolBase

# Characters allowed in find -name/-path patterns (glob metacharacters + safe chars)
_SAFE_PATTERN_RE = re.compile(r'^[a-zA-Z0-9_\-.*?\[\]/]+$')


# ---------------------------------------------------------------------------
# Pure utility functions (no state needed)
# ---------------------------------------------------------------------------
def _normalize_pattern(glob_pattern: str) -> str:
    """Normalize a glob pattern to ensure recursive matching."""
    if glob_pattern.startswith("**/"):
        return glob_pattern
    return f"**/{glob_pattern}"


def _ext_label(path: str) -> str:
    """Get the extension label for a path (without leading dot)."""
    if "." in path.rsplit("/", 1)[-1]:
        suffix = "." + path.rsplit(".", 1)[-1]
        return suffix[1:]
    return "noext"


def _has_allowed_ext(path: str, allowed_exts: set[str]) -> bool:
    """Check if a path has an allowed extension."""
    name = path.rsplit("/", 1)[-1] if "/" in path else path
    return any(name.lower().endswith(ext) for ext in allowed_exts)


def _summarize_file_exts(paths: Iterable[str], max_groups: Optional[int] = None) -> str:
    """Summarize remaining files by extension type."""
    counter: Counter[str] = Counter(_ext_label(p) for p in paths)
    if not counter:
        return ""
    groups = 3 if max_groups is None else max_groups
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(items) <= groups:
        return ", ".join(f"{n} more files of type {ext}" for ext, n in items)
    top = items[:groups]
    rest = items[groups:]
    other_count = sum(n for _, n in rest)
    parts = [f"{n} more files of type {ext}" for ext, n in top]
    parts.append(f"{other_count} more files of other types")
    return ", ".join(parts)


def _build_find_command(pattern: str, target_dir: str = ".") -> str:
    """Build a find command that mimics glob behavior.

    Translates glob patterns to find commands:
    - "**/*.py"     -> find . -type f -name '*.py'
    - "src/*.ts"    -> find . -type f -path '*/src/*.ts'
    - "*.md"        -> find . -type f -name '*.md'

    Uses only POSIX-compatible find flags (no GNU -printf).
    All arguments are shell-escaped to prevent injection.
    """
    safe_dir = shlex.quote(target_dir)

    # After normalization, pattern always starts with **/
    # Extract the name portion after the last **/
    if "**/" in pattern:
        name_part = pattern.rsplit("**/", 1)[-1]
        if not name_part or name_part == "**":
            # "**" or "**/**" — match all files
            cmd = f"find {safe_dir} -type f"
        elif "/" in name_part:
            # Pattern has a directory component: use -path
            safe_name = shlex.quote(f"*/{name_part}")
            cmd = f"find {safe_dir} -type f -path {safe_name}"
        else:
            # Just a filename pattern
            safe_name = shlex.quote(name_part)
            cmd = f"find {safe_dir} -type f -name {safe_name}"
    else:
        safe_name = shlex.quote(pattern)
        cmd = f"find {safe_dir} -type f -name {safe_name}"

    cmd += " 2>/dev/null"
    return cmd


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class GlobFileSearchTool(ConfigurableToolBase):
    """Configurable glob_file_search tool with sandbox-based I/O.

    All file access goes through the injected Sandbox instance.
    Uses `find` via sandbox.exec for file discovery.

    Example:
        >>> glob_tool = GlobFileSearchTool(max_results=100)
        >>> func = glob_tool.get_tool()
        >>> registry.register_tools([func])
        >>> registry.attach_sandbox(sandbox)
    """

    DOCSTRING_TEMPLATE = """Find files matching a glob pattern.

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
    Newline-separated file paths:
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
        max_results: int = 50,
        summary_max_ext_groups: int = 3,
        allowed_extensions: set[str] | None = None,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the GlobFileSearchTool.

        Args:
            max_results: Maximum number of results to return. Defaults to 50.
            summary_max_ext_groups: Maximum extension groups to show in summary. Defaults to 3.
            allowed_extensions: Set of allowed file extensions (with leading dot).
                               Defaults to {".md", ".mmd"} if None.
            docstring_template: Optional custom docstring template.
            schema_override: Optional complete Anthropic tool schema dict.
        """
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.max_results: int = max_results
        self.summary_max_ext_groups: int = summary_max_ext_groups
        self.allowed_extensions: set[str] = allowed_extensions or {".md", ".mmd"}

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "max_results": self.max_results,
            "allowed_extensions_str": ", ".join(sorted(self.allowed_extensions)),
        }

    def get_tool(self) -> Callable:
        instance = self

        async def glob_file_search(glob_pattern: str, target_directory: str | None = None) -> str:
            """Placeholder docstring - replaced by template."""
            target_dir = "." if target_directory is None else _posix_normpath(str(target_directory).replace("\\", "/"))
            rel_arg = "." if target_directory is None else target_dir

            # Verify target directory exists and is a directory
            if target_directory is not None:
                exists = await instance._sandbox.file_exists(target_dir)
                if not exists:
                    return f"Path does not exist: {rel_arg}. Use list_dir to explore available directories first."
                # Verify it's actually a directory (not a file)
                try:
                    await instance._sandbox.list_dir(target_dir)
                except NotADirectoryError:
                    return f"Path is not a directory: {rel_arg}. Remove the file name and search in its parent directory."

            pattern = _normalize_pattern(glob_pattern)
            cmd = _build_find_command(pattern, target_dir)

            result = await instance._sandbox.exec(cmd, timeout=15.0, cwd=".")
            if result.exit_code != 0 and not result.stdout.strip():
                return f"No matches found for pattern '{glob_pattern}'. Try a broader pattern (e.g., '*.md') or check the directory with list_dir."

            # Parse results, filter by allowed extensions
            raw_paths = [
                _posix_normpath(line.strip().removeprefix("./"))
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]

            # Filter by allowed extensions
            filtered_paths = [p for p in raw_paths if _has_allowed_ext(p, instance.allowed_extensions)]

            if not filtered_paths:
                return f"No matches found for pattern '{glob_pattern}'. Try a broader pattern (e.g., '*.md') or check the directory with list_dir."

            # Truncate and summarize
            shown_paths = filtered_paths[:instance.max_results]
            remaining_paths = filtered_paths[instance.max_results:]

            lines: List[str] = list(shown_paths)

            if remaining_paths:
                file_summary = _summarize_file_exts(remaining_paths, instance.summary_max_ext_groups)
                if file_summary:
                    lines.append(f"[{file_summary}]")

            return "\n".join(lines)

        func = self._apply_schema(glob_file_search)
        func.__tool_instance__ = instance
        return func
