"""ListDirTool — async, sandbox-based directory listing.

Migrated from anthropic_agent/common_tools/list_dir.py.
All filesystem access replaced with sandbox API calls.
The recursive _render_directory method is now async.
"""
from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath
from posixpath import normpath as _posix_normpath
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

from ..tools.base import ConfigurableToolBase

if TYPE_CHECKING:
    from ..sandbox.sandbox_types import FileEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INDENT_PER_LEVEL = " " * 3
SUMMARY_MAX_EXT_GROUPS = 3


# ---------------------------------------------------------------------------
# Pure utility functions (no state needed)
# ---------------------------------------------------------------------------
def format_dir_line(name: str, depth: int) -> str:
    bullet = "- " if depth > 0 else ""
    indent = INDENT_PER_LEVEL * depth
    return f"{indent}{bullet}{name}/"


def format_file_line(name: str, depth: int) -> str:
    indent = INDENT_PER_LEVEL * depth
    return f"{indent}- {name}"


def format_bracket_line(text: str, depth: int) -> str:
    indent = INDENT_PER_LEVEL * (depth + 1)
    return f"{indent}[{text}]"


def ext_label(name: str) -> str:
    if "." in name:
        suffix = name.rsplit(".", 1)[-1]
        return suffix
    return "noext"


def summarize_extension_groups(names: Iterable[str]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for n in names:
        counter[ext_label(n)] += 1
    return dict(counter)


def format_ext_groups(counts: Dict[str, int], max_groups: Optional[int] = None) -> str:
    if not counts:
        return ""
    groups = SUMMARY_MAX_EXT_GROUPS if max_groups is None else max_groups
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(items) <= groups:
        parts = [f"{n} more files of type {ext}" for ext, n in items]
        return ", ".join(parts)
    top = items[:groups]
    rest = items[groups:]
    parts = [f"{n} more files of type {ext}" for ext, n in top]
    other_count = sum(n for _, n in rest)
    parts.append(f"{other_count} more files of other types")
    return ", ".join(parts)


def _is_ignored(name: str, rel_posix: str, patterns: List[str]) -> bool:
    """Check if a path should be ignored based on patterns."""
    posix_path = PurePosixPath(rel_posix)
    for pattern in patterns:
        if pattern.endswith("/**"):
            dir_pat = pattern[:-3].rstrip("/")
            if posix_path.match(pattern):
                return True
            if pattern.startswith("**/"):
                base_dir = dir_pat.split("/")[-1]
                if rel_posix == base_dir or rel_posix.endswith("/" + base_dir):
                    return True
            continue
        else:
            if posix_path.match(pattern):
                return True
    return False


def _is_allowed_file(name: str, allowed_extensions: set[str]) -> bool:
    """Check if a filename has an allowed extension."""
    return any(name.lower().endswith(ext) for ext in allowed_extensions)


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class ListDirTool(ConfigurableToolBase):
    """Configurable list_dir tool with sandbox-based I/O.

    All directory access goes through the injected Sandbox instance.

    Example:
        >>> list_dir_tool = ListDirTool(max_depth=10)
        >>> func = list_dir_tool.get_tool()
        >>> registry.register_tools([func])
        >>> registry.attach_sandbox(sandbox)
    """

    DOCSTRING_TEMPLATE = """List the contents of a directory as an ASCII tree.

Use this tool to explore the structure of a codebase. Shows directories before files,
sorted alphabetically. Hidden files are included.

**Limits:**
- Max depth: {max_depth} levels (deeper directories show a summary count)
- Large directories (>{large_dir_threshold} entries): Shows first {large_dir_show_dirs} subdirs + {large_dir_show_files} files with summaries
- Allowed extensions: {allowed_extensions_str}

Args:
    target_directory: Path relative to the workspace root. Use "." for the root.
    ignore_globs: Glob patterns to exclude. Examples:
        - "**/node_modules/**" - hide node_modules and all contents anywhere
        - "**/__pycache__/**" - hide Python cache directories
        - "*.log" - hide all .log files
        - "name/**" - show directory 'name' but hide its contents

Returns:
    ASCII tree with "- " bullets. Directories end with "/".
    Example output:
    ```
    my_project/
       - src/
          - main.py
          - utils.py
       - tests/
       - README.md
    ```

    On depth/size limits: "[depth limit reached; 42 files (py: 30, md: 12), 5 subdirectories]"

**Error Recovery:**
- "Path does not exist" -> Check the parent directory with list_dir first
- "Path is not a directory" -> Use read_file to view the file contents instead
- "[permission denied]" -> Try a different directory or check file permissions
"""

    def __init__(
        self,
        max_depth: int = 5,
        large_dir_threshold: int = 50,
        large_dir_show_files: int = 5,
        large_dir_show_dirs: int = 5,
        allowed_extensions: set[str] | None = None,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.max_depth: int = max_depth
        self.large_dir_threshold: int = large_dir_threshold
        self.large_dir_show_files: int = large_dir_show_files
        self.large_dir_show_dirs: int = large_dir_show_dirs
        self.allowed_extensions: set[str] = allowed_extensions or {".md", ".mmd"}

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "large_dir_threshold": self.large_dir_threshold,
            "large_dir_show_files": self.large_dir_show_files,
            "large_dir_show_dirs": self.large_dir_show_dirs,
            "allowed_extensions_str": ", ".join(sorted(self.allowed_extensions)),
        }

    def get_tool(self) -> Callable:
        instance = self

        async def list_dir(target_directory: str, ignore_globs: List[str] | None = None) -> str:
            """Placeholder docstring - replaced by template."""
            rel_path = _posix_normpath(str(target_directory).replace("\\", "/"))

            # Check if path exists and is a directory
            exists = await instance._sandbox.file_exists(rel_path)
            if not exists:
                return f"Path does not exist: {rel_path}. Try list_dir on the parent directory to see available paths."

            # Try listing to verify it's a directory
            try:
                await instance._sandbox.list_dir(rel_path)
            except NotADirectoryError:
                return f"Path is not a directory: {rel_path}. Use read_file to view the file contents instead."

            patterns = ignore_globs or []
            # Track which directories contain allowed files (memoization)
            allowed_cache: Dict[str, bool] = {}

            async def has_allowed_in_subtree(dir_path: str) -> bool:
                """Check if directory contains at least one allowed file."""
                if dir_path in allowed_cache:
                    return allowed_cache[dir_path]

                try:
                    entries = await instance._sandbox.list_dir(dir_path)
                except Exception:
                    allowed_cache[dir_path] = False
                    return False

                for entry in entries:
                    child_path = f"{dir_path}/{entry.name}" if dir_path != "." else entry.name
                    child_rel = child_path

                    if _is_ignored(entry.name, child_rel, patterns):
                        continue

                    if entry.is_dir:
                        if await has_allowed_in_subtree(child_path):
                            allowed_cache[dir_path] = True
                            return True
                    else:
                        if _is_allowed_file(entry.name, instance.allowed_extensions):
                            allowed_cache[dir_path] = True
                            return True

                allowed_cache[dir_path] = False
                return False

            async def count_subtree(dir_path: str) -> Tuple[int, int, Dict[str, int]]:
                """Count files, subdirectories, and extension groups in subtree."""
                files_count = 0
                dirs_count = 0
                ext_counts: Counter[str] = Counter()

                stack: List[str] = [dir_path]
                visited: set[str] = set()

                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)

                    try:
                        entries = await instance._sandbox.list_dir(current)
                    except Exception:
                        continue

                    for entry in entries:
                        child_path = f"{current}/{entry.name}" if current != "." else entry.name

                        if _is_ignored(entry.name, child_path, patterns):
                            continue

                        if entry.is_dir:
                            if await has_allowed_in_subtree(child_path):
                                dirs_count += 1
                                stack.append(child_path)
                        else:
                            if _is_allowed_file(entry.name, instance.allowed_extensions):
                                files_count += 1
                                ext_counts[ext_label(entry.name)] += 1

                return files_count, dirs_count, dict(ext_counts)

            async def render_directory(dir_path: str, dir_name: str, depth: int, lines: List[str]) -> None:
                """Render a directory and its contents to output lines."""
                if depth == 0:
                    lines.append(format_dir_line(dir_name, depth))
                else:
                    if not await has_allowed_in_subtree(dir_path):
                        return
                    lines.append(format_dir_line(dir_name, depth))

                # If beyond depth cap, summarize
                if depth >= instance.max_depth:
                    files_total, dirs_total, ext_counts = await count_subtree(dir_path)
                    if files_total == 0 and dirs_total == 0:
                        return
                    groups = sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    if groups:
                        shown = groups[:SUMMARY_MAX_EXT_GROUPS]
                        rest_count = sum(n for _, n in groups[SUMMARY_MAX_EXT_GROUPS:])
                        files_part = ", ".join(f"{ext}: {n}" for ext, n in shown)
                        if rest_count:
                            files_part = f"{files_part}, other: {rest_count}"
                        summary = f"depth limit reached; {files_total} files ({files_part}), {dirs_total} subdirectories"
                    else:
                        summary = f"depth limit reached; {files_total} files, {dirs_total} subdirectories"
                    lines.append(format_bracket_line(summary, depth))
                    return

                # List immediate children
                try:
                    entries = await instance._sandbox.list_dir(dir_path)
                except Exception as exc:
                    lines.append(format_bracket_line(str(exc), depth))
                    return

                # Partition and filter
                immediate_dirs: List[Tuple[str, str]] = []  # (name, path)
                immediate_files: List[str] = []  # names

                for entry in entries:
                    child_path = f"{dir_path}/{entry.name}" if dir_path != "." else entry.name

                    if _is_ignored(entry.name, child_path, patterns):
                        continue

                    if entry.is_dir:
                        if await has_allowed_in_subtree(child_path):
                            immediate_dirs.append((entry.name, child_path))
                    else:
                        if _is_allowed_file(entry.name, instance.allowed_extensions):
                            immediate_files.append(entry.name)

                # Sort case-insensitively (matching original behavior)
                immediate_dirs.sort(key=lambda t: t[0].casefold())
                immediate_files.sort(key=str.casefold)

                total_entries = len(immediate_dirs) + len(immediate_files)
                is_large = total_entries > instance.large_dir_threshold

                # Subdirectories
                if is_large:
                    shown_dirs = immediate_dirs[:instance.large_dir_show_dirs]
                    remaining_dir_count = len(immediate_dirs) - instance.large_dir_show_dirs
                else:
                    shown_dirs = immediate_dirs
                    remaining_dir_count = 0

                for name, path in shown_dirs:
                    await render_directory(path, name, depth + 1, lines)

                if remaining_dir_count > 0:
                    lines.append(format_bracket_line(f"{remaining_dir_count} more subdirectories", depth))

                # Files
                if is_large:
                    shown_files = immediate_files[:instance.large_dir_show_files]
                    remaining_files = immediate_files[instance.large_dir_show_files:]
                else:
                    shown_files = immediate_files
                    remaining_files = []

                for f in shown_files:
                    lines.append(format_file_line(f, depth + 1))

                if remaining_files:
                    counts = summarize_extension_groups(remaining_files)
                    text = format_ext_groups(counts)
                    if text:
                        lines.append(format_bracket_line(text, depth))
                    else:
                        lines.append(format_bracket_line(f"{len(remaining_files)} more files", depth))

            # Render
            dir_name = rel_path.rsplit("/", 1)[-1] if "/" in rel_path else rel_path
            if rel_path == ".":
                dir_name = "."

            lines: List[str] = []
            await render_directory(rel_path, dir_name, 0, lines)
            return "\n".join(lines)

        func = self._apply_schema(list_dir)
        func.__tool_instance__ = instance
        return func
