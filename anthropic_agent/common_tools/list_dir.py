"""
### Spec for `list_dir` in `src/tools/list_dir.py`

- **Signature**
  - `def list_dir(target_directory: str, ignore_globs: list[str] | None = None) -> str`

- **Purpose**
  - Returns a pretty-printed, concise, user-readable ASCII tree of the entire folder hierarchy (subject to limits below).

- **Output format**
  - Root shown as the basename of `target_directory`; root line has no leading bullet.
  - ASCII tree using `- ` bullets; directories end with `/`.
  - Indentation: 3 spaces per depth level.
  - Alphabetical sort within directories: directories first, then files (case-insensitive).
  - Hidden files are included.
  - Symlinked files are listed as files.
  - Symlinked directories are listed with `/` but are not recursed into.

- **Ignore patterns**
  - `ignore_globs` is applied to POSIX-style paths relative to `target_directory`, supports `**` (glob semantics).
  - If a path matches a pattern, it is skipped.
  - Special cases for directory subtree patterns:
    - `**/name/**`: hide the directory named `name` anywhere and all of its children.
    - `name/**`: hide all children of `name` but still print the directory line itself.

- **Recursion depth**
  - Maximum recursion depth: 5 levels (root is depth 0).
  - If a directory would exceed this depth, do not descend into it. Instead append a single summary line under that directory:
    - Example: `[depth limit reached; 132 files (py: 80, txt: 30, csv: 22), 45 subdirectories]`
    - File-type counts are grouped by extension (label `noext` for files without an extension). List the top groups (sorted by count desc, then ext asc) up to a limit, and aggregate the rest under `other`.

- **Large directory handling (> 50 total entries)**
  - "Large" is defined as more than 50 immediate entries in a directory (files + subdirectories after applies ignores).
  - Listing order and summarization within a large directory:
    - First 5 subdirectories (alphabetical), each processed normally (recursing until depth or size limits hit).
    - Then a subdirectory summary line: `[K more subdirectories]` (if any remain).
    - First 5 files (alphabetical).
    - Then a file summary line grouped by extension: `[N more files of type ext1, M more files of type ext2, ...]` (top groups first; aggregate tail into `other` if needed).
  - For non-large directories, list all entries (respecting recursion cap), with directories before files.
  - Summary lines are indented one level under their directory.

- **Error handling**
  - If `target_directory` does not exist or is not a directory, return a one-line message indicating the issue.
  - Permission errors are handled gracefully by showing the directory name and a single line: `[permission denied]`.

- **General notes**
  - The output is concise and readable; there's no strict length limit.
  - The function always provides a view into the hierarchy down to files, limited only by:
    - The recursion cap (with summary at cap),
    - Large-directory summarization,
    - Ignore rules,
    - Permission constraints.

- **Filtering**
  - Only files with extensions `.md` or `.mmd` are listed.
  - Only directories that contain at least one `.md`/`.mmd` file in their subtree are listed (root is always shown).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
from posixpath import normpath as _posix_normpath
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ..tools.decorators import tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INDENT_PER_LEVEL = " " * 3
MAX_DEPTH = 5  # Root is depth 0; do not descend beyond this depth
LARGE_DIR_ENTRY_THRESHOLD = 50
LARGE_DIR_SHOW_FILES = 5
LARGE_DIR_SHOW_DIRS = 5
SUMMARY_MAX_EXT_GROUPS = 3
ALLOWED_EXTS: set[str] = {".md", ".mmd"}


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


def ext_label(path: Path) -> str:
    suffix = path.suffix
    if not suffix:
        return "noext"
    # Remove leading dot
    return suffix[1:]


def summarize_extension_groups(paths: Iterable[Path]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for p in paths:
        counter[ext_label(p)] += 1
    return dict(counter)


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def format_ext_groups(counts: Dict[str, int], max_groups: Optional[int] = None) -> str:
    if not counts:
        return ""
    groups = SUMMARY_MAX_EXT_GROUPS if max_groups is None else max_groups
    # Sort by count desc, then ext asc
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


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class ListDirTool:
    """Configurable list_dir tool with a sandboxed base path.
    
    This class encapsulates the list_dir functionality, allowing configuration
    of the base path at instantiation time. The tool returned by get_tool()
    can be registered with an AnthropicAgent.
    
    Example:
        >>> list_dir_tool = ListDirTool(base_path="/path/to/workspace")
        >>> agent = AnthropicAgent(tools=[list_dir_tool.get_tool()])
    """
    
    def __init__(self, base_path: str | Path):
        """Initialize the ListDirTool with a base path.
        
        Args:
            base_path: The root directory that list_dir operates within.
                       All target_directory paths will be relative to this.
        """
        self.search_root: Path = Path(base_path).resolve()
        self._allowed_cache: Dict[Path, bool] = {}
        self._patterns: List[str] = []
    
    def _is_ignored(self, path: Path) -> bool:
        """Check if a path should be ignored based on current patterns."""
        try:
            rel = path.relative_to(self.search_root)
        except Exception:
            # Fallback to name; should rarely happen for children
            rel_posix = path.name
        else:
            rel_posix = rel.as_posix()
        posix_path = PurePosixPath(rel_posix)
        for pattern in self._patterns:
            # Normalize directory subtree patterns (ending with '/**')
            if pattern.endswith("/**"):
                dir_pat = pattern[:-3].rstrip("/")
                # Child paths inside the subtree are ignored
                if posix_path.match(pattern):
                    return True
                # The directory itself for patterns like '**/name/**'
                if pattern.startswith("**/"):
                    base_dir = dir_pat.split("/")[-1]
                    rel_str = rel_posix
                    if rel_str == base_dir or rel_str.endswith("/" + base_dir):
                        return True
                # For 'name/**' (not starting with '**/'), keep the dir and only hide children
                continue
            else:
                # Regular glob
                if posix_path.match(pattern):
                    return True
        return False
    
    def _is_allowed_file(self, path: Path) -> bool:
        """Check if a file has an allowed extension."""
        try:
            if path.is_dir():
                return False
        except Exception:
            # If we cannot determine, treat as not allowed
            return False
        return path.suffix.lower() in ALLOWED_EXTS
    
    def _has_allowed_in_subtree(self, directory: Path) -> bool:
        """Check if a directory contains at least one allowed file in its subtree."""
        # Memoize results for performance
        if directory in self._allowed_cache:
            return self._allowed_cache[directory]
        try:
            children = list(directory.iterdir())
        except PermissionError:
            # If we cannot access the directory, include it so we can render a permission line
            self._allowed_cache[directory] = True
            return True
        except Exception:
            self._allowed_cache[directory] = False
            return False
        for child in children:
            if self._is_ignored(child):
                continue
            try:
                if child.is_dir():
                    if self._has_allowed_in_subtree(child):
                        self._allowed_cache[directory] = True
                        return True
                else:
                    if self._is_allowed_file(child):
                        self._allowed_cache[directory] = True
                        return True
            except Exception:
                continue
        self._allowed_cache[directory] = False
        return False
    
    def _safe_list_dir(self, directory: Path) -> Tuple[List[Path], Optional[str]]:
        """List directory contents safely, handling errors and applying filters."""
        try:
            entries = list(directory.iterdir())
        except PermissionError:
            return [], "permission denied"
        except Exception as exc:
            # Unexpected errors are reported tersely
            return [], str(exc)
        # Apply ignores to immediate children
        filtered = [p for p in entries if not self._is_ignored(p)]
        # Partition into dirs and files
        dirs = [p for p in filtered if p.is_dir()]
        files = [p for p in filtered if not p.is_dir() and self._is_allowed_file(p)]
        # Filter directories to only those that contain allowed files
        dirs = [d for d in dirs if self._has_allowed_in_subtree(d)]
        # Sort alpha (case-insensitive)
        dirs.sort(key=lambda p: p.name.casefold())
        files.sort(key=lambda p: p.name.casefold())
        return dirs + files, None
    
    def _count_subtree(self, directory: Path) -> Tuple[int, int, Dict[str, int]]:
        """Count files, subdirectories, and file extensions in the subtree.
        
        Respects ignores and avoids recursing into symlinked directories.
        Only counts files with allowed extensions and directories that contain
        at least one allowed file.
        """
        files_count = 0
        dirs_count = 0
        ext_counts: Counter[str] = Counter()

        stack: List[Path] = [directory]
        visited: set[Path] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            try:
                children = list(current.iterdir())
            except PermissionError:
                # Skip inaccessible subtrees in summary
                continue
            except Exception:
                continue
            for child in children:
                if self._is_ignored(child):
                    continue
                try:
                    if child.is_dir():
                        # Skip symlinked directories for counting
                        if child.is_symlink():
                            continue
                        # Only count directory if it contains allowed files somewhere beneath
                        if self._has_allowed_in_subtree(child):
                            dirs_count += 1
                            stack.append(child)
                    else:
                        if self._is_allowed_file(child):
                            files_count += 1
                            ext_counts[ext_label(child)] += 1
                except Exception:
                    continue
        return files_count, dirs_count, dict(ext_counts)
    
    def _render_directory(self, directory: Path, depth: int, lines: List[str]) -> None:
        """Render a directory and its contents to the output lines."""
        # Root is always shown; non-root directories are shown only if they contain allowed files
        if depth == 0:
            lines.append(format_dir_line(directory.name, depth))
        else:
            if not self._has_allowed_in_subtree(directory):
                return
            lines.append(format_dir_line(directory.name, depth))

        # If beyond depth cap, summarize contents without descending
        if depth >= MAX_DEPTH:
            files_total, dirs_total, ext_counts = self._count_subtree(directory)
            # Compose summary
            if files_total == 0 and dirs_total == 0:
                return
            groups = sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if groups:
                # Limit displayed groups for conciseness
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
        children, err = self._safe_list_dir(directory)
        if err is not None:
            lines.append(format_bracket_line(err, depth))
            return

        # Partition immediate children into dirs and files (preserve sorting)
        immediate_dirs = [c for c in children if c.is_dir()]
        immediate_files = [c for c in children if not c.is_dir()]

        total_entries = len(immediate_dirs) + len(immediate_files)
        is_large = total_entries > LARGE_DIR_ENTRY_THRESHOLD

        # Subdirectories section (directories first)
        if is_large:
            shown_dirs = immediate_dirs[:LARGE_DIR_SHOW_DIRS]
            remaining_dirs = immediate_dirs[LARGE_DIR_SHOW_DIRS:]
        else:
            shown_dirs = immediate_dirs
            remaining_dirs = []

        for d in shown_dirs:
            # Do not recurse into symlinked directories
            try:
                if d.is_symlink():
                    # Render as a directory without descending
                    lines.append(format_dir_line(d.name, depth + 1))
                    continue
            except Exception:
                lines.append(format_dir_line(d.name, depth + 1))
                continue
            self._render_directory(d, depth + 1, lines)

        if remaining_dirs:
            lines.append(format_bracket_line(f"{len(remaining_dirs)} more subdirectories", depth))

        # Files section (after directories)
        if is_large:
            shown_files = immediate_files[:LARGE_DIR_SHOW_FILES]
            remaining_files = immediate_files[LARGE_DIR_SHOW_FILES:]
        else:
            shown_files = immediate_files
            remaining_files = []

        for f in shown_files:
            lines.append(format_file_line(f.name, depth + 1))

        if remaining_files:
            counts = summarize_extension_groups(remaining_files)
            text = format_ext_groups(counts)
            if text:
                lines.append(format_bracket_line(text, depth))
            else:
                lines.append(format_bracket_line(f"{len(remaining_files)} more files", depth))
    
    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.
        
        Returns:
            A decorated list_dir function that operates within the configured base_path.
        """
        # Capture self in closure for the inner function
        instance = self
        
        @tool
        def list_dir(target_directory: str, ignore_globs: List[str] | None = None) -> str:
            """Render a concise ASCII tree of a directory with depth and size-aware summarization.

            Directories are listed before files, alphabetically (case-insensitive). Hidden files are shown.
            Symlinked directories are displayed but not traversed. Recursion is capped at MAX_DEPTH, with a
            summary when the cap is reached. Large directories (> LARGE_DIR_ENTRY_THRESHOLD entries) show only
            the first LARGE_DIR_SHOW_DIRS subdirectories and LARGE_DIR_SHOW_FILES files, followed by summaries.

            Args:
                target_directory: Path to the root directory to render, relative to the configured base path.
                ignore_globs: Optional list of POSIX-style glob patterns relative to
                    target_directory. Supports '**' and subtree forms:
                    - '**/name/**' to hide any directory named 'name' and all of its children anywhere
                    - 'name/**' to print the directory line but hide all of its children

            Returns:
                An ASCII tree using '- ' bullets, directories with trailing '/', and summary lines:
                    - "[K more subdirectories]" for elided subdirectories in large directories
                    - "[N more files of type ext1, M of type ext2, ...]" for elided files (grouped by ext)
                    - "[depth limit reached; X files (ext1: n1, ...), Y subdirectories]" at depth cap
                    On errors, returns a one-line message such as:
                    - "Path does not exist: …"
                    - "Path is not a directory: …"
                    - "[permission denied]"
            """
            base = instance.search_root / target_directory
            rel_arg = _posix_normpath(str(target_directory).replace("\\", "/"))
            if not _is_within(base, instance.search_root):
                return f"Base path escapes search root: {rel_arg}"
            if not base.exists():
                return f"Path does not exist: {rel_arg}"
            if not base.is_dir():
                return f"Path is not a directory: {rel_arg}"

            # Reset instance state for this invocation
            instance._patterns = ignore_globs or []
            instance._allowed_cache = {}

            # Begin rendering
            lines: List[str] = []
            instance._render_directory(base, 0, lines)
            return "\n".join(lines)
        
        return list_dir
