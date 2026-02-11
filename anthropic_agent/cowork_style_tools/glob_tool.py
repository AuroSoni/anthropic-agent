"""Cowork-style Glob tool — file discovery by glob pattern."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from ..tools.decorators import tool


_MAX_RESULTS = 2000


def _safe_mtime(path: Path) -> float:
    """Get file modification time, returning 0.0 on error."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def create_glob_tool() -> Callable:
    """Create the glob_search tool function.

    Returns:
        A @tool-decorated function for glob-based file discovery.
    """

    @tool
    def glob_search(pattern: str, path: str | None = None) -> str:
        """Find files by glob pattern, sorted by modification time (newest first).

        Args:
            pattern: Glob pattern to match files against (e.g. "**/*.py", "src/**/*.ts").
            path: Directory to search in. Defaults to the current working directory.

        Returns:
            Newline-separated list of matching absolute file paths sorted by
            modification time, or a message if no matches are found.
        """
        search_root = Path(path) if path else Path(os.getcwd())

        if not search_root.exists():
            return f"Error: Path does not exist: {search_root}"
        if not search_root.is_dir():
            return f"Error: Path is not a directory: {search_root}"

        try:
            matches = [p for p in search_root.glob(pattern) if p.is_file()]
        except ValueError as e:
            return f"Error: Invalid glob pattern: {e}"

        if not matches:
            return "No matches found."

        # Sort by modification time, newest first; break ties by path
        matches.sort(key=lambda p: (-_safe_mtime(p), str(p)))

        total = len(matches)
        truncated = matches[:_MAX_RESULTS]
        lines = [str(p.resolve()) for p in truncated]

        if total > _MAX_RESULTS:
            lines.append(f"\n[{total - _MAX_RESULTS} more files not shown]")

        return "\n".join(lines)

    return glob_search
