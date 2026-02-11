"""Cowork-style Grep tool — content search via ripgrep."""
from __future__ import annotations

import os
import subprocess
from typing import Callable

from ..tools.decorators import tool


def create_grep_tool() -> Callable:
    """Create the grep_search tool function.

    Returns:
        A @tool-decorated function for regex content search via ripgrep.
    """

    @tool
    def grep_search(
        pattern: str,
        path: str | None = None,
        output_mode: str = "files_with_matches",
        include_glob: str | None = None,
        file_type: str | None = None,
        after_context: int | None = None,
        before_context: int | None = None,
        context: int | None = None,
        case_insensitive: bool = False,
        line_numbers: bool = True,
        head_limit: int | None = None,
        offset: int | None = None,
        multiline: bool = False,
    ) -> str:
        """Search file contents using ripgrep regex.

        Args:
            pattern: The regular expression pattern to search for.
            path: File or directory to search in. Defaults to current working directory.
            output_mode: Output mode — "files_with_matches" (default, file paths only),
                "content" (matching lines with context), or "count" (match counts per file).
            include_glob: Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}").
            file_type: File type filter (e.g. "py", "js", "rust"). Uses ripgrep's built-in type definitions.
            after_context: Number of lines to show after each match (like rg -A).
            before_context: Number of lines to show before each match (like rg -B).
            context: Number of lines to show before and after each match (like rg -C).
            case_insensitive: Whether to search case-insensitively.
            line_numbers: Whether to show line numbers in content output. Defaults to true.
            head_limit: Limit output to first N lines/entries.
            offset: Skip first N lines/entries before applying head_limit.
            multiline: Enable multiline matching where . matches newlines.

        Returns:
            Search results based on the selected output_mode,
            or an error message if the search fails.
        """
        if not pattern:
            return "Error: Search pattern cannot be empty."

        search_path = path or os.getcwd()

        # Build ripgrep command
        cmd: list[str] = ["rg"]

        # Output mode flags
        if output_mode == "files_with_matches":
            cmd.append("--files-with-matches")
        elif output_mode == "count":
            cmd.append("--count")
        elif output_mode == "content":
            if line_numbers:
                cmd.append("-n")
        else:
            return f"Error: Invalid output_mode '{output_mode}'. Expected 'files_with_matches', 'content', or 'count'."

        # Context lines (only meaningful for content mode)
        if output_mode == "content":
            if context is not None:
                cmd.extend(["-C", str(context)])
            else:
                if after_context is not None:
                    cmd.extend(["-A", str(after_context)])
                if before_context is not None:
                    cmd.extend(["-B", str(before_context)])

        # Case sensitivity
        if case_insensitive:
            cmd.append("-i")

        # Multiline
        if multiline:
            cmd.extend(["-U", "--multiline-dotall"])

        # File filters
        if include_glob:
            cmd.extend(["--glob", include_glob])
        if file_type:
            cmd.extend(["--type", file_type])

        # Pattern and path
        cmd.append(pattern)
        cmd.append(search_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            return "Error: ripgrep (rg) is not installed. Install it from https://github.com/BurntSushi/ripgrep"
        except subprocess.TimeoutExpired:
            return "Error: Search timed out after 30 seconds."

        # rg returns exit code 1 for "no matches" (not an error)
        if result.returncode == 1:
            return "No matches found."
        if result.returncode == 2:
            return f"Error: {result.stderr.strip()}"

        output = result.stdout
        if not output.strip():
            return "No matches found."

        # Apply offset and head_limit
        lines = output.split("\n")
        # Remove trailing empty line from split
        if lines and lines[-1] == "":
            lines = lines[:-1]

        if offset:
            lines = lines[offset:]
        if head_limit:
            lines = lines[:head_limit]

        return "\n".join(lines)

    return grep_search
