"""Cowork-style Edit tool — exact string replacement in files."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

from ..tools.decorators import tool


def create_edit_tool() -> Callable:
    """Create the edit_file tool function.

    Returns:
        A @tool-decorated function for editing files via string replacement.
    """

    @tool
    def edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Perform exact string replacement in a file.

        Finds old_string in the file and replaces it with new_string.
        By default, old_string must appear exactly once (for safety).
        Set replace_all to true to replace every occurrence.

        Args:
            file_path: Absolute path to the file to modify.
            old_string: The exact text to find and replace. Must be unique
                in the file unless replace_all is true.
            new_string: The replacement text. Must differ from old_string.
            replace_all: If true, replace all occurrences. If false (default),
                old_string must appear exactly once.

        Returns:
            A success message with replacement count, or an error message.
        """
        p = Path(file_path)

        if not p.is_absolute():
            return f"Error: Path must be absolute, got relative path: {file_path}"
        if not p.exists():
            return f"Error: File does not exist: {file_path}"
        if not p.is_file():
            return f"Error: Path is not a file: {file_path}"
        if not old_string:
            return "Error: old_string cannot be empty."
        if old_string == new_string:
            return "Error: old_string and new_string are identical. No changes needed."

        # Read current content
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error reading file: {e}"

        # Count occurrences
        count = content.count(old_string)

        if count == 0:
            return f"Error: old_string not found in {file_path}."

        if not replace_all and count > 1:
            return (
                f"Error: old_string appears {count} times in {file_path}. "
                f"Provide more context to make it unique, or set replace_all=true."
            )

        # Perform replacement
        new_content = content.replace(old_string, new_string)
        replacements = count if replace_all else 1

        if not replace_all:
            # Single replacement: use replace with count=1 for safety
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1

        # Atomic write
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(p.parent), suffix=".tmp", prefix=".edit_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(new_content)
                os.replace(tmp_path, str(p))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            return f"Error writing file: {e}"

        return f"Successfully edited {file_path}. {replacements} replacement(s) made."

    return edit_file
