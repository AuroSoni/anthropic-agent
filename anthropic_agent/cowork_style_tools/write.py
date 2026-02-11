"""Cowork-style Write tool — create or overwrite files."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

from ..tools.decorators import tool


def create_write_tool() -> Callable:
    """Create the write_file tool function.

    Returns:
        A @tool-decorated function for writing files.
    """

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Create or overwrite a file on the local filesystem.

        Writes the full content to the specified file. If the file already
        exists, its contents are completely replaced. Parent directories
        are created automatically if they don't exist.

        Args:
            file_path: Absolute path to the file to write.
            content: The complete content to write to the file.

        Returns:
            A success message with the file path and line count, or an error message.
        """
        p = Path(file_path)

        if not p.is_absolute():
            return f"Error: Path must be absolute, got relative path: {file_path}"

        # Create parent directories if needed
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return f"Error creating parent directories: {e}"

        # Atomic write: temp file + os.replace
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(p.parent), suffix=".tmp", prefix=".write_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                os.replace(tmp_path, str(p))
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            return f"Error writing file: {e}"

        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"Successfully wrote {line_count} lines to {file_path}"

    return write_file
