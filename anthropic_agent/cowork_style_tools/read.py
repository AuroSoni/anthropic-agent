"""Cowork-style Read tool — read files with multimodal support."""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Callable

from ..tools.base import ToolResult, ImageBlock, DocumentBlock
from ..tools.decorators import tool


_DEFAULT_LIMIT = 2000
_MAX_LINE_LENGTH = 2000

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_PDF_EXTENSIONS = {".pdf"}
_MAX_PDF_SIZE = 32 * 1024 * 1024  # 32 MB API limit


def create_read_tool() -> Callable:
    """Create the read_file tool function.

    Returns:
        A @tool-decorated function for reading files.
    """

    @tool
    def read_file(
        file_path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        """Read a file from the local filesystem.

        Reads text files and returns content with line numbers (cat -n format).
        For images and PDFs, the file content is returned directly for the
        model to process visually.

        Args:
            file_path: Absolute path to the file to read.
            offset: Line number to start reading from (1-based). Defaults to 1.
            limit: Number of lines to read. Defaults to 2000.

        Returns:
            File contents with line numbers for text files, or the binary
            content for images and PDFs.
        """
        p = Path(file_path)

        if not p.is_absolute():
            return f"Error: Path must be absolute, got relative path: {file_path}"
        if not p.exists():
            return f"Error: File does not exist: {file_path}"
        if p.is_dir():
            return f"Error: Path is a directory, not a file: {file_path}. Use ls via the Bash tool to list directory contents."

        suffix = p.suffix.lower()

        # --- Image files ---
        if suffix in _IMAGE_EXTENSIONS:
            try:
                data = p.read_bytes()
                media_type = _IMAGE_MEDIA_TYPES[suffix]
                return ToolResult.with_image(
                    f"Image file: {file_path}", data, media_type
                )
            except OSError as e:
                return f"Error reading image file: {e}"

        # --- PDF files ---
        if suffix in _PDF_EXTENSIONS:
            try:
                data = p.read_bytes()
                if len(data) > _MAX_PDF_SIZE:
                    size_mb = len(data) / (1024 * 1024)
                    return f"Error: PDF file is {size_mb:.1f} MB, exceeding the 32 MB API limit."
                return ToolResult.with_document(
                    f"PDF document: {file_path}", data, "application/pdf"
                )
            except OSError as e:
                return f"Error reading PDF file: {e}"

        # --- Text files ---
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error reading file: {e}"

        if not text:
            return f"Warning: File exists but is empty: {file_path}"

        lines = text.split("\n")
        # Handle trailing newline: if file ends with \n, the last split
        # element is empty — don't count it as a separate line
        if text.endswith("\n") and lines and lines[-1] == "":
            lines = lines[:-1]

        total_lines = len(lines)

        # Apply offset (1-based)
        start = max(1, offset if offset is not None else 1)
        if start > total_lines:
            return f"Error: offset ({start}) exceeds total lines ({total_lines}) in {file_path}"

        # Apply limit
        end_limit = limit if limit is not None else _DEFAULT_LIMIT
        end = min(start - 1 + end_limit, total_lines)

        # Slice to the requested window (convert 1-based to 0-based)
        window = lines[start - 1 : end]

        # Format with line numbers (cat -n style)
        # Right-align line numbers to accommodate the max line number width
        width = len(str(end))
        formatted_lines = []
        for i, line in enumerate(window, start=start):
            # Truncate long lines
            if len(line) > _MAX_LINE_LENGTH:
                line = line[:_MAX_LINE_LENGTH] + " [truncated]"
            formatted_lines.append(f"{i:>{width}}\t{line}")

        return "\n".join(formatted_lines)

    return read_file
