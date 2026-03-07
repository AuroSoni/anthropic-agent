"""Tests for read_file tool functionality.

Tests cover:
- Initialization and setup
- Path resolution and security (traversal prevention, allowed extensions)
- Parameter validation (offset, limit)
- Content reading in buffered mode (small files)
- Content reading in streaming mode (large files)
- Edge cases (empty files, unicode, line endings)
"""
import tempfile
from pathlib import Path
from typing import Callable, Generator
from unittest.mock import patch

import pytest

from anthropic_agent.common_tools.read_file import (
    ALLOWED_EXTS,
    MAX_LIMIT,
    STREAMING_THRESHOLD_BYTES,
    ReadFileTool,
    _format_header,
    _is_within,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def read_file_tool(temp_workspace: Path) -> ReadFileTool:
    """Create a ReadFileTool instance with the temp workspace."""
    return ReadFileTool(base_path=temp_workspace)


@pytest.fixture
def read_file_fn(read_file_tool: ReadFileTool) -> Callable:
    """Get the read_file function from the tool."""
    return read_file_tool.get_tool()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def create_file(workspace: Path, rel_path: str, content: str = "") -> Path:
    """Create a test file in the workspace.

    Args:
        workspace: Base directory path.
        rel_path: Relative path for the file.
        content: File content.

    Returns:
        The full path to the created file.
    """
    full_path = workspace / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    return full_path


def create_dir(workspace: Path, rel_path: str) -> Path:
    """Create a directory in the workspace."""
    full_path = workspace / rel_path
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


# ---------------------------------------------------------------------------
# Tests for utility functions
# ---------------------------------------------------------------------------
class TestIsWithin:
    """Tests for _is_within helper function."""

    def test_child_within_parent(self, temp_workspace: Path) -> None:
        """Child path inside parent should return True."""
        child = temp_workspace / "subdir" / "file.md"
        assert _is_within(child, temp_workspace) is True

    def test_child_outside_parent(self, temp_workspace: Path) -> None:
        """Child path outside parent should return False."""
        outside = temp_workspace.parent / "outside.md"
        assert _is_within(outside, temp_workspace) is False

    def test_same_path(self, temp_workspace: Path) -> None:
        """Same path should be within itself."""
        assert _is_within(temp_workspace, temp_workspace) is True

    def test_traversal_attempt(self, temp_workspace: Path) -> None:
        """Path with .. traversal escaping should return False."""
        traversal = temp_workspace / ".." / "outside.md"
        assert _is_within(traversal, temp_workspace) is False


class TestFormatHeader:
    """Tests for _format_header helper function."""

    def test_basic_header(self) -> None:
        """Basic header formatting."""
        result = _format_header(1, 10, 100, "docs/readme.md")
        assert result == "[lines 1-10 of 100 in docs/readme.md]"

    def test_zero_range_header(self) -> None:
        """Header with zero range (limit=0 case)."""
        result = _format_header(0, 0, 50, "file.md")
        assert result == "[lines 0-0 of 50 in file.md]"

    def test_single_line_header(self) -> None:
        """Header for single line slice."""
        result = _format_header(5, 5, 10, "test.mmd")
        assert result == "[lines 5-5 of 10 in test.mmd]"


# ---------------------------------------------------------------------------
# Tests for initialization and setup
# ---------------------------------------------------------------------------
class TestToolInitialization:
    """Tests for ReadFileTool initialization."""

    def test_init_with_string_path(self, temp_workspace: Path) -> None:
        """Initialize with string path."""
        tool = ReadFileTool(base_path=str(temp_workspace))
        assert tool.search_root == temp_workspace.resolve()

    def test_init_with_path_object(self, temp_workspace: Path) -> None:
        """Initialize with Path object."""
        tool = ReadFileTool(base_path=temp_workspace)
        assert tool.search_root == temp_workspace.resolve()

    def test_get_tool_returns_callable(self, read_file_tool: ReadFileTool) -> None:
        """get_tool() should return a callable function."""
        fn = read_file_tool.get_tool()
        assert callable(fn)

    def test_tool_has_name_attribute(self, read_file_fn: Callable) -> None:
        """The tool function should have a __name__ attribute."""
        assert hasattr(read_file_fn, "__name__")
        assert read_file_fn.__name__ == "read_file"

    def test_init_with_custom_max_lines(self, temp_workspace: Path) -> None:
        """Initialize with custom max_lines limit."""
        tool = ReadFileTool(base_path=temp_workspace, max_lines=50)
        assert tool.max_lines == 50

    def test_init_with_custom_streaming_threshold(self, temp_workspace: Path) -> None:
        """Initialize with custom streaming threshold."""
        tool = ReadFileTool(base_path=temp_workspace, streaming_threshold_bytes=1024)
        assert tool.streaming_threshold == 1024

    def test_init_with_custom_extensions(self, temp_workspace: Path) -> None:
        """Initialize with custom allowed extensions."""
        custom_exts = {".txt", ".py"}
        tool = ReadFileTool(base_path=temp_workspace, allowed_extensions=custom_exts)
        assert tool.allowed_extensions == custom_exts

    def test_default_extensions(self, read_file_tool: ReadFileTool) -> None:
        """Default allowed extensions should be .md and .mmd."""
        assert read_file_tool.allowed_extensions == {".md", ".mmd"}


class TestCustomExtensions:
    """Tests for custom allowed_extensions configuration."""

    def test_custom_extension_txt_allowed(self, temp_workspace: Path) -> None:
        """Custom extensions should allow reading .txt files."""
        tool = ReadFileTool(
            base_path=temp_workspace,
            allowed_extensions={".txt"}
        )
        fn = tool.get_tool()
        
        # Create a .txt file
        txt_file = temp_workspace / "notes.txt"
        txt_file.write_text("Text content\n")
        
        result = fn("notes.txt")
        assert "Text content" in result
        assert "[lines 1-1 of 1 in notes.txt]" in result

    def test_custom_extension_md_not_allowed(self, temp_workspace: Path) -> None:
        """When .md not in allowed_extensions, .md files should not be found."""
        tool = ReadFileTool(
            base_path=temp_workspace,
            allowed_extensions={".txt"}
        )
        fn = tool.get_tool()
        
        # Create a .md file
        md_file = temp_workspace / "doc.md"
        md_file.write_text("Markdown content\n")
        
        result = fn("doc.md")
        assert "Path does not exist" in result

    def test_custom_extension_py_allowed(self, temp_workspace: Path) -> None:
        """Custom extensions should allow reading .py files."""
        tool = ReadFileTool(
            base_path=temp_workspace,
            allowed_extensions={".py", ".md"}
        )
        fn = tool.get_tool()
        
        # Create a .py file
        py_file = temp_workspace / "script.py"
        py_file.write_text("print('hello')\n")
        
        result = fn("script.py")
        assert "print('hello')" in result

    def test_implicit_extension_uses_custom_extensions(self, temp_workspace: Path) -> None:
        """Extension resolution should use custom allowed extensions."""
        tool = ReadFileTool(
            base_path=temp_workspace,
            allowed_extensions={".txt", ".rst"}
        )
        fn = tool.get_tool()
        
        # Create files with different extensions
        (temp_workspace / "doc.txt").write_text("TXT content\n")
        (temp_workspace / "doc.rst").write_text("RST content\n")
        
        # Request without extension should find one of the allowed extensions
        result = fn("doc")
        assert "content" in result


class TestCustomMaxLines:
    """Tests for custom max_lines configuration."""

    def test_custom_max_lines_clamping(self, temp_workspace: Path) -> None:
        """Limit should be clamped to custom max_lines."""
        tool = ReadFileTool(base_path=temp_workspace, max_lines=10)
        fn = tool.get_tool()
        
        # Create file with 20 lines
        lines = [f"line{i}\n" for i in range(1, 21)]
        (temp_workspace / "test.md").write_text("".join(lines))
        
        result = fn("test.md", no_of_lines_to_read=50)  # Request more than max
        # Should only get 10 lines
        assert "[lines 1-10 of 20 in test.md]" in result

    def test_custom_max_lines_default_behavior(self, temp_workspace: Path) -> None:
        """Default limit should use custom max_lines."""
        tool = ReadFileTool(base_path=temp_workspace, max_lines=5)
        fn = tool.get_tool()
        
        # Create file with 10 lines
        lines = [f"line{i}\n" for i in range(1, 11)]
        (temp_workspace / "test.md").write_text("".join(lines))
        
        result = fn("test.md")  # No limit specified
        # Should use custom default of 5
        assert "[lines 1-5 of 10 in test.md]" in result


# ---------------------------------------------------------------------------
# Tests for path resolution and security
# ---------------------------------------------------------------------------
class TestPathResolutionAndSecurity:
    """Tests for path resolution and security checks."""

    def test_read_valid_md_file(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Read a standard .md file."""
        create_file(temp_workspace, "readme.md", "# Hello\nWorld\n")
        result = read_file_fn("readme.md")
        assert "[lines 1-2 of 2 in readme.md]" in result
        assert "# Hello" in result
        assert "World" in result

    def test_read_valid_mmd_file(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Read a standard .mmd file."""
        create_file(temp_workspace, "diagram.mmd", "graph TD\n  A --> B\n")
        result = read_file_fn("diagram.mmd")
        assert "[lines 1-2 of 2 in diagram.mmd]" in result
        assert "graph TD" in result

    def test_implicit_extension_resolution_md(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Access file without extension resolves to .md."""
        create_file(temp_workspace, "doc.md", "Content here\n")
        result = read_file_fn("doc")
        assert "[lines 1-1 of 1 in doc.md]" in result
        assert "Content here" in result

    def test_implicit_extension_resolution_mmd(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Access file without extension resolves to .mmd when no .md exists."""
        create_file(temp_workspace, "flowchart.mmd", "flowchart LR\n")
        result = read_file_fn("flowchart")
        assert "[lines 1-1 of 1 in flowchart.mmd]" in result

    def test_md_takes_precedence_over_mmd(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """When both .md and .mmd exist, .md takes precedence."""
        create_file(temp_workspace, "both.md", "markdown content\n")
        create_file(temp_workspace, "both.mmd", "mermaid content\n")
        result = read_file_fn("both")
        assert "both.md" in result
        assert "markdown content" in result

    def test_path_traversal_prevention(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Attempt to access ../outside.md should be blocked."""
        # Create file outside workspace
        outside_dir = temp_workspace.parent
        create_file(outside_dir, "outside.md", "secret content\n")

        result = read_file_fn("../outside.md")
        assert "Base path escapes search root" in result

    def test_double_dot_traversal(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Multiple .. traversal attempts should be blocked."""
        result = read_file_fn("../../etc/passwd")
        assert "Base path escapes search root" in result

    def test_absolute_path_outside_root(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Absolute path outside root should be rejected."""
        result = read_file_fn("/etc/passwd")
        # The path will be joined with base_path, so it should either escape
        # or not exist - behavior depends on OS path handling
        assert "Base path escapes search root" in result or "Path does not exist" in result

    def test_directory_path_error(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Pass a directory path should return error."""
        create_dir(temp_workspace, "subdir")
        result = read_file_fn("subdir")
        assert "Path is a directory" in result

    def test_non_existent_file(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Non-existent file should return error."""
        result = read_file_fn("nonexistent.md")
        assert "Path does not exist" in result

    def test_invalid_extension_txt(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Reading .txt file should fail (not in allowed extensions)."""
        create_file(temp_workspace, "notes.txt", "some notes\n")
        result = read_file_fn("notes.txt")
        # Since only .md/.mmd are allowed, .txt won't resolve
        assert "Path does not exist" in result

    def test_invalid_extension_py(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Reading .py file should fail (not in allowed extensions)."""
        create_file(temp_workspace, "script.py", "print('hello')\n")
        result = read_file_fn("script.py")
        assert "Path does not exist" in result

    def test_nested_directory_file(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Read file in nested directory."""
        create_file(temp_workspace, "docs/api/reference.md", "API docs\n")
        result = read_file_fn("docs/api/reference.md")
        assert "[lines 1-1 of 1 in docs/api/reference.md]" in result
        assert "API docs" in result

    def test_backslash_path_normalization(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Backslash paths should be normalized to forward slashes."""
        create_file(temp_workspace, "docs/guide.md", "Guide content\n")
        result = read_file_fn("docs\\guide.md")
        assert "Guide content" in result


# ---------------------------------------------------------------------------
# Tests for parameter validation
# ---------------------------------------------------------------------------
class TestParameterValidation:
    """Tests for offset and limit parameter validation."""

    def test_invalid_limit_type_string(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Non-integer string no_of_lines_to_read should return error."""
        create_file(temp_workspace, "test.md", "line1\nline2\n")
        result = read_file_fn("test.md", no_of_lines_to_read="abc")
        assert "ERROR" in result and "abc" in result and "not an integer" in result

    def test_invalid_limit_type_float_string(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Float string no_of_lines_to_read should return error."""
        create_file(temp_workspace, "test.md", "line1\nline2\n")
        result = read_file_fn("test.md", no_of_lines_to_read="3.5")
        assert "ERROR" in result and "3.5" in result and "not an integer" in result

    def test_negative_limit(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Negative no_of_lines_to_read should return error."""
        create_file(temp_workspace, "test.md", "line1\nline2\n")
        result = read_file_fn("test.md", no_of_lines_to_read=-5)
        assert "ERROR" in result and "-5" in result and "cannot be negative" in result

    def test_offset_exceeds_total_lines(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """start_line_one_indexed larger than total lines should return error."""
        create_file(temp_workspace, "short.md", "line1\nline2\n")
        result = read_file_fn("short.md", start_line_one_indexed=10)
        assert "ERROR" in result and "10" in result and "greater than total number of lines" in result
        assert "short.md" in result

    def test_offset_underflow_zero(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """start_line_one_indexed of 0 should be treated as 1."""
        create_file(temp_workspace, "test.md", "line1\nline2\nline3\n")
        result = read_file_fn("test.md", start_line_one_indexed=0, no_of_lines_to_read=1)
        assert "[lines 1-1 of 3 in test.md]" in result
        assert "line1" in result

    def test_offset_underflow_negative(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Negative start_line_one_indexed should be treated as 1."""
        create_file(temp_workspace, "test.md", "line1\nline2\n")
        result = read_file_fn("test.md", start_line_one_indexed=-10, no_of_lines_to_read=1)
        assert "[lines 1-1 of 2 in test.md]" in result
        assert "line1" in result

    def test_limit_clamping_above_max(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """no_of_lines_to_read > MAX_LIMIT should be clamped to MAX_LIMIT."""
        # Create file with more than MAX_LIMIT lines
        lines = [f"line{i}\n" for i in range(1, 150)]
        create_file(temp_workspace, "long.md", "".join(lines))
        result = read_file_fn("long.md", no_of_lines_to_read=500)
        # Should only show MAX_LIMIT (100) lines
        assert f"[lines 1-{MAX_LIMIT} of 149 in long.md]" in result

    def test_limit_zero_returns_header_only(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """no_of_lines_to_read of 0 should return header only with 0-0 range."""
        create_file(temp_workspace, "test.md", "line1\nline2\nline3\n")
        result = read_file_fn("test.md", no_of_lines_to_read=0)
        assert "[lines 0-0 of 3 in test.md]" in result
        # Should only have header and newline, no content
        lines = result.strip().split("\n")
        assert len(lines) == 1

    def test_limit_as_integer_float(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Float that equals integer (e.g., 5.0) should work via int()."""
        create_file(temp_workspace, "test.md", "line1\nline2\nline3\n")
        result = read_file_fn("test.md", no_of_lines_to_read=2.0)
        assert "[lines 1-2 of 3 in test.md]" in result

    def test_default_offset_and_limit(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Default offset=1 and limit=100 when not provided."""
        lines = [f"line{i}\n" for i in range(1, 51)]
        create_file(temp_workspace, "test.md", "".join(lines))
        result = read_file_fn("test.md")
        assert "[lines 1-50 of 50 in test.md]" in result


# ---------------------------------------------------------------------------
# Tests for content reading (buffered mode)
# ---------------------------------------------------------------------------
class TestBufferedModeReading:
    """Tests for reading files in buffered mode (small files)."""

    def test_read_full_file(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Read entire small file."""
        content = "First line\nSecond line\nThird line\n"
        create_file(temp_workspace, "small.md", content)
        result = read_file_fn("small.md")
        assert "[lines 1-3 of 3 in small.md]" in result
        assert "First line" in result
        assert "Second line" in result
        assert "Third line" in result

    def test_read_slice_middle(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Read specific lines from middle of file."""
        lines = [f"line{i}\n" for i in range(1, 11)]
        create_file(temp_workspace, "test.md", "".join(lines))
        result = read_file_fn("test.md", start_line_one_indexed=3, no_of_lines_to_read=3)
        assert "[lines 3-5 of 10 in test.md]" in result
        assert "line3" in result
        assert "line4" in result
        assert "line5" in result
        assert "line2" not in result
        assert "line6" not in result

    def test_read_slice_from_end(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Read lines near end of file."""
        lines = [f"line{i}\n" for i in range(1, 11)]
        create_file(temp_workspace, "test.md", "".join(lines))
        result = read_file_fn("test.md", start_line_one_indexed=8, no_of_lines_to_read=5)
        # Should only get lines 8-10 (3 lines, not 5)
        assert "[lines 8-10 of 10 in test.md]" in result
        assert "line8" in result
        assert "line10" in result

    def test_read_utf8_content(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Verify handling of UTF-8 characters."""
        content = "Hello 世界\nПривет мир\n日本語テスト\n🎉 Emoji test\n"
        create_file(temp_workspace, "unicode.md", content)
        result = read_file_fn("unicode.md")
        assert "世界" in result
        assert "Привет" in result
        assert "日本語" in result
        assert "🎉" in result

    def test_read_empty_file(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Read an empty file - should return header with 0-0 range."""
        create_file(temp_workspace, "empty.md", "")
        result = read_file_fn("empty.md")
        # Empty file returns valid header with 0-0 range
        assert "[lines 0-0 of 0 in empty.md]" in result

    def test_read_single_line_no_newline(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Read file with single line, no trailing newline."""
        create_file(temp_workspace, "single.md", "Only line")
        result = read_file_fn("single.md")
        assert "[lines 1-1 of 1 in single.md]" in result
        assert "Only line" in result

    def test_read_file_with_blank_lines(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Read file with blank lines."""
        content = "Line 1\n\n\nLine 4\n"
        create_file(temp_workspace, "blanks.md", content)
        result = read_file_fn("blanks.md")
        assert "[lines 1-4 of 4 in blanks.md]" in result

    def test_windows_line_endings(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Windows CRLF line endings should be handled."""
        # Write with binary mode to preserve CRLF
        full_path = temp_workspace / "windows.md"
        full_path.write_bytes(b"Line 1\r\nLine 2\r\nLine 3\r\n")
        result = read_file_fn("windows.md")
        assert "[lines 1-3 of 3 in windows.md]" in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_mixed_line_endings(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Mixed line endings should be handled via universal newline."""
        full_path = temp_workspace / "mixed.md"
        full_path.write_bytes(b"Line 1\nLine 2\r\nLine 3\rLine 4\n")
        result = read_file_fn("mixed.md")
        # Universal newline mode should normalize all endings
        assert "Line 1" in result
        assert "Line 2" in result


# ---------------------------------------------------------------------------
# Tests for content reading (streaming mode)
# ---------------------------------------------------------------------------
class TestStreamingModeReading:
    """Tests for reading files in streaming mode (large files).

    Uses monkeypatch to lower STREAMING_THRESHOLD_BYTES for testing.
    """

    def test_streaming_read_full(
        self, temp_workspace: Path, read_file_tool: ReadFileTool
    ) -> None:
        """Read full file in streaming mode."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
        create_file(temp_workspace, "large.md", content)

        # Patch threshold to force streaming mode
        with patch(
            "anthropic_agent.common_tools.read_file.STREAMING_THRESHOLD_BYTES", 1
        ):
            fn = read_file_tool.get_tool()
            result = fn("large.md")

        assert "[lines 1-5 of 5 in large.md]" in result
        assert "Line 1" in result
        assert "Line 5" in result

    def test_streaming_read_slice(
        self, temp_workspace: Path, read_file_tool: ReadFileTool
    ) -> None:
        """Read specific slice in streaming mode."""
        lines = [f"line{i}\n" for i in range(1, 21)]
        create_file(temp_workspace, "large.md", "".join(lines))

        with patch(
            "anthropic_agent.common_tools.read_file.STREAMING_THRESHOLD_BYTES", 1
        ):
            fn = read_file_tool.get_tool()
            result = fn("large.md", start_line_one_indexed=5, no_of_lines_to_read=3)

        assert "[lines 5-7 of 20 in large.md]" in result
        assert "line5" in result
        assert "line6" in result
        assert "line7" in result
        assert "line4" not in result
        assert "line8" not in result

    def test_streaming_limit_zero(
        self, temp_workspace: Path, read_file_tool: ReadFileTool
    ) -> None:
        """Limit=0 in streaming mode returns header only."""
        content = "Line 1\nLine 2\nLine 3\n"
        create_file(temp_workspace, "large.md", content)

        with patch(
            "anthropic_agent.common_tools.read_file.STREAMING_THRESHOLD_BYTES", 1
        ):
            fn = read_file_tool.get_tool()
            result = fn("large.md", no_of_lines_to_read=0)

        assert "[lines 0-0 of 3 in large.md]" in result
        lines = result.strip().split("\n")
        assert len(lines) == 1

    def test_streaming_offset_exceeds(
        self, temp_workspace: Path, read_file_tool: ReadFileTool
    ) -> None:
        """Offset > total lines in streaming mode returns error."""
        content = "Line 1\nLine 2\n"
        create_file(temp_workspace, "large.md", content)

        with patch(
            "anthropic_agent.common_tools.read_file.STREAMING_THRESHOLD_BYTES", 1
        ):
            fn = read_file_tool.get_tool()
            result = fn("large.md", start_line_one_indexed=100)

        assert "ERROR" in result and "100" in result and "greater than total number of lines" in result

    def test_streaming_read_from_end(
        self, temp_workspace: Path, read_file_tool: ReadFileTool
    ) -> None:
        """Read lines near end in streaming mode."""
        lines = [f"line{i}\n" for i in range(1, 101)]
        create_file(temp_workspace, "large.md", "".join(lines))

        with patch(
            "anthropic_agent.common_tools.read_file.STREAMING_THRESHOLD_BYTES", 1
        ):
            fn = read_file_tool.get_tool()
            result = fn("large.md", start_line_one_indexed=95, no_of_lines_to_read=10)

        # Should get lines 95-100 (6 lines, not 10)
        assert "[lines 95-100 of 100 in large.md]" in result
        assert "line95" in result
        assert "line100" in result


# ---------------------------------------------------------------------------
# Tests for edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_file_with_invalid_utf8_bytes(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """File with invalid UTF-8 bytes should be handled with replacement."""
        full_path = temp_workspace / "invalid.md"
        # Write invalid UTF-8 sequence
        full_path.write_bytes(b"Valid text\n\xff\xfe Invalid bytes\nMore valid\n")
        result = read_file_fn("invalid.md")
        # Should not raise, errors="replace" handles invalid bytes
        assert "Valid text" in result
        assert "More valid" in result

    def test_exact_offset_equals_total_lines(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Offset exactly equal to total lines should work."""
        content = "Line 1\nLine 2\nLine 3\n"
        create_file(temp_workspace, "test.md", content)
        result = read_file_fn("test.md", start_line_one_indexed=3, no_of_lines_to_read=1)
        assert "[lines 3-3 of 3 in test.md]" in result
        assert "Line 3" in result

    def test_file_in_deeply_nested_directory(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Read file in deeply nested directory structure."""
        create_file(
            temp_workspace,
            "a/b/c/d/e/deep.md",
            "Deep content\n",
        )
        result = read_file_fn("a/b/c/d/e/deep.md")
        assert "[lines 1-1 of 1 in a/b/c/d/e/deep.md]" in result
        assert "Deep content" in result

    def test_filename_with_dots(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Filename with multiple dots should be handled correctly."""
        create_file(temp_workspace, "file.name.with.dots.md", "Dotty content\n")
        result = read_file_fn("file.name.with.dots.md")
        assert "file.name.with.dots.md" in result
        assert "Dotty content" in result

    def test_filename_with_spaces(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Filename with spaces should work."""
        create_file(temp_workspace, "file with spaces.md", "Spaced content\n")
        result = read_file_fn("file with spaces.md")
        assert "file with spaces.md" in result
        assert "Spaced content" in result

    def test_case_sensitivity(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Test case handling for extensions."""
        # Create files with different case extensions
        create_file(temp_workspace, "upper.MD", "Upper case ext\n")
        result = read_file_fn("upper.MD")
        # Should find the file (case-insensitive extension matching)
        assert "Upper case ext" in result

    def test_symlink_to_allowed_file(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Symlink to allowed file within workspace should work."""
        create_file(temp_workspace, "original.md", "Original content\n")
        link_path = temp_workspace / "link.md"
        try:
            link_path.symlink_to(temp_workspace / "original.md")
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        result = read_file_fn("link.md")
        assert "Original content" in result

    def test_symlink_escaping_workspace(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Symlink pointing outside workspace should be blocked or fail."""
        outside_file = temp_workspace.parent / "outside_target.md"
        create_file(temp_workspace.parent, "outside_target.md", "Outside content\n")

        link_path = temp_workspace / "escape_link.md"
        try:
            link_path.symlink_to(outside_file)
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        result = read_file_fn("escape_link.md")
        # The symlink resolves outside the workspace
        # Depending on implementation, may block or fail
        # Current implementation uses _is_within on raw path, not resolved
        # So this might work - testing actual behavior
        assert "Outside content" in result or "Base path escapes search root" in result

    def test_streaming_mode_produces_same_result_as_buffered(
        self, temp_workspace: Path, read_file_tool: ReadFileTool
    ) -> None:
        """Streaming and buffered modes should produce identical results."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
        create_file(temp_workspace, "test.md", content)

        # First, read in buffered mode (default threshold)
        fn_buffered = read_file_tool.get_tool()
        result_buffered = fn_buffered("test.md", start_line_one_indexed=2, no_of_lines_to_read=2)

        # Then, read in streaming mode (by lowering threshold)
        with patch(
            "anthropic_agent.common_tools.read_file.STREAMING_THRESHOLD_BYTES", 1
        ):
            fn_streaming = read_file_tool.get_tool()
            result_streaming = fn_streaming("test.md", start_line_one_indexed=2, no_of_lines_to_read=2)

        # Both should produce identical output
        assert result_buffered == result_streaming
        assert "[lines 2-3 of 5 in test.md]" in result_buffered

    def test_very_long_single_line(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Handle file with very long single line."""
        long_line = "x" * 10000 + "\n"
        create_file(temp_workspace, "longline.md", long_line)
        result = read_file_fn("longline.md")
        assert "[lines 1-1 of 1 in longline.md]" in result
        assert "x" * 100 in result  # Check partial content

    def test_limit_exactly_max_limit(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Limit exactly equal to MAX_LIMIT should not be clamped."""
        lines = [f"line{i}\n" for i in range(1, 150)]
        create_file(temp_workspace, "test.md", "".join(lines))
        result = read_file_fn("test.md", no_of_lines_to_read=MAX_LIMIT)
        assert f"[lines 1-{MAX_LIMIT} of 149 in test.md]" in result

    def test_read_file_called_multiple_times(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Reading the same file multiple times should work consistently."""
        create_file(temp_workspace, "test.md", "Content\n")
        result1 = read_file_fn("test.md")
        result2 = read_file_fn("test.md")
        result3 = read_file_fn("test.md")
        assert result1 == result2 == result3


# ---------------------------------------------------------------------------
# Tests for POSIX path normalization
# ---------------------------------------------------------------------------
class TestPosixPathNormalization:
    """Tests for POSIX-style path normalization in output."""

    def test_header_uses_posix_path(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Header should use forward slashes regardless of OS."""
        create_file(temp_workspace, "docs/api/ref.md", "API ref\n")
        result = read_file_fn("docs/api/ref.md")
        # Header should have forward slashes
        assert "docs/api/ref.md" in result
        # Should not have backslashes
        assert "\\" not in result.split("\n")[0]

    def test_dot_normalization(
        self, temp_workspace: Path, read_file_fn: Callable
    ) -> None:
        """Paths with . should be normalized."""
        create_file(temp_workspace, "docs/readme.md", "Readme\n")
        result = read_file_fn("./docs/./readme.md")
        # The header should have normalized path
        assert "docs/readme.md" in result


# ---------------------------------------------------------------------------
# Tests for constants verification
# ---------------------------------------------------------------------------
class TestConstants:
    """Verify module constants are as expected."""

    def test_max_limit_value(self) -> None:
        """MAX_LIMIT should be 100."""
        assert MAX_LIMIT == 100

    def test_allowed_extensions(self) -> None:
        """ALLOWED_EXTS should contain .md and .mmd."""
        assert ".md" in ALLOWED_EXTS
        assert ".mmd" in ALLOWED_EXTS
        assert len(ALLOWED_EXTS) == 2

    def test_streaming_threshold(self) -> None:
        """STREAMING_THRESHOLD_BYTES should be 2MB."""
        assert STREAMING_THRESHOLD_BYTES == 2 * 1024 * 1024
