"""Tests for the cowork-style Write tool."""
import os
import pytest
from pathlib import Path

from anthropic_agent.cowork_style_tools.write import create_write_tool


@pytest.fixture
def tool_fn():
    return create_write_tool()


@pytest.fixture
def tmp_workspace(tmp_path):
    return tmp_path


class TestWriteBasic:
    def test_create_new_file(self, tool_fn, tmp_workspace):
        target = tmp_workspace / "new_file.txt"
        result = tool_fn(file_path=str(target), content="hello\nworld\n")
        assert "Successfully" in result
        assert target.read_text() == "hello\nworld\n"

    def test_overwrite_existing_file(self, tool_fn, tmp_workspace):
        target = tmp_workspace / "existing.txt"
        target.write_text("old content")
        result = tool_fn(file_path=str(target), content="new content")
        assert "Successfully" in result
        assert target.read_text() == "new content"

    def test_creates_parent_directories(self, tool_fn, tmp_workspace):
        target = tmp_workspace / "deep" / "nested" / "dir" / "file.py"
        result = tool_fn(file_path=str(target), content="print('hi')\n")
        assert "Successfully" in result
        assert target.exists()
        assert target.read_text() == "print('hi')\n"

    def test_reports_line_count(self, tool_fn, tmp_workspace):
        target = tmp_workspace / "counted.txt"
        result = tool_fn(file_path=str(target), content="a\nb\nc\n")
        assert "3 lines" in result

    def test_empty_content(self, tool_fn, tmp_workspace):
        target = tmp_workspace / "empty.txt"
        result = tool_fn(file_path=str(target), content="")
        assert "Successfully" in result
        assert target.read_text() == ""

    def test_content_without_trailing_newline(self, tool_fn, tmp_workspace):
        target = tmp_workspace / "no_newline.txt"
        result = tool_fn(file_path=str(target), content="line1\nline2")
        assert "2 lines" in result
        assert target.read_text() == "line1\nline2"

    def test_unicode_content(self, tool_fn, tmp_workspace):
        target = tmp_workspace / "unicode.txt"
        content = "Hello \u4e16\u754c\nCaf\u00e9\n\U0001f600\n"
        result = tool_fn(file_path=str(target), content=content)
        assert "Successfully" in result
        assert target.read_text(encoding="utf-8") == content


class TestWriteErrors:
    def test_relative_path(self, tool_fn):
        result = tool_fn(file_path="relative/path.txt", content="test")
        assert "Error" in result
        assert "absolute" in result.lower()

    def test_permission_error(self, tool_fn, tmp_workspace):
        # Create a read-only directory
        readonly = tmp_workspace / "readonly"
        readonly.mkdir()
        readonly.chmod(0o444)
        try:
            target = readonly / "file.txt"
            result = tool_fn(file_path=str(target), content="test")
            assert "Error" in result
        finally:
            readonly.chmod(0o755)


class TestWriteAtomicity:
    def test_file_content_is_complete(self, tool_fn, tmp_workspace):
        """File should contain complete content after write, not partial."""
        target = tmp_workspace / "atomic.txt"
        content = "line\n" * 1000
        tool_fn(file_path=str(target), content=content)
        assert target.read_text() == content


class TestWriteSchema:
    def test_has_tool_schema(self, tool_fn):
        assert hasattr(tool_fn, "__tool_schema__")
        schema = tool_fn.__tool_schema__
        assert schema["name"] == "write_file"

    def test_schema_parameters(self, tool_fn):
        props = tool_fn.__tool_schema__["input_schema"]["properties"]
        assert "file_path" in props
        assert "content" in props
