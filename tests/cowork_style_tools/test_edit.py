"""Tests for the cowork-style Edit tool."""
import os
import pytest
from pathlib import Path

from anthropic_agent.cowork_style_tools.edit import create_edit_tool


@pytest.fixture
def tool_fn():
    return create_edit_tool()


@pytest.fixture
def tmp_workspace(tmp_path):
    return tmp_path


def create_file(workspace, name, content):
    p = workspace / name
    p.write_text(content)
    return p


class TestEditBasic:
    def test_single_replacement(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "test.py", "foo = 42\nbar = 99\n")
        result = tool_fn(file_path=str(f), old_string="foo = 42", new_string="foo = 100")
        assert "Successfully" in result
        assert "1 replacement" in result
        assert f.read_text() == "foo = 100\nbar = 99\n"

    def test_replace_all(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "test.txt", "hello world hello world hello\n")
        result = tool_fn(
            file_path=str(f),
            old_string="hello",
            new_string="hi",
            replace_all=True,
        )
        assert "Successfully" in result
        assert "3 replacement" in result
        assert f.read_text() == "hi world hi world hi\n"

    def test_multiline_replacement(self, tool_fn, tmp_workspace):
        content = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        f = create_file(tmp_workspace, "multi.py", content)
        result = tool_fn(
            file_path=str(f),
            old_string="def foo():\n    return 1",
            new_string="def foo():\n    return 42",
        )
        assert "Successfully" in result
        assert "return 42" in f.read_text()

    def test_preserves_rest_of_file(self, tool_fn, tmp_workspace):
        content = "aaa\nbbb\nccc\n"
        f = create_file(tmp_workspace, "preserve.txt", content)
        tool_fn(file_path=str(f), old_string="bbb", new_string="BBB")
        assert f.read_text() == "aaa\nBBB\nccc\n"


class TestEditErrors:
    def test_old_string_not_found(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "test.txt", "hello world\n")
        result = tool_fn(file_path=str(f), old_string="xyz", new_string="abc")
        assert "Error" in result
        assert "not found" in result

    def test_non_unique_old_string(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "dup.txt", "hello hello hello\n")
        result = tool_fn(file_path=str(f), old_string="hello", new_string="hi")
        assert "Error" in result
        assert "3 times" in result

    def test_identical_strings(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "same.txt", "test\n")
        result = tool_fn(file_path=str(f), old_string="test", new_string="test")
        assert "Error" in result
        assert "identical" in result

    def test_empty_old_string(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "empty.txt", "test\n")
        result = tool_fn(file_path=str(f), old_string="", new_string="new")
        assert "Error" in result
        assert "empty" in result.lower()

    def test_nonexistent_file(self, tool_fn):
        result = tool_fn(
            file_path="/nonexistent/file.txt",
            old_string="a",
            new_string="b",
        )
        assert "Error" in result
        assert "does not exist" in result

    def test_relative_path(self, tool_fn):
        result = tool_fn(
            file_path="relative/file.txt",
            old_string="a",
            new_string="b",
        )
        assert "Error" in result
        assert "absolute" in result.lower()

    def test_directory_path(self, tool_fn, tmp_workspace):
        result = tool_fn(
            file_path=str(tmp_workspace),
            old_string="a",
            new_string="b",
        )
        assert "Error" in result


class TestEditEdgeCases:
    def test_unicode_content(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "unicode.txt", "caf\u00e9 au lait\n")
        result = tool_fn(file_path=str(f), old_string="caf\u00e9", new_string="coffee")
        assert "Successfully" in result
        assert f.read_text() == "coffee au lait\n"

    def test_replace_with_empty_string(self, tool_fn, tmp_workspace):
        f = create_file(tmp_workspace, "remove.txt", "keep this remove this keep this too\n")
        result = tool_fn(file_path=str(f), old_string=" remove this", new_string="")
        assert "Successfully" in result
        assert f.read_text() == "keep this keep this too\n"

    def test_replace_all_single_occurrence(self, tool_fn, tmp_workspace):
        """replace_all=True should work fine with a single occurrence."""
        f = create_file(tmp_workspace, "single.txt", "one\n")
        result = tool_fn(
            file_path=str(f),
            old_string="one",
            new_string="1",
            replace_all=True,
        )
        assert "Successfully" in result
        assert "1 replacement" in result


class TestEditSchema:
    def test_has_tool_schema(self, tool_fn):
        assert hasattr(tool_fn, "__tool_schema__")
        schema = tool_fn.__tool_schema__
        assert schema["name"] == "edit_file"

    def test_schema_parameters(self, tool_fn):
        props = tool_fn.__tool_schema__["input_schema"]["properties"]
        assert "file_path" in props
        assert "old_string" in props
        assert "new_string" in props
        assert "replace_all" in props
