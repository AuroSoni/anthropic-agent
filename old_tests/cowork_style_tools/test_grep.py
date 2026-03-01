"""Tests for the cowork-style Grep tool."""
import os
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from anthropic_agent.cowork_style_tools.grep_tool import create_grep_tool


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with sample files."""
    (tmp_path / "src").mkdir()

    (tmp_path / "src" / "main.py").write_text(
        "import os\n"
        "def hello():\n"
        "    print('hello world')\n"
        "\n"
        "def goodbye():\n"
        "    print('goodbye world')\n"
    )
    (tmp_path / "src" / "utils.py").write_text(
        "def helper():\n"
        "    # TODO: implement\n"
        "    pass\n"
    )
    (tmp_path / "readme.md").write_text(
        "# Project\n"
        "This is a hello world project.\n"
    )
    return tmp_path


@pytest.fixture
def tool_fn():
    return create_grep_tool()


def _has_ripgrep():
    try:
        subprocess.run(["rg", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


needs_rg = pytest.mark.skipif(not _has_ripgrep(), reason="ripgrep not installed")


@needs_rg
class TestGrepFilesWithMatches:
    def test_basic_search(self, tool_fn, tmp_workspace):
        result = tool_fn(pattern="hello", path=str(tmp_workspace))
        assert "main.py" in result
        assert "readme.md" in result

    def test_no_matches(self, tool_fn, tmp_workspace):
        result = tool_fn(pattern="nonexistent_string_xyz", path=str(tmp_workspace))
        assert result == "No matches found."

    def test_glob_filter(self, tool_fn, tmp_workspace):
        result = tool_fn(
            pattern="hello",
            path=str(tmp_workspace),
            include_glob="*.py",
        )
        assert "main.py" in result
        assert "readme.md" not in result

    def test_type_filter(self, tool_fn, tmp_workspace):
        result = tool_fn(
            pattern="hello",
            path=str(tmp_workspace),
            file_type="py",
        )
        assert "main.py" in result
        assert "readme.md" not in result


@needs_rg
class TestGrepContentMode:
    def test_content_output(self, tool_fn, tmp_workspace):
        result = tool_fn(
            pattern="hello",
            path=str(tmp_workspace),
            output_mode="content",
        )
        assert "hello" in result

    def test_line_numbers(self, tool_fn, tmp_workspace):
        result = tool_fn(
            pattern="TODO",
            path=str(tmp_workspace),
            output_mode="content",
            line_numbers=True,
        )
        # ripgrep shows line numbers like "2:    # TODO: implement"
        assert "2:" in result or "TODO" in result

    def test_context_lines(self, tool_fn, tmp_workspace):
        result = tool_fn(
            pattern="TODO",
            path=str(tmp_workspace),
            output_mode="content",
            context=1,
        )
        # Should show lines around the match
        assert "helper" in result or "pass" in result


@needs_rg
class TestGrepCountMode:
    def test_count_output(self, tool_fn, tmp_workspace):
        result = tool_fn(
            pattern="hello",
            path=str(tmp_workspace),
            output_mode="count",
        )
        assert "main.py" in result


@needs_rg
class TestGrepOptions:
    def test_case_insensitive(self, tool_fn, tmp_workspace):
        result = tool_fn(
            pattern="HELLO",
            path=str(tmp_workspace),
            case_insensitive=True,
        )
        assert "main.py" in result

    def test_case_sensitive_no_match(self, tool_fn, tmp_workspace):
        result = tool_fn(
            pattern="HELLO",
            path=str(tmp_workspace),
            case_insensitive=False,
        )
        assert result == "No matches found."

    def test_head_limit(self, tool_fn, tmp_workspace):
        result = tool_fn(
            pattern="hello|goodbye|TODO|Project",
            path=str(tmp_workspace),
            head_limit=2,
        )
        lines = [l for l in result.strip().split("\n") if l]
        assert len(lines) <= 2

    def test_offset(self, tool_fn, tmp_workspace):
        # Get full results first
        full = tool_fn(
            pattern="def",
            path=str(tmp_workspace),
            output_mode="content",
        )
        full_lines = full.strip().split("\n")

        # Now with offset=1
        offset_result = tool_fn(
            pattern="def",
            path=str(tmp_workspace),
            output_mode="content",
            offset=1,
        )
        offset_lines = offset_result.strip().split("\n")
        assert len(offset_lines) == len(full_lines) - 1


class TestGrepErrorHandling:
    def test_empty_pattern(self, tool_fn):
        result = tool_fn(pattern="")
        assert "Error" in result

    def test_invalid_output_mode(self, tool_fn, tmp_workspace):
        result = tool_fn(
            pattern="test",
            path=str(tmp_workspace),
            output_mode="invalid",
        )
        assert "Error" in result
        assert "Invalid output_mode" in result

    def test_ripgrep_not_installed(self, tool_fn, tmp_workspace):
        with patch("anthropic_agent.cowork_style_tools.grep_tool.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = tool_fn(pattern="test", path=str(tmp_workspace))
            assert "ripgrep" in result.lower() or "rg" in result


class TestGrepSchema:
    def test_has_tool_schema(self, tool_fn):
        assert hasattr(tool_fn, "__tool_schema__")
        schema = tool_fn.__tool_schema__
        assert schema["name"] == "grep_search"

    def test_schema_parameters(self, tool_fn):
        props = tool_fn.__tool_schema__["input_schema"]["properties"]
        assert "pattern" in props
        assert "output_mode" in props
        assert "include_glob" in props
        assert "case_insensitive" in props
