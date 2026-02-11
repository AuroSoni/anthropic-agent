"""Tests for the cowork-style Glob tool."""
import os
import time
import pytest
from pathlib import Path

from anthropic_agent.cowork_style_tools.glob_tool import create_glob_tool


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with sample files."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "utils").mkdir()
    (tmp_path / "docs").mkdir()

    # Create files with staggered mtime
    files = [
        ("src/main.py", "print('main')"),
        ("src/utils/helpers.py", "def helper(): pass"),
        ("src/app.ts", "console.log('app')"),
        ("docs/readme.md", "# Readme"),
        ("config.json", '{"key": "val"}'),
    ]
    for rel_path, content in files:
        p = tmp_path / rel_path
        p.write_text(content)
        # Small sleep to get distinct mtime ordering
        time.sleep(0.05)

    return tmp_path


@pytest.fixture
def tool_fn():
    return create_glob_tool()


class TestGlobBasic:
    def test_find_python_files(self, tool_fn, tmp_workspace):
        result = tool_fn(pattern="**/*.py", path=str(tmp_workspace))
        assert "main.py" in result
        assert "helpers.py" in result

    def test_find_all_files(self, tool_fn, tmp_workspace):
        result = tool_fn(pattern="**/*.*", path=str(tmp_workspace))
        assert "main.py" in result
        assert "app.ts" in result
        assert "readme.md" in result
        assert "config.json" in result

    def test_no_extension_filtering(self, tool_fn, tmp_workspace):
        """Cowork glob has no extension filtering — all types returned."""
        result = tool_fn(pattern="**/*.*", path=str(tmp_workspace))
        lines = [l for l in result.strip().split("\n") if l]
        assert len(lines) == 5  # all 5 files

    def test_sorted_by_mtime_newest_first(self, tool_fn, tmp_workspace):
        result = tool_fn(pattern="**/*.*", path=str(tmp_workspace))
        lines = result.strip().split("\n")
        # Last created file (config.json) should be first
        assert lines[0].endswith("config.json")

    def test_returns_absolute_paths(self, tool_fn, tmp_workspace):
        result = tool_fn(pattern="**/*.py", path=str(tmp_workspace))
        for line in result.strip().split("\n"):
            assert os.path.isabs(line.strip())

    def test_specific_directory(self, tool_fn, tmp_workspace):
        result = tool_fn(pattern="*.py", path=str(tmp_workspace / "src"))
        assert "main.py" in result
        assert "helpers.py" not in result  # not in src/ directly

    def test_defaults_to_cwd(self, tool_fn, tmp_workspace):
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_workspace))
            result = tool_fn(pattern="**/*.py")
            assert "main.py" in result
        finally:
            os.chdir(original_cwd)


class TestGlobEdgeCases:
    def test_no_matches(self, tool_fn, tmp_workspace):
        result = tool_fn(pattern="**/*.xyz", path=str(tmp_workspace))
        assert result == "No matches found."

    def test_nonexistent_path(self, tool_fn):
        result = tool_fn(pattern="**/*", path="/nonexistent/path/abc123")
        assert "Error" in result
        assert "does not exist" in result

    def test_path_is_file(self, tool_fn, tmp_workspace):
        result = tool_fn(pattern="*", path=str(tmp_workspace / "config.json"))
        assert "Error" in result
        assert "not a directory" in result

    def test_empty_directory(self, tool_fn, tmp_path):
        result = tool_fn(pattern="**/*", path=str(tmp_path))
        assert result == "No matches found."


class TestGlobSchema:
    def test_has_tool_schema(self, tool_fn):
        assert hasattr(tool_fn, "__tool_schema__")
        schema = tool_fn.__tool_schema__
        assert schema["name"] == "glob_search"
        assert "input_schema" in schema

    def test_schema_parameters(self, tool_fn):
        props = tool_fn.__tool_schema__["input_schema"]["properties"]
        assert "pattern" in props
        assert "path" in props
