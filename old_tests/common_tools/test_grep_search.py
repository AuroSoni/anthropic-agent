"""Tests for grep_search tool functionality.

Tests cover:
- Initialization and internal utilities
- Command construction (subprocess arguments)
- Output parsing, formatting, and filtering
- Truncation logic
- Error handling and edge cases
"""
import json
import tempfile
from pathlib import Path
from typing import Callable, Generator, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from anthropic_agent.common_tools.grep_search import (
    ALLOWED_EXTS,
    CONTEXT_LINES,
    MAX_MATCH_LINES,
    GrepSearchTool,
    _byte_ranges_to_char_ranges,
    _highlight_line,
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
def grep_tool(temp_workspace: Path) -> GrepSearchTool:
    """Create a GrepSearchTool instance with the temp workspace."""
    return GrepSearchTool(base_path=temp_workspace)


@pytest.fixture
def search_fn(grep_tool: GrepSearchTool) -> Callable:
    """Get the grep_search function from the tool."""
    return grep_tool.get_tool()


# ---------------------------------------------------------------------------
# Helper functions for generating mock ripgrep JSON output
# ---------------------------------------------------------------------------
def make_rg_match(
    path: str,
    line_number: int,
    text: str,
    submatches: List[Tuple[int, int]],
) -> str:
    """Generate a ripgrep JSON match event.
    
    Args:
        path: The file path.
        line_number: The line number of the match.
        text: The line text (without trailing newline).
        submatches: List of (start, end) byte offset tuples for matched ranges.
    
    Returns:
        JSON string for a match event.
    """
    submatch_data = [
        {"match": {"text": text[s:e]}, "start": s, "end": e}
        for s, e in submatches
    ]
    event = {
        "type": "match",
        "data": {
            "path": {"text": path},
            "line_number": line_number,
            "lines": {"text": text + "\n"},
            "submatches": submatch_data,
        },
    }
    return json.dumps(event)


def make_rg_context(path: str, line_number: int, text: str) -> str:
    """Generate a ripgrep JSON context event.
    
    Args:
        path: The file path.
        line_number: The line number.
        text: The line text (without trailing newline).
    
    Returns:
        JSON string for a context event.
    """
    event = {
        "type": "context",
        "data": {
            "path": {"text": path},
            "line_number": line_number,
            "lines": {"text": text + "\n"},
        },
    }
    return json.dumps(event)


def make_rg_begin(path: str) -> str:
    """Generate a ripgrep JSON begin event."""
    event = {"type": "begin", "data": {"path": {"text": path}}}
    return json.dumps(event)


def make_rg_end(path: str) -> str:
    """Generate a ripgrep JSON end event."""
    event = {"type": "end", "data": {"path": {"text": path}}}
    return json.dumps(event)


def make_mock_proc(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> MagicMock:
    """Create a mock subprocess result."""
    mock = MagicMock()
    mock.stdout = stdout
    mock.stderr = stderr
    mock.returncode = returncode
    return mock


# ---------------------------------------------------------------------------
# Tests for utility functions
# ---------------------------------------------------------------------------
class TestHighlightLine:
    def test_single_match(self) -> None:
        text = "hello world"
        ranges = [(0, 5)]  # "hello"
        result = _highlight_line(text, ranges)
        assert result == "<match>hello</match> world"

    def test_multiple_matches(self) -> None:
        text = "hello world hello"
        ranges = [(0, 5), (12, 17)]  # both "hello"
        result = _highlight_line(text, ranges)
        assert result == "<match>hello</match> world <match>hello</match>"

    def test_adjacent_matches(self) -> None:
        text = "foobar"
        ranges = [(0, 3), (3, 6)]  # "foo", "bar"
        result = _highlight_line(text, ranges)
        assert result == "<match>foo</match><match>bar</match>"

    def test_empty_ranges(self) -> None:
        text = "hello world"
        ranges: List[Tuple[int, int]] = []
        result = _highlight_line(text, ranges)
        assert result == "hello world"

    def test_out_of_bounds_ignored(self) -> None:
        text = "hello"
        ranges = [(10, 20)]  # out of bounds
        result = _highlight_line(text, ranges)
        # Should not crash, just return original
        assert result == "hello"


class TestByteRangesToCharRanges:
    def test_ascii_text(self) -> None:
        text = "hello"
        byte_ranges = [(0, 5)]
        result = _byte_ranges_to_char_ranges(text, byte_ranges)
        assert result == [(0, 5)]

    def test_utf8_multibyte_chars(self) -> None:
        # "café" - 'é' is 2 bytes in UTF-8
        text = "café"
        # byte offsets: c=0, a=1, f=2, é=3-4
        byte_ranges = [(3, 5)]  # 'é' in bytes
        result = _byte_ranges_to_char_ranges(text, byte_ranges)
        assert result == [(3, 4)]  # char index 3-4

    def test_emoji(self) -> None:
        # "a😀b" - 😀 is 4 bytes in UTF-8
        text = "a😀b"
        # byte offsets: a=0, 😀=1-4, b=5
        byte_ranges = [(1, 5)]  # the emoji in bytes
        result = _byte_ranges_to_char_ranges(text, byte_ranges)
        assert result == [(1, 2)]  # char index 1-2

    def test_empty_byte_ranges(self) -> None:
        text = "hello"
        result = _byte_ranges_to_char_ranges(text, [])
        assert result == []

    def test_chinese_characters(self) -> None:
        # "你好" - each char is 3 bytes
        text = "你好"
        # byte offsets: 你=0-2, 好=3-5
        byte_ranges = [(0, 3)]  # first char
        result = _byte_ranges_to_char_ranges(text, byte_ranges)
        assert result == [(0, 1)]


# ---------------------------------------------------------------------------
# Tests for GrepSearchTool initialization
# ---------------------------------------------------------------------------
class TestGrepSearchToolInit:
    def test_initializes_with_string_path(self, temp_workspace: Path) -> None:
        tool = GrepSearchTool(base_path=str(temp_workspace))
        assert tool.search_root == temp_workspace.resolve()

    def test_initializes_with_path_object(self, temp_workspace: Path) -> None:
        tool = GrepSearchTool(base_path=temp_workspace)
        assert tool.search_root == temp_workspace.resolve()

    def test_search_root_is_absolute(self, temp_workspace: Path) -> None:
        tool = GrepSearchTool(base_path=temp_workspace)
        assert tool.search_root.is_absolute()


class TestRelPosix:
    def test_relative_path_inside_root(self, temp_workspace: Path) -> None:
        tool = GrepSearchTool(base_path=temp_workspace)
        # Create a path inside the search root
        test_path = str(temp_workspace / "subdir" / "file.md")
        result = tool._rel_posix(test_path)
        assert result == "subdir/file.md"

    def test_path_at_root(self, temp_workspace: Path) -> None:
        tool = GrepSearchTool(base_path=temp_workspace)
        test_path = str(temp_workspace / "file.md")
        result = tool._rel_posix(test_path)
        assert result == "file.md"

    def test_already_relative_path(self, temp_workspace: Path) -> None:
        tool = GrepSearchTool(base_path=temp_workspace)
        # ripgrep returns paths like "./subdir/file.md" when run with cwd
        result = tool._rel_posix("./subdir/file.md")
        assert result == "subdir/file.md"

    def test_posix_style_output(self, temp_workspace: Path) -> None:
        tool = GrepSearchTool(base_path=temp_workspace)
        # Even on Windows, output should use forward slashes
        result = tool._rel_posix("subdir/nested/file.md")
        assert "\\" not in result
        assert "/" in result


# ---------------------------------------------------------------------------
# Tests for command construction
# ---------------------------------------------------------------------------
class TestCommandConstruction:
    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_default_command_args(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify default command includes required flags."""
        mock_run.return_value = make_mock_proc(returncode=1)
        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()

        search_fn("test_pattern")

        args = mock_run.call_args
        cmd = args[0][0]

        assert "rg" in cmd
        assert "--json" in cmd
        assert "-n" in cmd
        assert "-C" in cmd
        assert str(CONTEXT_LINES) in cmd
        assert "--ignore-case" in cmd
        # Default globs for allowed extensions
        assert "--glob" in cmd
        assert "**/*.md" in cmd
        assert "**/*.mmd" in cmd

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_case_sensitive_flag(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify case_sensitive=True uses --case-sensitive."""
        mock_run.return_value = make_mock_proc(returncode=1)
        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()

        search_fn("pattern", case_sensitive=True)

        cmd = mock_run.call_args[0][0]
        assert "--case-sensitive" in cmd
        assert "--ignore-case" not in cmd

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_case_insensitive_flag(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify case_sensitive=False (default) uses --ignore-case."""
        mock_run.return_value = make_mock_proc(returncode=1)
        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()

        search_fn("pattern", case_sensitive=False)

        cmd = mock_run.call_args[0][0]
        assert "--ignore-case" in cmd
        assert "--case-sensitive" not in cmd

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_include_pattern_adds_glob(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify include_pattern adds --glob and --no-ignore."""
        mock_run.return_value = make_mock_proc(returncode=1)
        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()

        search_fn("pattern", include_pattern="**/*.py")

        cmd = mock_run.call_args[0][0]
        assert "--no-ignore" in cmd
        assert "--glob" in cmd
        # Find the glob value
        glob_idx = cmd.index("--glob")
        assert cmd[glob_idx + 1] == "**/*.py"

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_exclude_pattern_adds_negated_glob(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify exclude_pattern adds negated --glob."""
        mock_run.return_value = make_mock_proc(returncode=1)
        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()

        search_fn("pattern", exclude_pattern="**/test_*")

        cmd = mock_run.call_args[0][0]
        # Find negated glob
        glob_indices = [i for i, x in enumerate(cmd) if x == "--glob"]
        negated_globs = [cmd[i + 1] for i in glob_indices if cmd[i + 1].startswith("!")]
        assert any("test_" in g for g in negated_globs)

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_gitignore_used_when_no_include_pattern(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify .gitignore is used via --ignore-file when include_pattern is None."""
        # Create a .gitignore in the workspace
        gitignore = temp_workspace / ".gitignore"
        gitignore.write_text("node_modules/\n")

        mock_run.return_value = make_mock_proc(returncode=1)
        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()

        search_fn("pattern")

        cmd = mock_run.call_args[0][0]
        assert "--ignore-file" in cmd
        ignore_idx = cmd.index("--ignore-file")
        assert str(gitignore) in cmd[ignore_idx + 1]

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_gitignore_not_used_with_include_pattern(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify .gitignore is NOT used when include_pattern is provided."""
        gitignore = temp_workspace / ".gitignore"
        gitignore.write_text("node_modules/\n")

        mock_run.return_value = make_mock_proc(returncode=1)
        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()

        search_fn("pattern", include_pattern="**/*.txt")

        cmd = mock_run.call_args[0][0]
        assert "--ignore-file" not in cmd

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_cwd_set_to_search_root(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify subprocess runs with cwd set to search_root."""
        mock_run.return_value = make_mock_proc(returncode=1)
        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()

        search_fn("pattern")

        kwargs = mock_run.call_args[1]
        # Use resolve() to handle macOS /var -> /private/var symlink
        assert kwargs["cwd"] == str(temp_workspace.resolve())


# ---------------------------------------------------------------------------
# Tests for output parsing and formatting
# ---------------------------------------------------------------------------
class TestOutputParsing:
    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_basic_match_formatting(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify basic match output format."""
        stdout = make_rg_match("./test.md", 5, "hello world", [(0, 5)])
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("hello")

        assert "test.md:" in result
        assert "5: <match>hello</match> world" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_context_line_formatting(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify context lines use NNN- prefix."""
        lines = [
            make_rg_context("./test.md", 4, "before line"),
            make_rg_match("./test.md", 5, "match line", [(0, 5)]),
            make_rg_context("./test.md", 6, "after line"),
        ]
        stdout = "\n".join(lines)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("match")

        assert "4- before line" in result
        assert "5: <match>match</match> line" in result
        assert "6- after line" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_multiple_submatches_highlighted(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify multiple matches on same line are highlighted."""
        # "test foo test" with "test" matched at positions 0-4 and 9-13
        stdout = make_rg_match("./file.md", 1, "test foo test", [(0, 4), (9, 13)])
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("test")

        assert "<match>test</match> foo <match>test</match>" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_multiple_files_grouped(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify output is grouped by file."""
        lines = [
            make_rg_match("./first.md", 1, "match one", [(0, 5)]),
            make_rg_match("./second.md", 1, "match two", [(0, 5)]),
        ]
        stdout = "\n".join(lines)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("match")

        assert "first.md:" in result
        assert "second.md:" in result
        # first.md should appear before second.md (order of encounter)
        assert result.index("first.md:") < result.index("second.md:")

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_extension_filtering_allows_md(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify .md files are included in output."""
        stdout = make_rg_match("./docs.md", 1, "content", [(0, 7)])
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("content")

        assert "docs.md:" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_extension_filtering_allows_mmd(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify .mmd files are included in output."""
        stdout = make_rg_match("./diagram.mmd", 1, "content", [(0, 7)])
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("content")

        assert "diagram.mmd:" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_extension_filtering_rejects_txt(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify .txt files are filtered out from results."""
        # Even if rg returns .txt matches, they should be filtered
        lines = [
            make_rg_match("./allowed.md", 1, "match", [(0, 5)]),
            make_rg_match("./disallowed.txt", 1, "match", [(0, 5)]),
        ]
        stdout = "\n".join(lines)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("match")

        assert "allowed.md:" in result
        assert "disallowed.txt" not in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_extension_filtering_rejects_py(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify .py files are filtered out from results."""
        lines = [
            make_rg_match("./allowed.md", 1, "match", [(0, 5)]),
            make_rg_match("./disallowed.py", 1, "match", [(0, 5)]),
        ]
        stdout = "\n".join(lines)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("match")

        assert "allowed.md:" in result
        assert "disallowed.py" not in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_utf8_content_with_highlighting(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify UTF-8 content is highlighted correctly with byte offsets."""
        # Text: "你好世界" (hello world in Chinese)
        # Each char is 3 bytes. Match 你 (bytes 0-3)
        text = "你好世界"
        # The submatch returns byte offsets
        event = {
            "type": "match",
            "data": {
                "path": {"text": "./chinese.md"},
                "line_number": 1,
                "lines": {"text": text + "\n"},
                "submatches": [{"match": {"text": "你"}, "start": 0, "end": 3}],
            },
        }
        stdout = json.dumps(event)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("你")

        assert "<match>你</match>好世界" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_crlf_line_endings_handled(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify CRLF line endings are stripped."""
        event = {
            "type": "match",
            "data": {
                "path": {"text": "./test.md"},
                "line_number": 1,
                "lines": {"text": "hello world\r\n"},
                "submatches": [{"match": {"text": "hello"}, "start": 0, "end": 5}],
            },
        }
        stdout = json.dumps(event)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("hello")

        # Should not have trailing \r
        assert "world\r" not in result
        assert "<match>hello</match> world" in result


# ---------------------------------------------------------------------------
# Tests for truncation logic
# ---------------------------------------------------------------------------
class TestTruncation:
    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_max_matches_limit(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify only MAX_MATCH_LINES matches are shown."""
        # Create more than MAX_MATCH_LINES matches
        # Use consistent text where "match" is always at bytes 0-5
        num_matches = MAX_MATCH_LINES + 10
        lines = [
            make_rg_match("./file.md", i, "match in this line", [(0, 5)])
            for i in range(1, num_matches + 1)
        ]
        stdout = "\n".join(lines)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("match")

        # Count actual match lines (lines with ": <match>")
        match_count = result.count(": <match>")
        assert match_count == MAX_MATCH_LINES

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_omission_message_appended(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify omission message shows correct count."""
        num_matches = MAX_MATCH_LINES + 5
        lines = [
            make_rg_match("./file.md", i, "match in this line", [(0, 5)])
            for i in range(1, num_matches + 1)
        ]
        stdout = "\n".join(lines)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("match")

        assert "[... 5 more matches omitted]" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_context_draining_after_limit(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify context lines after the last shown match are included."""
        # Create exactly MAX_MATCH_LINES matches plus context
        lines = []
        for i in range(1, MAX_MATCH_LINES + 3):
            if i <= MAX_MATCH_LINES:
                lines.append(make_rg_match("./file.md", i * 5, f"match {i}", [(0, 5)]))
            else:
                # These are extra matches that should be counted but not shown
                lines.append(make_rg_match("./file.md", i * 5, f"extra {i}", [(0, 5)]))

        # Add context after the MAX_MATCH_LINES-th match
        last_match_line = MAX_MATCH_LINES * 5
        lines.insert(
            MAX_MATCH_LINES,
            make_rg_context("./file.md", last_match_line + 1, "context after"),
        )
        lines.insert(
            MAX_MATCH_LINES + 1,
            make_rg_context("./file.md", last_match_line + 2, "context after 2"),
        )

        stdout = "\n".join(lines)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("match")

        # The last printed match should have its context
        assert "context after" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_no_omission_when_under_limit(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify no omission message when matches are under limit."""
        lines = [
            make_rg_match("./file.md", i, "match in this line", [(0, 5)])
            for i in range(1, 6)  # 5 matches
        ]
        stdout = "\n".join(lines)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("match")

        assert "[..." not in result
        assert "omitted" not in result


# ---------------------------------------------------------------------------
# Tests for error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_empty_query_returns_error(self, temp_workspace: Path) -> None:
        """Verify empty query returns error message."""
        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()

        result = search_fn("")
        assert "cannot be empty" in result

    def test_none_query_returns_error(self, temp_workspace: Path) -> None:
        """Verify None query returns error message."""
        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()

        result = search_fn(None)
        assert "cannot be empty" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_missing_rg_returns_friendly_error(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify FileNotFoundError returns friendly message."""
        mock_run.side_effect = FileNotFoundError("rg not found")

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("pattern")

        assert "ripgrep (rg) is not installed or not found in PATH" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_invalid_regex_returns_stderr(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify invalid regex returns stderr from ripgrep."""
        error_msg = "error: regex parse error"
        mock_run.return_value = make_mock_proc(
            stdout="", stderr=error_msg, returncode=2
        )

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("[invalid(regex")

        assert error_msg in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_no_matches_with_returncode_1(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify returncode 1 (no matches) returns proper message."""
        mock_run.return_value = make_mock_proc(stdout="", returncode=1)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("nonexistent")

        assert result.startswith("No matches found")

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_no_matches_with_returncode_0_empty_stdout(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify empty stdout with returncode 0 returns no matches."""
        mock_run.return_value = make_mock_proc(stdout="", returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("pattern")

        assert result.startswith("No matches found")

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_generic_exception_returns_message(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify generic exceptions are returned as strings."""
        mock_run.side_effect = Exception("Something went wrong")

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("pattern")

        assert "Something went wrong" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_unicode_filename_handling(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify unicode filenames are handled correctly."""
        stdout = make_rg_match("./文档.md", 1, "content", [(0, 7)])
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("content")

        assert "文档.md:" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_malformed_json_line_skipped(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify malformed JSON lines are skipped gracefully."""
        lines = [
            "not valid json",
            make_rg_match("./file.md", 1, "valid match", [(0, 5)]),
            "{incomplete json",
        ]
        stdout = "\n".join(lines)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("valid")

        # Should not crash and should include valid match
        assert "file.md:" in result
        assert "<match>valid</match>" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_rg_error_without_stderr(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify rg error without stderr returns generic message."""
        mock_run.return_value = make_mock_proc(stdout="", stderr="", returncode=2)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        result = search_fn("pattern")

        assert result.startswith("ripgrep failed")


# ---------------------------------------------------------------------------
# Tests for configurable limits
# ---------------------------------------------------------------------------
class TestConfigurableLimits:
    """Tests for instance-configurable limits."""

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_custom_max_match_lines_truncates(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify custom max_match_lines truncates at the specified limit."""
        # Create 10 matches
        lines = [
            make_rg_match("./file.md", i, f"match line {i}", [(0, 5)])
            for i in range(1, 11)
        ]
        stdout = "\n".join(lines)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        # Create tool with custom limit of 3
        tool = GrepSearchTool(base_path=temp_workspace, max_match_lines=3)
        search_fn = tool.get_tool()
        result = search_fn("match")

        # Should show exactly 3 matches
        match_count = result.count(": <match>")
        assert match_count == 3

        # Should have omission message for remaining 7
        assert "[... 7 more matches omitted]" in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_custom_context_lines_in_command(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify custom context_lines is passed to ripgrep command."""
        mock_run.return_value = make_mock_proc(returncode=1)

        tool = GrepSearchTool(base_path=temp_workspace, context_lines=5)
        search_fn = tool.get_tool()
        search_fn("pattern")

        cmd = mock_run.call_args[0][0]
        # Find -C flag and verify its value
        c_idx = cmd.index("-C")
        assert cmd[c_idx + 1] == "5"

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_zero_context_lines(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify context_lines=0 shows no context."""
        mock_run.return_value = make_mock_proc(returncode=1)

        tool = GrepSearchTool(base_path=temp_workspace, context_lines=0)
        search_fn = tool.get_tool()
        search_fn("pattern")

        cmd = mock_run.call_args[0][0]
        c_idx = cmd.index("-C")
        assert cmd[c_idx + 1] == "0"

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_default_limits(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify default limits use module constants."""
        mock_run.return_value = make_mock_proc(returncode=1)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        search_fn("pattern")

        cmd = mock_run.call_args[0][0]
        c_idx = cmd.index("-C")
        assert cmd[c_idx + 1] == str(CONTEXT_LINES)


# ---------------------------------------------------------------------------
# Tests for custom allowed extensions
# ---------------------------------------------------------------------------
class TestCustomExtensions:
    """Tests for custom allowed_extensions configuration."""

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_custom_extension_py_in_globs(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify custom allowed_extensions adds correct globs for .py."""
        mock_run.return_value = make_mock_proc(returncode=1)

        tool = GrepSearchTool(
            base_path=temp_workspace,
            allowed_extensions={".py"},
        )
        search_fn = tool.get_tool()
        search_fn("pattern")

        cmd = mock_run.call_args[0][0]
        # Should have glob for .py
        assert "**/*.py" in cmd
        # Should NOT have default .md glob
        assert "**/*.md" not in cmd

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_custom_extension_multiple_globs(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify multiple custom extensions create multiple globs."""
        mock_run.return_value = make_mock_proc(returncode=1)

        tool = GrepSearchTool(
            base_path=temp_workspace,
            allowed_extensions={".py", ".js", ".ts"},
        )
        search_fn = tool.get_tool()
        search_fn("pattern")

        cmd = mock_run.call_args[0][0]
        assert "**/*.py" in cmd
        assert "**/*.js" in cmd
        assert "**/*.ts" in cmd

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_custom_extension_filters_results(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify custom extensions filter results from ripgrep output."""
        # Simulate ripgrep returning matches from multiple file types
        lines = [
            make_rg_match("./allowed.py", 1, "match", [(0, 5)]),
            make_rg_match("./disallowed.txt", 1, "match", [(0, 5)]),
        ]
        stdout = "\n".join(lines)
        mock_run.return_value = make_mock_proc(stdout=stdout, returncode=0)

        tool = GrepSearchTool(
            base_path=temp_workspace,
            allowed_extensions={".py"},
        )
        search_fn = tool.get_tool()
        result = search_fn("match")

        assert "allowed.py:" in result
        assert "disallowed.txt" not in result

    @patch("anthropic_agent.common_tools.grep_search.subprocess.run")
    def test_default_extensions_md_mmd(
        self, mock_run: MagicMock, temp_workspace: Path
    ) -> None:
        """Verify default extensions are .md and .mmd."""
        mock_run.return_value = make_mock_proc(returncode=1)

        tool = GrepSearchTool(base_path=temp_workspace)
        search_fn = tool.get_tool()
        search_fn("pattern")

        cmd = mock_run.call_args[0][0]
        assert "**/*.md" in cmd
        assert "**/*.mmd" in cmd
