"""Tests for list_dir tool functionality.

Tests cover:
- Basic structure and ASCII tree rendering
- Extension filtering (.md, .mmd only)
- Directory pruning (exclude dirs without allowed files)
- Recursion depth limits
- Large directory handling
- Ignore pattern logic
- Symlink handling
- Error handling and edge cases
"""
import tempfile
from pathlib import Path
from typing import Callable, Generator
from unittest.mock import patch

import pytest

from anthropic_agent.common_tools.list_dir import (
    ALLOWED_EXTS,
    INDENT_PER_LEVEL,
    LARGE_DIR_ENTRY_THRESHOLD,
    LARGE_DIR_SHOW_DIRS,
    LARGE_DIR_SHOW_FILES,
    MAX_DEPTH,
    SUMMARY_MAX_EXT_GROUPS,
    ListDirTool,
    ext_label,
    format_bracket_line,
    format_dir_line,
    format_ext_groups,
    format_file_line,
    summarize_extension_groups,
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
def list_dir_tool(temp_workspace: Path) -> ListDirTool:
    """Create a ListDirTool instance with the temp workspace."""
    return ListDirTool(base_path=temp_workspace)


@pytest.fixture
def list_dir_fn(list_dir_tool: ListDirTool) -> Callable:
    """Get the list_dir function from the tool."""
    return list_dir_tool.get_tool()


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
class TestFormatDirLine:
    def test_root_no_bullet(self) -> None:
        """Root directory (depth 0) should have no bullet."""
        result = format_dir_line("mydir", depth=0)
        assert result == "mydir/"
        assert "- " not in result

    def test_nested_has_bullet(self) -> None:
        """Nested directories should have bullet prefix."""
        result = format_dir_line("subdir", depth=1)
        assert result == "   - subdir/"

    def test_indentation_increases_with_depth(self) -> None:
        """Indentation should increase by INDENT_PER_LEVEL per depth."""
        result_d2 = format_dir_line("dir", depth=2)
        result_d3 = format_dir_line("dir", depth=3)
        assert result_d2 == "      - dir/"
        assert result_d3 == "         - dir/"


class TestFormatFileLine:
    def test_always_has_bullet(self) -> None:
        """File lines always have a bullet."""
        result = format_file_line("file.md", depth=1)
        assert result == "   - file.md"

    def test_indentation(self) -> None:
        """Indentation should match depth."""
        result = format_file_line("test.md", depth=2)
        assert result == "      - test.md"


class TestFormatBracketLine:
    def test_indentation_one_level_deeper(self) -> None:
        """Bracket lines are indented one level deeper than their parent."""
        result = format_bracket_line("summary", depth=1)
        # depth+1 = 2, so 6 spaces
        assert result == "      [summary]"

    def test_brackets_present(self) -> None:
        """Result should be wrapped in square brackets."""
        result = format_bracket_line("test text", depth=0)
        assert result.startswith("   [")
        assert result.endswith("]")


class TestExtLabel:
    def test_returns_extension_without_dot(self) -> None:
        assert ext_label(Path("file.md")) == "md"
        assert ext_label(Path("file.mmd")) == "mmd"
        assert ext_label(Path("file.txt")) == "txt"

    def test_returns_noext_for_no_extension(self) -> None:
        assert ext_label(Path("README")) == "noext"
        assert ext_label(Path(".hidden")) == "noext"


class TestSummarizeExtensionGroups:
    def test_counts_extensions(self) -> None:
        paths = [Path("a.md"), Path("b.md"), Path("c.mmd")]
        result = summarize_extension_groups(paths)
        assert result == {"md": 2, "mmd": 1}

    def test_empty_list(self) -> None:
        result = summarize_extension_groups([])
        assert result == {}


class TestFormatExtGroups:
    def test_single_group(self) -> None:
        counts = {"md": 5}
        result = format_ext_groups(counts)
        assert result == "5 more files of type md"

    def test_multiple_groups(self) -> None:
        counts = {"md": 3, "mmd": 2}
        result = format_ext_groups(counts)
        assert "3 more files of type md" in result
        assert "2 more files of type mmd" in result

    def test_other_bucket_when_exceeding_max(self) -> None:
        counts = {"md": 10, "mmd": 5, "txt": 3, "py": 2}
        result = format_ext_groups(counts, max_groups=2)
        assert "10 more files of type md" in result
        assert "5 more files of type mmd" in result
        assert "5 more files of other types" in result

    def test_empty_counts(self) -> None:
        result = format_ext_groups({})
        assert result == ""


# ---------------------------------------------------------------------------
# Tests for ListDirTool initialization
# ---------------------------------------------------------------------------
class TestListDirToolInit:
    def test_initializes_with_string_path(self, temp_workspace: Path) -> None:
        tool = ListDirTool(base_path=str(temp_workspace))
        assert tool.search_root == temp_workspace.resolve()

    def test_initializes_with_path_object(self, temp_workspace: Path) -> None:
        tool = ListDirTool(base_path=temp_workspace)
        assert tool.search_root == temp_workspace.resolve()


# ---------------------------------------------------------------------------
# Tests for basic functionality and filtering
# ---------------------------------------------------------------------------
class TestBasicStructure:
    def test_root_shown_without_bullet(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Root directory should be shown without a bullet."""
        create_file(temp_workspace, "file.md")

        result = list_dir_fn(".")
        lines = result.split("\n")

        # Root line should not have a bullet
        assert not lines[0].startswith("-")
        assert lines[0].endswith("/")

    def test_alphabetical_sort_dirs_before_files(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Directories should come before files, both alphabetically sorted."""
        create_file(temp_workspace, "zebra.md")
        create_file(temp_workspace, "alpha.md")
        create_dir(temp_workspace, "zdir")
        create_file(temp_workspace, "zdir/file.md")  # so zdir is included
        create_dir(temp_workspace, "adir")
        create_file(temp_workspace, "adir/file.md")  # so adir is included

        result = list_dir_fn(".")
        lines = result.split("\n")

        # Find indices (skip root line at index 0)
        adir_idx = next(i for i, l in enumerate(lines) if "adir/" in l)
        zdir_idx = next(i for i, l in enumerate(lines) if "zdir/" in l)
        alpha_idx = next(i for i, l in enumerate(lines) if "alpha.md" in l)
        zebra_idx = next(i for i, l in enumerate(lines) if "zebra.md" in l)

        # Directories before files
        assert adir_idx < alpha_idx
        assert zdir_idx < alpha_idx
        # Alphabetical within category
        assert adir_idx < zdir_idx
        assert alpha_idx < zebra_idx

    def test_case_insensitive_sorting(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Sorting should be case-insensitive."""
        create_file(temp_workspace, "Zebra.md")
        create_file(temp_workspace, "apple.md")
        create_file(temp_workspace, "Banana.md")

        result = list_dir_fn(".")
        lines = [l.strip() for l in result.split("\n") if ".md" in l]

        # Case-insensitive: apple, Banana, Zebra
        assert lines[0] == "- apple.md"
        assert lines[1] == "- Banana.md"
        assert lines[2] == "- Zebra.md"

    def test_indentation_per_level(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Each nesting level should add 3 spaces of indentation."""
        create_file(temp_workspace, "level1/level2/file.md")

        result = list_dir_fn(".")
        lines = result.split("\n")

        # Find the file line (depth 3: root=0, level1=1, level2=2, file at level2 -> depth 3)
        file_line = next(l for l in lines if "file.md" in l)
        # depth=3 means 9 spaces + "- "
        assert file_line == "         - file.md"


class TestExtensionFiltering:
    def test_only_md_and_mmd_files(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Only .md and .mmd files should be listed."""
        create_file(temp_workspace, "readme.md")
        create_file(temp_workspace, "diagram.mmd")
        create_file(temp_workspace, "script.py")
        create_file(temp_workspace, "data.json")
        create_file(temp_workspace, "notes.txt")

        result = list_dir_fn(".")

        assert "readme.md" in result
        assert "diagram.mmd" in result
        assert "script.py" not in result
        assert "data.json" not in result
        assert "notes.txt" not in result

    def test_case_insensitive_extension(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Extension filtering should be case-insensitive."""
        create_file(temp_workspace, "upper.MD")
        create_file(temp_workspace, "lower.md")
        create_file(temp_workspace, "mixed.Md")

        result = list_dir_fn(".")

        assert "upper.MD" in result
        assert "lower.md" in result
        assert "mixed.Md" in result


class TestDirectoryPruning:
    def test_empty_dirs_excluded(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Directories with no allowed files should be excluded."""
        create_dir(temp_workspace, "empty_dir")
        create_file(temp_workspace, "file.md")

        result = list_dir_fn(".")

        assert "empty_dir" not in result
        assert "file.md" in result

    def test_dirs_with_only_disallowed_files_excluded(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Directories with only non-.md/.mmd files should be excluded."""
        create_file(temp_workspace, "excluded_dir/script.py")
        create_file(temp_workspace, "included_dir/doc.md")

        result = list_dir_fn(".")

        assert "excluded_dir" not in result
        assert "included_dir" in result

    def test_nested_allowed_file_includes_parent_dirs(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """A deeply nested .md file should cause all parent dirs to be shown."""
        create_file(temp_workspace, "a/b/c/deep.md")

        result = list_dir_fn(".")

        assert "a/" in result
        assert "b/" in result
        assert "c/" in result
        assert "deep.md" in result


# ---------------------------------------------------------------------------
# Tests for recursion depth and large directory constraints
# ---------------------------------------------------------------------------
class TestRecursionDepthLimit:
    def test_depth_limit_summary(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Directories at MAX_DEPTH should show summary instead of contents."""
        # Create structure: root/d1/d2/d3/d4/d5/d6/file.md
        # MAX_DEPTH is 5, so d5 is at depth 5 and should show summary
        path = "d1/d2/d3/d4/d5/d6/file.md"
        create_file(temp_workspace, path)

        result = list_dir_fn(".")

        # d5/ should appear with a summary line
        assert "d5/" in result
        assert "depth limit reached" in result
        # d6/ should NOT appear as a separate directory
        lines = [l for l in result.split("\n") if "d6/" in l and "- d6/" in l]
        assert len(lines) == 0

    def test_depth_summary_counts(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Depth limit summary should show correct file/dir counts."""
        # Create structure at depth 5 with multiple files
        base = "d1/d2/d3/d4/d5"
        create_file(temp_workspace, f"{base}/subdir/a.md")
        create_file(temp_workspace, f"{base}/subdir/b.mmd")
        create_file(temp_workspace, f"{base}/another/c.md")

        result = list_dir_fn(".")

        # Should show counts for files and subdirectories
        assert "depth limit reached" in result
        assert "3 files" in result
        assert "2 subdirectories" in result


class TestLargeDirectoryHandling:
    def test_large_dir_shows_first_n_dirs(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Large directories should show only first 5 subdirectories."""
        # Create >50 entries to trigger large dir handling
        for i in range(10):
            create_file(temp_workspace, f"dir_{i:02d}/file.md")
        for i in range(45):
            create_file(temp_workspace, f"file_{i:02d}.md")

        result = list_dir_fn(".")

        # First 5 dirs should be shown
        assert "dir_00/" in result
        assert "dir_04/" in result
        # 6th dir should NOT be shown directly
        assert "- dir_05/" not in result
        # Summary should indicate remaining dirs
        assert "5 more subdirectories" in result

    def test_large_dir_shows_first_n_files(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Large directories should show only first 5 files."""
        # Create >50 files
        for i in range(60):
            create_file(temp_workspace, f"file_{i:02d}.md")

        result = list_dir_fn(".")

        # First 5 files should be shown
        assert "file_00.md" in result
        assert "file_04.md" in result
        # 6th file should NOT be shown directly
        lines = result.split("\n")
        file_05_lines = [l for l in lines if "- file_05.md" in l]
        assert len(file_05_lines) == 0
        # Summary should show remaining files
        assert "more files of type md" in result

    def test_large_dir_file_summary_groups_by_ext(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """File summary in large dirs should group by extension."""
        # Create mixed .md and .mmd files (>50)
        for i in range(35):
            create_file(temp_workspace, f"doc_{i:02d}.md")
        for i in range(20):
            create_file(temp_workspace, f"diagram_{i:02d}.mmd")

        result = list_dir_fn(".")

        # Should have extension breakdown
        assert "more files of type md" in result
        assert "more files of type mmd" in result


# ---------------------------------------------------------------------------
# Tests for ignore patterns
# ---------------------------------------------------------------------------
class TestIgnoreGlobsFiles:
    def test_ignore_specific_file(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Specific file patterns should be ignored."""
        create_file(temp_workspace, "visible.md")
        create_file(temp_workspace, "secret.md")

        result = list_dir_fn(".", ignore_globs=["secret.md"])

        assert "visible.md" in result
        assert "secret.md" not in result

    def test_ignore_pattern_with_wildcard(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Wildcard patterns should work."""
        create_file(temp_workspace, "keep.md")
        create_file(temp_workspace, "temp_1.md")
        create_file(temp_workspace, "temp_2.md")

        result = list_dir_fn(".", ignore_globs=["temp_*.md"])

        assert "keep.md" in result
        assert "temp_1.md" not in result
        assert "temp_2.md" not in result


class TestIgnoreGlobsDirectories:
    def test_star_star_name_star_star_hides_dir_and_children(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Pattern **/name/** should hide directory and all children."""
        create_file(temp_workspace, "keep/file.md")
        create_file(temp_workspace, "node_modules/pkg/file.md")

        result = list_dir_fn(".", ignore_globs=["**/node_modules/**"])

        assert "keep/" in result
        assert "node_modules" not in result

    def test_name_star_star_hides_children_but_shows_dir(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Pattern name/** hides children; dir shown only if it has remaining allowed files."""
        # hidden_content/ has a direct file AND a nested child
        create_file(temp_workspace, "hidden_content/direct.md")
        create_file(temp_workspace, "hidden_content/child/nested.md")
        create_file(temp_workspace, "other/file.md")

        result = list_dir_fn(".", ignore_globs=["hidden_content/**"])

        # hidden_content/ is pruned since all its content is ignored by pattern
        # (both direct.md and child/nested.md match hidden_content/**)
        assert "hidden_content/" not in result
        assert "other/" in result
        assert "child/" not in result


# ---------------------------------------------------------------------------
# Tests for symlinks
# ---------------------------------------------------------------------------
class TestSymlinks:
    def test_symlinked_file_listed(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Symlinked files should be listed as normal files."""
        real_file = create_file(temp_workspace, "real.md", content="content")
        symlink_path = temp_workspace / "link.md"
        try:
            symlink_path.symlink_to(real_file)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        result = list_dir_fn(".")

        assert "real.md" in result
        assert "link.md" in result

    def test_symlinked_dir_shown_but_not_descended(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Symlinked directories should be shown but not recursed into."""
        real_dir = create_dir(temp_workspace, "real_dir")
        create_file(temp_workspace, "real_dir/nested.md")
        symlink_dir = temp_workspace / "link_dir"
        try:
            symlink_dir.symlink_to(real_dir)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        result = list_dir_fn(".")

        # real_dir should show contents
        assert "real_dir/" in result
        assert "nested.md" in result
        # link_dir should appear as a directory
        assert "link_dir/" in result
        # But link_dir's contents should NOT be shown (no second nested.md)
        lines = result.split("\n")
        nested_lines = [l for l in lines if "nested.md" in l]
        # Only one nested.md line (from real_dir)
        assert len(nested_lines) == 1


# ---------------------------------------------------------------------------
# Tests for error handling and edge cases
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_path_not_exists(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Non-existent path should return error message."""
        result = list_dir_fn("nonexistent")

        assert "does not exist" in result

    def test_path_is_file_not_directory(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Path pointing to a file should return error message."""
        create_file(temp_workspace, "afile.txt")

        result = list_dir_fn("afile.txt")

        assert "not a directory" in result

    def test_path_traversal_rejected(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Path traversal attempts should be rejected."""
        result = list_dir_fn("../outside")

        assert "escapes search root" in result

    def test_double_dot_escape_rejected(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Path with .. that escapes root should be rejected."""
        create_dir(temp_workspace, "subdir")

        result = list_dir_fn("subdir/../../outside")

        assert "escapes search root" in result

    def test_permission_error_handled(
        self, temp_workspace: Path, list_dir_tool: ListDirTool
    ) -> None:
        """Permission errors should be handled gracefully."""
        create_dir(temp_workspace, "restricted")
        create_file(temp_workspace, "ok.md")

        # Mock iterdir to raise PermissionError for the restricted directory
        original_iterdir = Path.iterdir

        def mock_iterdir(self):
            if self.name == "restricted":
                raise PermissionError("Access denied")
            return original_iterdir(self)

        # The _has_allowed_in_subtree catches PermissionError and returns True
        # (treating inaccessible dirs as potentially having content)
        # Then _safe_list_dir will also catch PermissionError
        with patch.object(Path, "iterdir", mock_iterdir):
            list_dir_fn = list_dir_tool.get_tool()
            result = list_dir_fn(".")

        assert "restricted/" in result
        assert "permission denied" in result


class TestEdgeCases:
    def test_empty_directory(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Empty directory with no .md files shows just the root."""
        result = list_dir_fn(".")
        lines = result.strip().split("\n")

        # Just the root line
        assert len(lines) == 1
        assert lines[0].endswith("/")

    def test_hidden_files_included(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Hidden files (starting with .) should be included."""
        create_file(temp_workspace, ".hidden.md")
        create_file(temp_workspace, "visible.md")

        result = list_dir_fn(".")

        assert ".hidden.md" in result
        assert "visible.md" in result

    def test_unicode_filenames(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Unicode filenames should be handled correctly."""
        create_file(temp_workspace, "文档.md")
        create_file(temp_workspace, "émoji_📝.md")

        result = list_dir_fn(".")

        assert "文档.md" in result
        assert "émoji_📝.md" in result

    def test_filenames_with_spaces(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Filenames with spaces should be handled correctly."""
        create_file(temp_workspace, "file with spaces.md")

        result = list_dir_fn(".")

        assert "file with spaces.md" in result

    def test_deeply_nested_single_file(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """A single deeply nested file should show the full path hierarchy."""
        create_file(temp_workspace, "a/b/c/d/only.md")

        result = list_dir_fn(".")

        assert "a/" in result
        assert "b/" in result
        assert "c/" in result
        assert "d/" in result
        assert "only.md" in result

    def test_current_dir_as_target(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Target directory '.' should work."""
        create_file(temp_workspace, "file.md")

        result = list_dir_fn(".")

        assert "file.md" in result

    def test_subdirectory_as_target(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Subdirectory as target should scope the listing."""
        create_file(temp_workspace, "root.md")
        create_file(temp_workspace, "subdir/nested.md")

        result = list_dir_fn("subdir")

        assert "nested.md" in result
        assert "root.md" not in result


# ---------------------------------------------------------------------------
# Tests for configurable limits
# ---------------------------------------------------------------------------
class TestConfigurableLimits:
    """Tests for instance-configurable limits."""

    def test_custom_max_depth_shows_summary(self, temp_workspace: Path) -> None:
        """Verify custom max_depth shows summary at specified depth."""
        # Create structure: root/d1/d2/d3/file.md
        # With max_depth=2, d2 should show summary instead of contents
        create_file(temp_workspace, "d1/d2/d3/file.md")

        tool = ListDirTool(base_path=temp_workspace, max_depth=2)
        list_dir_fn = tool.get_tool()

        result = list_dir_fn(".")

        assert "d1/" in result
        assert "d2/" in result
        assert "depth limit reached" in result
        # d3/ should NOT appear as a separate directory line
        lines = [l for l in result.split("\n") if "- d3/" in l]
        assert len(lines) == 0

    def test_custom_large_dir_threshold(self, temp_workspace: Path) -> None:
        """Verify custom large_dir_threshold triggers truncation earlier."""
        # Create 15 files (more than custom threshold of 10)
        for i in range(15):
            create_file(temp_workspace, f"file_{i:02d}.md")

        tool = ListDirTool(
            base_path=temp_workspace,
            large_dir_threshold=10,
            large_dir_show_files=3,
        )
        list_dir_fn = tool.get_tool()

        result = list_dir_fn(".")

        # Should show only 3 files
        file_lines = [l for l in result.split("\n") if ".md" in l and "- " in l]
        assert len(file_lines) == 3

        # Should have summary for remaining 12 files
        assert "12 more files of type md" in result

    def test_custom_large_dir_show_dirs(self, temp_workspace: Path) -> None:
        """Verify custom large_dir_show_dirs limits shown directories."""
        # Create 15 directories with files (more than custom threshold)
        for i in range(15):
            create_file(temp_workspace, f"dir_{i:02d}/file.md")

        tool = ListDirTool(
            base_path=temp_workspace,
            large_dir_threshold=10,
            large_dir_show_dirs=2,
        )
        list_dir_fn = tool.get_tool()

        result = list_dir_fn(".")

        # Should show only 2 directories
        assert "dir_00/" in result
        assert "dir_01/" in result
        # 3rd dir should be in summary
        assert "- dir_02/" not in result
        assert "13 more subdirectories" in result

    def test_default_max_depth(self, temp_workspace: Path) -> None:
        """Verify default max_depth (5) allows deep nesting."""
        # Create structure at depth 4 (under default limit)
        create_file(temp_workspace, "d1/d2/d3/d4/file.md")

        tool = ListDirTool(base_path=temp_workspace)
        list_dir_fn = tool.get_tool()

        result = list_dir_fn(".")

        # All directories should be shown
        assert "d1/" in result
        assert "d2/" in result
        assert "d3/" in result
        assert "d4/" in result
        assert "file.md" in result
        # No depth limit message at this depth
        lines_with_depth = [l for l in result.split("\n") if "depth limit" in l]
        assert len(lines_with_depth) == 0


# ---------------------------------------------------------------------------
# Tests for custom allowed extensions
# ---------------------------------------------------------------------------
class TestCustomExtensions:
    """Tests for custom allowed_extensions configuration."""

    def test_custom_extension_py_allowed(self, temp_workspace: Path) -> None:
        """Verify custom allowed_extensions enables .py files."""
        create_file(temp_workspace, "script.py")
        create_file(temp_workspace, "readme.md")

        tool = ListDirTool(
            base_path=temp_workspace,
            allowed_extensions={".py"},
        )
        list_dir_fn = tool.get_tool()

        result = list_dir_fn(".")

        assert "script.py" in result
        assert "readme.md" not in result

    def test_custom_extension_multiple(self, temp_workspace: Path) -> None:
        """Verify multiple custom extensions work together."""
        create_file(temp_workspace, "script.py")
        create_file(temp_workspace, "config.yaml")
        create_file(temp_workspace, "readme.md")
        create_file(temp_workspace, "data.json")

        tool = ListDirTool(
            base_path=temp_workspace,
            allowed_extensions={".py", ".yaml"},
        )
        list_dir_fn = tool.get_tool()

        result = list_dir_fn(".")

        assert "script.py" in result
        assert "config.yaml" in result
        assert "readme.md" not in result
        assert "data.json" not in result

    def test_custom_extension_prunes_dirs_without_matches(
        self, temp_workspace: Path
    ) -> None:
        """Verify directories without matching extensions are pruned."""
        create_file(temp_workspace, "docs/readme.md")
        create_file(temp_workspace, "src/main.py")

        tool = ListDirTool(
            base_path=temp_workspace,
            allowed_extensions={".py"},
        )
        list_dir_fn = tool.get_tool()

        result = list_dir_fn(".")

        assert "src/" in result
        assert "main.py" in result
        # docs/ should be pruned (no .py files)
        assert "docs/" not in result
        assert "readme.md" not in result

    def test_default_extensions_md_mmd(
        self, temp_workspace: Path, list_dir_fn: Callable
    ) -> None:
        """Verify default extensions are .md and .mmd only."""
        create_file(temp_workspace, "readme.md")
        create_file(temp_workspace, "diagram.mmd")
        create_file(temp_workspace, "script.py")

        result = list_dir_fn(".")

        assert "readme.md" in result
        assert "diagram.mmd" in result
        assert "script.py" not in result

    def test_custom_extension_txt(self, temp_workspace: Path) -> None:
        """Verify .txt can be allowed via custom extensions."""
        create_file(temp_workspace, "notes.txt")
        create_file(temp_workspace, "readme.md")

        tool = ListDirTool(
            base_path=temp_workspace,
            allowed_extensions={".txt"},
        )
        list_dir_fn = tool.get_tool()

        result = list_dir_fn(".")

        assert "notes.txt" in result
        assert "readme.md" not in result
