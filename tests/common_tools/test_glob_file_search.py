"""Tests for glob_file_search tool functionality.

Tests cover:
- Basic search and pattern matching
- Extension filtering (.md, .mmd only)
- Path handling and target directory
- Sorting by mtime and name
- Truncation and summaries
- Error handling and security
- Edge cases (empty dirs, symlinks)
"""
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Generator, Optional

import pytest

from anthropic_agent.common_tools.glob_file_search import (
    ALLOWED_EXTS,
    MAX_RESULTS,
    SUMMARY_MAX_EXT_GROUPS,
    GlobFileSearchTool,
    _ext_label,
    _has_allowed_ext,
    _is_within,
    _normalize_pattern,
    _safe_stat_mtime,
    _summarize_file_exts,
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
def glob_tool(temp_workspace: Path) -> GlobFileSearchTool:
    """Create a GlobFileSearchTool instance with the temp workspace."""
    return GlobFileSearchTool(base_path=temp_workspace)


@pytest.fixture
def search_fn(glob_tool: GlobFileSearchTool) -> Callable:
    """Get the glob_file_search function from the tool."""
    return glob_tool.get_tool()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def create_file(
    workspace: Path,
    rel_path: str,
    content: str = "",
    mtime: Optional[float] = None,
) -> Path:
    """Create a test file in the workspace with optional modification time.
    
    Args:
        workspace: Base directory path.
        rel_path: Relative path for the file.
        content: File content.
        mtime: Optional modification time (epoch). If None, uses current time.
    
    Returns:
        The full path to the created file.
    """
    full_path = workspace / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    if mtime is not None:
        os.utime(full_path, (mtime, mtime))
    return full_path


def create_dir(workspace: Path, rel_path: str) -> Path:
    """Create a directory in the workspace."""
    full_path = workspace / rel_path
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


# ---------------------------------------------------------------------------
# Tests for utility functions
# ---------------------------------------------------------------------------
class TestNormalizePattern:
    def test_prepends_recursive_glob(self) -> None:
        assert _normalize_pattern("*.md") == "**/*.md"
        assert _normalize_pattern("file.md") == "**/file.md"
    
    def test_preserves_existing_recursive_glob(self) -> None:
        assert _normalize_pattern("**/test.md") == "**/test.md"
        assert _normalize_pattern("**/subdir/*.md") == "**/subdir/*.md"


class TestExtLabel:
    def test_returns_extension_without_dot(self) -> None:
        assert _ext_label(Path("file.md")) == "md"
        assert _ext_label(Path("file.mmd")) == "mmd"
        assert _ext_label(Path("file.txt")) == "txt"
    
    def test_returns_noext_for_no_extension(self) -> None:
        assert _ext_label(Path("README")) == "noext"
        assert _ext_label(Path(".hidden")) == "noext"


class TestHasAllowedExt:
    def test_allowed_extensions(self) -> None:
        assert _has_allowed_ext(Path("file.md")) is True
        assert _has_allowed_ext(Path("file.mmd")) is True
        assert _has_allowed_ext(Path("file.MD")) is True  # case insensitive
        assert _has_allowed_ext(Path("file.MMD")) is True
    
    def test_disallowed_extensions(self) -> None:
        assert _has_allowed_ext(Path("file.txt")) is False
        assert _has_allowed_ext(Path("file.py")) is False
        assert _has_allowed_ext(Path("file.json")) is False
        assert _has_allowed_ext(Path("README")) is False


class TestIsWithin:
    def test_child_within_parent(self, temp_workspace: Path) -> None:
        child = temp_workspace / "subdir" / "file.txt"
        assert _is_within(child, temp_workspace) is True
    
    def test_child_outside_parent(self, temp_workspace: Path) -> None:
        child = temp_workspace.parent / "outside"
        assert _is_within(child, temp_workspace) is False
    
    def test_same_path(self, temp_workspace: Path) -> None:
        assert _is_within(temp_workspace, temp_workspace) is True


class TestSafeStatMtime:
    def test_returns_mtime_for_existing_file(self, temp_workspace: Path) -> None:
        file_path = create_file(temp_workspace, "test.md")
        mtime = _safe_stat_mtime(file_path)
        assert mtime is not None
        assert isinstance(mtime, float)
    
    def test_returns_none_for_nonexistent_file(self, temp_workspace: Path) -> None:
        file_path = temp_workspace / "nonexistent.md"
        assert _safe_stat_mtime(file_path) is None


class TestSummarizeFileExts:
    def test_single_extension_group(self) -> None:
        paths = [Path("a.md"), Path("b.md"), Path("c.md")]
        result = _summarize_file_exts(paths)
        assert result == "3 more files of type md"
    
    def test_multiple_extension_groups(self) -> None:
        paths = [Path("a.md"), Path("b.md"), Path("c.mmd")]
        result = _summarize_file_exts(paths)
        assert "2 more files of type md" in result
        assert "1 more files of type mmd" in result
    
    def test_empty_paths(self) -> None:
        result = _summarize_file_exts([])
        assert result == ""
    
    def test_other_bucket_when_exceeding_max_groups(self) -> None:
        # Create more than SUMMARY_MAX_EXT_GROUPS different extensions
        paths = [
            Path("a.md"), Path("b.md"),
            Path("c.mmd"), Path("d.mmd"),
            Path("e.txt"), Path("f.txt"),
            Path("g.py"),
        ]
        result = _summarize_file_exts(paths, max_groups=2)
        assert "other types" in result


# ---------------------------------------------------------------------------
# Tests for GlobFileSearchTool initialization
# ---------------------------------------------------------------------------
class TestGlobFileSearchToolInit:
    def test_initializes_with_string_path(self, temp_workspace: Path) -> None:
        tool = GlobFileSearchTool(base_path=str(temp_workspace))
        assert tool.search_root == temp_workspace.resolve()
    
    def test_initializes_with_path_object(self, temp_workspace: Path) -> None:
        tool = GlobFileSearchTool(base_path=temp_workspace)
        assert tool.search_root == temp_workspace.resolve()
    
    def test_resolves_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a relative path scenario
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                tool = GlobFileSearchTool(base_path=".")
                assert tool.search_root == Path(tmpdir).resolve()
            finally:
                os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Tests for basic search and pattern matching
# ---------------------------------------------------------------------------
class TestBasicSearch:
    def test_recursive_search_default(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Pattern without **/ should still find files in subdirectories."""
        create_file(temp_workspace, "root.md")
        create_file(temp_workspace, "subdir/nested.md")
        create_file(temp_workspace, "subdir/deep/file.md")
        
        result = search_fn("*.md")
        
        assert "root.md" in result
        assert "subdir/nested.md" in result
        assert "subdir/deep/file.md" in result
    
    def test_explicit_recursive_pattern(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Explicit **/ pattern should work."""
        create_file(temp_workspace, "root.md")
        create_file(temp_workspace, "subdir/file.md")
        
        result = search_fn("**/file.md")
        
        assert "subdir/file.md" in result
        # root.md doesn't match **/file.md pattern
        assert "root.md" not in result
    
    def test_no_matches_returns_message(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """When no files match, return 'No matches found.'"""
        create_file(temp_workspace, "file.txt")  # Not .md or .mmd
        
        result = search_fn("*.md")
        
        assert result == "No matches found."
    
    def test_hidden_files_included(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Hidden files (starting with .) should be found."""
        create_file(temp_workspace, ".hidden.md")
        create_file(temp_workspace, "visible.md")
        
        result = search_fn("*.md")
        
        assert ".hidden.md" in result
        assert "visible.md" in result
    
    def test_specific_filename_pattern(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Search for specific filename."""
        create_file(temp_workspace, "README.md")
        create_file(temp_workspace, "other.md")
        
        result = search_fn("README.md")
        
        assert "README.md" in result
        assert "other.md" not in result


# ---------------------------------------------------------------------------
# Tests for extension filtering
# ---------------------------------------------------------------------------
class TestExtensionFiltering:
    def test_only_md_files_returned(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Only .md files should be returned."""
        create_file(temp_workspace, "file.md")
        create_file(temp_workspace, "file.txt")
        create_file(temp_workspace, "file.py")
        
        result = search_fn("file.*")
        
        assert "file.md" in result
        assert "file.txt" not in result
        assert "file.py" not in result
    
    def test_only_mmd_files_returned(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Only .mmd files should be returned."""
        create_file(temp_workspace, "diagram.mmd")
        create_file(temp_workspace, "diagram.txt")
        
        result = search_fn("diagram.*")
        
        assert "diagram.mmd" in result
        assert "diagram.txt" not in result
    
    def test_both_md_and_mmd_returned(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Both .md and .mmd files should be returned."""
        create_file(temp_workspace, "doc.md")
        create_file(temp_workspace, "flow.mmd")
        
        result = search_fn("*")
        
        assert "doc.md" in result
        assert "flow.mmd" in result
    
    def test_directories_not_included(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Directories should not be included in results."""
        create_dir(temp_workspace, "docs.md")  # Directory with .md name
        create_file(temp_workspace, "file.md")
        
        result = search_fn("*.md")
        
        lines = result.strip().split("\n")
        # Only the file should be included, not the directory
        assert "file.md" in result
        # Check docs.md is not in result or is filtered out
        assert all("docs.md" not in line or line == "file.md" for line in lines)
    
    def test_case_insensitive_extension(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Extension matching should be case-insensitive."""
        create_file(temp_workspace, "upper.MD")
        create_file(temp_workspace, "lower.md")
        create_file(temp_workspace, "mixed.Md")
        
        result = search_fn("*.MD")
        
        # All should be found due to case-insensitive extension checking
        # Note: glob pattern itself may be case-sensitive on some OS
        assert "lower.md" in result or "upper.MD" in result


# ---------------------------------------------------------------------------
# Tests for path handling and target directory
# ---------------------------------------------------------------------------
class TestPathHandling:
    def test_target_directory_scopes_search(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """target_directory should scope search to subdirectory."""
        create_file(temp_workspace, "root.md")
        create_file(temp_workspace, "docs/nested.md")
        create_file(temp_workspace, "other/file.md")
        
        result = search_fn("*.md", target_directory="docs")
        
        assert "docs/nested.md" in result
        assert "root.md" not in result
        assert "other/file.md" not in result
    
    def test_paths_relative_to_search_root(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Output paths should be relative to search_root, not target_directory."""
        create_file(temp_workspace, "subdir/deep/file.md")
        
        result = search_fn("*.md", target_directory="subdir")
        
        # Path should be relative to search_root
        assert "subdir/deep/file.md" in result
        # Not just "deep/file.md"
        assert result.strip().startswith("subdir")
    
    def test_posix_style_paths(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Output paths should use forward slashes (POSIX style)."""
        create_file(temp_workspace, "dir/subdir/file.md")
        
        result = search_fn("*.md")
        
        assert "\\" not in result  # No backslashes
        assert "dir/subdir/file.md" in result
    
    def test_dot_in_target_directory(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Current directory (.) should work as target_directory."""
        create_file(temp_workspace, "file.md")
        
        result = search_fn("*.md", target_directory=".")
        
        assert "file.md" in result
    
    def test_normalized_paths(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Paths should be normalized (collapse . and ..)."""
        create_file(temp_workspace, "dir/file.md")
        
        result = search_fn("*.md", target_directory="dir/./")
        
        # Path should be normalized
        assert "dir/file.md" in result
        assert "//" not in result
        assert "/./" not in result


# ---------------------------------------------------------------------------
# Tests for sorting logic
# ---------------------------------------------------------------------------
class TestSorting:
    def test_sorted_by_mtime_newest_first(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Files should be sorted by modification time, newest first."""
        base_time = time.time()
        create_file(temp_workspace, "old.md", mtime=base_time - 100)
        create_file(temp_workspace, "middle.md", mtime=base_time - 50)
        create_file(temp_workspace, "new.md", mtime=base_time)
        
        result = search_fn("*.md")
        lines = result.strip().split("\n")
        
        assert lines[0] == "new.md"
        assert lines[1] == "middle.md"
        assert lines[2] == "old.md"
    
    def test_alphabetic_tiebreaker_case_insensitive(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Files with same mtime should be sorted alphabetically (case-insensitive)."""
        same_time = time.time()
        create_file(temp_workspace, "Zebra.md", mtime=same_time)
        create_file(temp_workspace, "apple.md", mtime=same_time)
        create_file(temp_workspace, "Banana.md", mtime=same_time)
        
        result = search_fn("*.md")
        lines = result.strip().split("\n")
        
        # Case-insensitive alphabetical: apple, Banana, Zebra
        assert lines[0] == "apple.md"
        assert lines[1] == "Banana.md"
        assert lines[2] == "Zebra.md"


# ---------------------------------------------------------------------------
# Tests for truncation and summaries
# ---------------------------------------------------------------------------
class TestTruncation:
    def test_max_results_limit(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Only MAX_RESULTS files should be shown."""
        # Create more than MAX_RESULTS files
        num_files = MAX_RESULTS + 10
        for i in range(num_files):
            create_file(temp_workspace, f"file_{i:03d}.md")
        
        result = search_fn("*.md")
        lines = result.strip().split("\n")
        
        # First MAX_RESULTS lines should be file paths
        file_lines = [l for l in lines if not l.startswith("[")]
        assert len(file_lines) == MAX_RESULTS
    
    def test_summary_appended_when_truncated(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Summary should be appended when results are truncated."""
        num_files = MAX_RESULTS + 5
        for i in range(num_files):
            create_file(temp_workspace, f"file_{i:03d}.md")
        
        result = search_fn("*.md")
        
        assert "[5 more files of type md]" in result
    
    def test_summary_with_multiple_extensions(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Summary should group by extension type."""
        # Create MAX_RESULTS .md files first (these will be shown)
        base_time = time.time()
        for i in range(MAX_RESULTS):
            create_file(temp_workspace, f"shown_{i:03d}.md", mtime=base_time)
        
        # Create extra files with different extensions (these will be summarized)
        for i in range(3):
            create_file(temp_workspace, f"extra_{i}.md", mtime=base_time - 100)
        for i in range(2):
            create_file(temp_workspace, f"extra_{i}.mmd", mtime=base_time - 100)
        
        result = search_fn("*")
        
        assert "more files of type md" in result
        assert "more files of type mmd" in result
    
    def test_no_summary_when_under_limit(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """No summary when results are under MAX_RESULTS."""
        for i in range(5):
            create_file(temp_workspace, f"file_{i}.md")
        
        result = search_fn("*.md")
        
        assert "[" not in result  # No bracketed summary


# ---------------------------------------------------------------------------
# Tests for error handling and security
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_nonexistent_target_directory(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Non-existent target_directory should return error."""
        result = search_fn("*.md", target_directory="nonexistent")
        
        assert "does not exist" in result
    
    def test_target_is_file_not_directory(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """target_directory pointing to a file should return error."""
        create_file(temp_workspace, "notadir.txt")
        
        result = search_fn("*.md", target_directory="notadir.txt")
        
        assert "not a directory" in result
    
    def test_path_traversal_rejected(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Path traversal attempts should be rejected."""
        result = search_fn("*.md", target_directory="../outside")
        
        assert "escapes search root" in result
    
    def test_double_dot_in_path(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Path with .. that escapes root should be rejected."""
        create_dir(temp_workspace, "subdir")
        
        result = search_fn("*.md", target_directory="subdir/../../outside")
        
        assert "escapes search root" in result


# ---------------------------------------------------------------------------
# Tests for edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_directory(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Empty directory should return 'No matches found.'"""
        create_dir(temp_workspace, "empty")
        
        result = search_fn("*.md", target_directory="empty")
        
        assert result == "No matches found."
    
    def test_empty_glob_pattern(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Empty or minimal glob pattern."""
        create_file(temp_workspace, "test.md")
        
        # Pattern "*" should find all .md files
        result = search_fn("*")
        
        assert "test.md" in result
    
    def test_complex_glob_pattern(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Complex glob patterns should work."""
        create_file(temp_workspace, "doc_v1.md")
        create_file(temp_workspace, "doc_v2.md")
        create_file(temp_workspace, "readme.md")
        
        result = search_fn("doc_*.md")
        
        assert "doc_v1.md" in result
        assert "doc_v2.md" in result
        assert "readme.md" not in result
    
    def test_symlink_to_file_included(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Symlinks to files should be included if they match."""
        real_file = create_file(temp_workspace, "real.md", content="content")
        symlink_path = temp_workspace / "link.md"
        try:
            symlink_path.symlink_to(real_file)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")
        
        result = search_fn("*.md")
        
        assert "real.md" in result
        assert "link.md" in result
    
    def test_broken_symlink_ignored(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Broken symlinks should be gracefully ignored."""
        symlink_path = temp_workspace / "broken.md"
        try:
            symlink_path.symlink_to(temp_workspace / "nonexistent.md")
        except OSError:
            pytest.skip("Symlinks not supported on this platform")
        
        create_file(temp_workspace, "valid.md")
        
        result = search_fn("*.md")
        
        # Should not crash, and valid file should be found
        assert "valid.md" in result
        # Broken symlink should be omitted (no mtime available)
        assert "broken.md" not in result
    
    def test_deeply_nested_files(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Deeply nested files should be found."""
        deep_path = "a/b/c/d/e/f/g/h/deep.md"
        create_file(temp_workspace, deep_path)
        
        result = search_fn("*.md")
        
        assert deep_path in result
    
    def test_unicode_filenames(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Unicode filenames should be handled correctly."""
        create_file(temp_workspace, "文档.md")
        create_file(temp_workspace, "émoji_📝.md")
        
        result = search_fn("*.md")
        
        assert "文档.md" in result
        assert "émoji_📝.md" in result
    
    def test_filename_with_spaces(
        self, temp_workspace: Path, search_fn: Callable
    ) -> None:
        """Filenames with spaces should be handled correctly."""
        create_file(temp_workspace, "file with spaces.md")
        
        result = search_fn("*.md")
        
        assert "file with spaces.md" in result
