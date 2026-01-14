"""Tests for apply_patch tool functionality.

Tests cover:
- Basic Add/Update operations
- Delete file operations
- Move/Rename operations
- Fuzzy context matching
- Error handling and validation
"""
import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from anthropic_agent.common_tools.apply_patch import (
    ApplyPatchTool,
    Chunk,
    PatchError,
    _find_context,
    _find_scope,
    _format_context_mismatch,
    _norm,
    _parse_hunk_into_chunks,
    _parse_patch,
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
def patch_tool(temp_workspace: Path):
    """Create an ApplyPatchTool instance with the temp workspace."""
    return ApplyPatchTool(base_path=temp_workspace)


@pytest.fixture
def apply_patch_fn(patch_tool: ApplyPatchTool):
    """Get the apply_patch function from the tool."""
    return patch_tool.get_tool()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def parse_response(response: str) -> dict:
    """Parse JSON response from apply_patch."""
    return json.loads(response)


def create_test_file(workspace: Path, rel_path: str, content: str) -> Path:
    """Create a test file in the workspace."""
    full_path = workspace / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    return full_path


# ---------------------------------------------------------------------------
# Tests for _norm helper
# ---------------------------------------------------------------------------
class TestNormHelper:
    def test_norm_strips_trailing_cr(self) -> None:
        # _norm strips trailing \r (for CRLF split lines)
        assert _norm("hello\r") == "hello"
        assert _norm("hello\r\r") == "hello"
    
    def test_norm_preserves_other_content(self) -> None:
        # \r not at end is not stripped
        assert _norm("hello\r\n") == "hello\r\n"
        assert _norm("hello\n") == "hello\n"
        assert _norm("hello") == "hello"


# ---------------------------------------------------------------------------
# Tests for _find_context fuzzy matching
# ---------------------------------------------------------------------------
class TestFindContext:
    def test_exact_match(self) -> None:
        lines = ["line 1", "line 2", "line 3", "line 4"]
        old_lines = ["line 2", "line 3"]
        index, fuzz = _find_context(lines, old_lines, 0)
        assert index == 1
        assert fuzz == 0
    
    def test_rstrip_match(self) -> None:
        """Test matching with trailing whitespace differences."""
        lines = ["line 1  ", "line 2  ", "line 3"]
        old_lines = ["line 1", "line 2"]
        index, fuzz = _find_context(lines, old_lines, 0)
        assert index == 0
        assert fuzz == 1
    
    def test_strip_match(self) -> None:
        """Test matching with leading/trailing whitespace differences."""
        lines = ["  line 1  ", "  line 2  ", "line 3"]
        old_lines = ["line 1", "line 2"]
        index, fuzz = _find_context(lines, old_lines, 0)
        assert index == 0
        assert fuzz == 100
    
    def test_no_match(self) -> None:
        lines = ["line 1", "line 2", "line 3"]
        old_lines = ["not found", "anywhere"]
        index, fuzz = _find_context(lines, old_lines, 0)
        assert index == -1
        assert fuzz == 0
    
    def test_empty_old_lines(self) -> None:
        lines = ["line 1", "line 2"]
        old_lines: list[str] = []
        index, fuzz = _find_context(lines, old_lines, 0)
        assert index == 0
        assert fuzz == 0
    
    def test_search_from_start_index(self) -> None:
        lines = ["line 1", "line 2", "line 1", "line 2"]
        old_lines = ["line 1", "line 2"]
        # Start from index 2, should find second occurrence
        index, fuzz = _find_context(lines, old_lines, 2)
        assert index == 2
        assert fuzz == 0


# ---------------------------------------------------------------------------
# Tests for Add File operation
# ---------------------------------------------------------------------------
class TestAddFile:
    def test_add_new_file(self, temp_workspace: Path, apply_patch_fn) -> None:
        patch = """*** Begin Patch
*** Add File: new_file.py
+def hello():
+    return "world"
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        assert result["op"] == "add"
        assert result["path"] == "new_file.py"
        
        # Verify file was created
        created_file = temp_workspace / "new_file.py"
        assert created_file.exists()
        assert 'def hello():' in created_file.read_text()
    
    def test_add_file_in_subdirectory(self, temp_workspace: Path, apply_patch_fn) -> None:
        patch = """*** Begin Patch
*** Add File: subdir/nested/file.py
+content
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        assert (temp_workspace / "subdir/nested/file.py").exists()
    
    def test_add_existing_file_fails(self, temp_workspace: Path, apply_patch_fn) -> None:
        create_test_file(temp_workspace, "existing.py", "old content")
        
        patch = """*** Begin Patch
*** Add File: existing.py
+new content
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "error"
        assert "already exists" in result["error"]
    
    def test_add_file_dry_run(self, temp_workspace: Path, apply_patch_fn) -> None:
        patch = """*** Begin Patch
*** Add File: new_file.py
+content
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch, dry_run=True))
        
        assert result["status"] == "ok"
        assert result["dry_run"] is True
        assert not (temp_workspace / "new_file.py").exists()


# ---------------------------------------------------------------------------
# Tests for Update File operation
# ---------------------------------------------------------------------------
class TestUpdateFile:
    def test_update_file_basic(self, temp_workspace: Path, apply_patch_fn) -> None:
        create_test_file(temp_workspace, "test.py", "line 1\nline 2\nline 3\n")
        
        patch = """*** Begin Patch
*** Update File: test.py
@@
 line 1
-line 2
+modified line 2
 line 3
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        assert result["op"] == "update"
        assert result["hunks_applied"] == 1
        assert result["lines_added"] == 1
        assert result["lines_removed"] == 1
        
        content = (temp_workspace / "test.py").read_text()
        assert "modified line 2" in content
        assert "line 2" not in content or "modified" in content
    
    def test_update_nonexistent_file_fails(self, temp_workspace: Path, apply_patch_fn) -> None:
        patch = """*** Begin Patch
*** Update File: nonexistent.py
@@
 context
-old
+new
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "error"
        assert "does not exist" in result["error"]
    
    def test_update_with_fuzzy_matching(self, temp_workspace: Path, apply_patch_fn) -> None:
        # File has trailing spaces
        create_test_file(temp_workspace, "test.py", "line 1  \nline 2  \nline 3\n")
        
        # Patch expects lines without trailing spaces
        patch = """*** Begin Patch
*** Update File: test.py
@@
 line 1
-line 2
+modified line 2
 line 3
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        # Fuzz level should be present and > 0
        assert result.get("fuzz_level", 0) > 0
    
    def test_update_dry_run(self, temp_workspace: Path, apply_patch_fn) -> None:
        create_test_file(temp_workspace, "test.py", "line 1\nline 2\nline 3\n")
        
        patch = """*** Begin Patch
*** Update File: test.py
@@
 line 1
-line 2
+modified
 line 3
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch, dry_run=True))
        
        assert result["status"] == "ok"
        assert result["dry_run"] is True
        
        # File should be unchanged
        content = (temp_workspace / "test.py").read_text()
        assert "line 2" in content
        assert "modified" not in content


# ---------------------------------------------------------------------------
# Tests for Delete File operation
# ---------------------------------------------------------------------------
class TestDeleteFile:
    def test_delete_file(self, temp_workspace: Path, apply_patch_fn) -> None:
        file_path = create_test_file(temp_workspace, "to_delete.py", "content\n")
        assert file_path.exists()
        
        patch = """*** Begin Patch
*** Delete File: to_delete.py
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        assert result["op"] == "delete"
        assert result["path"] == "to_delete.py"
        assert not file_path.exists()
    
    def test_delete_nonexistent_file_fails(self, temp_workspace: Path, apply_patch_fn) -> None:
        patch = """*** Begin Patch
*** Delete File: nonexistent.py
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "error"
        assert "does not exist" in result["error"]
    
    def test_delete_file_dry_run(self, temp_workspace: Path, apply_patch_fn) -> None:
        file_path = create_test_file(temp_workspace, "to_delete.py", "content")
        
        patch = """*** Begin Patch
*** Delete File: to_delete.py
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch, dry_run=True))
        
        assert result["status"] == "ok"
        assert result["dry_run"] is True
        assert file_path.exists()  # File should still exist
    
    def test_delete_with_body_content_fails(self) -> None:
        patch = """*** Begin Patch
*** Delete File: file.py
+should not be here
*** End Patch"""
        
        with pytest.raises(PatchError) as exc_info:
            _parse_patch(patch)
        
        assert "should not contain body content" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests for Move/Rename operation
# ---------------------------------------------------------------------------
class TestMoveFile:
    def test_move_file_basic(self, temp_workspace: Path, apply_patch_fn) -> None:
        create_test_file(temp_workspace, "old_name.py", "line 1\nline 2\n")
        
        patch = """*** Begin Patch
*** Update File: old_name.py
*** Move to: new_name.py
@@
 line 1
-line 2
+modified line 2
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        assert result["path"] == "new_name.py"
        assert result["moved_from"] == "old_name.py"
        
        # Old file should be gone, new file should exist
        assert not (temp_workspace / "old_name.py").exists()
        assert (temp_workspace / "new_name.py").exists()
        
        content = (temp_workspace / "new_name.py").read_text()
        assert "modified line 2" in content
    
    def test_move_to_new_directory(self, temp_workspace: Path, apply_patch_fn) -> None:
        create_test_file(temp_workspace, "file.py", "content\n")
        
        patch = """*** Begin Patch
*** Update File: file.py
*** Move to: subdir/file.py
@@
-content
+new content
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        assert result["path"] == "subdir/file.py"
        assert not (temp_workspace / "file.py").exists()
        assert (temp_workspace / "subdir/file.py").exists()
    
    def test_move_to_existing_file_fails(self, temp_workspace: Path, apply_patch_fn) -> None:
        create_test_file(temp_workspace, "source.py", "line 1\n")
        create_test_file(temp_workspace, "target.py", "existing content")
        
        patch = """*** Begin Patch
*** Update File: source.py
*** Move to: target.py
@@
-line 1
+new line
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "error"
        assert "already exists" in result["error"]
    
    def test_move_dry_run(self, temp_workspace: Path, apply_patch_fn) -> None:
        create_test_file(temp_workspace, "old.py", "content\n")
        
        patch = """*** Begin Patch
*** Update File: old.py
*** Move to: new.py
@@
-content
+new content
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch, dry_run=True))
        
        assert result["status"] == "ok"
        assert result["dry_run"] is True
        # Both should remain unchanged
        assert (temp_workspace / "old.py").exists()
        assert not (temp_workspace / "new.py").exists()
    
    def test_move_with_add_file_fails(self) -> None:
        patch = """*** Begin Patch
*** Add File: file.py
*** Move to: other.py
+content
*** End Patch"""
        
        with pytest.raises(PatchError) as exc_info:
            _parse_patch(patch)
        
        assert "Move to" in str(exc_info.value) and "Update File" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests for path security
# ---------------------------------------------------------------------------
class TestPathSecurity:
    def test_absolute_path_rejected(self, temp_workspace: Path, apply_patch_fn) -> None:
        patch = """*** Begin Patch
*** Add File: /etc/passwd
+malicious
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "error"
        assert "must be relative" in result["error"]
    
    def test_path_traversal_rejected(self, temp_workspace: Path, apply_patch_fn) -> None:
        patch = """*** Begin Patch
*** Add File: ../outside.py
+malicious
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "error"
    
    def test_disallowed_extension_rejected(self, temp_workspace: Path, apply_patch_fn) -> None:
        patch = """*** Begin Patch
*** Add File: script.exe
+binary
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "error"
        assert "extension not allowed" in result["error"]


# ---------------------------------------------------------------------------
# Tests for patch parsing errors
# ---------------------------------------------------------------------------
class TestPatchParsing:
    def test_missing_begin_marker(self) -> None:
        patch = """*** Update File: test.py
@@
 content
*** End Patch"""
        
        with pytest.raises(PatchError) as exc_info:
            _parse_patch(patch)
        
        assert "Begin Patch" in str(exc_info.value)
    
    def test_missing_end_marker(self) -> None:
        patch = """*** Begin Patch
*** Update File: test.py
@@
 content"""
        
        with pytest.raises(PatchError) as exc_info:
            _parse_patch(patch)
        
        assert "End Patch" in str(exc_info.value)
    
    def test_no_file_operation(self) -> None:
        patch = """*** Begin Patch
@@
 content
*** End Patch"""
        
        with pytest.raises(PatchError) as exc_info:
            _parse_patch(patch)
        
        assert "No file operation found" in str(exc_info.value)
    
    def test_multiple_file_operations(self) -> None:
        patch = """*** Begin Patch
*** Add File: file1.py
*** Update File: file2.py
*** End Patch"""
        
        with pytest.raises(PatchError) as exc_info:
            _parse_patch(patch)
        
        assert "Multiple file operations" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests for Scope Lines
# ---------------------------------------------------------------------------
class TestScopeLines:
    def test_find_scope_exact_match(self) -> None:
        """Test scope finding with exact function definition match."""
        lines = [
            "import os",
            "",
            "def helper():",
            "    pass",
            "",
            "def target_function():",
            "    x = 1",
            "    return x",
        ]
        scope_lines = ["def target_function"]
        index, fuzz = _find_scope(lines, scope_lines, 0)
        assert index == 6  # Position after "def target_function():"
        assert fuzz == 0
    
    def test_find_scope_class(self) -> None:
        """Test scope finding with class definition."""
        lines = [
            "class MyClass:",
            "    def __init__(self):",
            "        pass",
            "    ",
            "    def method(self):",
            "        return 42",
        ]
        scope_lines = ["class MyClass"]
        index, fuzz = _find_scope(lines, scope_lines, 0)
        assert index == 1  # Position after "class MyClass:"
        assert fuzz == 0
    
    def test_find_scope_not_found(self) -> None:
        """Test scope finding when scope doesn't exist."""
        lines = ["def other_function():", "    pass"]
        scope_lines = ["def nonexistent"]
        index, fuzz = _find_scope(lines, scope_lines, 0)
        assert index == -1
        assert fuzz == 0
    
    def test_find_scope_whitespace_tolerance(self) -> None:
        """Test scope finding with whitespace differences.
        
        Scope finding uses substring matching, so 'def indented_func' will
        match inside '  def indented_func():' without needing fuzz.
        """
        lines = ["  def indented_func():", "    pass"]
        scope_lines = ["def indented_func"]
        index, fuzz = _find_scope(lines, scope_lines, 0)
        assert index == 1
        assert fuzz == 0  # Substring match succeeds without fuzz
    
    def test_scope_in_update_patch(self, temp_workspace: Path, apply_patch_fn) -> None:
        """Test applying a patch with scope line to target specific function."""
        content = '''def first_func():
    x = 1
    return x

def second_func():
    x = 1
    return x
'''
        create_test_file(temp_workspace, "test.py", content)
        
        # Patch targets second_func specifically using scope
        # Note: The scope line narrows search context; hunk context is WITHIN that scope
        patch = """*** Begin Patch
*** Update File: test.py
@@ def second_func
-    x = 1
+    x = 42
     return x
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        new_content = (temp_workspace / "test.py").read_text()
        # First function should be unchanged
        assert "def first_func():\n    x = 1" in new_content
        # Second function should be modified
        assert "def second_func():\n    x = 42" in new_content
    
    def test_nested_scope(self, temp_workspace: Path, apply_patch_fn) -> None:
        """Test nested scope lines for class method."""
        content = '''class MyClass:
    def method_one(self):
        return 1
    
    def method_two(self):
        return 2

class OtherClass:
    def method_one(self):
        return 100
'''
        create_test_file(temp_workspace, "test.py", content)
        
        # Target method_one in OtherClass using nested scope
        # Note: Scope lines narrow search; hunk context is WITHIN that scope
        patch = """*** Begin Patch
*** Update File: test.py
@@ class OtherClass
@@ def method_one
-        return 100
+        return 999
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        new_content = (temp_workspace / "test.py").read_text()
        # MyClass method should be unchanged
        assert "class MyClass:" in new_content
        # OtherClass method should be modified
        assert "return 999" in new_content


# ---------------------------------------------------------------------------
# Tests for EOF Marker
# ---------------------------------------------------------------------------
class TestEOFMarker:
    def test_eof_marker_in_find_context(self) -> None:
        """Test EOF flag in _find_context prioritizes end of file."""
        lines = ["line 1", "line 2", "target", "line 3", "target"]
        old_lines = ["target"]
        
        # Without EOF, finds first occurrence
        index, fuzz = _find_context(lines, old_lines, 0, eof=False)
        assert index == 2
        
        # With EOF, finds last occurrence at end
        index, fuzz = _find_context(lines, old_lines, 0, eof=True)
        assert index == 4  # Last position
        assert fuzz == 0  # Exact match at end
    
    def test_eof_not_at_end_adds_fuzz(self) -> None:
        """Test that EOF marker adds fuzz penalty when match isn't at end."""
        lines = ["target", "other", "other2"]
        old_lines = ["target"]
        
        # With EOF but match not at end
        index, fuzz = _find_context(lines, old_lines, 0, eof=True)
        assert index == 0
        assert fuzz >= 10000  # Large penalty
    
    def test_eof_patch_application(self, temp_workspace: Path, apply_patch_fn) -> None:
        """Test applying a patch with EOF marker."""
        content = '''def func():
    pass

# End of file marker test
'''
        create_test_file(temp_workspace, "test.py", content)
        
        patch = """*** Begin Patch
*** Update File: test.py
@@
 # End of file marker test
+# New line at end
*** End of File
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        new_content = (temp_workspace / "test.py").read_text()
        assert "# New line at end" in new_content
    
    def test_eof_parsed_in_hunk(self) -> None:
        """Test that EOF marker is correctly parsed into hunk."""
        patch = """*** Begin Patch
*** Update File: test.py
@@
 context
+added
*** End of File
*** End Patch"""
        
        parsed = _parse_patch(patch)
        assert len(parsed.hunks) == 1
        assert parsed.hunks[0].is_eof is True


# ---------------------------------------------------------------------------
# Tests for Chunk-based Parsing
# ---------------------------------------------------------------------------
class TestChunkParsing:
    def test_parse_simple_chunk(self) -> None:
        """Test parsing a simple hunk into chunks."""
        hunk_lines = [
            " context 1",
            "-deleted",
            "+inserted",
            " context 2",
        ]
        
        context_lines, chunks = _parse_hunk_into_chunks(hunk_lines, "test.py")
        
        assert context_lines == ["context 1", "deleted", "context 2"]
        assert len(chunks) == 1
        assert chunks[0].orig_index == 1
        assert chunks[0].del_lines == ["deleted"]
        assert chunks[0].ins_lines == ["inserted"]
    
    def test_parse_multiple_chunks(self) -> None:
        """Test parsing hunk with multiple separate changes."""
        hunk_lines = [
            " context 1",
            "-del1",
            "+ins1",
            " context 2",
            " context 3",
            "-del2",
            "+ins2",
            " context 4",
        ]
        
        context_lines, chunks = _parse_hunk_into_chunks(hunk_lines, "test.py")
        
        assert len(chunks) == 2
        # First chunk
        assert chunks[0].orig_index == 1
        assert chunks[0].del_lines == ["del1"]
        assert chunks[0].ins_lines == ["ins1"]
        # Second chunk
        assert chunks[1].orig_index == 4
        assert chunks[1].del_lines == ["del2"]
        assert chunks[1].ins_lines == ["ins2"]
    
    def test_parse_pure_insertion(self) -> None:
        """Test parsing hunk with only insertions."""
        hunk_lines = [
            " context",
            "+new line 1",
            "+new line 2",
            " more context",
        ]
        
        context_lines, chunks = _parse_hunk_into_chunks(hunk_lines, "test.py")
        
        assert context_lines == ["context", "more context"]
        assert len(chunks) == 1
        assert chunks[0].del_lines == []
        assert chunks[0].ins_lines == ["new line 1", "new line 2"]
    
    def test_parse_pure_deletion(self) -> None:
        """Test parsing hunk with only deletions."""
        hunk_lines = [
            " context",
            "-removed 1",
            "-removed 2",
            " more context",
        ]
        
        context_lines, chunks = _parse_hunk_into_chunks(hunk_lines, "test.py")
        
        assert context_lines == ["context", "removed 1", "removed 2", "more context"]
        assert len(chunks) == 1
        assert chunks[0].del_lines == ["removed 1", "removed 2"]
        assert chunks[0].ins_lines == []


# ---------------------------------------------------------------------------
# Tests for Blank Line Handling
# ---------------------------------------------------------------------------
class TestBlankLines:
    def test_blank_line_without_prefix(self, temp_workspace: Path, apply_patch_fn) -> None:
        """Test that blank lines without prefix are treated as context."""
        content = '''def func():
    line 1

    line 2
'''
        create_test_file(temp_workspace, "test.py", content)
        
        # Patch with blank line (no prefix)
        patch = """*** Begin Patch
*** Update File: test.py
@@
 def func():
-    line 1
+    modified 1

     line 2
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        new_content = (temp_workspace / "test.py").read_text()
        assert "modified 1" in new_content
    
    def test_empty_line_in_hunk(self, temp_workspace: Path, apply_patch_fn) -> None:
        """Test truly empty line within hunk."""
        content = '''first

second
third
'''
        create_test_file(temp_workspace, "test.py", content)
        
        patch = """*** Begin Patch
*** Update File: test.py
@@
 first

-second
+modified
 third
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        new_content = (temp_workspace / "test.py").read_text()
        assert "modified" in new_content
        assert "second" not in new_content
    
    def test_parse_blank_lines_in_chunks(self) -> None:
        """Test chunk parsing handles blank lines correctly."""
        hunk_lines = [
            " context",
            "",  # Empty line
            "-deleted",
            "+inserted",
        ]
        
        context_lines, chunks = _parse_hunk_into_chunks(hunk_lines, "test.py")
        
        assert context_lines == ["context", "", "deleted"]
        assert len(chunks) == 1
        assert chunks[0].orig_index == 2  # After blank line


# ---------------------------------------------------------------------------
# Tests for detailed error context (Improvement 1)
# ---------------------------------------------------------------------------
class TestErrorContext:
    """Tests for _format_context_mismatch helper and detailed error messages."""
    
    def test_format_context_mismatch_basic(self) -> None:
        """Test basic formatting of context mismatch."""
        expected = ["line 1", "line 2", "line 3"]
        file_lines = ["actual 1", "actual 2", "actual 3", "actual 4"]
        
        result = _format_context_mismatch(expected, file_lines, 0)
        
        assert "Expected context:" in result
        assert "line 1" in result
        assert "line 2" in result
        assert "File content near line 1:" in result
        assert "actual 1" in result
    
    def test_format_context_mismatch_truncates_long_expected(self) -> None:
        """Test that long expected context is truncated."""
        expected = [f"line {i}" for i in range(10)]
        file_lines = ["actual"]
        
        result = _format_context_mismatch(expected, file_lines, 0, max_lines=3)
        
        assert "line 0" in result
        assert "line 1" in result
        assert "line 2" in result
        assert "... (7 more lines)" in result
    
    def test_format_context_mismatch_shows_line_numbers(self) -> None:
        """Test that file content includes line numbers."""
        expected = ["expected"]
        file_lines = ["line 1", "line 2", "line 3"]
        
        result = _format_context_mismatch(expected, file_lines, 1)
        
        assert "2:" in result  # Line number should be present
        assert "File content near line 2:" in result
    
    def test_context_error_includes_detail(self, temp_workspace: Path, apply_patch_fn) -> None:
        """Test that context mismatch errors include detailed information."""
        content = "actual line 1\nactual line 2\nactual line 3"
        create_test_file(temp_workspace, "test.py", content)
        
        patch = """*** Begin Patch
*** Update File: test.py
@@
 wrong context
-remove this
+add this
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "error"
        assert "hint" in result
        assert "Expected context:" in result["hint"]
        assert "File content near line" in result["hint"]


# ---------------------------------------------------------------------------
# Tests for overlap detection (Improvement 2)
# ---------------------------------------------------------------------------
class TestOverlapDetection:
    """Tests for overlapping or out-of-order hunk detection.
    
    The overlap detection validates that found_at >= search_start, which is
    a safety check for edge cases. In practice, _find_context searches from
    search_start, so this rarely triggers, but it protects against malformed
    patches in EOF mode where matching occurs at file end regardless of position.
    """
    
    def test_non_overlapping_hunks_succeed(self, temp_workspace: Path, apply_patch_fn) -> None:
        """Test that sequential non-overlapping hunks work correctly."""
        content = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7"
        create_test_file(temp_workspace, "test.py", content)
        
        # Two hunks that don't overlap - applied in file order
        # First hunk modifies line 2, second hunk modifies line 6
        patch = """*** Begin Patch
*** Update File: test.py
@@
 line 1
-line 2
+modified line 2
 line 3
@@
 line 5
-line 6
+modified line 6
 line 7
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        new_content = (temp_workspace / "test.py").read_text()
        assert "modified line 2" in new_content
        assert "modified line 6" in new_content


# ---------------------------------------------------------------------------
# Tests for improved scope matching (Improvement 3)
# ---------------------------------------------------------------------------
class TestScopeMatching:
    """Tests for improved scope signature matching."""
    
    def test_scope_startswith_preferred(self) -> None:
        """Test that startswith matching is preferred over substring."""
        lines = [
            "# def old_function  # commented out",
            "def new_function():",
            "    pass",
        ]
        scope_lines = ["def new_function"]
        
        # Should find "def new_function" at line 1, not match the comment at line 0
        pos, fuzz = _find_scope(lines, scope_lines, 0)
        
        assert pos == 2  # Position after the scope line
        assert fuzz == 0  # Should be exact (startswith) match
    
    def test_scope_does_not_match_comments(self) -> None:
        """Test that scope lines don't match commented-out code."""
        lines = [
            "# def foo():  # old implementation",
            "# def foo():  # another comment",
            "def foo():",
            "    return 42",
        ]
        scope_lines = ["def foo"]
        
        pos, fuzz = _find_scope(lines, scope_lines, 0)
        
        assert pos == 3  # Should match actual def, not comments
        assert fuzz == 0
    
    def test_scope_fallback_to_substring(self) -> None:
        """Test fallback to substring matching with fuzz penalty."""
        lines = [
            "some prefix def foo() some suffix",
            "    pass",
        ]
        scope_lines = ["def foo"]
        
        pos, fuzz = _find_scope(lines, scope_lines, 0)
        
        assert pos == 1  # Should still find it
        assert fuzz == 1  # But with fuzz penalty for substring match
    
    def test_scope_with_indentation(self, temp_workspace: Path, apply_patch_fn) -> None:
        """Test scope matching with indented scope signatures."""
        content = '''class MyClass:
    # def method():  # commented out
    def method(self):
        old_line = 1
        return old_line
'''
        create_test_file(temp_workspace, "test.py", content)
        
        patch = """*** Begin Patch
*** Update File: test.py
@@ def method
 old_line = 1
-return old_line
+return old_line * 2
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        new_content = (temp_workspace / "test.py").read_text()
        assert "return old_line * 2" in new_content


# ---------------------------------------------------------------------------
# Tests for line ending preservation (Improvement 4)
# ---------------------------------------------------------------------------
class TestLineEndingPreservation:
    """Tests for preserving original line ending style."""
    
    def test_preserves_lf_endings(self, temp_workspace: Path, apply_patch_fn) -> None:
        """Test that LF line endings are preserved."""
        content = "line 1\nline 2\nline 3\n"
        (temp_workspace / "test.py").write_bytes(content.encode("utf-8"))
        
        patch = """*** Begin Patch
*** Update File: test.py
@@
 line 1
-line 2
+modified line 2
 line 3
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        new_bytes = (temp_workspace / "test.py").read_bytes()
        assert b"\r\n" not in new_bytes  # No CRLF introduced
        assert b"\n" in new_bytes  # LF still present
    
    def test_preserves_crlf_endings(self, temp_workspace: Path, apply_patch_fn) -> None:
        """Test that CRLF line endings are preserved."""
        content = "line 1\r\nline 2\r\nline 3\r\n"
        (temp_workspace / "test.py").write_bytes(content.encode("utf-8"))
        
        patch = """*** Begin Patch
*** Update File: test.py
@@
 line 1
-line 2
+modified line 2
 line 3
*** End Patch"""
        
        result = parse_response(apply_patch_fn(patch))
        
        assert result["status"] == "ok"
        new_bytes = (temp_workspace / "test.py").read_bytes()
        assert b"\r\n" in new_bytes  # CRLF preserved
        # Count CRLF occurrences - should have them for all line endings
        crlf_count = new_bytes.count(b"\r\n")
        lf_only_count = new_bytes.count(b"\n") - crlf_count
        assert crlf_count >= 2  # At least 2 CRLF endings
        assert lf_only_count == 0  # No stray LF-only endings


# ---------------------------------------------------------------------------
# Tests for lenient sentinel parsing (Improvement 5)
# ---------------------------------------------------------------------------
class TestLenientParsing:
    """Tests for optional lenient sentinel parsing."""
    
    def test_strict_mode_requires_begin_sentinel(self) -> None:
        """Test that strict mode requires Begin Patch sentinel."""
        patch_text = """*** Update File: test.py
@@
 line 1
-old
+new
*** End Patch"""
        
        with pytest.raises(PatchError) as exc_info:
            _parse_patch(patch_text, strict=True)
        
        assert "Begin Patch" in str(exc_info.value)
    
    def test_strict_mode_requires_end_sentinel(self) -> None:
        """Test that strict mode requires End Patch sentinel."""
        patch_text = """*** Begin Patch
*** Update File: test.py
@@
 line 1
-old
+new"""
        
        with pytest.raises(PatchError) as exc_info:
            _parse_patch(patch_text, strict=True)
        
        assert "End Patch" in str(exc_info.value)
    
    def test_lenient_mode_parses_without_begin_sentinel(self) -> None:
        """Test that lenient mode parses without Begin Patch."""
        patch_text = """*** Update File: test.py
@@
 line 1
-old
+new
*** End Patch"""
        
        parsed = _parse_patch(patch_text, strict=False)
        
        assert parsed.op == "update"
        assert parsed.path == "test.py"
    
    def test_lenient_mode_parses_without_end_sentinel(self) -> None:
        """Test that lenient mode parses without End Patch."""
        patch_text = """*** Begin Patch
*** Update File: test.py
@@
 line 1
-old
+new"""
        
        parsed = _parse_patch(patch_text, strict=False)
        
        assert parsed.op == "update"
        assert parsed.path == "test.py"
    
    def test_lenient_mode_parses_without_any_sentinels(self) -> None:
        """Test that lenient mode parses without any sentinels."""
        patch_text = """*** Update File: test.py
@@
 line 1
-old
+new"""
        
        parsed = _parse_patch(patch_text, strict=False)
        
        assert parsed.op == "update"
        assert parsed.path == "test.py"
    
    def test_lenient_mode_add_file(self) -> None:
        """Test lenient mode with Add File operation."""
        patch_text = """*** Add File: new.py
+line 1
+line 2"""
        
        parsed = _parse_patch(patch_text, strict=False)
        
        assert parsed.op == "add"
        assert parsed.path == "new.py"
        assert parsed.add_content == "line 1\nline 2"
    
    def test_lenient_mode_rejects_non_patch_content(self) -> None:
        """Test that lenient mode still rejects non-patch content."""
        patch_text = """This is just some random text
with no patch markers at all
just regular content"""
        
        with pytest.raises(PatchError) as exc_info:
            _parse_patch(patch_text, strict=False)
        
        assert "does not appear to be a valid patch" in str(exc_info.value)
    
    def test_apply_patch_strict_parameter(self, temp_workspace: Path, patch_tool: ApplyPatchTool) -> None:
        """Test that apply_patch accepts strict parameter."""
        content = "line 1\nold\nline 3"
        create_test_file(temp_workspace, "test.py", content)
        
        # Patch without sentinels
        patch = """*** Update File: test.py
@@
 line 1
-old
+new
 line 3"""
        
        apply_fn = patch_tool.get_tool()
        
        # Should fail in strict mode (default)
        result = parse_response(apply_fn(patch))
        assert result["status"] == "error"
        
        # Should succeed in lenient mode
        result = parse_response(apply_fn(patch, strict=False))
        assert result["status"] == "ok"
        
        new_content = (temp_workspace / "test.py").read_text()
        assert "new" in new_content
