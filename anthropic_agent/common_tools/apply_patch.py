"""
### Spec for `apply_patch` in `anthropic_agent/common_tools/apply_patch.py`

- **Signature**
  - `def apply_patch(patch: str, dry_run: bool = False, strict: bool = True) -> str`

- **Purpose**
  - Apply changes to files using a patch format. Supports create, modify, delete, and rename.

- **Operations**
  - `*** Add File: path` - Create new file (all lines start with +)
  - `*** Update File: path` - Modify existing file (use @@ hunks)
  - `*** Delete File: path` - Remove file (no body content)
  - `*** Move to: new_path` - Rename/move during update

- **Hunk Prefixes**
  - ` ` (space) - Context line that must match
  - `-` - Line to remove
  - `+` - Line to add

- **Error Recovery Guidance**
  - "could not find matching context" -> Re-read file with read_file, content may have changed
  - "File already exists" -> Use Update File instead of Add File
  - "File does not exist" -> Use Add File instead of Update File
  - "scope signature not found" -> Check function/class name, re-read file

  - Paths escaping via `..` or absolute paths are rejected.
  - All output paths are normalized.

- **Allowed file extensions**
  - `.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.md`, `.mdx`, `.mmd`, `.json`, `.yaml`, `.yml`,
    `.toml`, `.txt`, `.csv`, `.html`, `.css`, `.scss`, `.sh`, `.bash`, `.zsh`,
    `.sql`, `.graphql`, `.xml`, `.ini`, `.cfg`, `.conf`, `.env`, `.gitignore`,
    `.dockerignore`, `.editorconfig`, `.prettierrc`, `.eslintrc`
  - Files without extensions are rejected.
  - Binary file extensions are rejected.

- **Binary protection**
  - Content containing null bytes (`\\x00`) is rejected.
  - Maximum file size after patch: 10 MB.
  - Maximum single patch size: 1 MB.

- **Operation semantics**
  - `Add File`: fails if file already exists.
  - `Update File`: fails if file does not exist.
  - `Delete File`: fails if file does not exist.
  - `Move to`: fails if target file already exists (unless same as source).
  - Exactly one file operation per patch call.

- **Hunk application algorithm (with fuzzy matching)**
  - Parse each hunk into "old lines" (context + removed) and "new lines" (context + added).
  - Search for old lines in the file content starting from the last match position.
  - Fuzzy matching is applied in order: exact match (fuzz=0), trailing whitespace
    tolerance (fuzz=1), full whitespace tolerance (fuzz=100).
  - Replace old lines with new lines.
  - If old lines cannot be found, fail with structured error.

- **Return format (JSON string)**
  - Success:
    ```json
    {
      "status": "ok",
      "op": "add" | "update" | "delete",
      "path": "relative/path",
      "hunks_applied": 3,
      "lines_added": 15,
      "lines_removed": 5,
      "dry_run": false,
      "fuzz_level": 0,
      "moved_from": "old/path" | null
    }
    ```
    Note: `fuzz_level` only present if >0, `moved_from` only present if file was moved.
  - Error:
    ```json
    {
      "status": "error",
      "error": "Description of error",
      "path": "relative/path" | null,
      "hint": "Suggestion to fix" | null
    }
    ```

- **dry_run mode**
  - When `dry_run=True`, parse and validate the patch, compute changes, but do not write.
  - Returns success/error as if the patch were applied, with `"dry_run": true`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from posixpath import normpath as _posix_normpath
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Tuple

from ..tools.base import ConfigurableToolBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_PATCH_SIZE_BYTES: int = 1 * 1024 * 1024  # 1 MB
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS: set[str] = {
    ".py", ".pyi", ".pyx",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".md", ".mdx", ".mmd", ".rst", ".txt",
    ".json", ".jsonc", ".json5",
    ".yaml", ".yml",
    ".toml",
    ".csv", ".tsv",
    ".html", ".htm", ".xml", ".svg",
    ".css", ".scss", ".sass", ".less",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql", ".gql",
    ".ini", ".cfg", ".conf", ".config",
    ".env", ".env.local", ".env.example",
    ".gitignore", ".gitattributes", ".gitmodules",
    ".dockerignore", ".dockerfile",
    ".editorconfig", ".prettierrc", ".eslintrc",
    ".makefile", ".cmake",
    ".r", ".R", ".rmd",
    ".java", ".kt", ".kts", ".scala",
    ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".swift", ".m", ".mm",
    ".rb", ".rake", ".gemspec",
    ".php", ".pl", ".pm",
    ".lua", ".vim", ".el",
    ".tf", ".hcl",
    ".proto",
}

# Regex patterns for parsing
RE_BEGIN_PATCH = re.compile(r"^\*\*\*\s*Begin\s+Patch\s*$", re.IGNORECASE)
RE_END_PATCH = re.compile(r"^\*\*\*\s*End\s+Patch\s*$", re.IGNORECASE)
RE_ADD_FILE = re.compile(r"^\*\*\*\s*Add\s+File:\s*(.+)$", re.IGNORECASE)
RE_UPDATE_FILE = re.compile(r"^\*\*\*\s*Update\s+File:\s*(.+)$", re.IGNORECASE)
RE_DELETE_FILE = re.compile(r"^\*\*\*\s*Delete\s+File:\s*(.+)$", re.IGNORECASE)
RE_MOVE_TO = re.compile(r"^\*\*\*\s*Move\s+to:\s*(.+)$", re.IGNORECASE)
RE_HUNK_HEADER = re.compile(r"^@@")
RE_EOF_MARKER = re.compile(r"^\*\*\*\s*End\s+of\s+File\s*$", re.IGNORECASE)
RE_PATCH_ACTION = re.compile(r"^\*\*\*\s*(Add|Update|Delete)\s+File:", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A single change point within a hunk.
    
    Tracks individual deletions and insertions at a specific position,
    enabling more precise change tracking with absolute line indices.
    """
    orig_index: int  # Line index relative to context start (converted to absolute during apply)
    del_lines: List[str] = field(default_factory=list)
    ins_lines: List[str] = field(default_factory=list)


class Hunk(NamedTuple):
    """Represents a single hunk in a patch."""
    header: str  # The @@ line (may be empty after @@)
    old_lines: List[str]  # Lines to find (context + removed, without prefix)
    new_lines: List[str]  # Lines to replace with (context + added, without prefix)
    lines_added: int
    lines_removed: int
    scope_lines: List[str] = []  # Scope context from @@ lines (e.g., def func, class Class)
    is_eof: bool = False  # Whether *** End of File marker is present
    chunks: List[Chunk] = []  # Fine-grained change points for precise tracking


class ParsedPatch(NamedTuple):
    """Represents a fully parsed patch."""
    op: Literal["add", "update", "delete"]
    path: str  # Relative POSIX path
    hunks: List[Hunk]
    add_content: Optional[str]  # For "add" operations, the full file content
    move_to: Optional[str] = None  # For "update" operations with rename/move


class PatchError(Exception):
    """Exception raised during patch parsing or application."""
    def __init__(self, error: str, path: Optional[str] = None, hint: Optional[str] = None):
        super().__init__(error)
        self.error = error
        self.path = path
        self.hint = hint


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------
def _is_within(child: Path, parent: Path) -> bool:
    """Check if a child path is within a parent directory."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _normalize_path(raw_path: str) -> str:
    """Normalize a path to POSIX style, stripping leading/trailing whitespace."""
    cleaned = raw_path.strip()
    # Convert backslashes to forward slashes
    cleaned = cleaned.replace("\\", "/")
    # Normalize (collapse . and ..)
    return _posix_normpath(cleaned)


def _has_allowed_extension(path: str, allowed_exts: set[str]) -> bool:
    """Check if the file has an allowed extension."""
    p = Path(path)
    # Check for compound extensions like .env.local
    name = p.name.lower()
    for ext in allowed_exts:
        if name.endswith(ext):
            return True
    # Check standard suffix
    suffix = p.suffix.lower()
    return suffix in allowed_exts


def _contains_null_bytes(content: str) -> bool:
    """Check if content contains null bytes (binary indicator)."""
    return "\x00" in content


def _norm(line: str) -> str:
    """Strip CR for CRLF/LF compatibility."""
    return line.rstrip("\r")


def _find_context(
    lines: List[str],
    old_lines: List[str],
    start: int,
    eof: bool = False,
) -> Tuple[int, int]:
    """Find old_lines in lines starting from start with fuzzy matching.
    
    Tries matching in order of decreasing strictness:
    1. Exact match (fuzz=0)
    2. Right-strip match (fuzz=1) - handles trailing whitespace
    3. Full-strip match (fuzz=100) - handles indentation changes
    
    When eof=True, prioritizes matching at end of file with large fuzz penalty
    if not found at exact end position.
    
    Args:
        lines: The file content split into lines.
        old_lines: The lines to search for.
        start: The starting index for the search.
        eof: If True, prioritize matching at end of file.
        
    Returns:
        Tuple of (index, fuzz_level) where index is the position found
        or (-1, 0) if not found.
    """
    n = len(old_lines)
    if n == 0:
        return start, 0
    
    # EOF mode: first try matching at end of file
    if eof:
        # We try two positions: 
        # 1. The absolute end of the lines list
        # 2. The position before a trailing empty line (result of file ending in \n)
        potential_end_positions = [len(lines) - n]
        if len(lines) > 0 and lines[-1] == "" and n > 0:
            potential_end_positions.append(len(lines) - n - 1)
        
        for end_pos in potential_end_positions:
            if end_pos >= 0:
                # Try exact match at end
                if lines[end_pos:end_pos + n] == old_lines:
                    return end_pos, 0
                # Try right-strip at end
                norm_old = [_norm(s).rstrip() for s in old_lines]
                if [_norm(s).rstrip() for s in lines[end_pos:end_pos + n]] == norm_old:
                    return end_pos, 1
                # Try full-strip at end
                norm_old_strip = [_norm(s).strip() for s in old_lines]
                if [_norm(s).strip() for s in lines[end_pos:end_pos + n]] == norm_old_strip:
                    return end_pos, 100
    
    # Level 0: Exact match
    for i in range(start, len(lines) - n + 1):
        if lines[i:i + n] == old_lines:
            # In EOF mode, add large penalty if not at end
            is_at_end = False
            if eof:
                if i == len(lines) - n:
                    is_at_end = True
                elif len(lines) > 0 and lines[-1] == "" and i == len(lines) - n - 1:
                    is_at_end = True
            
            fuzz = 10000 if eof and not is_at_end else 0
            return i, fuzz
    
    # Level 1: Right-strip match (trailing whitespace tolerance)
    norm_old = [_norm(s).rstrip() for s in old_lines]
    for i in range(start, len(lines) - n + 1):
        if [_norm(s).rstrip() for s in lines[i:i + n]] == norm_old:
            is_at_end = False
            if eof:
                if i == len(lines) - n:
                    is_at_end = True
                elif len(lines) > 0 and lines[-1] == "" and i == len(lines) - n - 1:
                    is_at_end = True

            fuzz = 10001 if eof and not is_at_end else 1
            return i, fuzz
    
    # Level 100: Full-strip match (leading/trailing whitespace tolerance)
    norm_old_strip = [_norm(s).strip() for s in old_lines]
    for i in range(start, len(lines) - n + 1):
        if [_norm(s).strip() for s in lines[i:i + n]] == norm_old_strip:
            is_at_end = False
            if eof:
                if i == len(lines) - n:
                    is_at_end = True
                elif len(lines) > 0 and lines[-1] == "" and i == len(lines) - n - 1:
                    is_at_end = True

            fuzz = 10100 if eof and not is_at_end else 100
            return i, fuzz
    
    return -1, 0


def _find_single_scope(
    lines: List[str],
    scope_pattern: str,
    start: int,
) -> Tuple[int, int]:
    """Find a single scope signature in file and return position after it.
    
    Args:
        lines: The file content split into lines.
        scope_pattern: The scope signature to search for (e.g., "def func_name").
        start: The starting index for the search.
        
    Returns:
        Tuple of (index_after_scope, fuzz_level) where index_after_scope is the
        position immediately after the scope signature, or (-1, 0) if not found.
    """
    scope_stripped = scope_pattern.strip()
    
    # Level 0: Match at line start (after stripping leading whitespace)
    for i in range(start, len(lines)):
        line_stripped = lines[i].lstrip()
        if line_stripped.startswith(scope_stripped):
            return i + 1, 0
    
    # Level 1: Fallback to substring match with fuzz penalty
    for i in range(start, len(lines)):
        if scope_stripped in lines[i]:
            return i + 1, 1
    
    return -1, 0


def _find_scope(
    lines: List[str],
    scope_lines: List[str],
    start: int,
) -> Tuple[int, int]:
    """Find scope signatures in file and return position after the last one.
    
    Scope lines narrow the search context to within a specific function/class.
    For nested scopes (e.g., class then method), each scope is searched
    sequentially from the position after the previous scope was found.
    
    Args:
        lines: The file content split into lines.
        scope_lines: The scope signature lines to search for (e.g., ["class Foo", "def bar"]).
        start: The starting index for the search.
        
    Returns:
        Tuple of (index_after_scope, fuzz_level) where index_after_scope is the
        position immediately after the last scope signature, or (-1, 0) if not found.
    """
    if not scope_lines:
        return start, 0
    
    current_pos = start
    total_fuzz = 0
    
    # Find each scope line sequentially, starting from where the previous one was found
    for scope_pattern in scope_lines:
        pos, fuzz = _find_single_scope(lines, scope_pattern, current_pos)
        if pos == -1:
            return -1, 0
        current_pos = pos
        total_fuzz += fuzz
    
    return current_pos, total_fuzz


def _format_context_mismatch(
    expected: List[str],
    file_lines: List[str],
    search_start: int,
    max_lines: int = 5,
) -> str:
    """Format detailed context mismatch error for diagnostics.
    
    Provides both the expected context lines and the actual file content
    near the search position to help diagnose why matching failed.
    
    Args:
        expected: The context lines that were being searched for.
        file_lines: The file content split into lines.
        search_start: The position in the file where search began.
        max_lines: Maximum number of lines to show in each preview.
        
    Returns:
        Formatted string with expected vs actual context for error messages.
    """
    expected_preview = "\n".join(f"  {s}" for s in expected[:max_lines])
    if len(expected) > max_lines:
        expected_preview += f"\n  ... ({len(expected) - max_lines} more lines)"
    
    nearby_start = max(0, search_start)
    nearby_end = min(len(file_lines), search_start + max_lines + 3)
    nearby_preview = "\n".join(
        f"  {i+1}: {file_lines[i]}" for i in range(nearby_start, nearby_end)
    )
    
    return f"Expected context:\n{expected_preview}\n\nFile content near line {search_start + 1}:\n{nearby_preview}"


def _parse_hunk_into_chunks(
    hunk_lines: List[str],
    file_path: str,
) -> Tuple[List[str], List[Chunk]]:
    """Parse hunk lines into context block and chunks for finer-grained tracking.
    
    Parses the `+`/`-`/` ` prefixed lines into:
    - context_lines: The original file content to match (context + deleted lines)
    - chunks: List of Chunk objects with relative orig_index and ins/del lines
    
    Args:
        hunk_lines: Lines from the hunk (with +/-/space prefixes).
        file_path: File path for error messages.
        
    Returns:
        Tuple of (context_lines, chunks) where context_lines is the full
        original content block and chunks contain individual change points.
        
    Raises:
        PatchError: If the hunk lines have invalid format.
    """
    context_lines: List[str] = []  # Original content (context + deleted)
    chunks: List[Chunk] = []
    
    current_chunk: Optional[Chunk] = None
    orig_index = 0  # Tracks position in original content
    
    for hl in hunk_lines:
        if not hl or hl.strip() == "":
            # Empty/blank line - treat as context
            context_lines.append("")
            if current_chunk is not None:
                # Close current chunk
                chunks.append(current_chunk)
                current_chunk = None
            orig_index += 1
            
        elif hl.startswith(" "):
            # Context line
            context_lines.append(hl[1:])
            if current_chunk is not None:
                # Close current chunk
                chunks.append(current_chunk)
                current_chunk = None
            orig_index += 1
            
        elif hl.startswith("-"):
            # Deleted line
            context_lines.append(hl[1:])
            if current_chunk is None:
                current_chunk = Chunk(orig_index=orig_index)
            current_chunk.del_lines.append(hl[1:])
            orig_index += 1
            
        elif hl.startswith("+"):
            # Inserted line
            if current_chunk is None:
                current_chunk = Chunk(orig_index=orig_index)
            current_chunk.ins_lines.append(hl[1:])
            # Note: insertions don't advance orig_index
            
        else:
            raise PatchError(
                f"Invalid hunk line: '{hl[:50]}...'",
                path=file_path,
                hint="Hunk lines must start with ' ', '+', or '-'"
            )
    
    # Close final chunk if any
    if current_chunk is not None:
        chunks.append(current_chunk)
    
    return context_lines, chunks


def _make_error_response(error: str, path: Optional[str] = None, hint: Optional[str] = None) -> str:
    """Create a JSON error response."""
    result = {"status": "error", "error": error}
    if path is not None:
        result["path"] = path
    if hint is not None:
        result["hint"] = hint
    return json.dumps(result)


def _make_success_response(
    op: str,
    path: str,
    hunks_applied: int,
    lines_added: int,
    lines_removed: int,
    dry_run: bool,
    fuzz_level: int = 0,
    moved_from: Optional[str] = None,
) -> str:
    """Create a JSON success response."""
    result: dict = {
        "status": "ok",
        "op": op,
        "path": path,
        "hunks_applied": hunks_applied,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "dry_run": dry_run,
    }
    # Only include fuzz_level if non-zero (indicates fuzzy matching was used)
    if fuzz_level > 0:
        result["fuzz_level"] = fuzz_level
    # Only include moved_from if file was moved/renamed
    if moved_from is not None:
        result["moved_from"] = moved_from
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------
def _parse_patch(patch_text: str, strict: bool = True) -> ParsedPatch:
    """Parse a Cursor-style patch envelope.
    
    Args:
        patch_text: The full patch text including envelope markers.
        strict: If True (default), require Begin/End Patch sentinels.
                If False, attempt to parse patch-like content without sentinels.
        
    Returns:
        ParsedPatch with operation type, path, hunks, and optional add content.
        
    Raises:
        PatchError: If the patch format is invalid.
    """
    lines = patch_text.split("\n")
    
    # Find envelope markers
    begin_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if RE_BEGIN_PATCH.match(line.strip()):
            begin_idx = i
        elif RE_END_PATCH.match(line.strip()):
            end_idx = i
            break
    
    # Lenient mode: if sentinels missing but content looks like a patch
    if not strict and (begin_idx is None or end_idx is None):
        has_action = any(RE_PATCH_ACTION.match(line.strip()) for line in lines)
        if has_action:
            # Only set missing sentinel positions, preserve found ones
            if begin_idx is None:
                begin_idx = -1  # Will add 1 to get 0
            if end_idx is None:
                end_idx = len(lines)
        else:
            raise PatchError(
                "Content does not appear to be a valid patch",
                hint="Missing sentinels and no file operations found"
            )
    elif begin_idx is None:
        raise PatchError("Missing '*** Begin Patch' marker", hint="Patch must start with '*** Begin Patch'")
    elif end_idx is None:
        raise PatchError("Missing '*** End Patch' marker", hint="Patch must end with '*** End Patch'")
    
    if end_idx <= begin_idx + 1:
        raise PatchError("Empty patch (no content between markers)")
    
    # Extract content between markers
    content_lines = lines[begin_idx + 1 : end_idx]
    
    # Find file operation header
    op: Optional[Literal["add", "update", "delete"]] = None
    file_path: Optional[str] = None
    op_line_idx: Optional[int] = None
    move_to: Optional[str] = None
    
    for i, line in enumerate(content_lines):
        add_match = RE_ADD_FILE.match(line.strip())
        if add_match:
            if op is not None:
                raise PatchError("Multiple file operations in patch", hint="Only one file operation per patch")
            op = "add"
            file_path = _normalize_path(add_match.group(1))
            op_line_idx = i
            continue
        
        update_match = RE_UPDATE_FILE.match(line.strip())
        if update_match:
            if op is not None:
                raise PatchError("Multiple file operations in patch", hint="Only one file operation per patch")
            op = "update"
            file_path = _normalize_path(update_match.group(1))
            op_line_idx = i
            continue
        
        delete_match = RE_DELETE_FILE.match(line.strip())
        if delete_match:
            if op is not None:
                raise PatchError("Multiple file operations in patch", hint="Only one file operation per patch")
            op = "delete"
            file_path = _normalize_path(delete_match.group(1))
            op_line_idx = i
            continue
        
        # Check for Move to directive (only valid after Update File)
        move_match = RE_MOVE_TO.match(line.strip())
        if move_match:
            if op != "update":
                raise PatchError(
                    "'*** Move to:' can only be used with '*** Update File:'",
                    hint="Move directive must follow an Update File operation"
                )
            if move_to is not None:
                raise PatchError("Multiple '*** Move to:' directives", hint="Only one move target per patch")
            move_to = _normalize_path(move_match.group(1))
            continue
    
    if op is None or file_path is None or op_line_idx is None:
        raise PatchError(
            "No file operation found",
            hint="Patch must contain '*** Add File:', '*** Update File:', or '*** Delete File:'"
        )
    
    # Get lines after the operation header (filtering out Move to directive if present)
    body_lines = [
        line for line in content_lines[op_line_idx + 1 :]
        if not RE_MOVE_TO.match(line.strip())
    ]
    
    # Handle Delete File operation
    if op == "delete":
        # Delete operations should have no body content
        non_empty_body = [line for line in body_lines if line.strip()]
        if non_empty_body:
            raise PatchError(
                "Delete File patch should not contain body content",
                path=file_path,
                hint="Remove any content after '*** Delete File:'"
            )
        return ParsedPatch(
            op="delete",
            path=file_path,
            hunks=[],
            add_content=None,
            move_to=None,
        )
    
    if op == "add":
        # For add operations, all lines must start with '+'
        add_content_lines: List[str] = []
        for i, line in enumerate(body_lines):
            if not line:
                # Empty line - treat as empty added line
                add_content_lines.append("")
            elif line.startswith("+"):
                add_content_lines.append(line[1:])
            else:
                raise PatchError(
                    f"Invalid line in Add File patch at line {i + 1}: must start with '+'",
                    path=file_path,
                    hint="All content lines in Add File must start with '+'"
                )
        
        add_content = "\n".join(add_content_lines)
        return ParsedPatch(
            op="add",
            path=file_path,
            hunks=[],
            add_content=add_content,
            move_to=None,
        )
    
    # For update operations, parse hunks
    hunks: List[Hunk] = []
    current_hunk_header: Optional[str] = None
    current_hunk_lines: List[str] = []
    current_scope_lines: List[str] = []
    current_is_eof: bool = False
    
    def _finalize_hunk():
        nonlocal current_hunk_header, current_hunk_lines, current_scope_lines, current_is_eof
        if current_hunk_header is None:
            return
        
        old_lines: List[str] = []
        new_lines: List[str] = []
        lines_added = 0
        lines_removed = 0
        
        for hl in current_hunk_lines:
            if not hl:
                # Empty line within hunk - treat as context
                old_lines.append("")
                new_lines.append("")
            elif hl.strip() == "":
                # Blank line without prefix (just whitespace) - treat as context
                old_lines.append("")
                new_lines.append("")
            elif hl.startswith(" "):
                # Context line
                old_lines.append(hl[1:])
                new_lines.append(hl[1:])
            elif hl.startswith("-"):
                # Removed line
                old_lines.append(hl[1:])
                lines_removed += 1
            elif hl.startswith("+"):
                # Added line
                new_lines.append(hl[1:])
                lines_added += 1
            else:
                raise PatchError(
                    f"Invalid hunk line: '{hl[:50]}...'",
                    path=file_path,
                    hint="Hunk lines must start with ' ', '+', or '-'"
                )
        
        hunks.append(Hunk(
            header=current_hunk_header,
            old_lines=old_lines,
            new_lines=new_lines,
            lines_added=lines_added,
            lines_removed=lines_removed,
            scope_lines=current_scope_lines,
            is_eof=current_is_eof,
            chunks=[],  # Chunks populated during apply if needed
        ))
        
        current_hunk_header = None
        current_hunk_lines = []
        current_scope_lines = []
        current_is_eof = False
    
    i = 0
    while i < len(body_lines):
        line = body_lines[i]
        
        if RE_HUNK_HEADER.match(line):
            # Finalize previous hunk if any
            _finalize_hunk()
            
            # Start new hunk - collect scope lines from consecutive @@ lines
            current_scope_lines = []
            
            # Extract scope text from this @@ line (everything after @@)
            scope_text = line[2:].strip()  # Skip "@@" prefix
            if scope_text:
                current_scope_lines.append(scope_text)
            
            current_hunk_header = line
            current_hunk_lines = []
            
            # Check for nested scope lines (@@  @@  content)
            i += 1
            while i < len(body_lines) and RE_HUNK_HEADER.match(body_lines[i]):
                # This is a nested scope line (e.g., @@ @@ method_name)
                nested_scope = body_lines[i][2:].strip()
                if nested_scope:
                    current_scope_lines.append(nested_scope)
                current_hunk_header = body_lines[i]  # Update header to latest
                i += 1
            continue  # Don't increment i again at end of loop
            
        elif RE_EOF_MARKER.match(line.strip()):
            # EOF marker - set flag for current hunk
            current_is_eof = True
            
        elif current_hunk_header is not None:
            # Inside a hunk
            current_hunk_lines.append(line)
            
        elif line.strip():
            # Non-empty line outside of hunk
            raise PatchError(
                f"Content outside of hunk: '{line[:50]}...'",
                path=file_path,
                hint="Update patches must have content inside @@ hunks"
            )
        
        i += 1
    
    # Finalize last hunk
    _finalize_hunk()
    
    if not hunks:
        raise PatchError(
            "No hunks found in Update File patch",
            path=file_path,
            hint="Update patches must contain at least one @@ hunk"
        )
    
    return ParsedPatch(
        op="update",
        path=file_path,
        hunks=hunks,
        add_content=None,
        move_to=move_to,
    )


# ---------------------------------------------------------------------------
# Hunk application
# ---------------------------------------------------------------------------
def _apply_hunks(content: str, hunks: List[Hunk], file_path: str) -> Tuple[str, int]:
    """Apply hunks to file content with fuzzy matching.
    
    Supports scope lines to narrow context search and EOF markers
    for end-of-file context matching. Preserves original line ending
    style (LF or CRLF).
    
    Args:
        content: Current file content.
        hunks: List of hunks to apply.
        file_path: File path for error messages.
        
    Returns:
        Tuple of (modified_content, total_fuzz_level).
        fuzz_level indicates how much whitespace normalization was needed:
        - 0: exact match
        - 1-99: trailing whitespace differences
        - 100+: indentation/leading whitespace differences
        - 10000+: EOF marker was used but context not at file end
        
    Raises:
        PatchError: If a hunk cannot be applied.
    """
    # Detect original line ending style and preserve it
    line_ending = "\r\n" if "\r\n" in content else "\n"
    
    # Normalize to \n for processing, then split
    lines = content.replace("\r\n", "\n").split("\n")
    
    search_start = 0
    total_fuzz = 0
    
    for hunk_idx, hunk in enumerate(hunks):
        old_lines = hunk.old_lines
        
        # Handle scope lines - narrow search to within specific function/class
        if hunk.scope_lines:
            scope_pos, scope_fuzz = _find_scope(lines, hunk.scope_lines, search_start)
            if scope_pos == -1:
                scope_preview = hunk.scope_lines[0][:60] if hunk.scope_lines else "(empty)"
                raise PatchError(
                    f"Hunk {hunk_idx + 1} failed: could not find scope signature",
                    path=file_path,
                    hint=f"Scope mismatch. Expected to find: '{scope_preview}'. Re-read the file to get current content."
                )
            search_start = scope_pos
            total_fuzz += scope_fuzz
        
        if not old_lines:
            # Pure addition hunk (no context or removed lines)
            # Insert at current position
            new_lines_to_insert = hunk.new_lines
            lines = lines[:search_start] + new_lines_to_insert + lines[search_start:]
            search_start += len(new_lines_to_insert)
            continue
        
        # Find the old_lines sequence using fuzzy matching
        # Pass EOF flag to prioritize end-of-file matching
        found_at, fuzz = _find_context(lines, old_lines, search_start, eof=hunk.is_eof)
        total_fuzz += fuzz
        
        if found_at == -1:
            eof_hint = " (with EOF marker)" if hunk.is_eof else ""
            detail = _format_context_mismatch(old_lines, lines, search_start)
            raise PatchError(
                f"Hunk {hunk_idx + 1} failed: could not find matching context{eof_hint}",
                path=file_path,
                hint=f"Context mismatch. Re-read the file to get current content.\n{detail}"
            )
        
        # Overlap detection: ensure hunks don't overlap or go backwards
        if found_at < search_start:
            raise PatchError(
                f"Hunk {hunk_idx + 1} failed: overlapping or out-of-order context",
                path=file_path,
                hint=f"Hunk context found at line {found_at + 1} but previous hunk ended at line {search_start + 1}. Ensure hunks are in file order and don't overlap."
            )
        
        # Replace old_lines with new_lines
        lines = lines[:found_at] + hunk.new_lines + lines[found_at + len(old_lines):]
        
        # Update search position to after the replacement
        search_start = found_at + len(hunk.new_lines)
    
    return line_ending.join(lines), total_fuzz


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class ApplyPatchTool(ConfigurableToolBase):
    """Configurable apply_patch tool with a sandboxed base path and templated docstrings.
    
    This class encapsulates the apply_patch functionality, allowing configuration
    of the base path at instantiation time. The tool returned by get_tool()
    can be registered with an AnthropicAgent.
    
    The docstring uses {placeholder} syntax that gets replaced with actual
    configured values at schema generation time.
    
    Example:
        >>> # Default usage - docstring reflects actual config
        >>> patch_tool = ApplyPatchTool(base_path="/path/to/workspace")
        >>> agent = AnthropicAgent(tools=[patch_tool.get_tool()])
        
        >>> # Custom limits
        >>> patch_tool = ApplyPatchTool(
        ...     base_path="/workspace",
        ...     max_patch_size_bytes=2 * 1024 * 1024,  # 2MB
        ...     max_file_size_bytes=20 * 1024 * 1024,  # 20MB
        ... )
    """
    
    DOCSTRING_TEMPLATE = """Apply changes to a file using a patch format.
            
            Use this tool to create, modify, or delete files. Supports fuzzy matching
            to tolerate minor whitespace differences when locating context.
            
            **Patch Format:**
            ```
            *** Begin Patch
            *** Update File: path/to/file.py
            @@ def function_name
             context line (unchanged, space prefix)
            -line to remove
            +line to add
             more context
            *** End Patch
            ```
            
            **Operations:**
            - `*** Add File: path` - Create new file (all lines must start with +)
            - `*** Update File: path` - Modify existing file (use @@ hunks)
            - `*** Delete File: path` - Remove file (no body content allowed)
            - `*** Move to: new_path` - Rename/move during update (optional, after Update File)
            
            **Hunk Prefixes:**
            - ` ` (space) - Context line that must match the file
            - `-` - Line to remove from the file
            - `+` - Line to add to the file
            
            **Scope Lines (narrow search):**
            - `@@ def function_name` - Search within this function
            - `@@ class ClassName` then `@@ def method` - Nested scope
            
            Args:
                patch: Complete patch text with Begin/End markers.
                dry_run: If True, validate without writing. Useful for testing patches.
                strict: If False, allows patches without Begin/End markers.
            
            Returns:
                JSON with result:
                - Success: `{{"status": "ok", "op": "update", "path": "...", "hunks_applied": 2, ...}}`
                - Error: `{{"status": "error", "error": "...", "hint": "..."}}`
            
            **Error Recovery:**
            - "could not find matching context" -> Re-read the file with read_file, the content may have changed
            - "File already exists" -> Use `*** Update File:` instead of `*** Add File:`
            - "File does not exist" -> Use `*** Add File:` instead of `*** Update File:`
            - "Invalid hunk line" -> Ensure all lines start with space, +, or -
            - "scope signature not found" -> Check function/class name spelling, re-read file
"""
    
    def __init__(
        self,
        base_path: str | Path,
        max_patch_size_bytes: int = 1 * 1024 * 1024,
        max_file_size_bytes: int = 10 * 1024 * 1024,
        allowed_extensions: set[str] | None = None,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the ApplyPatchTool with a base path and configurable limits.
        
        Args:
            base_path: The root directory that apply_patch operates within.
                       All file paths in patches must be relative to this directory.
            max_patch_size_bytes: Maximum size of a single patch. Defaults to 1MB.
            max_file_size_bytes: Maximum file size after patch application. Defaults to 10MB.
            allowed_extensions: Set of allowed file extensions (with leading dot).
                               Defaults to a comprehensive set of text/code extensions if None.
            docstring_template: Optional custom docstring template with {placeholder} syntax.
                               Available placeholders: {max_patch_size_mb}, {max_file_size_mb}.
            schema_override: Optional complete Anthropic tool schema dict for full control.
        """
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.base_path: Path = Path(base_path).resolve()
        self.max_patch_size: int = max_patch_size_bytes
        self.max_file_size: int = max_file_size_bytes
        self.allowed_extensions: set[str] = allowed_extensions or ALLOWED_EXTENSIONS
    
    def _get_template_context(self) -> Dict[str, Any]:
        """Return placeholder values for docstring template."""
        return {
            "max_patch_size_mb": self.max_patch_size // 1024 // 1024,
            "max_file_size_mb": self.max_file_size // 1024 // 1024,
        }
    
    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.
        
        Returns:
            A decorated apply_patch function that operates within the configured base_path.
            The docstring will reflect the actual configured limits.
        """
        instance = self
        
        def apply_patch(patch: str, dry_run: bool = False, strict: bool = True) -> str:
            """Placeholder docstring - replaced by template."""
            # Check patch size
            if len(patch.encode("utf-8")) > instance.max_patch_size:
                return _make_error_response(
                    f"Patch exceeds maximum size ({instance.max_patch_size // 1024 // 1024} MB)",
                    hint="Split into smaller patches"
                )
            
            # Check for null bytes in patch
            if _contains_null_bytes(patch):
                return _make_error_response(
                    "Patch contains null bytes (binary content not allowed)",
                    hint="Ensure patch contains only text content"
                )
            
            # Parse the patch
            try:
                parsed = _parse_patch(patch, strict=strict)
            except PatchError as e:
                return _make_error_response(e.error, e.path, e.hint)
            
            file_path = parsed.path
            
            # Security: check path doesn't escape base
            if file_path.startswith("/") or file_path.startswith(".."):
                return _make_error_response(
                    "Invalid path: must be relative and cannot start with '..'",
                    path=file_path,
                    hint="Use paths relative to the workspace root"
                )
            
            full_path = instance.base_path / file_path
            
            if not _is_within(full_path, instance.base_path):
                return _make_error_response(
                    "Path escapes base directory",
                    path=file_path,
                    hint="Path cannot use '..' to escape the workspace"
                )
            
            # Check file extension
            if not _has_allowed_extension(file_path, instance.allowed_extensions):
                return _make_error_response(
                    f"File extension not allowed for editing",
                    path=file_path,
                    hint="Only common text/code file extensions are supported"
                )
            
            # Handle Add File operation
            if parsed.op == "add":
                if full_path.exists():
                    return _make_error_response(
                        "Cannot add file: already exists",
                        path=file_path,
                        hint="Use '*** Update File:' to modify existing files"
                    )
                
                content = parsed.add_content or ""
                
                # Check content size
                if len(content.encode("utf-8")) > instance.max_file_size:
                    return _make_error_response(
                        f"Content exceeds maximum file size ({instance.max_file_size // 1024 // 1024} MB)",
                        path=file_path
                    )
                
                lines_added = content.count("\n") + (1 if content else 0)
                
                if not dry_run:
                    # Ensure parent directory exists
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(content, encoding="utf-8")
                
                return _make_success_response(
                    op="add",
                    path=file_path,
                    hunks_applied=0,
                    lines_added=lines_added,
                    lines_removed=0,
                    dry_run=dry_run,
                )
            
            # Handle Delete File operation
            if parsed.op == "delete":
                if not full_path.exists():
                    return _make_error_response(
                        "Cannot delete file: does not exist",
                        path=file_path,
                        hint="File may have already been deleted"
                    )
                
                if not full_path.is_file():
                    return _make_error_response(
                        "Cannot delete: path is not a regular file",
                        path=file_path,
                        hint="Only regular files can be deleted"
                    )
                
                # Count lines for stats
                try:
                    content = full_path.read_text(encoding="utf-8")
                    lines_removed = content.count("\n") + (1 if content else 0)
                except Exception:
                    lines_removed = 0
                
                if not dry_run:
                    full_path.unlink()
                
                return _make_success_response(
                    op="delete",
                    path=file_path,
                    hunks_applied=0,
                    lines_added=0,
                    lines_removed=lines_removed,
                    dry_run=dry_run,
                )
            
            # Handle Update File operation
            if not full_path.exists():
                return _make_error_response(
                    "Cannot update file: does not exist",
                    path=file_path,
                    hint="Use '*** Add File:' to create new files"
                )
            
            if not full_path.is_file():
                return _make_error_response(
                    "Path is not a regular file",
                    path=file_path
                )
            
            # Read current content (using read_bytes to preserve line endings)
            try:
                current_content = full_path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                return _make_error_response(
                    "Cannot read file: not valid UTF-8 text",
                    path=file_path,
                    hint="This tool only supports text files"
                )
            except Exception as e:
                return _make_error_response(
                    f"Cannot read file: {e}",
                    path=file_path
                )
            
            # Apply hunks
            try:
                new_content, fuzz_level = _apply_hunks(current_content, parsed.hunks, file_path)
            except PatchError as e:
                return _make_error_response(e.error, e.path, e.hint)
            
            # Check result size
            if len(new_content.encode("utf-8")) > instance.max_file_size:
                return _make_error_response(
                    f"Result exceeds maximum file size ({instance.max_file_size // 1024 // 1024} MB)",
                    path=file_path
                )
            
            # Calculate stats
            total_added = sum(h.lines_added for h in parsed.hunks)
            total_removed = sum(h.lines_removed for h in parsed.hunks)
            
            # Handle move_to if specified
            target_path = file_path
            moved_from: Optional[str] = None
            
            if parsed.move_to:
                move_to_path = parsed.move_to
                
                # Validate move_to path
                if move_to_path.startswith("/") or move_to_path.startswith(".."):
                    return _make_error_response(
                        "Invalid move target: must be relative and cannot start with '..'",
                        path=move_to_path,
                        hint="Use paths relative to the workspace root"
                    )
                
                full_move_path = instance.base_path / move_to_path
                
                if not _is_within(full_move_path, instance.base_path):
                    return _make_error_response(
                        "Move target path escapes base directory",
                        path=move_to_path,
                        hint="Path cannot use '..' to escape the workspace"
                    )
                
                if not _has_allowed_extension(move_to_path, instance.allowed_extensions):
                    return _make_error_response(
                        "Move target file extension not allowed",
                        path=move_to_path,
                        hint="Only common text/code file extensions are supported"
                    )
                
                # Check if target already exists (and is different from source)
                if full_move_path.exists() and full_move_path.resolve() != full_path.resolve():
                    return _make_error_response(
                        "Cannot move: target file already exists",
                        path=move_to_path,
                        hint="Delete the target file first or choose a different name"
                    )
                
                if not dry_run:
                    # Ensure parent directory exists for target
                    full_move_path.parent.mkdir(parents=True, exist_ok=True)
                    # Write to new location (using write_bytes to preserve line endings)
                    full_move_path.write_bytes(new_content.encode("utf-8"))
                    # Remove original file (only if different from target)
                    if full_path.resolve() != full_move_path.resolve():
                        full_path.unlink()
                
                target_path = move_to_path
                moved_from = file_path
            else:
                if not dry_run:
                    full_path.write_bytes(new_content.encode("utf-8"))
            
            return _make_success_response(
                op="update",
                path=target_path,
                hunks_applied=len(parsed.hunks),
                lines_added=total_added,
                lines_removed=total_removed,
                dry_run=dry_run,
                fuzz_level=fuzz_level,
                moved_from=moved_from,
            )
        
        return self._apply_schema(apply_patch)
