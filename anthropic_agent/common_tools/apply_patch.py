"""anthropic_agent/common_tools/apply_patch.py

### Tool spec (high level)

- Signature
  - ``def apply_patch(patch: str, dry_run: bool = False, strict: bool = True) -> str``

- Purpose
  - Apply changes to files using a patch format. Supports create, modify, delete, and rename.
  - Designed to be **safe** and **deterministic** when used by automated agents.

- Operations
  - ``*** Add File: path``    - Create a new file (body lines start with ``+``)
  - ``*** Update File: path`` - Modify an existing file (use ``@@`` hunks)
  - ``*** Delete File: path`` - Remove a file (no body content)
  - ``*** Move to: new_path`` - Rename/move during update

- Matching semantics
  - Hunks locate the *old* block (context + removed lines) in the current file.
  - Matching is allowed to be whitespace-fuzzy, but **must be unambiguous**.

    Matching modes (in order):
      1) exact match
      2) ignore trailing whitespace
      3) ignore all whitespace (collapse runs + trim)

  - If a hunk context matches multiple locations (for the first mode that yields any
    match), the operation fails with an "ambiguous match" error.

  - ``*** End of File`` marker on a hunk tells the matcher to prefer the match closest
    to EOF. This can disambiguate repeated blocks.

- Safety constraints
  - Path traversal / absolute paths are rejected.
  - Null bytes are rejected (patch and file content).
  - Only UTF-8 text is supported.
  - Default allowlist for file extensions / common config filenames.
  - Maximum patch size: 1 MB (configurable).
  - Maximum file size after patch: 10 MB (configurable).

- Output format
  - Returns a JSON string.

"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from posixpath import normpath as _posix_normpath
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from ..tools.base import ConfigurableToolBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_PATCH_SIZE_BYTES: int = 1 * 1024 * 1024  # 1 MB
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

UTF8_BOM: bytes = b"\xef\xbb\xbf"

# NOTE: This tool is intended for text editing. We keep a conservative default allowlist.
# Users of ApplyPatchTool can override/extend allowed_extensions and allowed_basenames.
ALLOWED_EXTENSIONS: set[str] = {
    ".py",
    ".pyi",
    ".pyx",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".md",
    ".mdx",
    ".mmd",
    ".rst",
    ".txt",
    ".json",
    ".jsonc",
    ".json5",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
    ".xml",
    ".svg",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".graphql",
    ".gql",
    ".ini",
    ".cfg",
    ".conf",
    ".config",
    ".env",
    ".env.local",
    ".env.example",
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".dockerignore",
    ".editorconfig",
    ".prettierrc",
    ".eslintrc",
    ".cmake",
    ".r",
    ".rmd",
    ".java",
    ".kt",
    ".kts",
    ".scala",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".swift",
    ".m",
    ".mm",
    ".rb",
    ".rake",
    ".gemspec",
    ".php",
    ".pl",
    ".pm",
    ".lua",
    ".vim",
    ".el",
    ".tf",
    ".hcl",
    ".proto",
}

# Common extensionless filenames that are typically text.
# (Still subject to UTF-8 + null-byte checks.)
ALLOWED_BASENAMES: set[str] = {
    "dockerfile",
    "makefile",
    "license",
    "notice",
    "authors",
    "copying",
    "changelog",
    "readme",
    "requirements",
    "pipfile",
    "gemfile",
}

# Common binary extensions to reject early when allowlist enforcement is disabled.
# (Even if a binary file sneaks past this list, we still reject null bytes and
# require valid UTF-8 decoding for updates.)
BINARY_EXTENSIONS: set[str] = {
    '.png',
    '.jpg',
    '.jpeg',
    '.gif',
    '.webp',
    '.bmp',
    '.ico',
    '.pdf',
    '.zip',
    '.tar',
    '.gz',
    '.bz2',
    '.xz',
    '.7z',
    '.rar',
    '.exe',
    '.dll',
    '.so',
    '.dylib',
    '.class',
    '.pyc',
    '.pyo',
    '.wasm',
    '.woff',
    '.woff2',
    '.ttf',
    '.otf',
    '.eot',
    '.mp3',
    '.wav',
    '.flac',
    '.mp4',
    '.mov',
    '.avi',
    '.mkv',
    '.sqlite',
    '.db',
    '.bin',
    '.dmg',
    '.iso',
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

_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A fine-grained change point within a hunk."""

    orig_index: int
    del_lines: List[str] = field(default_factory=list)
    ins_lines: List[str] = field(default_factory=list)


@dataclass
class Hunk:
    """Represents a single hunk in a patch."""

    header: str
    old_lines: List[str]
    new_lines: List[str]
    lines_added: int
    lines_removed: int
    scope_lines: List[str] = field(default_factory=list)
    is_eof: bool = False
    chunks: List[Chunk] = field(default_factory=list)


@dataclass
class ParsedPatch:
    """Represents a fully parsed patch."""

    op: Literal["add", "update", "delete"]
    path: str  # Relative POSIX path
    hunks: List[Hunk]
    add_content: Optional[str]
    move_to: Optional[str] = None


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
    """Normalize an incoming path to a POSIX-style relative path string."""

    cleaned = raw_path.strip()
    cleaned = cleaned.replace("\\", "/")
    cleaned = _posix_normpath(cleaned)
    return cleaned


def _validate_rel_path(path: str) -> None:
    """Validate that a normalized path is safe and relative."""

    if not path or path == ".":
        raise PatchError("Invalid path", path=path or None, hint="Path must be a non-empty relative file path")

    # Absolute paths and traversal.
    if path.startswith("/") or path.startswith(".."):
        raise PatchError(
            "Invalid path: must be relative and cannot start with '..'",
            path=path,
            hint="Use paths relative to the workspace root",
        )

    # Reject home expansion, drive letters, and other suspicious characters.
    if path.startswith("~"):
        raise PatchError("Invalid path", path=path, hint="'~' is not allowed in paths")
    if ":" in path:
        raise PatchError("Invalid path", path=path, hint="':' is not allowed in paths")
    if "\x00" in path:
        raise PatchError("Invalid path", path=path, hint="NUL bytes are not allowed in paths")

    # Must be a file path, not a directory path.
    if path.endswith("/"):
        raise PatchError("Invalid path", path=path, hint="Path must refer to a file, not end with '/'")



def _is_allowed_text_path(
    path: str,
    allowed_exts: set[str],
    allowed_basenames: set[str],
    *,
    enforce_allowlist: bool = True,
    binary_exts: Optional[set[str]] = None,
) -> bool:
    """Return True if the path is permitted for editing/creation.

    If ``enforce_allowlist`` is True:
      - Only paths matching ``allowed_basenames`` or ``allowed_exts`` are permitted.

    If ``enforce_allowlist`` is False:
      - All paths are permitted except those with known-binary extensions
        (``binary_exts`` / ``BINARY_EXTENSIONS``).

    Note:
      - Content-level checks (no NUL bytes + valid UTF-8) are still enforced for
        updates; this function is an early filter only.
    """

    p = Path(path)
    name_lower = p.name.lower()
    suffix = p.suffix.lower()

    if not enforce_allowlist:
        bin_exts = binary_exts or BINARY_EXTENSIONS
        if suffix and suffix in bin_exts:
            return False
        return True

    # Allow common extensionless text filenames.
    if name_lower in allowed_basenames:
        return True

    # Allow dotfiles / compound extensions via suffix matching.
    for ext in allowed_exts:
        if name_lower.endswith(ext):
            return True

    # Fallback: allow by standard suffix.
    return suffix in allowed_exts


def _contains_null_bytes_str(content: str) -> bool:
    return "\x00" in content


def _contains_null_bytes_bytes(content: bytes) -> bool:
    return b"\x00" in content


def _split_text_preserve_line_endings(text: str) -> Tuple[List[str], str]:
    """Split text into lines while preserving the dominant line ending style.

    Returns:
        (lines, line_ending)

    The returned ``lines`` are split on ``\n`` after normalization.
    The join operation ``line_ending.join(lines)`` reproduces the original
    line ending style (LF, CRLF, or CR).
    """

    if "\r\n" in text:
        return text.replace("\r\n", "\n").split("\n"), "\r\n"
    if "\r" in text:
        # Old Mac-style CR line endings.
        return text.replace("\r", "\n").split("\n"), "\r"
    return text.split("\n"), "\n"


def _decode_utf8_preserve_bom(data: bytes) -> Tuple[str, bytes]:
    """Decode UTF-8 while preserving a UTF-8 BOM if present."""

    if data.startswith(UTF8_BOM):
        return data[len(UTF8_BOM) :].decode("utf-8"), UTF8_BOM
    return data.decode("utf-8"), b""


def _atomic_write_bytes(path: Path, data: bytes, *, mode: Optional[int] = None) -> None:
    """Atomically replace ``path`` with ``data``.

    Writes to a temporary file in the same directory and then uses ``os.replace``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.apply_patch.", dir=str(path.parent))
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        if mode is not None:
            os.chmod(tmp_path, mode)

        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            # Best effort cleanup.
            pass


def _atomic_create_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    """Create ``path`` with ``data`` without overwriting an existing file.

    The creation of the target name is atomic (via hardlink) when supported.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.apply_patch.", dir=str(path.parent))
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        os.chmod(tmp_path, mode)

        try:
            os.link(tmp_path, path)  # fails if path already exists
        except FileExistsError:
            raise
        except OSError:
            # Fallback for filesystems that don't support hardlinks.
            # Use exclusive create (not fully atomic w.r.t. crash mid-write, but never overwrites).
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd2 = os.open(path, flags, mode)
            try:
                with os.fdopen(fd2, "wb") as f2:
                    f2.write(data)
                    f2.flush()
                    os.fsync(f2.fileno())
            finally:
                # tmp file removed below
                pass
        else:
            # Remove tmp file (target now exists).
            pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _make_error_response(error: str, path: Optional[str] = None, hint: Optional[str] = None) -> str:
    result: dict[str, Any] = {"status": "error", "error": error}
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
    result: dict[str, Any] = {
        "status": "ok",
        "op": op,
        "path": path,
        "hunks_applied": hunks_applied,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "dry_run": dry_run,
    }
    if fuzz_level > 0:
        result["fuzz_level"] = fuzz_level
    if moved_from is not None:
        result["moved_from"] = moved_from
    return json.dumps(result)


def _format_context_mismatch(
    expected: Sequence[str],
    file_lines: Sequence[str],
    search_start: int,
    max_lines: int = 5,
) -> str:
    """Format mismatch details for diagnostics."""

    expected_preview = "\n".join(f"  {s}" for s in expected[:max_lines])
    if len(expected) > max_lines:
        expected_preview += f"\n  ... ({len(expected) - max_lines} more lines)"

    nearby_start = max(0, search_start)
    nearby_end = min(len(file_lines), search_start + max_lines + 3)
    nearby_preview = "\n".join(f"  {i + 1}: {file_lines[i]}" for i in range(nearby_start, nearby_end))

    return (
        f"Expected context:\n{expected_preview}\n\n"
        f"File content near line {search_start + 1}:\n{nearby_preview}"
    )


def _format_ambiguous_locations(
    file_lines: Sequence[str],
    positions: Sequence[int],
    *,
    block_len: int,
    max_candidates: int = 5,
) -> str:
    """Format a short summary of ambiguous match candidates."""

    parts: List[str] = []
    for pos in list(positions)[:max_candidates]:
        first_line = file_lines[pos] if 0 <= pos < len(file_lines) else ""
        parts.append(f"  - line {pos + 1}: {first_line[:200]}")

    remaining = len(positions) - max_candidates
    if remaining > 0:
        parts.append(f"  ... and {remaining} more matches")

    # Include a tiny preview of the matched block length.
    parts.append(f"  (matched block length: {block_len} lines)")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Matching utilities (unambiguous matching)
# ---------------------------------------------------------------------------

def _collapse_ws(line: str) -> str:
    """Normalize a line by collapsing all whitespace runs and trimming."""

    # Collapse tabs/spaces/etc to a single space, then trim.
    return _WS_RE.sub(" ", line).strip()


def _collect_matching_positions(lines: Sequence[str], pattern: Sequence[str], start: int) -> List[int]:
    """Return all indices >= start where pattern matches lines."""

    n = len(pattern)
    if n == 0:
        return [start]

    if start < 0:
        start = 0

    max_i = len(lines) - n
    if max_i < start:
        return []

    first = pattern[0]
    positions: List[int] = []

    for i in range(start, max_i + 1):
        if lines[i] != first:
            continue
        for j in range(1, n):
            if lines[i + j] != pattern[j]:
                break
        else:
            positions.append(i)

    return positions


def _is_at_eof(lines: Sequence[str], start: int, n: int) -> bool:
    """Return True if the block (start..start+n) is at EOF (with or without trailing newline)."""

    if start < 0:
        return False

    if start + n == len(lines):
        return True

    # If file ends with a trailing empty element (common when content ends with newline),
    # then a match that ends one element earlier is also logically "at EOF".
    if len(lines) > 0 and lines[-1] == "" and start + n == len(lines) - 1:
        return True

    return False


def _find_context(
    lines: List[str],
    old_lines: List[str],
    start: int,
    *,
    eof: bool,
    file_path: str,
    hunk_idx: int,
) -> Tuple[int, int]:
    """Find an unambiguous match for ``old_lines`` in ``lines``.

    Returns:
        (index, fuzz_level)

    Raises:
        PatchError on ambiguous matches.
    """

    n = len(old_lines)
    if n == 0:
        return start, 0

    # Matching modes in strictness order.
    # fuzz_level indicates the normalization used.
    modes: List[Tuple[str, int]] = [
        ("exact", 0),
        ("rstrip", 1),
        ("collapse_ws", 100),
    ]

    # Precompute candidate EOF positions for fast success in EOF mode.
    eof_positions: List[int] = []
    if eof:
        eof_positions.append(len(lines) - n)
        if len(lines) > 0 and lines[-1] == "":
            eof_positions.append(len(lines) - n - 1)
        # Keep only valid positions and those >= start.
        eof_positions = sorted({p for p in eof_positions if 0 <= p <= len(lines) - n and p >= start})

    for mode, fuzz in modes:
        if mode == "exact":
            haystack = lines
            needle = old_lines
        elif mode == "rstrip":
            haystack = [ln.rstrip() for ln in lines]
            needle = [ln.rstrip() for ln in old_lines]
        elif mode == "collapse_ws":
            haystack = [_collapse_ws(ln) for ln in lines]
            needle = [_collapse_ws(ln) for ln in old_lines]
        else:
            # Should never happen.
            continue

        # EOF mode: if we can match at EOF, prefer it immediately.
        if eof and eof_positions:
            matches_at_eof = [p for p in eof_positions if haystack[p : p + n] == needle]
            if matches_at_eof:
                # Choose the match closest to the true end.
                best = max(matches_at_eof)
                return best, fuzz

        positions = _collect_matching_positions(haystack, needle, start)

        if not positions:
            continue

        if eof:
            # EOF marker disambiguates: choose the match closest to EOF.
            best = max(positions)
            penalty = 0 if _is_at_eof(lines, best, n) else 10000
            return best, fuzz + penalty

        # Non-EOF: must be unambiguous.
        if len(positions) == 1:
            return positions[0], fuzz

        # Ambiguous.
        preview = _format_ambiguous_locations(lines, positions, block_len=n)
        raise PatchError(
            f"Hunk {hunk_idx + 1} failed: ambiguous context match ({len(positions)} matches) using mode '{mode}'",
            path=file_path,
            hint=(
                "The hunk's context matches multiple locations in the file. "
                "Add more unique context lines to the hunk or use @@ scope lines to narrow the search.\n"
                f"Candidate locations:\n{preview}"
            ),
        )

    return -1, 0


def _find_single_scope(
    lines: Sequence[str],
    scope_pattern: str,
    start: int,
    *,
    file_path: str,
    hunk_idx: int,
) -> Tuple[int, int]:
    """Find a unique scope signature and return the position after it."""

    wanted = scope_pattern.strip()
    if not wanted:
        return start, 0

    # Prefer 'line startswith' after lstrip.
    prefix_hits = [i for i in range(start, len(lines)) if lines[i].lstrip().startswith(wanted)]
    if len(prefix_hits) == 1:
        return prefix_hits[0] + 1, 0
    if len(prefix_hits) > 1:
        preview = _format_ambiguous_locations(lines, prefix_hits, block_len=1)
        raise PatchError(
            f"Hunk {hunk_idx + 1} failed: ambiguous scope signature",
            path=file_path,
            hint=(
                f"Scope line '{wanted}' matched multiple locations. "
                "Make the scope line more specific (e.g., include full function signature) or use nested scopes.\n"
                f"Candidate locations:\n{preview}"
            ),
        )

    # Fallback to substring match (fuzz=1).
    substring_hits = [i for i in range(start, len(lines)) if wanted in lines[i]]
    if len(substring_hits) == 1:
        return substring_hits[0] + 1, 1
    if len(substring_hits) > 1:
        preview = _format_ambiguous_locations(lines, substring_hits, block_len=1)
        raise PatchError(
            f"Hunk {hunk_idx + 1} failed: ambiguous scope signature",
            path=file_path,
            hint=(
                f"Scope line '{wanted}' matched multiple locations (substring match). "
                "Make the scope line more specific.\n"
                f"Candidate locations:\n{preview}"
            ),
        )

    return -1, 0


def _find_scope(
    lines: Sequence[str],
    scope_lines: Sequence[str],
    start: int,
    *,
    file_path: str,
    hunk_idx: int,
) -> Tuple[int, int]:
    """Find nested scope signatures sequentially and return position after last."""

    if not scope_lines:
        return start, 0

    pos = start
    max_fuzz = 0

    for scope in scope_lines:
        next_pos, fuzz = _find_single_scope(lines, scope, pos, file_path=file_path, hunk_idx=hunk_idx)
        if next_pos == -1:
            return -1, 0
        pos = next_pos
        if fuzz > max_fuzz:
            max_fuzz = fuzz

    return pos, max_fuzz


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------

def _parse_patch(patch_text: str, strict: bool = True) -> ParsedPatch:
    """Parse a Cursor-style patch envelope."""

    # Normalize line endings for parsing.
    # (Patch content is text; CR-only is treated as line breaks.)
    normalized = patch_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    # Find envelope markers
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if RE_BEGIN_PATCH.match(line.strip()):
            begin_idx = i
        elif RE_END_PATCH.match(line.strip()):
            end_idx = i
            break

    # Lenient mode: accept missing sentinels if content looks like a patch.
    if not strict and (begin_idx is None or end_idx is None):
        has_action = any(RE_PATCH_ACTION.match(line.strip()) for line in lines)
        if has_action:
            if begin_idx is None:
                begin_idx = -1
            if end_idx is None:
                end_idx = len(lines)
        else:
            raise PatchError("Content does not appear to be a valid patch", hint="Missing sentinels and no file operations found")
    elif begin_idx is None:
        raise PatchError("Missing '*** Begin Patch' marker", hint="Patch must start with '*** Begin Patch'")
    elif end_idx is None:
        raise PatchError("Missing '*** End Patch' marker", hint="Patch must end with '*** End Patch'")

    if end_idx <= begin_idx + 1:
        raise PatchError("Empty patch (no content between markers)")

    content_lines = lines[begin_idx + 1 : end_idx]

    op: Optional[Literal["add", "update", "delete"]] = None
    file_path: Optional[str] = None
    op_line_idx: Optional[int] = None
    move_to: Optional[str] = None

    for i, line in enumerate(content_lines):
        s = line.strip()

        add_match = RE_ADD_FILE.match(s)
        if add_match:
            if op is not None:
                raise PatchError("Multiple file operations in patch", hint="Only one file operation per patch")
            op = "add"
            file_path = _normalize_path(add_match.group(1))
            op_line_idx = i
            continue

        update_match = RE_UPDATE_FILE.match(s)
        if update_match:
            if op is not None:
                raise PatchError("Multiple file operations in patch", hint="Only one file operation per patch")
            op = "update"
            file_path = _normalize_path(update_match.group(1))
            op_line_idx = i
            continue

        delete_match = RE_DELETE_FILE.match(s)
        if delete_match:
            if op is not None:
                raise PatchError("Multiple file operations in patch", hint="Only one file operation per patch")
            op = "delete"
            file_path = _normalize_path(delete_match.group(1))
            op_line_idx = i
            continue

        move_match = RE_MOVE_TO.match(s)
        if move_match:
            if op != "update":
                raise PatchError("'*** Move to:' can only be used with '*** Update File:'", hint="Move directive must follow an Update File operation")
            if move_to is not None:
                raise PatchError("Multiple '*** Move to:' directives", hint="Only one move target per patch")
            move_to = _normalize_path(move_match.group(1))
            continue

    if op is None or file_path is None or op_line_idx is None:
        raise PatchError("No file operation found", hint="Patch must contain '*** Add File:', '*** Update File:', or '*** Delete File:'")

    # Validate paths early.
    _validate_rel_path(file_path)
    if move_to is not None:
        _validate_rel_path(move_to)

    # Filter out Move to directive lines from body.
    body_lines = [line for line in content_lines[op_line_idx + 1 :] if not RE_MOVE_TO.match(line.strip())]

    if op == "delete":
        non_empty = [ln for ln in body_lines if ln.strip()]
        if non_empty:
            raise PatchError("Delete File patch should not contain body content", path=file_path, hint="Remove any content after '*** Delete File:'")
        return ParsedPatch(op="delete", path=file_path, hunks=[], add_content=None, move_to=None)

    if op == "add":
        add_lines: List[str] = []
        for i, line in enumerate(body_lines):
            if line == "":
                add_lines.append("")
                continue
            if line.startswith("+"):
                add_lines.append(line[1:])
                continue
            raise PatchError(
                f"Invalid line in Add File patch at line {i + 1}: must start with '+'",
                path=file_path,
                hint="All content lines in Add File must start with '+'",
            )

        return ParsedPatch(op="add", path=file_path, hunks=[], add_content="\n".join(add_lines), move_to=None)

    # Update: parse hunks.
    hunks: List[Hunk] = []
    current_header: Optional[str] = None
    current_lines: List[str] = []
    current_scope: List[str] = []
    current_is_eof = False

    def finalize_hunk() -> None:
        nonlocal current_header, current_lines, current_scope, current_is_eof

        if current_header is None:
            return

        old_lines: List[str] = []
        new_lines: List[str] = []
        lines_added = 0
        lines_removed = 0

        for hl in current_lines:
            # Treat a raw blank line as an empty context line.
            if hl == "":
                old_lines.append("")
                new_lines.append("")
                continue

            if hl.startswith(" "):
                old_lines.append(hl[1:])
                new_lines.append(hl[1:])
            elif hl.startswith("-"):
                old_lines.append(hl[1:])
                lines_removed += 1
            elif hl.startswith("+"):
                new_lines.append(hl[1:])
                lines_added += 1
            else:
                raise PatchError(
                    f"Invalid hunk line: '{hl[:50]}...'",
                    path=file_path,
                    hint="Hunk lines must start with ' ', '+', or '-'",
                )

        hunks.append(
            Hunk(
                header=current_header,
                old_lines=old_lines,
                new_lines=new_lines,
                lines_added=lines_added,
                lines_removed=lines_removed,
                scope_lines=current_scope,
                is_eof=current_is_eof,
            )
        )

        current_header = None
        current_lines = []
        current_scope = []
        current_is_eof = False

    i = 0
    while i < len(body_lines):
        line = body_lines[i]

        if RE_HUNK_HEADER.match(line):
            finalize_hunk()

            current_scope = []
            scope_text = line[2:].strip()
            if scope_text:
                current_scope.append(scope_text)

            current_header = line
            current_lines = []

            i += 1
            while i < len(body_lines) and RE_HUNK_HEADER.match(body_lines[i]):
                nested = body_lines[i][2:].strip()
                if nested:
                    current_scope.append(nested)
                current_header = body_lines[i]
                i += 1
            continue

        if RE_EOF_MARKER.match(line.strip()):
            current_is_eof = True
            i += 1
            continue

        if current_header is not None:
            current_lines.append(line)
        elif line.strip():
            raise PatchError(
                f"Content outside of hunk: '{line[:50]}...'",
                path=file_path,
                hint="Update patches must have content inside @@ hunks",
            )

        i += 1

    finalize_hunk()

    if not hunks:
        raise PatchError("No hunks found in Update File patch", path=file_path, hint="Update patches must contain at least one @@ hunk")

    return ParsedPatch(op="update", path=file_path, hunks=hunks, add_content=None, move_to=move_to)


# ---------------------------------------------------------------------------
# Hunk application
# ---------------------------------------------------------------------------

def _apply_hunks(content: str, hunks: List[Hunk], file_path: str) -> Tuple[str, int]:
    """Apply hunks to file content with unambiguous fuzzy matching."""

    lines, line_ending = _split_text_preserve_line_endings(content)

    search_start = 0
    max_fuzz = 0

    for hunk_idx, hunk in enumerate(hunks):
        # Narrow search using scopes if provided.
        if hunk.scope_lines:
            scope_pos, scope_fuzz = _find_scope(lines, hunk.scope_lines, search_start, file_path=file_path, hunk_idx=hunk_idx)
            if scope_pos == -1:
                scope_preview = hunk.scope_lines[0][:80] if hunk.scope_lines else "(empty)"
                raise PatchError(
                    f"Hunk {hunk_idx + 1} failed: could not find scope signature",
                    path=file_path,
                    hint=f"Scope mismatch. Expected to find: '{scope_preview}'. Re-read the file to get current content.",
                )
            search_start = scope_pos
            max_fuzz = max(max_fuzz, scope_fuzz)

        if not hunk.old_lines:
            # Pure insertion hunk.
            insert_at = search_start
            # EOF marker can disambiguate insertion at file end.
            if hunk.is_eof and not hunk.scope_lines:
                insert_at = len(lines) - 1 if len(lines) > 0 and lines[-1] == "" else len(lines)
            lines = list(lines[:insert_at]) + hunk.new_lines + list(lines[insert_at:])
            search_start = insert_at + len(hunk.new_lines)
            continue

        found_at, fuzz = _find_context(lines, hunk.old_lines, search_start, eof=hunk.is_eof, file_path=file_path, hunk_idx=hunk_idx)
        max_fuzz = max(max_fuzz, fuzz)

        if found_at == -1:
            eof_hint = " (with EOF marker)" if hunk.is_eof else ""
            detail = _format_context_mismatch(hunk.old_lines, lines, search_start)
            raise PatchError(
                f"Hunk {hunk_idx + 1} failed: could not find matching context{eof_hint}",
                path=file_path,
                hint=f"Context mismatch. Re-read the file to get current content.\n{detail}",
            )

        if found_at < search_start:
            raise PatchError(
                f"Hunk {hunk_idx + 1} failed: overlapping or out-of-order context",
                path=file_path,
                hint=(
                    f"Hunk context found at line {found_at + 1} but previous hunk ended at line {search_start + 1}. "
                    "Ensure hunks are in file order and don't overlap."
                ),
            )

        # Replace the matched block.
        lines = list(lines[:found_at]) + hunk.new_lines + list(lines[found_at + len(hunk.old_lines) :])
        search_start = found_at + len(hunk.new_lines)

    return line_ending.join(lines), max_fuzz


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------


class ApplyPatchTool(ConfigurableToolBase):
    """Configurable apply_patch tool.

    This class encapsulates the apply_patch functionality, allowing configuration
    of:
    - base path / sandbox root
    - size limits
    - allowed file types

    The function returned by get_tool() is intended to be registered as an agent tool.
    """

    DOCSTRING_TEMPLATE = """Apply changes to a file using a patch format.

Use this tool to create, modify, delete, or rename files. Matching is allowed to be
whitespace-fuzzy but must be **unambiguous**.

Patch Format:
```
*** Begin Patch
*** Update File: path/to/file.py
@@ def function_name
 context line (space prefix)
-line to remove
+line to add
*** End Patch
```

Args:
    patch: Complete patch text with Begin/End markers.
    dry_run: If True, validate without writing.
    strict: If False, allows patches without Begin/End markers.

Returns:
    JSON with result:
    - Success: {{"status": "ok", "op": "update", "path": "...", ...}}
    - Error:   {{"status": "error", "error": "...", "hint": "..."}}
"""

    def __init__(
        self,
        base_path: str | Path,
        max_patch_size_bytes: int = MAX_PATCH_SIZE_BYTES,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
        allowed_extensions: Optional[set[str]] = None,
        allowed_basenames: Optional[set[str]] = None,
        enforce_allowlist: bool = True,
        docstring_template: Optional[str] = None,
        schema_override: Optional[dict] = None,
    ):
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.base_path = Path(base_path).resolve()
        self.max_patch_size = int(max_patch_size_bytes)
        self.max_file_size = int(max_file_size_bytes)
        self.allowed_extensions = allowed_extensions or set(ALLOWED_EXTENSIONS)
        self.allowed_basenames = allowed_basenames or set(ALLOWED_BASENAMES)
        self.enforce_allowlist = bool(enforce_allowlist)

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "max_patch_size_mb": self.max_patch_size // 1024 // 1024,
            "max_file_size_mb": self.max_file_size // 1024 // 1024,
        }

    def get_tool(self) -> Callable:
        instance = self

        def apply_patch(patch: str, dry_run: bool = False, strict: bool = True) -> str:
            """Placeholder docstring - replaced by template."""

            # Size / binary checks for the incoming patch text.
            if len(patch.encode("utf-8")) > instance.max_patch_size:
                return _make_error_response(
                    f"Patch exceeds maximum size ({instance.max_patch_size // 1024 // 1024} MB)",
                    hint="Split into smaller patches",
                )

            if _contains_null_bytes_str(patch):
                return _make_error_response(
                    "Patch contains null bytes (binary content not allowed)",
                    hint="Ensure patch contains only text content",
                )

            # Parse patch.
            try:
                parsed = _parse_patch(patch, strict=strict)
            except PatchError as e:
                return _make_error_response(e.error, e.path, e.hint)

            # Validate & resolve paths.
            try:
                _validate_rel_path(parsed.path)
                if parsed.move_to is not None:
                    _validate_rel_path(parsed.move_to)
            except PatchError as e:
                return _make_error_response(e.error, e.path, e.hint)

            file_path = parsed.path
            full_path = (instance.base_path / file_path).resolve()

            if not _is_within(full_path, instance.base_path):
                return _make_error_response(
                    "Path escapes base directory",
                    path=file_path,
                    hint="Path cannot use '..' to escape the workspace",
                )

            # File type guard. By default we enforce an allowlist of common text/code
            # extensions, but this can be relaxed via ApplyPatchTool(..., enforce_allowlist=False).
            if not _is_allowed_text_path(
                file_path,
                instance.allowed_extensions,
                instance.allowed_basenames,
                enforce_allowlist=instance.enforce_allowlist,
            ):
                if instance.enforce_allowlist:
                    return _make_error_response(
                        "File type not allowed for editing",
                        path=file_path,
                        hint=(
                            "This tool only edits common text/code files by default. "
                            "Configure ApplyPatchTool(allowed_extensions=..., allowed_basenames=...) to extend."
                        ),
                    )
                return _make_error_response(
                    "File extension appears to be binary and is not allowed",
                    path=file_path,
                    hint="This tool only supports UTF-8 text files (no NUL bytes).",
                )

            # Add file.
            if parsed.op == "add":
                if full_path.exists():
                    return _make_error_response(
                        "Cannot add file: already exists",
                        path=file_path,
                        hint="Use '*** Update File:' to modify existing files",
                    )

                content = parsed.add_content or ""

                if len(content.encode("utf-8")) > instance.max_file_size:
                    return _make_error_response(
                        f"Content exceeds maximum file size ({instance.max_file_size // 1024 // 1024} MB)",
                        path=file_path,
                    )

                lines_added = content.count("\n") + (1 if content else 0)

                if not dry_run:
                    try:
                        _atomic_create_bytes(full_path, content.encode("utf-8"))
                    except FileExistsError:
                        return _make_error_response(
                            "Cannot add file: already exists",
                            path=file_path,
                            hint="Use '*** Update File:' to modify existing files",
                        )
                    except Exception as e:
                        return _make_error_response(f"Failed to create file: {e}", path=file_path)

                return _make_success_response(
                    op="add",
                    path=file_path,
                    hunks_applied=0,
                    lines_added=lines_added,
                    lines_removed=0,
                    dry_run=dry_run,
                )

            # Delete file.
            if parsed.op == "delete":
                if not full_path.exists():
                    return _make_error_response(
                        "Cannot delete file: does not exist",
                        path=file_path,
                        hint="File may have already been deleted",
                    )

                if not full_path.is_file():
                    return _make_error_response(
                        "Cannot delete: path is not a regular file",
                        path=file_path,
                        hint="Only regular files can be deleted",
                    )

                lines_removed = 0
                try:
                    data = full_path.read_bytes()
                    if _contains_null_bytes_bytes(data):
                        return _make_error_response(
                            "Cannot delete: file appears to be binary (contains NUL bytes)",
                            path=file_path,
                        )
                    txt, _ = _decode_utf8_preserve_bom(data)
                    lines_removed = txt.count("\n") + (1 if txt else 0)
                except Exception:
                    lines_removed = 0

                if not dry_run:
                    try:
                        full_path.unlink()
                    except Exception as e:
                        return _make_error_response(f"Failed to delete file: {e}", path=file_path)

                return _make_success_response(
                    op="delete",
                    path=file_path,
                    hunks_applied=0,
                    lines_added=0,
                    lines_removed=lines_removed,
                    dry_run=dry_run,
                )

            # Update file.
            if not full_path.exists():
                return _make_error_response(
                    "Cannot update file: does not exist",
                    path=file_path,
                    hint="Use '*** Add File:' to create new files",
                )

            if not full_path.is_file():
                return _make_error_response("Path is not a regular file", path=file_path)

            try:
                raw = full_path.read_bytes()
            except Exception as e:
                return _make_error_response(f"Cannot read file: {e}", path=file_path)

            if _contains_null_bytes_bytes(raw):
                return _make_error_response(
                    "Cannot read file: appears to be binary (contains NUL bytes)",
                    path=file_path,
                    hint="This tool only supports UTF-8 text files",
                )

            try:
                current_content, bom = _decode_utf8_preserve_bom(raw)
            except UnicodeDecodeError:
                return _make_error_response(
                    "Cannot read file: not valid UTF-8 text",
                    path=file_path,
                    hint="This tool only supports UTF-8 text files",
                )

            # Apply hunks.
            try:
                new_content, fuzz_level = _apply_hunks(current_content, parsed.hunks, file_path)
            except PatchError as e:
                return _make_error_response(e.error, e.path, e.hint)

            if len(new_content.encode("utf-8")) > instance.max_file_size:
                return _make_error_response(
                    f"Result exceeds maximum file size ({instance.max_file_size // 1024 // 1024} MB)",
                    path=file_path,
                )

            total_added = sum(h.lines_added for h in parsed.hunks)
            total_removed = sum(h.lines_removed for h in parsed.hunks)

            # Determine destination.
            target_path = file_path
            moved_from: Optional[str] = None

            if parsed.move_to:
                move_to_path = parsed.move_to

                if not _is_allowed_text_path(
                    move_to_path,
                    instance.allowed_extensions,
                    instance.allowed_basenames,
                    enforce_allowlist=instance.enforce_allowlist,
                ):
                    if instance.enforce_allowlist:
                        return _make_error_response(
                            "Move target file type not allowed",
                            path=move_to_path,
                            hint=(
                                "Only common text/code file types are supported by default. "
                                "Configure ApplyPatchTool(..., enforce_allowlist=False) to allow any UTF-8 text file."
                            ),
                        )
                    return _make_error_response(
                        "Move target extension appears to be binary and is not allowed",
                        path=move_to_path,
                        hint="This tool only supports UTF-8 text files (no NUL bytes).",
                    )

                full_move = (instance.base_path / move_to_path).resolve()

                if not _is_within(full_move, instance.base_path):
                    return _make_error_response(
                        "Move target path escapes base directory",
                        path=move_to_path,
                        hint="Path cannot use '..' to escape the workspace",
                    )

                # Check target existence.
                if full_move.exists() and full_move.resolve() != full_path.resolve():
                    return _make_error_response(
                        "Cannot move: target file already exists",
                        path=move_to_path,
                        hint="Delete the target file first or choose a different name",
                    )

                target_path = move_to_path
                moved_from = file_path

                if not dry_run:
                    mode = full_path.stat().st_mode & 0o777
                    try:
                        _atomic_write_bytes(full_move, bom + new_content.encode("utf-8"), mode=mode)
                    except Exception as e:
                        return _make_error_response(f"Failed to write move target: {e}", path=move_to_path)

                    # Remove original if different.
                    if full_move.resolve() != full_path.resolve():
                        try:
                            full_path.unlink()
                        except Exception as e:
                            # Best effort rollback: try to remove new file to preserve semantics.
                            try:
                                full_move.unlink()
                            except Exception:
                                pass
                            return _make_error_response(
                                f"Failed to remove original file after move: {e}",
                                path=file_path,
                                hint="No changes were committed if rollback succeeded; otherwise both files may exist.",
                            )

            else:
                if not dry_run:
                    mode = full_path.stat().st_mode & 0o777
                    try:
                        _atomic_write_bytes(full_path, bom + new_content.encode("utf-8"), mode=mode)
                    except Exception as e:
                        return _make_error_response(f"Failed to write file: {e}", path=file_path)

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

