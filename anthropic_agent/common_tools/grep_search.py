"""
### Spec for `grep_search` in `src/tools/grep_search.py`

- **Signature**
  - `def grep_search(query: str, include_pattern: str | None = None, exclude_pattern: str | None = None, case_sensitive: bool = False) -> str`

- **Purpose**
  - Execute a ripgrep search under the shared search root, including match lines and 2 context lines before and after each match, and return a pretty-printed string.

- **Search root**
  - Uses configurable `base_path` provided at tool instantiation.
  - The ripgrep process runs with `cwd=base_path` so paths are naturally relative.

- **Input parameters**
  - `query`: regex pattern to search for.
  - `include_pattern`: a glob to include files. If `None` or empty, rg's default include applies (respect `.gitignore`).
  - `exclude_pattern`: a glob to exclude files (prefixed as `!` glob to rg).
  - `case_sensitive`: when `True`, enforce case-sensitive; when `False` (default), use ignore-case.

- **Execution details**
  - Invokes `rg` with `--json`, `-n` (line numbers), `-C 2` (context lines), and case flags.
  - Include/exclude patterns are applied via repeated `--glob` options (`pattern` and `!pattern`).
  - When no `include_pattern` is provided, restrict search to files with `.md` or `.mmd` extensions via `--glob '**/*.md' --glob '**/*.mmd'`.

- **Output formatting**
  - Group by file, header line: `relative/posix/path:`.
  - Lines are prefixed by `NNN:` for match lines and `NNN-` for context lines.
  - Matches in lines are wrapped with `<match>…</match>` (no ANSI colors).
  - If more than 20 match lines exist, truncate after printing the 20th match and its after-context, then append `[...]` indicator at the end with omitted count.
  - If there are no matches: return `No matches found.`
  - If ripgrep is missing: `ripgrep (rg) is not installed or not found in PATH.`
  - If ripgrep fails (e.g., bad regex): return stderr text.
"""
from __future__ import annotations

import json
import subprocess
from bisect import bisect_left
from pathlib import Path
from posixpath import normpath as _posix_normpath
from typing import Callable, Dict, List, Tuple

from ..tools.decorators import tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTEXT_LINES: int = 2
MAX_MATCH_LINES: int = 20
ALLOWED_EXTS: set[str] = {".md", ".mmd"}


# ---------------------------------------------------------------------------
# Pure utility functions (no state needed)
# ---------------------------------------------------------------------------
def _highlight_line(text: str, ranges: List[Tuple[int, int]]) -> str:
    """Insert <match>...</match> tags around matched ranges in text."""
    # Insert tags from right to left to preserve offsets
    highlighted = text
    for start, end in sorted(ranges, key=lambda r: r[0], reverse=True):
        if 0 <= start <= end <= len(highlighted):
            highlighted = highlighted[:start] + "<match>" + highlighted[start:end] + "</match>" + highlighted[end:]
    return highlighted


def _byte_ranges_to_char_ranges(text: str, byte_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Convert UTF-8 byte offset ranges to Python string character index ranges."""
    if not byte_ranges:
        return []
    # Build cumulative byte offsets for each character boundary
    byte_offsets: List[int] = [0]
    for ch in text:
        byte_offsets.append(byte_offsets[-1] + len(ch.encode("utf-8")))

    char_ranges: List[Tuple[int, int]] = []
    for bstart, bend in byte_ranges:
        # Map byte offsets to nearest character boundaries
        cs = bisect_left(byte_offsets, bstart)
        ce = bisect_left(byte_offsets, bend)
        # Clamp within valid range
        cs = max(0, min(cs, len(text)))
        ce = max(cs, min(ce, len(text)))
        char_ranges.append((cs, ce))
    return char_ranges


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class GrepSearchTool:
    """Configurable grep_search tool with a sandboxed base path.
    
    This class encapsulates the grep_search functionality using ripgrep,
    allowing configuration of the base path at instantiation time. The tool
    returned by get_tool() can be registered with an AnthropicAgent.
    
    Example:
        >>> grep_tool = GrepSearchTool(base_path="/path/to/workspace")
        >>> agent = AnthropicAgent(tools=[grep_tool.get_tool()])
    """
    
    def __init__(self, base_path: str | Path):
        """Initialize the GrepSearchTool with a base path.
        
        Args:
            base_path: The root directory that grep_search operates within.
                       All searches are executed relative to this directory.
        """
        self.search_root: Path = Path(base_path).resolve()
    
    def _rel_posix(self, path_str: str) -> str:
        """Convert a path string to a POSIX-style path relative to search_root."""
        p = Path(path_str)
        try:
            rel = p.resolve().relative_to(self.search_root.resolve())
            return _posix_normpath(rel.as_posix())
        except Exception:
            # If already relative to cwd=search_root, just normalize to POSIX
            return _posix_normpath(p.as_posix())
    
    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.
        
        Returns:
            A decorated grep_search function that operates within the configured base_path.
        """
        # Capture self in closure for the inner function
        instance = self
        
        @tool
        def grep_search(
            query: str,
            include_pattern: str | None = None,
            exclude_pattern: str | None = None,
            case_sensitive: bool = False,
        ) -> str:
            """Search for a regex using ripgrep under the configured base path and pretty-print results with context.

            Each match is shown with 2 lines of context before and after. Output is grouped by file and
            formatted with:
              - "NNN:" for matched lines (with <match>…</match> tags around matches)
              - "NNN-" for context lines
            At most MAX_MATCH_LINES match events are shown; trailing after-context for the final shown
            match is included, and an omission indicator is appended if more matches exist.

            Args:
                query: Regular expression to search for.
                include_pattern: Optional POSIX-style glob to include files. When set,
                    ignore rules are disabled and the glob is passed via '--glob'.
                exclude_pattern: Optional POSIX-style glob to exclude files, applied as
                    a negated '--glob' (e.g., '!pattern').
                case_sensitive: Whether the regex match is case-sensitive. Defaults to False.

            Returns:
                A formatted, multi-file result. Each file section starts with "relative/path:" and
                    contains interleaved match/context lines as described above. Returns:
                    - "No matches found." when there are no results
                    - "ripgrep (rg) is not installed or not found in PATH." if rg is missing
                    - ripgrep's stderr text if the command fails (e.g., bad regex)
            """
            if query is None or str(query) == "":
                return "pattern cannot be empty"

            cmd: List[str] = [
                "rg",
                "--json",
                "-n",
                "-C",
                str(CONTEXT_LINES),
            ]

            # Case sensitivity
            if case_sensitive:
                cmd.append("--case-sensitive")
            else:
                cmd.append("--ignore-case")

            # Include/exclude globs
            if include_pattern:
                norm_inc = _posix_normpath(str(include_pattern).replace("\\", "/"))
                # Allow include to override ignore rules by disabling ignore files
                cmd.append("--no-ignore")
                cmd.extend(["--glob", norm_inc])
            else:
                # Default restriction to allowed extensions
                cmd.extend(["--glob", "**/*.md", "--glob", "**/*.mmd"])

            if exclude_pattern:
                norm_exc = _posix_normpath(str(exclude_pattern).replace("\\", "/"))
                cmd.extend(["--glob", f"!{norm_exc}"])

            # Ensure .gitignore at root is honored by default; if include_pattern is provided and
            # we used --no-ignore to override, do not re-apply ignore-file.
            gi = instance.search_root / ".gitignore"
            if include_pattern is None and gi.exists():
                cmd.extend(["--ignore-file", str(gi)])

            cmd.append("--")
            cmd.append(query)
            cmd.append(".")

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(instance.search_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
            except FileNotFoundError:
                return "ripgrep (rg) is not installed or not found in PATH."
            except Exception as exc:
                return str(exc)

            stdout = proc.stdout.splitlines()
            stderr = proc.stderr.strip()

            file_to_lines: Dict[str, List[str]] = {}
            file_order: List[str] = []
            total_match_events: int = 0
            printed_match_events: int = 0
            # Track whether to collect trailing after-context after reaching cap
            draining_after_context: bool = False
            drain_file: str = ""
            last_match_line_no: int = -1
            remaining_after_context: int = 0

            for line in stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Skip non-JSON lines
                    continue

                typ = event.get("type")
                data = event.get("data", {})

                if typ not in {"match", "context"}:
                    # Ignore other events for output
                    continue

                path_text = data.get("path", {}).get("text")
                if not path_text:
                    # Some rg versions put path under data["path"]["text"]
                    path_text = (event.get("data", {}).get("path") or {}).get("text")
                if not path_text:
                    continue
                rel_path = instance._rel_posix(path_text)

                # Enforce allowed extension filter even when include_pattern is broad
                if Path(rel_path).suffix.lower() not in ALLOWED_EXTS:
                    # Skip any non-allowed file from output
                    continue

                header_rel = rel_path

                if header_rel not in file_to_lines:
                    file_to_lines[header_rel] = []
                    file_order.append(header_rel)

                line_number = data.get("line_number")
                line_obj = data.get("lines", {})
                text = line_obj.get("text", "")
                # Normalize text: ripgrep includes trailing newline in JSON; handle CRLF
                if text.endswith("\n"):
                    text = text[:-1]
                if text.endswith("\r"):
                    text = text[:-1]

                if typ == "match":
                    total_match_events += 1
                    if draining_after_context and header_rel == drain_file and int(line_number) > last_match_line_no and remaining_after_context > 0:
                        # Treat subsequent lines as after-context even if they are matches
                        file_to_lines[header_rel].append(f"  {line_number}- {text}")
                        remaining_after_context -= 1
                        # Do not count as printed match; it is part of context drain
                        continue
                    if printed_match_events < MAX_MATCH_LINES:
                        # Highlight submatch ranges
                        submatches = data.get("submatches", [])
                        byte_ranges: List[Tuple[int, int]] = []
                        for sm in submatches:
                            start = int(sm.get("start", 0))
                            end = int(sm.get("end", start))
                            byte_ranges.append((start, end))
                        char_ranges = _byte_ranges_to_char_ranges(text, byte_ranges)
                        highlighted = _highlight_line(text, char_ranges)
                        file_to_lines[header_rel].append(f"  {line_number}: {highlighted}")
                        printed_match_events += 1
                        # Prepare to drain after-context if we just hit the cap
                        if printed_match_events == MAX_MATCH_LINES:
                            draining_after_context = True
                            drain_file = header_rel
                            last_match_line_no = int(line_number)
                            remaining_after_context = CONTEXT_LINES
                    # Do not collect further matches when at/over cap; keep counting
                    continue

                # typ == "context"
                if draining_after_context:
                    # Only collect context for the same file and after the last match line
                    if header_rel == drain_file and int(line_number) > last_match_line_no and remaining_after_context > 0:
                        file_to_lines[header_rel].append(f"  {line_number}- {text}")
                        remaining_after_context -= 1
                    # Skip other context lines while draining
                    continue

                # Regular context lines interleaved before/after matches
                file_to_lines[header_rel].append(f"  {line_number}- {text}")

            # Interpret ripgrep return codes: 0 (matches), 1 (no matches), >1 (error)
            if printed_match_events == 0:
                if proc.returncode in (0, 1):
                    return "No matches found."
                return stderr or "ripgrep failed."

            # Build final formatted output
            out_lines: List[str] = []
            for rel_path in file_order:
                lines = file_to_lines.get(rel_path)
                if not lines:
                    continue
                out_lines.append(f"{rel_path}:")
                out_lines.extend(lines)

            omitted = max(0, total_match_events - printed_match_events)
            if omitted > 0:
                out_lines.append(f"[... {omitted} more matches omitted]")

            return "\n".join(out_lines)
        
        return grep_search
