"""GrepSearchTool — async, sandbox-based content search.

Migrated from anthropic_agent/common_tools/grep_search.py.
subprocess.run(["rg", ...]) replaced with sandbox.exec("rg ...").
"""
from __future__ import annotations

import json
import shlex
from bisect import bisect_left
from posixpath import normpath as _posix_normpath
from typing import Any, Callable, Dict, List, Tuple

from ..tools.base import ConfigurableToolBase


# ---------------------------------------------------------------------------
# Pure utility functions (no state needed)
# ---------------------------------------------------------------------------
def _highlight_line(text: str, ranges: List[Tuple[int, int]]) -> str:
    """Insert <match>...</match> tags around matched ranges in text."""
    highlighted = text
    for start, end in sorted(ranges, key=lambda r: r[0], reverse=True):
        if 0 <= start <= end <= len(highlighted):
            highlighted = highlighted[:start] + "<match>" + highlighted[start:end] + "</match>" + highlighted[end:]
    return highlighted


def _byte_ranges_to_char_ranges(text: str, byte_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Convert UTF-8 byte offset ranges to Python string character index ranges."""
    if not byte_ranges:
        return []
    byte_offsets: List[int] = [0]
    for ch in text:
        byte_offsets.append(byte_offsets[-1] + len(ch.encode("utf-8")))

    char_ranges: List[Tuple[int, int]] = []
    for bstart, bend in byte_ranges:
        cs = bisect_left(byte_offsets, bstart)
        ce = bisect_left(byte_offsets, bend)
        cs = max(0, min(cs, len(text)))
        ce = max(cs, min(ce, len(text)))
        char_ranges.append((cs, ce))
    return char_ranges


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class GrepSearchTool(ConfigurableToolBase):
    """Configurable grep_search tool with sandbox-based execution.

    All search execution goes through the injected Sandbox instance
    via sandbox.exec("rg ...").

    Example:
        >>> grep_tool = GrepSearchTool(max_match_lines=50)
        >>> func = grep_tool.get_tool()
        >>> registry.register_tools([func])
        >>> registry.attach_sandbox(sandbox)
    """

    DOCSTRING_TEMPLATE = """Search file contents using regular expressions (powered by ripgrep).

Use this tool to find text patterns across files. Returns matches with
context lines before and after.

**Limits:**
- Max matches shown: {max_match_lines} (with count of omitted matches)
- Context: {context_lines} lines before and after each match
- Allowed extensions: {allowed_extensions_str}

Args:
    query: Regular expression to search for. Common patterns:
        - "TODO" - literal text search
        - "def \\\\w+\\\\(" - function definitions
        - "import.*requests" - import statements
        - "\\\\bword\\\\b" - whole word match
    include_pattern: Optional glob to restrict which files to search (e.g., "docs/*.md").
        When set, .gitignore rules are bypassed.
    exclude_pattern: Optional glob to exclude files (e.g., "test_*.md").
    case_sensitive: Whether to match case. Defaults to False (case-insensitive).

Returns:
    Results grouped by file with line numbers:
    ```
    docs/api.md:
      10- context before
      11: This line has a <match>TODO</match> marker
      12- context after

    src/main.md:
      25: Another <match>TODO</match> item
    ```

**Regex Tips:**
- Escape special chars: \\\\. \\\\( \\\\) \\\\[ \\\\]
- Word boundary: \\\\bword\\\\b
- Any whitespace: \\\\s+
- One or more: pattern+
- Zero or more: pattern*

**Error Recovery:**
- "No matches found" -> Try case_sensitive=False, or broaden the pattern
- "ripgrep failed" -> Check regex syntax, escape special characters
- "pattern cannot be empty" -> Provide a non-empty search pattern
"""

    def __init__(
        self,
        max_match_lines: int = 20,
        context_lines: int = 2,
        allowed_extensions: set[str] | None = None,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the GrepSearchTool.

        Args:
            max_match_lines: Maximum number of match lines to show. Defaults to 20.
            context_lines: Number of context lines before/after each match. Defaults to 2.
            allowed_extensions: Set of allowed file extensions (with leading dot).
                               Defaults to {".md", ".mmd"} if None.
            docstring_template: Optional custom docstring template.
            schema_override: Optional complete Anthropic tool schema dict.
        """
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)
        self.max_match_lines: int = max_match_lines
        self.context_lines: int = context_lines
        self.allowed_extensions: set[str] = allowed_extensions or {".md", ".mmd"}

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "max_match_lines": self.max_match_lines,
            "context_lines": self.context_lines,
            "allowed_extensions_str": ", ".join(sorted(self.allowed_extensions)),
        }

    def get_tool(self) -> Callable:
        instance = self

        async def grep_search(
            query: str,
            include_pattern: str | None = None,
            exclude_pattern: str | None = None,
            case_sensitive: bool = False,
        ) -> str:
            """Placeholder docstring - replaced by template."""
            if query is None or str(query) == "":
                return "Query pattern cannot be empty. Provide a regex pattern to search for (e.g., 'TODO', 'def \\w+')."

            # Build rg command
            cmd_parts: List[str] = [
                "rg",
                "--json",
                "-n",
                "-C",
                str(instance.context_lines),
            ]

            if case_sensitive:
                cmd_parts.append("--case-sensitive")
            else:
                cmd_parts.append("--ignore-case")

            if include_pattern:
                norm_inc = _posix_normpath(str(include_pattern).replace("\\", "/"))
                cmd_parts.append("--no-ignore")
                cmd_parts.extend(["--glob", shlex.quote(norm_inc)])
            else:
                for ext in sorted(instance.allowed_extensions):
                    cmd_parts.extend(["--glob", shlex.quote(f"**/*{ext}")])

            if exclude_pattern:
                norm_exc = _posix_normpath(str(exclude_pattern).replace("\\", "/"))
                cmd_parts.extend(["--glob", shlex.quote(f"!{norm_exc}")])

            cmd_parts.append("--")
            cmd_parts.append(shlex.quote(query))
            cmd_parts.append(".")

            cmd = " ".join(cmd_parts)

            result = await instance._sandbox.exec(cmd, timeout=15.0, cwd=".")

            stdout = result.stdout.splitlines()
            stderr = result.stderr.strip()

            file_to_lines: Dict[str, List[str]] = {}
            file_order: List[str] = []
            total_match_events: int = 0
            printed_match_events: int = 0
            draining_after_context: bool = False
            drain_file: str = ""
            last_match_line_no: int = -1
            remaining_after_context: int = 0

            for line in stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                typ = event.get("type")
                data = event.get("data", {})

                if typ not in {"match", "context"}:
                    continue

                path_text = data.get("path", {}).get("text")
                if not path_text:
                    path_text = (event.get("data", {}).get("path") or {}).get("text")
                if not path_text:
                    continue

                # Paths from rg running inside sandbox are already relative
                rel_path = _posix_normpath(path_text.removeprefix("./"))

                # Enforce allowed extension filter
                name = rel_path.rsplit("/", 1)[-1] if "/" in rel_path else rel_path
                if not any(name.lower().endswith(ext) for ext in instance.allowed_extensions):
                    continue

                header_rel = rel_path

                if header_rel not in file_to_lines:
                    file_to_lines[header_rel] = []
                    file_order.append(header_rel)

                line_number = data.get("line_number")
                line_obj = data.get("lines", {})
                text = line_obj.get("text", "")
                if text.endswith("\n"):
                    text = text[:-1]
                if text.endswith("\r"):
                    text = text[:-1]

                if typ == "match":
                    total_match_events += 1
                    if draining_after_context and header_rel == drain_file and int(line_number) > last_match_line_no and remaining_after_context > 0:
                        file_to_lines[header_rel].append(f"  {line_number}- {text}")
                        remaining_after_context -= 1
                        continue
                    if printed_match_events < instance.max_match_lines:
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
                        if printed_match_events == instance.max_match_lines:
                            draining_after_context = True
                            drain_file = header_rel
                            last_match_line_no = int(line_number)
                            remaining_after_context = instance.context_lines
                    continue

                # typ == "context"
                if draining_after_context:
                    if header_rel == drain_file and int(line_number) > last_match_line_no and remaining_after_context > 0:
                        file_to_lines[header_rel].append(f"  {line_number}- {text}")
                        remaining_after_context -= 1
                    continue

                file_to_lines[header_rel].append(f"  {line_number}- {text}")

            # Interpret ripgrep return codes: 0 (matches), 1 (no matches), >1 (error)
            if printed_match_events == 0:
                if result.exit_code in (0, 1):
                    return f"No matches found for pattern '{query}'. Try case_sensitive=False, broaden the pattern, or check if files exist with glob_file_search."
                return stderr or "ripgrep failed. Check regex syntax - special characters like ( ) [ ] . need escaping with \\."

            # Build final formatted output
            out_lines: List[str] = []
            for rel_path in file_order:
                lines_list = file_to_lines.get(rel_path)
                if not lines_list:
                    continue
                out_lines.append(f"{rel_path}:")
                out_lines.extend(lines_list)

            omitted = max(0, total_match_events - printed_match_events)
            if omitted > 0:
                out_lines.append(f"[... {omitted} more matches omitted]")

            return "\n".join(out_lines)

        func = self._apply_schema(grep_search)
        func.__tool_instance__ = instance
        return func
