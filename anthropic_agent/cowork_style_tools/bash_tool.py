"""
Bash Tool for AnthropicAgent.

This module provides a shell command execution tool that:
- Executes bash commands via subprocess
- Maintains working directory state between calls
- Captures combined stdout/stderr output
- Truncates output to a configurable limit (tail-keeping)
- Supports path-based sandboxing of the working directory
- Supports dynamic agent UUID injection post-initialization
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..tools.base import ConfigurableToolBase


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT_MS: int = 120_000        # 2 minutes
MAX_TIMEOUT_MS: int = 600_000            # 10 minutes
MAX_OUTPUT_CHARS: int = 30_000           # 30k character output cap
CWD_PROBE_MARKER: str = "__BASH_CWD_PROBE_a7f3e9__"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _is_within(child: Path, parent: Path) -> bool:
    """Check whether *child* is equal to or a descendant of *parent*.

    Both paths are resolved to absolute form before comparison so that
    symlinks and relative segments (``..``) cannot bypass the check.
    """
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class BashTool(ConfigurableToolBase):
    """Shell command execution tool with sandboxed working directory and output truncation.

    This tool provides bash command execution where:
    - Each invocation spawns a fresh shell process (no env/alias persistence)
    - The working directory persists across calls within the same session
    - stdout and stderr are captured together
    - Output is truncated (tail-kept) when it exceeds the configured limit

    Example:
        >>> bash_tool = BashTool(base_path="/workspace")
        >>> tool_fn = bash_tool.get_tool()
        >>> result = tool_fn(command="echo hello")
        >>> # result contains "hello\\n\\n[exit_code: 0]\\n[cwd: /workspace]"
    """

    DOCSTRING_TEMPLATE = """Execute a shell command in a bash subprocess.

Each command runs in a fresh shell process, but the working directory persists
between calls. Combined stdout and stderr are returned.

**Configuration:**
- Default timeout: {default_timeout_ms}ms ({default_timeout_sec}s)
- Maximum timeout: {max_timeout_ms}ms ({max_timeout_sec}s)
- Output truncation: {max_output_chars} characters (tail-kept if exceeded)
- Sandbox: {sandbox_status}

Args:
    command: The bash command to execute. Quote paths with spaces using
        double quotes. Chain sequential commands with '&&', use ';' when
        earlier failures don't matter.
    description: A brief human-readable description of what this command
        does. Used for logging and observability only — does not affect
        execution.
    timeout: Timeout in milliseconds. Defaults to {default_timeout_ms}ms.
        Maximum: {max_timeout_ms}ms. Commands exceeding the timeout are
        killed and an error is returned.
    dangerouslyDisableSandbox: When True, allows the working directory to
        move outside the sandboxed base directory. Use with caution.

Returns:
    Command output containing combined stdout/stderr, followed by metadata
    lines showing the exit code and current working directory.
"""

    def __init__(
        self,
        base_path: str | Path,
        default_timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_timeout_ms: int = MAX_TIMEOUT_MS,
        max_output_chars: int = MAX_OUTPUT_CHARS,
        sandbox_enabled: bool = True,
        agent_uuid: str | None = None,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the BashTool.

        Args:
            base_path: Root directory for sandboxed execution. The working
                directory starts here and (when sandboxed) must stay within it.
            default_timeout_ms: Default command timeout in milliseconds.
            max_timeout_ms: Hard upper limit on timeout in milliseconds.
            max_output_chars: Maximum characters in returned output. If
                exceeded, only the tail is kept.
            sandbox_enabled: When True (default), the working directory is
                constrained to remain within base_path.
            agent_uuid: Optional agent UUID for session scoping.
            docstring_template: Optional custom docstring template.
            schema_override: Optional complete Anthropic tool schema dict.
        """
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )

        self.base_path: Path = Path(base_path).resolve()
        self.default_timeout_ms = default_timeout_ms
        self.max_timeout_ms = max_timeout_ms
        self.max_output_chars = max_output_chars
        self.sandbox_enabled = sandbox_enabled
        self.agent_uuid = agent_uuid

        # Persistent working directory — survives across calls
        self._cwd: Path = self.base_path

    # ------------------------------------------------------------------
    # Template helpers
    # ------------------------------------------------------------------

    def _get_template_context(self) -> Dict[str, Any]:
        """Return placeholder values for docstring template."""
        return {
            "default_timeout_ms": self.default_timeout_ms,
            "default_timeout_sec": self.default_timeout_ms // 1000,
            "max_timeout_ms": self.max_timeout_ms,
            "max_timeout_sec": self.max_timeout_ms // 1000,
            "max_output_chars": self.max_output_chars,
            "sandbox_status": (
                f"commands execute within {self.base_path}"
                if self.sandbox_enabled
                else "unrestricted"
            ),
        }

    # ------------------------------------------------------------------
    # Agent UUID injection (duck-typed protocol)
    # ------------------------------------------------------------------

    def set_agent_uuid(self, agent_uuid: str) -> None:
        """Set the agent UUID for session scoping.

        Called by the agent after UUID assignment via the
        ``__tool_instance__`` / ``set_agent_uuid`` duck-typed protocol
        in ``AnthropicAgent._inject_agent_uuid_to_tools()``.

        Args:
            agent_uuid: The agent's UUID string.
        """
        self.agent_uuid = agent_uuid

    # ------------------------------------------------------------------
    # Sandbox validation
    # ------------------------------------------------------------------

    def _validate_cwd(self) -> str | None:
        """Validate that the current working directory is within the sandbox.

        Returns:
            An error message string if validation fails, or None if valid.
            If validation fails, ``_cwd`` is reset to ``base_path``.
        """
        if not self.sandbox_enabled:
            return None
        if not _is_within(self._cwd, self.base_path):
            self._cwd = self.base_path
            return (
                f"Working directory escaped sandbox boundary. "
                f"Reset to {self.base_path}."
            )
        if not self._cwd.is_dir():
            self._cwd = self.base_path
            return (
                f"Working directory no longer exists. "
                f"Reset to {self.base_path}."
            )
        return None

    # ------------------------------------------------------------------
    # Output truncation
    # ------------------------------------------------------------------

    def _truncate_output(self, content: str) -> str:
        """Truncate output to keep only the tail if it exceeds max_output_chars.

        Args:
            content: The raw command output.

        Returns:
            The content, possibly truncated with a notice header.
        """
        if len(content) <= self.max_output_chars:
            return content

        truncate_notice = (
            f"\n... [truncated, showing last {self.max_output_chars} "
            f"chars of {len(content)} total] ...\n"
        )
        tail_size = self.max_output_chars - len(truncate_notice)
        return truncate_notice + content[-tail_size:]

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def _execute_command(
        self,
        command: str,
        timeout_seconds: float,
        disable_sandbox: bool = False,
    ) -> tuple[str, int, Path]:
        """Execute a shell command and return (output, exit_code, new_cwd).

        The command is wrapped with a cwd probe suffix so that any ``cd``
        effects are detected and the persistent ``_cwd`` can be updated.

        Args:
            command: The raw command string from the caller.
            timeout_seconds: Maximum execution time in seconds.
            disable_sandbox: If True, allow cwd to move outside base_path.

        Returns:
            A tuple of (combined_output, exit_code, resolved_new_cwd).
        """
        # Append a cwd probe that runs after the user's command
        probe = f'; echo "{CWD_PROBE_MARKER}$(pwd){CWD_PROBE_MARKER}"'
        wrapped_command = command + probe

        try:
            proc = subprocess.run(
                ["bash", "-c", wrapped_command],
                cwd=str(self._cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
            combined_output = proc.stdout or ""
            exit_code = proc.returncode

        except subprocess.TimeoutExpired:
            return (
                f"Command timed out after {timeout_seconds:.0f}s. "
                f"Consider increasing the timeout parameter.",
                -1,
                self._cwd,
            )
        except FileNotFoundError:
            return (
                "bash is not available on this system.",
                -1,
                self._cwd,
            )
        except Exception as exc:
            return (
                f"Error executing command: {type(exc).__name__}: {exc}",
                -1,
                self._cwd,
            )

        # Parse the cwd probe marker from the output
        new_cwd = self._cwd
        pattern = re.escape(CWD_PROBE_MARKER) + r"(.+?)" + re.escape(CWD_PROBE_MARKER)
        cwd_match = re.search(pattern, combined_output)
        if cwd_match:
            candidate = Path(cwd_match.group(1)).resolve()
            if not self.sandbox_enabled or disable_sandbox or _is_within(candidate, self.base_path):
                new_cwd = candidate
            # Strip the probe line from output
            combined_output = combined_output[: cwd_match.start()].rstrip("\n")

        return combined_output, exit_code, new_cwd

    # ------------------------------------------------------------------
    # Tool factory
    # ------------------------------------------------------------------

    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.

        Returns:
            A decorated bash execution function. The function has a
            ``__tool_instance__`` attribute for agent UUID injection.
        """
        instance = self

        def bash(
            command: str,
            description: str | None = None,
            timeout: int | None = None,
            dangerouslyDisableSandbox: bool = False,
        ) -> str:
            """Placeholder docstring — replaced by template."""

            # 1. Resolve timeout
            if timeout is None:
                timeout_ms = instance.default_timeout_ms
            else:
                timeout_ms = max(1, min(int(timeout), instance.max_timeout_ms))
            timeout_seconds = timeout_ms / 1000.0

            # 2. Pre-execution sandbox check
            if instance.sandbox_enabled and not dangerouslyDisableSandbox:
                sandbox_err = instance._validate_cwd()
                if sandbox_err:
                    return sandbox_err

            # 3. Execute the command
            output, exit_code, new_cwd = instance._execute_command(
                command, timeout_seconds, dangerouslyDisableSandbox,
            )

            # 4. Update persistent working directory
            instance._cwd = new_cwd

            # 5. Truncate output
            truncated = instance._truncate_output(output)

            # 6. Format result with metadata
            parts = []
            if truncated.strip():
                parts.append(truncated)

            parts.append(f"\n[exit_code: {exit_code}]")
            parts.append(f"[cwd: {instance._cwd}]")

            return "\n".join(parts)

        # Attach instance reference for agent UUID injection
        bash.__tool_instance__ = instance

        return self._apply_schema(bash)
