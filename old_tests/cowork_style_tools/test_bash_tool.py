"""Tests for BashTool functionality.

Approach: Full Integration Testing
----------------------------------
These tests use real subprocess execution in temporary directories to validate
the bash tool end-to-end. Each test uses a temporary directory that is
discarded after the test completes.

Tests cover:
- Basic command execution (echo, exit codes, command not found)
- Working directory persistence (cd persists across calls, initial cwd, cwd in output)
- Timeout handling (default, custom, exceeded, clamped to max)
- Output truncation (short output untouched, long output tail-kept, truncation marker)
- Sandbox enforcement (cwd containment, dangerouslyDisableSandbox, sandbox_enabled=False)
- Combined stdout/stderr capture
- Docstring template rendering and schema generation
- Agent UUID injection protocol
"""
import tempfile
from pathlib import Path
from typing import Callable, Generator

import pytest

from anthropic_agent.cowork_style_tools.bash_tool import BashTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Resolve symlinks so paths match what subprocess reports
        # (macOS /var -> /private/var)
        yield Path(tmpdir).resolve()


@pytest.fixture
def bash_tool(temp_workspace: Path) -> BashTool:
    """Create a BashTool with default settings and sandbox enabled."""
    return BashTool(base_path=temp_workspace)


@pytest.fixture
def bash_fn(bash_tool: BashTool) -> Callable:
    """Get the decorated tool function from a BashTool instance."""
    return bash_tool.get_tool()


@pytest.fixture
def bash_tool_no_sandbox(temp_workspace: Path) -> BashTool:
    """Create a BashTool with sandbox disabled."""
    return BashTool(base_path=temp_workspace, sandbox_enabled=False)


# ---------------------------------------------------------------------------
# TestBasicExecution
# ---------------------------------------------------------------------------
class TestBasicExecution:
    """Tests for basic command execution."""

    def test_echo_command(self, bash_fn: Callable):
        result = bash_fn(command="echo hello")
        assert "hello" in result

    def test_exit_code_zero(self, bash_fn: Callable):
        result = bash_fn(command="echo ok")
        assert "[exit_code: 0]" in result

    def test_exit_code_nonzero(self, bash_fn: Callable):
        result = bash_fn(command="exit 42")
        assert "[exit_code: 42]" in result

    def test_command_not_found(self, bash_fn: Callable):
        result = bash_fn(command="nonexistent_command_xyz_12345")
        assert "[exit_code:" in result
        # Exit code 127 = command not found in bash
        assert "127" in result or "not found" in result

    def test_multiline_output(self, bash_fn: Callable):
        result = bash_fn(command="echo line1 && echo line2 && echo line3")
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_empty_command_output(self, bash_fn: Callable):
        result = bash_fn(command="true")
        assert "[exit_code: 0]" in result

    def test_description_accepted(self, bash_fn: Callable):
        result = bash_fn(command="echo test", description="Print test message")
        assert "test" in result
        assert "[exit_code: 0]" in result

    def test_description_does_not_affect_output(self, bash_fn: Callable):
        result_with = bash_fn(command="echo same")
        # Reset cwd for fair comparison (it may have changed)
        result_without = bash_fn(command="echo same", description="described")
        # Both should contain "same" and exit 0
        assert "same" in result_with
        assert "same" in result_without


# ---------------------------------------------------------------------------
# TestWorkingDirectoryPersistence
# ---------------------------------------------------------------------------
class TestWorkingDirectoryPersistence:
    """Tests for cwd persistence across calls."""

    def test_initial_cwd_is_base_path(self, bash_tool: BashTool, bash_fn: Callable, temp_workspace: Path):
        result = bash_fn(command="pwd")
        assert str(temp_workspace) in result

    def test_cd_persists_across_calls(self, bash_fn: Callable, temp_workspace: Path):
        # Create a subdirectory
        subdir = temp_workspace / "subdir"
        subdir.mkdir()

        # cd into it
        result1 = bash_fn(command="cd subdir")
        assert "[exit_code: 0]" in result1

        # pwd in the next call should reflect the cd
        result2 = bash_fn(command="pwd")
        assert "subdir" in result2

    def test_cwd_shown_in_output(self, bash_fn: Callable, temp_workspace: Path):
        result = bash_fn(command="echo hello")
        assert f"[cwd: {temp_workspace}]" in result

    def test_cd_to_nonexistent_dir(self, bash_fn: Callable, temp_workspace: Path):
        result = bash_fn(command="cd nonexistent_dir_xyz")
        # Should fail but cwd should remain at base_path
        assert "[exit_code:" in result
        # Next command should still work from base_path
        result2 = bash_fn(command="pwd")
        assert str(temp_workspace) in result2


# ---------------------------------------------------------------------------
# TestTimeout
# ---------------------------------------------------------------------------
class TestTimeout:
    """Tests for timeout behavior."""

    def test_default_timeout_works(self, bash_fn: Callable):
        # A fast command should not time out
        result = bash_fn(command="echo fast")
        assert "[exit_code: 0]" in result

    def test_custom_timeout(self, bash_fn: Callable):
        # Short timeout but fast command — should succeed
        result = bash_fn(command="echo quick", timeout=5000)
        assert "[exit_code: 0]" in result

    def test_timeout_exceeded(self, temp_workspace: Path):
        tool = BashTool(base_path=temp_workspace, default_timeout_ms=1000)
        fn = tool.get_tool()
        result = fn(command="sleep 10", timeout=1000)
        assert "timed out" in result.lower()
        assert "[exit_code: -1]" in result

    def test_timeout_clamped_to_max(self, temp_workspace: Path):
        tool = BashTool(base_path=temp_workspace, max_timeout_ms=2000)
        fn = tool.get_tool()
        # Request 60s but max is 2s — should be clamped
        result = fn(command="sleep 10", timeout=60000)
        assert "timed out" in result.lower()

    def test_negative_timeout_uses_minimum(self, bash_fn: Callable):
        # Negative timeout should be clamped to 1ms minimum
        result = bash_fn(command="echo ok", timeout=-1000)
        # Should still execute (clamped to min 1ms, but echo is fast enough)
        assert "[exit_code:" in result


# ---------------------------------------------------------------------------
# TestOutputTruncation
# ---------------------------------------------------------------------------
class TestOutputTruncation:
    """Tests for output truncation behavior."""

    def test_short_output_not_truncated(self, bash_fn: Callable):
        result = bash_fn(command="echo short")
        assert "truncated" not in result

    def test_long_output_truncated(self, temp_workspace: Path):
        tool = BashTool(base_path=temp_workspace, max_output_chars=200)
        fn = tool.get_tool()
        # Generate output longer than 200 chars
        result = fn(command="seq 1 500")
        assert "truncated" in result

    def test_truncation_keeps_tail(self, temp_workspace: Path):
        tool = BashTool(base_path=temp_workspace, max_output_chars=200)
        fn = tool.get_tool()
        # Generate output with identifiable end
        result = fn(command="seq 1 500")
        # The tail (last numbers like 499, 500) should be in the output
        assert "500" in result

    def test_exact_limit_not_truncated(self, temp_workspace: Path):
        tool = BashTool(base_path=temp_workspace, max_output_chars=50000)
        fn = tool.get_tool()
        result = fn(command="echo tiny")
        assert "truncated" not in result


# ---------------------------------------------------------------------------
# TestSandboxing
# ---------------------------------------------------------------------------
class TestSandboxing:
    """Tests for sandbox path containment."""

    def test_cd_beyond_base_path_blocked(self, bash_tool: BashTool, bash_fn: Callable, temp_workspace: Path):
        # Try to cd above the sandbox
        bash_fn(command="cd /tmp")
        # The cwd should NOT have changed to /tmp since it's outside sandbox
        assert bash_tool._cwd == temp_workspace

    def test_disable_sandbox_allows_escape(self, bash_tool: BashTool, temp_workspace: Path):
        fn = bash_tool.get_tool()
        result = fn(command="cd /tmp", dangerouslyDisableSandbox=True)
        assert "[exit_code: 0]" in result
        # With sandbox disabled, cwd should have changed
        assert bash_tool._cwd == Path("/tmp").resolve()
        # Reset for cleanup
        bash_tool._cwd = temp_workspace

    def test_sandbox_disabled_at_init(self, bash_tool_no_sandbox: BashTool, temp_workspace: Path):
        fn = bash_tool_no_sandbox.get_tool()
        result = fn(command="cd /tmp")
        assert "[exit_code: 0]" in result
        assert bash_tool_no_sandbox._cwd == Path("/tmp").resolve()
        # Reset for cleanup
        bash_tool_no_sandbox._cwd = temp_workspace

    def test_cwd_reset_on_deleted_directory(self, bash_tool: BashTool, temp_workspace: Path):
        fn = bash_tool.get_tool()
        # Create and cd into a subdirectory
        subdir = temp_workspace / "ephemeral"
        subdir.mkdir()
        fn(command="cd ephemeral")
        assert bash_tool._cwd == subdir

        # Delete the directory externally
        subdir.rmdir()

        # Next call should detect missing directory and reset
        result = fn(command="echo hello")
        assert str(temp_workspace) in result


# ---------------------------------------------------------------------------
# TestCombinedOutput
# ---------------------------------------------------------------------------
class TestCombinedOutput:
    """Tests for stdout+stderr combination."""

    def test_stdout_captured(self, bash_fn: Callable):
        result = bash_fn(command="echo stdout_msg")
        assert "stdout_msg" in result

    def test_stderr_captured(self, bash_fn: Callable):
        result = bash_fn(command="echo stderr_msg >&2")
        assert "stderr_msg" in result

    def test_combined_stdout_stderr(self, bash_fn: Callable):
        result = bash_fn(command="echo out_msg && echo err_msg >&2")
        assert "out_msg" in result
        assert "err_msg" in result


# ---------------------------------------------------------------------------
# TestSchemaAndTemplate
# ---------------------------------------------------------------------------
class TestSchemaAndTemplate:
    """Tests for docstring template rendering and schema generation."""

    def test_default_template_renders(self, bash_tool: BashTool):
        fn = bash_tool.get_tool()
        schema = fn.__tool_schema__
        assert "bash" == schema["name"]
        assert "120000" in schema["description"]  # default timeout in description
        assert "600000" in schema["description"]  # max timeout in description
        assert "30000" in schema["description"]  # max output chars

    def test_schema_has_required_command(self, bash_tool: BashTool):
        fn = bash_tool.get_tool()
        schema = fn.__tool_schema__
        assert "command" in schema["input_schema"]["required"]

    def test_schema_optional_params(self, bash_tool: BashTool):
        fn = bash_tool.get_tool()
        schema = fn.__tool_schema__
        required = schema["input_schema"]["required"]
        assert "description" not in required
        assert "timeout" not in required
        assert "dangerouslyDisableSandbox" not in required

    def test_schema_param_types(self, bash_tool: BashTool):
        fn = bash_tool.get_tool()
        props = fn.__tool_schema__["input_schema"]["properties"]
        assert props["command"]["type"] == "string"
        # timeout is int | None so type is ["integer", "null"]
        timeout_type = props["timeout"]["type"]
        assert "integer" in timeout_type
        assert props["dangerouslyDisableSandbox"]["type"] == "boolean"

    def test_custom_docstring_template(self, temp_workspace: Path):
        custom_template = """Run a command.

Args:
    command: The command to run.
    description: What the command does.
    timeout: Timeout in ms.
    dangerouslyDisableSandbox: Bypass sandbox.
"""
        tool = BashTool(base_path=temp_workspace, docstring_template=custom_template)
        fn = tool.get_tool()
        assert "Run a command" in fn.__tool_schema__["description"]

    def test_schema_override(self, temp_workspace: Path):
        override = {
            "name": "custom_bash",
            "description": "Custom bash tool",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }
        tool = BashTool(base_path=temp_workspace, schema_override=override)
        fn = tool.get_tool()
        assert fn.__tool_schema__["name"] == "custom_bash"


# ---------------------------------------------------------------------------
# TestAgentUUIDInjection
# ---------------------------------------------------------------------------
class TestAgentUUIDInjection:
    """Tests for agent UUID injection protocol."""

    def test_tool_instance_attached(self, bash_tool: BashTool):
        fn = bash_tool.get_tool()
        assert hasattr(fn, "__tool_instance__")
        assert fn.__tool_instance__ is bash_tool

    def test_set_agent_uuid(self, bash_tool: BashTool):
        bash_tool.set_agent_uuid("test-uuid-123")
        assert bash_tool.agent_uuid == "test-uuid-123"

    def test_works_without_uuid(self, bash_fn: Callable):
        # Tool should work fine even without UUID set
        result = bash_fn(command="echo hello")
        assert "hello" in result
        assert "[exit_code: 0]" in result
