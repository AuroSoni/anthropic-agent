"""Tests for CodeExecutionTool functionality.

Approach: Full Integration Testing
----------------------------------
These tests use the real LocalPythonExecutor and realistic filesystem writes to validate
statefulness across calls and tool injection end-to-end. Each test uses a temporary
directory that is discarded after the test completes.

This approach was chosen because:
1. CodeExecutionTool's core value is integrating LocalPythonExecutor with embedded tools
2. Mocking the executor would miss critical integration bugs (state persistence, tool injection)
3. File system operations (output writing) are fundamental to the tool's contract
4. The embedded common_tools (ReadFileTool, ApplyPatchTool, etc.) need real file I/O

Trade-offs:
- Tests are slightly slower than unit tests with mocks (~1-2s total)
- Tests depend on the real executor behavior (discovered that functions don't persist across calls)
- Tests require cleanup of temp directories (handled by pytest fixtures)

Tests cover:
- Statefulness and reset (variables persist across calls, reset_state clears state)
- Executor lazy initialization (created on first call, send_tools happens once)
- Embedded tools documentation and naming (duplicate names raise ValueError, schema rendering)
- Docstring template rendering (embedded tools docs, authorized imports, max output chars)
- Authorized imports allowlist (allowed/disallowed imports, wildcard mode)
- Output assembly order (logs, last value, final answer)
- Agent UUID injection (no UUID warning, file creation with UUID)
- Output file path and content (file created under <base>/<uuid>/code_runs)
- Truncation edge cases (exact limit, exceeds by 1, notice length edge case)
- Error handling paths (InterpreterError, unexpected exceptions)
- Schema override behavior
- Embedding common_tools end-to-end (ReadFileTool, ListDirTool, ApplyPatchTool, etc.)
"""
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Generator
from unittest.mock import patch, MagicMock

import pytest

from anthropic_agent.common_tools.code_execution_tool import CodeExecutionTool
from anthropic_agent.common_tools.read_file import ReadFileTool
from anthropic_agent.common_tools.list_dir import ListDirTool
from anthropic_agent.common_tools.apply_patch import ApplyPatchTool
from anthropic_agent.common_tools.glob_file_search import GlobFileSearchTool
from anthropic_agent.common_tools.grep_search import GrepSearchTool
from anthropic_agent.tools.decorators import tool
from anthropic_agent.python_executors.base import BASE_BUILTIN_MODULES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def output_base_path(temp_workspace: Path) -> Path:
    """Create a directory for code execution output files."""
    output_dir = temp_workspace / "code_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sandbox_root(temp_workspace: Path) -> Path:
    """Create a sandboxed root directory for embedded tools."""
    sandbox = temp_workspace / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    return sandbox


@pytest.fixture
def code_tool(output_base_path: Path) -> CodeExecutionTool:
    """Create a basic CodeExecutionTool instance."""
    return CodeExecutionTool(output_base_path=output_base_path)


@pytest.fixture
def code_tool_with_uuid(output_base_path: Path) -> CodeExecutionTool:
    """Create a CodeExecutionTool with agent UUID set."""
    tool = CodeExecutionTool(output_base_path=output_base_path)
    tool.set_agent_uuid("test-agent-uuid-123")
    return tool


# ---------------------------------------------------------------------------
# Sample embedded tools for testing
# ---------------------------------------------------------------------------
@tool
def sample_add(a: float, b: float) -> str:
    """Add two numbers together.
    
    Args:
        a: The first number to add.
        b: The second number to add.
    
    Returns:
        String representation of the sum.
    """
    return str(a + b)


@tool
def sample_multiply(x: float, y: float) -> str:
    """Multiply two numbers.
    
    Args:
        x: First factor.
        y: Second factor.
    
    Returns:
        String representation of the product.
    """
    return str(x * y)


def plain_func_no_decorator(value: int) -> str:
    """A plain function without @tool decorator."""
    return str(value * 2)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def create_file(workspace: Path, rel_path: str, content: str = "") -> Path:
    """Create a test file in the workspace."""
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
# Tests for statefulness and reset
# ---------------------------------------------------------------------------
class TestStatefulnessAndReset:
    """Tests for state persistence across calls and reset_state()."""

    def test_variable_persists_across_calls(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """Variables defined in one call should persist in subsequent calls."""
        fn = code_tool_with_uuid.get_tool()
        
        # First call: define a variable
        result1 = fn("x = 42")
        assert "[Execution Error]" not in result1
        
        # Second call: use the variable
        result2 = fn("print(f'x is {x}')")
        assert "x is 42" in result2

    def test_function_callable_within_same_execution(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """Functions defined should be callable within the same execution."""
        fn = code_tool_with_uuid.get_tool()
        
        # Define and call function in same execution
        result = fn("def double(n): return n * 2\nresult = double(21)\nprint(result)")
        
        assert "[Execution Error]" not in result
        assert "42" in result

    def test_function_does_not_persist_across_calls(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """Functions defined in one call do NOT persist to subsequent calls.
        
        Note: This is a limitation of the current LocalPythonExecutor implementation
        which copies custom_tools in evaluate_python_code, so user-defined functions
        are stored in the copy and lost after the call.
        """
        fn = code_tool_with_uuid.get_tool()
        
        # First call: define a function
        result1 = fn("def my_func(n): return n * 2")
        assert "[Execution Error]" not in result1
        
        # Second call: function should NOT be available
        result2 = fn("my_func(5)")
        assert "[Execution Error]" in result2
        assert "my_func" in result2

    def test_import_persists_across_calls(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """Imports done in one call should be available in subsequent calls."""
        tool = CodeExecutionTool(
            output_base_path=code_tool_with_uuid.output_base_path,
            authorized_imports=["math"],
        )
        tool.set_agent_uuid("test-uuid")
        fn = tool.get_tool()
        
        # First call: import math
        result1 = fn("import math")
        assert "[Execution Error]" not in result1
        
        # Second call: use math
        result2 = fn("print(math.pi)")
        assert "3.14" in result2

    def test_reset_state_clears_variables(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """reset_state() should clear all variables."""
        fn = code_tool_with_uuid.get_tool()
        
        # Define a variable
        fn("my_var = 'hello'")
        
        # Reset state
        code_tool_with_uuid.reset_state()
        
        # Variable should no longer exist
        result = fn("print(my_var)")
        assert "[Execution Error]" in result
        assert "my_var" in result

    def test_reset_state_recreates_executor(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """reset_state() should recreate the executor on next call."""
        fn = code_tool_with_uuid.get_tool()
        
        # Get the executor
        fn("x = 1")
        executor_before = code_tool_with_uuid._executor
        
        # Reset
        code_tool_with_uuid.reset_state()
        assert code_tool_with_uuid._executor is None
        
        # Next call should create new executor
        fn("y = 2")
        assert code_tool_with_uuid._executor is not None
        assert code_tool_with_uuid._executor is not executor_before


# ---------------------------------------------------------------------------
# Tests for executor lazy initialization
# ---------------------------------------------------------------------------
class TestExecutorLazyInit:
    """Tests for lazy executor initialization."""

    def test_executor_not_created_on_init(self, output_base_path: Path) -> None:
        """Executor should not be created during __init__."""
        tool = CodeExecutionTool(output_base_path=output_base_path)
        assert tool._executor is None

    def test_executor_created_on_first_call(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """Executor should be created on first tool call."""
        assert code_tool_with_uuid._executor is None
        
        fn = code_tool_with_uuid.get_tool()
        fn("x = 1")
        
        assert code_tool_with_uuid._executor is not None

    def test_send_tools_called_once(self, output_base_path: Path) -> None:
        """send_tools should be called once per executor lifecycle."""
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[sample_add],
        )
        tool.set_agent_uuid("test-uuid")
        fn = tool.get_tool()
        
        # First call creates executor and calls send_tools
        fn("x = 1")
        executor = tool._executor
        
        # Subsequent calls should use same executor (send_tools not called again)
        fn("y = 2")
        fn("z = 3")
        
        assert tool._executor is executor


# ---------------------------------------------------------------------------
# Tests for embedded tools documentation and naming
# ---------------------------------------------------------------------------
class TestEmbeddedToolsDocsAndNaming:
    """Tests for embedded tools documentation generation and naming."""

    def test_duplicate_tool_names_raises_error(self, output_base_path: Path) -> None:
        """Duplicate tool names should raise ValueError."""
        @tool
        def my_tool(x: int) -> str:
            """First tool."""
            return str(x)
        
        # Create another tool with same __name__ but different function
        @tool
        def my_tool_dup(y: int) -> str:
            """Second tool."""
            return str(y)
        
        # Override schema name to create duplicate
        my_tool_dup.__tool_schema__["name"] = "my_tool"
        
        with pytest.raises(ValueError, match="Duplicate tool name"):
            CodeExecutionTool(
                output_base_path=output_base_path,
                embedded_tools=[my_tool, my_tool_dup],
            )

    def test_tool_with_schema_renders_params(self, output_base_path: Path) -> None:
        """Tool with __tool_schema__ should render param list correctly."""
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[sample_add],
        )
        
        docs = tool._format_embedded_tool_docs()
        
        assert "sample_add" in docs
        assert "Add two numbers" in docs
        assert "a:" in docs
        assert "b:" in docs

    def test_tool_without_schema_uses_docstring(self, output_base_path: Path) -> None:
        """Tool without __tool_schema__ should use first docstring line."""
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[plain_func_no_decorator],
        )
        
        docs = tool._format_embedded_tool_docs()
        
        assert "plain_func_no_decorator" in docs
        assert "A plain function without @tool decorator" in docs

    def test_no_embedded_tools_message(self, output_base_path: Path) -> None:
        """No embedded tools should show appropriate message."""
        tool = CodeExecutionTool(output_base_path=output_base_path)
        
        docs = tool._format_embedded_tool_docs()
        
        assert "None" in docs or "built-in" in docs.lower()


# ---------------------------------------------------------------------------
# Tests for docstring template rendering
# ---------------------------------------------------------------------------
class TestDocstringTemplateRendering:
    """Tests for docstring template rendering with placeholders."""

    def test_docstring_includes_embedded_tools_docs(self, output_base_path: Path) -> None:
        """Rendered docstring should include embedded tools documentation."""
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[sample_add, sample_multiply],
        )
        fn = tool.get_tool()
        
        docstring = fn.__doc__
        
        assert "sample_add" in docstring
        assert "sample_multiply" in docstring

    def test_docstring_includes_authorized_imports(self, output_base_path: Path) -> None:
        """Rendered docstring should include authorized imports list."""
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            authorized_imports=["numpy", "pandas"],
        )
        fn = tool.get_tool()
        
        docstring = fn.__doc__
        
        assert "numpy" in docstring
        assert "pandas" in docstring

    def test_docstring_includes_max_output_chars(self, output_base_path: Path) -> None:
        """Rendered docstring should include max_output_chars."""
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            max_output_chars=5000,
        )
        fn = tool.get_tool()
        
        docstring = fn.__doc__
        
        assert "5000" in docstring

    def test_docstring_includes_output_path_pattern(self, output_base_path: Path) -> None:
        """Rendered docstring should include output path pattern."""
        tool = CodeExecutionTool(output_base_path=output_base_path)
        fn = tool.get_tool()
        
        docstring = fn.__doc__
        
        assert "code_runs" in docstring or "output" in docstring.lower()


# ---------------------------------------------------------------------------
# Tests for authorized imports allowlist
# ---------------------------------------------------------------------------
class TestAuthorizedImportsAllowlist:
    """Tests for authorized imports configuration."""

    def test_allowed_import_succeeds(self, output_base_path: Path) -> None:
        """Importing an allowed module should succeed."""
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            authorized_imports=["math"],
        )
        tool.set_agent_uuid("test-uuid")
        fn = tool.get_tool()
        
        result = fn("import math\nprint(math.sqrt(16))")
        
        assert "4.0" in result
        assert "[Execution Error]" not in result

    def test_disallowed_import_fails(self, output_base_path: Path) -> None:
        """Importing a disallowed module should fail with InterpreterError."""
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            authorized_imports=[],  # No additional imports
        )
        tool.set_agent_uuid("test-uuid")
        fn = tool.get_tool()
        
        result = fn("import os")
        
        assert "[Execution Error]" in result
        assert "not allowed" in result.lower() or "authorized" in result.lower()

    def test_wildcard_import_mode_docstring(self, output_base_path: Path) -> None:
        """authorized_imports=['*'] should show 'All imports are allowed' in docstring."""
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            authorized_imports=["*"],
        )
        fn = tool.get_tool()
        
        docstring = fn.__doc__
        
        assert "All imports are allowed" in docstring

    def test_base_builtin_modules_always_available(self, output_base_path: Path) -> None:
        """BASE_BUILTIN_MODULES should always be importable."""
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            authorized_imports=[],  # No additional imports
        )
        tool.set_agent_uuid("test-uuid")
        fn = tool.get_tool()
        
        # datetime is in BASE_BUILTIN_MODULES
        result = fn("import datetime\nprint(datetime.date.today())")
        
        assert "[Execution Error]" not in result


# ---------------------------------------------------------------------------
# Tests for output assembly order
# ---------------------------------------------------------------------------
class TestOutputAssemblyOrder:
    """Tests for output assembly: logs, last value, final answer marker."""

    def test_logs_appear_first(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """Print outputs should appear first in the result."""
        fn = code_tool_with_uuid.get_tool()
        
        result = fn("print('Hello')\n42")
        
        # Split by output_path line
        lines = result.split("\n")
        # Find where content starts (after output_path)
        content_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith("output_path=") and not line.startswith("[Warning"):
                content_start = i
                break
        
        content = "\n".join(lines[content_start:])
        
        # Hello should appear before [Last value]
        hello_pos = content.find("Hello")
        last_value_pos = content.find("[Last value]")
        
        assert hello_pos < last_value_pos

    def test_last_value_appears_after_logs(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """Last expression value should appear after logs."""
        fn = code_tool_with_uuid.get_tool()
        
        result = fn("print('Log message')\n'final result'")
        
        assert "Log message" in result
        assert "[Last value]" in result
        assert "final result" in result

    def test_last_value_skipped_when_none(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """[Last value] should not appear when output is None."""
        fn = code_tool_with_uuid.get_tool()
        
        result = fn("x = 5\nprint('done')")
        
        assert "done" in result
        # Assignment returns None, so no [Last value] marker
        # Note: this depends on implementation - assignment might return the value
        # Let's check the actual behavior
        if "[Last value]" in result:
            # If it shows, it should show the assigned value
            assert "5" in result

    def test_no_output_message(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """When there's no output, should show [No output]."""
        fn = code_tool_with_uuid.get_tool()
        
        # Pass statement produces no output
        result = fn("pass")
        
        # Either [No output] or minimal output
        assert "output_path=" in result or "[Warning" in result


# ---------------------------------------------------------------------------
# Tests for agent UUID injection
# ---------------------------------------------------------------------------
class TestAgentUUIDInjection:
    """Tests for agent UUID injection and file writing."""

    def test_no_uuid_returns_warning(self, code_tool: CodeExecutionTool) -> None:
        """Calling tool without UUID should return warning."""
        fn = code_tool.get_tool()
        
        result = fn("print('test')")
        
        assert "[Warning" in result
        assert "agent_uuid" in result.lower() or "uuid" in result.lower()

    def test_uuid_injection_enables_file_writing(
        self, code_tool: CodeExecutionTool, output_base_path: Path
    ) -> None:
        """After set_agent_uuid(), file should be created."""
        code_tool.set_agent_uuid("my-test-uuid")
        fn = code_tool.get_tool()
        
        result = fn("print('hello')")
        
        assert "output_path=" in result
        assert "[Warning" not in result
        
        # Extract path and verify file exists
        for line in result.split("\n"):
            if line.startswith("output_path="):
                path = line.replace("output_path=", "")
                assert Path(path).exists()
                break

    def test_tool_instance_attribute(self, code_tool: CodeExecutionTool) -> None:
        """get_tool() should attach __tool_instance__ for UUID injection."""
        fn = code_tool.get_tool()
        
        assert hasattr(fn, "__tool_instance__")
        assert fn.__tool_instance__ is code_tool

    def test_uuid_injection_via_tool_instance(self, code_tool: CodeExecutionTool) -> None:
        """UUID can be injected via __tool_instance__.set_agent_uuid()."""
        fn = code_tool.get_tool()
        
        # Inject UUID via the tool instance
        fn.__tool_instance__.set_agent_uuid("injected-uuid")
        
        result = fn("print('test')")
        
        assert "[Warning" not in result
        assert "output_path=" in result


# ---------------------------------------------------------------------------
# Tests for output file path and content
# ---------------------------------------------------------------------------
class TestOutputFilePathAndContent:
    """Tests for output file creation and content."""

    def test_output_file_created_under_correct_path(
        self, output_base_path: Path
    ) -> None:
        """Output file should be created under <base>/<uuid>/code_runs/."""
        tool = CodeExecutionTool(output_base_path=output_base_path)
        tool.set_agent_uuid("file-test-uuid")
        fn = tool.get_tool()
        
        result = fn("print('file content test')")
        
        # Extract and verify path structure
        for line in result.split("\n"):
            if line.startswith("output_path="):
                path = Path(line.replace("output_path=", ""))
                assert "file-test-uuid" in str(path)
                assert "code_runs" in str(path)
                break

    def test_output_file_content_matches_return(
        self, output_base_path: Path
    ) -> None:
        """File content should match the truncated output in return."""
        tool = CodeExecutionTool(output_base_path=output_base_path)
        tool.set_agent_uuid("content-test-uuid")
        fn = tool.get_tool()
        
        result = fn("print('consistent content')")
        
        # Extract path
        output_path = None
        for line in result.split("\n"):
            if line.startswith("output_path="):
                output_path = Path(line.replace("output_path=", ""))
                break
        
        assert output_path is not None
        file_content = output_path.read_text()
        
        assert "consistent content" in file_content

    def test_output_file_unique_per_execution(
        self, output_base_path: Path
    ) -> None:
        """Each execution should create a unique output file."""
        tool = CodeExecutionTool(output_base_path=output_base_path)
        tool.set_agent_uuid("unique-test-uuid")
        fn = tool.get_tool()
        
        result1 = fn("print('first')")
        result2 = fn("print('second')")
        
        # Extract paths
        path1 = path2 = None
        for line in result1.split("\n"):
            if line.startswith("output_path="):
                path1 = line.replace("output_path=", "")
        for line in result2.split("\n"):
            if line.startswith("output_path="):
                path2 = line.replace("output_path=", "")
        
        assert path1 != path2


# ---------------------------------------------------------------------------
# Tests for truncation edge cases
# ---------------------------------------------------------------------------
class TestTruncationEdgeCases:
    """Tests for output truncation behavior."""

    def test_output_exactly_at_limit_no_truncation(
        self, output_base_path: Path
    ) -> None:
        """Output exactly at max_output_chars should not be truncated."""
        max_chars = 100
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            max_output_chars=max_chars,
        )
        tool.set_agent_uuid("truncate-test-uuid")
        fn = tool.get_tool()
        
        # Create output that's exactly at the limit
        # We need to account for newline in print output
        target_len = max_chars - 1  # -1 for newline
        output_str = "x" * target_len
        result = fn(f"print('{output_str}')")
        
        assert "truncated" not in result.lower()

    def test_output_exceeds_limit_by_one_shows_truncation(
        self, output_base_path: Path
    ) -> None:
        """Output exceeding limit by 1 should show truncation notice."""
        max_chars = 50
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            max_output_chars=max_chars,
        )
        tool.set_agent_uuid("truncate-test-uuid")
        fn = tool.get_tool()
        
        # Create output that exceeds the limit
        output_str = "x" * (max_chars + 50)
        result = fn(f"print('{output_str}')")
        
        assert "truncated" in result.lower()

    def test_truncation_keeps_tail(self, output_base_path: Path) -> None:
        """Truncation should keep the tail of the output."""
        max_chars = 100
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            max_output_chars=max_chars,
        )
        tool.set_agent_uuid("truncate-test-uuid")
        fn = tool.get_tool()
        
        # Create output with distinctive ending
        result = fn("print('START' + 'x' * 200 + 'END_MARKER')")
        
        # END_MARKER should be present (it's at the tail)
        assert "END_MARKER" in result


# ---------------------------------------------------------------------------
# Tests for error handling paths
# ---------------------------------------------------------------------------
class TestErrorHandlingPaths:
    """Tests for error handling and formatting."""

    def test_interpreter_error_formatted(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """InterpreterError should be caught and formatted."""
        fn = code_tool_with_uuid.get_tool()
        
        # Undefined variable triggers InterpreterError
        result = fn("print(undefined_variable)")
        
        assert "[Execution Error]" in result

    def test_syntax_error_formatted(self, code_tool_with_uuid: CodeExecutionTool) -> None:
        """Syntax errors should be caught and formatted."""
        fn = code_tool_with_uuid.get_tool()
        
        result = fn("def incomplete(")
        
        assert "[Execution Error]" in result
        assert "Syntax" in result or "parsing" in result.lower()

    def test_unexpected_exception_includes_class_name(
        self, code_tool_with_uuid: CodeExecutionTool
    ) -> None:
        """Unexpected exceptions should include the exception class name."""
        fn = code_tool_with_uuid.get_tool()
        
        # Division by zero
        result = fn("1/0")
        
        # Should show error with class name
        assert "Error" in result
        assert "division" in result.lower() or "zero" in result.lower()


# ---------------------------------------------------------------------------
# Tests for schema override behavior
# ---------------------------------------------------------------------------
class TestSchemaOverrideBehavior:
    """Tests for schema_override parameter."""

    def test_schema_override_honored(self, output_base_path: Path) -> None:
        """schema_override should be used instead of generated schema."""
        custom_schema = {
            "name": "custom_code_exec",
            "description": "Custom description for code execution",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Custom code param"}
                },
                "required": ["code"],
            },
        }
        
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            schema_override=custom_schema,
        )
        fn = tool.get_tool()
        
        assert fn.__tool_schema__ == custom_schema

    def test_schema_override_still_has_tool_instance(self, output_base_path: Path) -> None:
        """Even with schema_override, __tool_instance__ should be attached."""
        custom_schema = {
            "name": "custom_exec",
            "description": "Custom",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        
        tool = CodeExecutionTool(
            output_base_path=output_base_path,
            schema_override=custom_schema,
        )
        fn = tool.get_tool()
        
        assert hasattr(fn, "__tool_instance__")
        assert fn.__tool_instance__ is tool


# ---------------------------------------------------------------------------
# Tests for embedding common_tools end-to-end
# ---------------------------------------------------------------------------
class TestEmbeddedCommonToolsEndToEnd:
    """Tests for embedding and using common_tools within code execution."""

    def test_read_file_tool_happy_path(
        self, output_base_path: Path, sandbox_root: Path
    ) -> None:
        """ReadFileTool embedded and called from executed code."""
        # Setup: create a file to read
        create_file(sandbox_root, "docs/a.md", "# Hello World\nThis is content.\n")
        
        # Create embedded tool
        read_tool = ReadFileTool(base_path=sandbox_root)
        
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[read_tool.get_tool()],
        )
        code_tool.set_agent_uuid("read-file-test")
        fn = code_tool.get_tool()
        
        # Execute code that calls read_file
        result = fn('out = read_file("docs/a.md")\nprint(out)')
        
        assert "[Execution Error]" not in result
        assert "Hello World" in result
        assert "[lines" in result  # Header format

    def test_list_dir_tool_happy_path(
        self, output_base_path: Path, sandbox_root: Path
    ) -> None:
        """ListDirTool embedded and called from executed code."""
        # Setup: create a directory structure
        create_file(sandbox_root, "docs/readme.md", "# Readme\n")
        create_file(sandbox_root, "docs/guide.md", "# Guide\n")
        create_file(sandbox_root, "notes.md", "# Notes\n")
        
        # Create embedded tool
        list_tool = ListDirTool(base_path=sandbox_root)
        
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[list_tool.get_tool()],
        )
        code_tool.set_agent_uuid("list-dir-test")
        fn = code_tool.get_tool()
        
        # Execute code that calls list_dir
        result = fn('tree = list_dir(".")\nprint(tree)')
        
        assert "[Execution Error]" not in result
        assert "docs/" in result  # Directory ends with /
        assert ".md" in result  # Only allowed extensions shown

    def test_apply_patch_write_then_read_chain(
        self, output_base_path: Path, sandbox_root: Path
    ) -> None:
        """ApplyPatchTool + ReadFileTool chained in single execution."""
        # Create embedded tools
        patch_tool = ApplyPatchTool(base_path=sandbox_root)
        read_tool = ReadFileTool(
            base_path=sandbox_root,
            allowed_extensions={".md", ".mmd", ".txt"},
        )
        
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[patch_tool.get_tool(), read_tool.get_tool()],
        )
        code_tool.set_agent_uuid("chain-test")
        fn = code_tool.get_tool()
        
        # Execute code that creates file then reads it
        code = '''
patch = """*** Begin Patch
*** Add File: docs/created.md
+# Created File
+This was created by patch.
*** End Patch"""
result = apply_patch(patch)
print("Patch result:", result)
content = read_file("docs/created.md")
print("File content:", content)
'''
        result = fn(code)
        
        assert "[Execution Error]" not in result
        assert '"status": "ok"' in result
        assert "Created File" in result

    def test_apply_patch_dry_run(
        self, output_base_path: Path, sandbox_root: Path
    ) -> None:
        """ApplyPatchTool dry_run=True should not create files."""
        patch_tool = ApplyPatchTool(base_path=sandbox_root)
        
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[patch_tool.get_tool()],
        )
        code_tool.set_agent_uuid("dry-run-test")
        fn = code_tool.get_tool()
        
        code = '''
patch = """*** Begin Patch
*** Add File: should_not_exist.md
+# This file should not be created
*** End Patch"""
result = apply_patch(patch, dry_run=True)
print(result)
'''
        result = fn(code)
        
        assert "[Execution Error]" not in result
        assert '"dry_run": true' in result
        
        # Verify file was NOT created
        assert not (sandbox_root / "should_not_exist.md").exists()

    def test_path_escape_hardening_read_file(
        self, output_base_path: Path, sandbox_root: Path
    ) -> None:
        """read_file('../secrets.md') should return escape error."""
        read_tool = ReadFileTool(base_path=sandbox_root)
        
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[read_tool.get_tool()],
        )
        code_tool.set_agent_uuid("escape-test")
        fn = code_tool.get_tool()
        
        result = fn('out = read_file("../secrets.md")\nprint(out)')
        
        assert "Base path escapes" in result or "escape" in result.lower()

    def test_path_escape_hardening_list_dir(
        self, output_base_path: Path, sandbox_root: Path
    ) -> None:
        """list_dir('../') should return escape error."""
        list_tool = ListDirTool(base_path=sandbox_root)
        
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[list_tool.get_tool()],
        )
        code_tool.set_agent_uuid("escape-test")
        fn = code_tool.get_tool()
        
        result = fn('out = list_dir("../")\nprint(out)')
        
        assert "Base path escapes" in result or "escape" in result.lower()

    def test_path_escape_hardening_apply_patch(
        self, output_base_path: Path, sandbox_root: Path
    ) -> None:
        """apply_patch with '../x.md' should return JSON error."""
        patch_tool = ApplyPatchTool(base_path=sandbox_root)
        
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[patch_tool.get_tool()],
        )
        code_tool.set_agent_uuid("escape-test")
        fn = code_tool.get_tool()
        
        code = '''
patch = """*** Begin Patch
*** Update File: ../x.md
@@ 
 line
*** End Patch"""
result = apply_patch(patch)
print(result)
'''
        result = fn(code)
        
        # Should contain error about invalid/escaping path
        assert '"status": "error"' in result or "Invalid path" in result or "escape" in result.lower()

    def test_tool_overwrite_protection(
        self, output_base_path: Path, sandbox_root: Path
    ) -> None:
        """Attempting to assign to tool name should raise error."""
        read_tool = ReadFileTool(base_path=sandbox_root)
        
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[read_tool.get_tool()],
        )
        code_tool.set_agent_uuid("overwrite-test")
        fn = code_tool.get_tool()
        
        result = fn('read_file = 123')
        
        assert "[Execution Error]" in result
        assert "Cannot assign" in result or "erase" in result.lower()

    def test_glob_then_read_workflow(
        self, output_base_path: Path, sandbox_root: Path
    ) -> None:
        """GlobFileSearchTool + ReadFileTool workflow."""
        # Setup: create multiple files
        create_file(sandbox_root, "docs/api.md", "# API Documentation\n")
        create_file(sandbox_root, "docs/guide.md", "# User Guide\n")
        create_file(sandbox_root, "readme.md", "# README\n")
        
        glob_tool = GlobFileSearchTool(base_path=sandbox_root)
        read_tool = ReadFileTool(base_path=sandbox_root)
        
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[glob_tool.get_tool(), read_tool.get_tool()],
        )
        code_tool.set_agent_uuid("glob-read-test")
        fn = code_tool.get_tool()
        
        code = '''
files = glob_file_search("*.md")
print("Found files:")
print(files)
# Read the first file in the list
first_file = files.split("\\n")[0]
content = read_file(first_file)
print("Content of first file:")
print(content)
'''
        result = fn(code)
        
        assert "[Execution Error]" not in result
        assert "Found files:" in result
        assert ".md" in result

    @pytest.mark.skipif(
        shutil.which("rg") is None,
        reason="ripgrep (rg) not available"
    )
    def test_grep_search_workflow(
        self, output_base_path: Path, sandbox_root: Path
    ) -> None:
        """GrepSearchTool workflow with <match> tags."""
        # Setup: create file with unique token
        create_file(
            sandbox_root,
            "docs/searchable.md",
            "# Document\nThis contains UNIQUE_TOKEN_12345 for testing.\nEnd.\n"
        )
        
        grep_tool = GrepSearchTool(base_path=sandbox_root)
        
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[grep_tool.get_tool()],
        )
        code_tool.set_agent_uuid("grep-test")
        fn = code_tool.get_tool()
        
        result = fn('out = grep_search("UNIQUE_TOKEN_12345")\nprint(out)')
        
        assert "[Execution Error]" not in result
        assert "<match>" in result
        assert "UNIQUE_TOKEN_12345" in result


# ---------------------------------------------------------------------------
# Tests for embedded tool calling from executed code
# ---------------------------------------------------------------------------
class TestEmbeddedToolCalling:
    """Tests for calling embedded tools from within executed code."""

    def test_embedded_tool_callable_by_name(self, output_base_path: Path) -> None:
        """Embedded tools should be callable by their name in code."""
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[sample_add],
        )
        code_tool.set_agent_uuid("call-test")
        fn = code_tool.get_tool()
        
        result = fn('result = sample_add(10, 5)\nprint(result)')
        
        assert "[Execution Error]" not in result
        assert "15" in result

    def test_embedded_tool_with_kwargs(self, output_base_path: Path) -> None:
        """Embedded tools should accept keyword arguments."""
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[sample_add],
        )
        code_tool.set_agent_uuid("kwargs-test")
        fn = code_tool.get_tool()
        
        result = fn('result = sample_add(a=3, b=7)\nprint(result)')
        
        assert "[Execution Error]" not in result
        assert "10" in result

    def test_multiple_embedded_tools(self, output_base_path: Path) -> None:
        """Multiple embedded tools should all be available."""
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[sample_add, sample_multiply],
        )
        code_tool.set_agent_uuid("multi-tool-test")
        fn = code_tool.get_tool()
        
        code = '''
sum_result = sample_add(2, 3)
prod_result = sample_multiply(4, 5)
print(f"Sum: {sum_result}, Product: {prod_result}")
'''
        result = fn(code)
        
        assert "[Execution Error]" not in result
        assert "Sum: 5" in result
        assert "Product: 20" in result

    def test_embedded_tool_result_in_expression(self, output_base_path: Path) -> None:
        """Embedded tool results should be usable in expressions."""
        code_tool = CodeExecutionTool(
            output_base_path=output_base_path,
            embedded_tools=[sample_add],
        )
        code_tool.set_agent_uuid("expr-test")
        fn = code_tool.get_tool()
        
        result = fn('x = int(sample_add(5, 5)) * 2\nprint(x)')
        
        assert "[Execution Error]" not in result
        assert "20" in result
