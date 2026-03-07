"""Tests for agent_base.common_tools.code_execution_tool."""

import pytest

from agent_base.common_tools.code_execution_tool import CodeExecutionTool
from agent_base.sandbox.local import LocalSandbox
from agent_base.tools.decorators import tool


@pytest.fixture()
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="test-code-exec", base_dir=str(tmp_path))
    async with sb:
        yield sb


@pytest.fixture()
def code_tool(sandbox):
    t = CodeExecutionTool(
        output_dir="code_runs",
        max_output_chars=5_000,
        agent_uuid="test-agent-uuid",
    )
    t.set_sandbox(sandbox)
    return t


# ─── Registration ──────────────────────────────────────────────────


def test_tool_has_schema(code_tool) -> None:
    func = code_tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__.name == "code_execution"


def test_tool_has_instance_ref(code_tool) -> None:
    func = code_tool.get_tool()
    assert hasattr(func, "__tool_instance__")
    assert func.__tool_instance__ is code_tool


# ─── Basic Execution ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_simple_print(sandbox, code_tool) -> None:
    func = code_tool.get_tool()
    result = await func(code='print("hello world")')
    assert "hello world" in result
    assert "output_path=" in result


@pytest.mark.asyncio
async def test_last_expression_value(sandbox, code_tool) -> None:
    func = code_tool.get_tool()
    result = await func(code="2 + 3")
    assert "[Last value]:" in result
    assert "5" in result


@pytest.mark.asyncio
async def test_state_persists_between_calls(sandbox, code_tool) -> None:
    func = code_tool.get_tool()
    await func(code="x = 42")
    result = await func(code="print(x)")
    assert "42" in result


# ─── Error Handling ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execution_error(sandbox, code_tool) -> None:
    func = code_tool.get_tool()
    result = await func(code="1 / 0")
    assert "Error" in result


@pytest.mark.asyncio
async def test_undefined_variable_error(sandbox, code_tool) -> None:
    func = code_tool.get_tool()
    result = await func(code="print(undefined_var)")
    assert "Error" in result


# ─── Output File Writing ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_output_file_written(sandbox, code_tool) -> None:
    func = code_tool.get_tool()
    result = await func(code='print("file test")')
    # Extract the output path from the result
    assert "output_path=code_runs/test-agent-uuid/" in result
    assert result.split("\n")[0].endswith(".txt")


@pytest.mark.asyncio
async def test_no_uuid_warns(sandbox) -> None:
    t = CodeExecutionTool(output_dir="code_runs", max_output_chars=5_000)
    t.set_sandbox(sandbox)
    func = t.get_tool()
    result = await func(code='print("no uuid")')
    assert "Warning" in result
    assert "agent_uuid is not set" in result


# ─── Agent UUID Injection ────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_agent_uuid(sandbox) -> None:
    t = CodeExecutionTool(output_dir="code_runs", max_output_chars=5_000)
    t.set_sandbox(sandbox)
    t.set_agent_uuid("injected-uuid")
    func = t.get_tool()
    result = await func(code='print("uuid test")')
    assert "output_path=code_runs/injected-uuid/" in result


# ─── Truncation ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_output_truncation(sandbox) -> None:
    t = CodeExecutionTool(
        output_dir="code_runs",
        max_output_chars=100,
        agent_uuid="trunc-uuid",
    )
    t.set_sandbox(sandbox)
    func = t.get_tool()
    # Generate output longer than max_output_chars
    result = await func(code='print("A" * 500)')
    assert "truncated" in result


# ─── Reset State ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_state(sandbox, code_tool) -> None:
    func = code_tool.get_tool()
    await func(code="x = 99")
    code_tool.reset_state()
    result = await func(code="print(x)")
    assert "Error" in result


# ─── Embedded Tools ──────────────────────────────────────────────


def test_embedded_tool_docs_in_schema() -> None:
    @tool
    def add(a: float, b: float) -> str:
        """Add two numbers."""
        return str(a + b)

    t = CodeExecutionTool(
        output_dir="code_runs",
        embedded_tools=[add],
        max_output_chars=5_000,
        agent_uuid="test",
    )
    func = t.get_tool()
    desc = func.__tool_schema__.description
    assert "add" in desc


def test_duplicate_embedded_tool_names_raises() -> None:
    @tool
    def my_func() -> str:
        """A tool."""
        return "a"

    with pytest.raises(ValueError, match="Duplicate tool name"):
        CodeExecutionTool(
            output_dir="code_runs",
            embedded_tools=[my_func, my_func],
        )


# ─── Docstring Template ──────────────────────────────────────────


def test_docstring_includes_authorized_imports() -> None:
    t = CodeExecutionTool(
        output_dir="code_runs",
        authorized_imports=["numpy", "pandas"],
        max_output_chars=5_000,
    )
    func = t.get_tool()
    desc = func.__tool_schema__.description
    assert "numpy" in desc
    assert "pandas" in desc


def test_docstring_unrestricted_imports() -> None:
    t = CodeExecutionTool(
        output_dir="code_runs",
        authorized_imports=["*"],
        max_output_chars=5_000,
    )
    func = t.get_tool()
    desc = func.__tool_schema__.description
    assert "unrestricted" in desc.lower()
