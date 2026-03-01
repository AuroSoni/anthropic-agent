"""Tests for agent_base.common_tools.read_file."""

import pytest

from agent_base.common_tools.read_file import ReadFileTool
from agent_base.sandbox.local import LocalSandbox


@pytest.fixture()
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="test-read", base_dir=str(tmp_path))
    async with sb:
        yield sb


@pytest.fixture()
def tool(sandbox):
    t = ReadFileTool(max_lines=10, allowed_extensions={".md", ".mmd"})
    t.set_sandbox(sandbox)
    return t


# ─── Registration ──────────────────────────────────────────────────


def test_tool_has_schema(tool) -> None:
    func = tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__.name == "read_file"


def test_tool_has_instance(tool) -> None:
    func = tool.get_tool()
    assert func.__tool_instance__ is tool


# ─── Happy Path ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_file_basic(sandbox, tool) -> None:
    await sandbox.write_file("workspace/hello.md", "line1\nline2\nline3\n")
    func = tool.get_tool()
    result = await func(target_file="workspace/hello.md")
    assert "[lines 1-3 of 3 in workspace/hello.md]" in result
    assert "line1" in result
    assert "line3" in result


@pytest.mark.asyncio
async def test_read_file_with_offset(sandbox, tool) -> None:
    lines = "\n".join(f"line{i}" for i in range(1, 21)) + "\n"
    await sandbox.write_file("workspace/big.md", lines)
    func = tool.get_tool()
    result = await func(target_file="workspace/big.md", start_line_one_indexed=5, no_of_lines_to_read=3)
    assert "[lines 5-7 of 20 in workspace/big.md]" in result
    assert "line5" in result
    assert "line7" in result


@pytest.mark.asyncio
async def test_read_file_clamps_max_lines(sandbox, tool) -> None:
    lines = "\n".join(f"line{i}" for i in range(1, 50)) + "\n"
    await sandbox.write_file("workspace/many.md", lines)
    func = tool.get_tool()
    result = await func(target_file="workspace/many.md")
    assert "[lines 1-10 of 49 in workspace/many.md]" in result


@pytest.mark.asyncio
async def test_read_file_auto_extension(sandbox, tool) -> None:
    await sandbox.write_file("workspace/readme.md", "# Hello\n")
    func = tool.get_tool()
    result = await func(target_file="workspace/readme")
    assert "Hello" in result


@pytest.mark.asyncio
async def test_read_empty_file(sandbox, tool) -> None:
    await sandbox.write_file("workspace/empty.md", "")
    func = tool.get_tool()
    result = await func(target_file="workspace/empty.md")
    assert "[lines 0-0 of 0 in workspace/empty.md]" in result


# ─── Error Cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_file_not_found(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(target_file="workspace/nonexistent.md")
    assert "Path does not exist" in result


@pytest.mark.asyncio
async def test_read_file_is_directory(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(target_file="workspace")
    assert "directory" in result.lower()


@pytest.mark.asyncio
async def test_read_file_offset_beyond_end(sandbox, tool) -> None:
    await sandbox.write_file("workspace/short.md", "one\ntwo\n")
    func = tool.get_tool()
    result = await func(target_file="workspace/short.md", start_line_one_indexed=100)
    assert "ERROR" in result
    assert "start_line_one_indexed(100)" in result


@pytest.mark.asyncio
async def test_read_file_negative_limit(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(target_file="workspace/any.md", no_of_lines_to_read=-5)
    assert "ERROR" in result
    assert "cannot be negative" in result


@pytest.mark.asyncio
async def test_read_file_zero_limit(sandbox, tool) -> None:
    await sandbox.write_file("workspace/data.md", "line1\nline2\n")
    func = tool.get_tool()
    result = await func(target_file="workspace/data.md", no_of_lines_to_read=0)
    assert "[lines 0-0 of 2 in workspace/data.md]" in result
