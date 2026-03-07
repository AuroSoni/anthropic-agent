"""Tests for agent_base.common_tools.list_dir."""

import pytest

from agent_base.common_tools.list_dir import ListDirTool
from agent_base.sandbox.local import LocalSandbox


@pytest.fixture()
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="test-listdir", base_dir=str(tmp_path))
    async with sb:
        await sb.write_file("workspace/README.md", "# README\n")
        await sb.write_file("workspace/docs/guide.md", "# Guide\n")
        await sb.write_file("workspace/docs/api.md", "# API\n")
        await sb.write_file("workspace/src/main.py", "print('hi')\n")
        await sb.write_file("workspace/src/utils.md", "# Utils\n")
        yield sb


@pytest.fixture()
def tool(sandbox):
    t = ListDirTool(max_depth=3, allowed_extensions={".md"})
    t.set_sandbox(sandbox)
    return t


# ─── Registration ──────────────────────────────────────────────────


def test_tool_has_schema(tool) -> None:
    func = tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__.name == "list_dir"


# ─── Happy Path ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_dir_basic(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(target_directory="workspace")
    assert "workspace/" in result
    assert "README.md" in result


@pytest.mark.asyncio
async def test_list_dir_shows_subdirectories(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(target_directory="workspace")
    assert "docs/" in result


@pytest.mark.asyncio
async def test_list_dir_filters_by_extension(sandbox, tool) -> None:
    """Should only show .md files (not .py)."""
    func = tool.get_tool()
    result = await func(target_directory="workspace")
    assert "main.py" not in result


@pytest.mark.asyncio
async def test_list_dir_shows_allowed_in_subdirs(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(target_directory="workspace")
    # src/ should be shown because it contains utils.md
    assert "src/" in result
    assert "utils.md" in result


# ─── Error Cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_dir_not_found(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(target_directory="nonexistent")
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_list_dir_file_not_dir(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(target_directory="workspace/README.md")
    assert "not a directory" in result.lower()
