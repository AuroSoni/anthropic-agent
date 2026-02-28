"""Tests for agent_base.common_tools.glob_file_search."""

import pytest

from agent_base.common_tools.glob_file_search import GlobFileSearchTool
from agent_base.sandbox.local import LocalSandbox


@pytest.fixture()
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="test-glob", base_dir=str(tmp_path))
    async with sb:
        # Create test files in the workspace
        await sb.write_file("workspace/README.md", "# README\n")
        await sb.write_file("workspace/docs/guide.md", "# Guide\n")
        await sb.write_file("workspace/docs/api.mmd", "graph\n")
        await sb.write_file("workspace/src/main.py", "print('hi')\n")
        yield sb


@pytest.fixture()
def tool(sandbox):
    t = GlobFileSearchTool(max_results=10, allowed_extensions={".md", ".mmd"})
    t.set_sandbox(sandbox)
    return t


# ─── Registration ──────────────────────────────────────────────────


def test_tool_has_schema(tool) -> None:
    func = tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__["name"] == "glob_file_search"


# ─── Happy Path ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_find_md_files(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(glob_pattern="*.md", target_directory="workspace")
    assert "README.md" in result
    assert "guide.md" in result


@pytest.mark.asyncio
async def test_find_mmd_files(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(glob_pattern="*.mmd", target_directory="workspace")
    assert "api.mmd" in result


@pytest.mark.asyncio
async def test_filters_non_allowed_extensions(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(glob_pattern="*.py", target_directory="workspace")
    assert "No matches found" in result


# ─── Error Cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_target_directory_not_found(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(glob_pattern="*.md", target_directory="nonexistent")
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_no_matches(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(glob_pattern="*.xyz", target_directory="workspace")
    assert "No matches found" in result
