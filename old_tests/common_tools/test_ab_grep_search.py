"""Tests for agent_base.common_tools.grep_search."""

import pytest

from agent_base.common_tools.grep_search import GrepSearchTool
from agent_base.sandbox.local import LocalSandbox


@pytest.fixture()
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="test-grep", base_dir=str(tmp_path))
    async with sb:
        await sb.write_file("workspace/notes.md", "# TODO: fix this\n\nSome text here.\n\n# TODO: refactor\n")
        await sb.write_file("workspace/guide.md", "No matches in this file.\n")
        await sb.write_file("workspace/code.py", "# TODO in python\n")
        yield sb


@pytest.fixture()
def tool(sandbox):
    t = GrepSearchTool(max_match_lines=5, context_lines=1, allowed_extensions={".md"})
    t.set_sandbox(sandbox)
    return t


# ─── Registration ──────────────────────────────────────────────────


def test_tool_has_schema(tool) -> None:
    func = tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__.name == "grep_search"


# ─── Happy Path ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_grep_finds_matches(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(query="TODO", include_pattern="workspace/**/*.md")
    assert "TODO" in result
    assert "notes.md" in result


@pytest.mark.asyncio
async def test_grep_match_tags(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(query="TODO", include_pattern="workspace/**/*.md")
    assert "<match>" in result
    assert "</match>" in result


@pytest.mark.asyncio
async def test_grep_filters_by_extension(sandbox, tool) -> None:
    """Tool should only show results from allowed extensions."""
    func = tool.get_tool()
    result = await func(query="TODO")
    # .py is not in allowed_extensions (.md only), so code.py should not appear
    assert "code.py" not in result


# ─── Error Cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_grep_empty_query(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(query="")
    assert "cannot be empty" in result


@pytest.mark.asyncio
async def test_grep_no_matches(sandbox, tool) -> None:
    func = tool.get_tool()
    result = await func(query="NONEXISTENT_PATTERN_XYZ")
    assert "No matches found" in result
