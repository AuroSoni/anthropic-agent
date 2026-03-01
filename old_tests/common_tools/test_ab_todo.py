"""Tests for agent_base.common_tools.todo_tool."""

import json

import pytest

from agent_base.common_tools.todo_tool import TodoWriteTool, CheckTodoTool
from agent_base.sandbox.local import LocalSandbox


@pytest.fixture()
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="test-todo", base_dir=str(tmp_path))
    async with sb:
        yield sb


@pytest.fixture()
def write_tool(sandbox):
    t = TodoWriteTool()
    t.set_sandbox(sandbox)
    return t


@pytest.fixture()
def check_tool(sandbox):
    t = CheckTodoTool()
    t.set_sandbox(sandbox)
    return t


# ─── Registration ──────────────────────────────────────────────────


def test_write_tool_has_schema(write_tool) -> None:
    func = write_tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__["name"] == "todo_write"


def test_check_tool_has_schema(check_tool) -> None:
    func = check_tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__["name"] == "check_todo"


# ─── Happy Path ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_and_check_todos(sandbox, write_tool, check_tool) -> None:
    write_fn = write_tool.get_tool()
    check_fn = check_tool.get_tool()

    todos = json.dumps([
        {"id": "task-1", "content": "Do something", "status": "pending", "activeForm": "Doing something"},
        {"id": "task-2", "content": "Do more", "status": "in_progress", "activeForm": "Doing more"},
    ])

    result = await write_fn(todos=todos)
    assert "Saved 2 todo(s)" in result
    assert "1 in progress" in result
    assert "1 pending" in result

    check_result = await check_fn()
    assert "task-1" in check_result
    assert "task-2" in check_result
    assert "[ ]" in check_result  # pending
    assert "[~]" in check_result  # in_progress


@pytest.mark.asyncio
async def test_check_empty_todos(sandbox, check_tool) -> None:
    check_fn = check_tool.get_tool()
    result = await check_fn()
    assert "No todos found" in result


# ─── Error Cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_invalid_json(sandbox, write_tool) -> None:
    write_fn = write_tool.get_tool()
    result = await write_fn(todos="not json")
    assert "Error" in result
    assert "parse" in result.lower()


@pytest.mark.asyncio
async def test_write_invalid_status(sandbox, write_tool) -> None:
    write_fn = write_tool.get_tool()
    todos = json.dumps([
        {"id": "t1", "content": "Task", "status": "invalid_status", "activeForm": "Tasking"},
    ])
    result = await write_fn(todos=todos)
    assert "Error" in result
    assert "invalid_status" in result.lower()


@pytest.mark.asyncio
async def test_write_missing_fields(sandbox, write_tool) -> None:
    write_fn = write_tool.get_tool()
    todos = json.dumps([{"id": "t1"}])
    result = await write_fn(todos=todos)
    assert "Error" in result
    assert "missing" in result.lower()


@pytest.mark.asyncio
async def test_write_duplicate_ids(sandbox, write_tool) -> None:
    write_fn = write_tool.get_tool()
    todos = json.dumps([
        {"id": "dup", "content": "First", "status": "pending", "activeForm": "First"},
        {"id": "dup", "content": "Second", "status": "pending", "activeForm": "Second"},
    ])
    result = await write_fn(todos=todos)
    assert "Error" in result
    assert "Duplicate" in result
