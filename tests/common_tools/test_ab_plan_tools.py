"""Tests for agent_base.common_tools.plan_tools."""

import json

import pytest
import yaml

from agent_base.common_tools.plan_tools import (
    EnterPlanModeTool,
    ExitPlanModeTool,
    CreatePlanTool,
    EditPlanTool,
)
from agent_base.sandbox.local import LocalSandbox


@pytest.fixture()
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="test-plan", base_dir=str(tmp_path))
    async with sb:
        yield sb


# ─── Frontend Tools ────────────────────────────────────────────────


def test_enter_plan_mode_schema() -> None:
    tool = EnterPlanModeTool()
    func = tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__["name"] == "enter_plan_mode"
    assert func.__tool_executor__ == "frontend"


def test_exit_plan_mode_schema() -> None:
    tool = ExitPlanModeTool()
    func = tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__["name"] == "exit_plan_mode"
    assert func.__tool_executor__ == "frontend"


# ─── CreatePlanTool ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_plan(sandbox) -> None:
    tool = CreatePlanTool()
    tool.set_sandbox(sandbox)
    func = tool.get_tool()

    todos_yaml = yaml.dump([
        {"id": "step-1", "content": "Research", "status": "pending", "activeForm": "Researching"},
    ])

    plan_id = await func(title="Test Plan", overview="A test plan.", todos=todos_yaml)
    assert "test-plan" in plan_id

    # Verify file was created
    plan_path = f"plans/{plan_id}.yaml"
    exists = await sandbox.file_exists(plan_path)
    assert exists

    content = await sandbox.read_file(plan_path)
    data = yaml.safe_load(content)
    assert data["title"] == "Test Plan"
    assert data["overview"] == "A test plan."
    assert len(data["todos"]) == 1


@pytest.mark.asyncio
async def test_create_plan_empty_title(sandbox) -> None:
    tool = CreatePlanTool()
    tool.set_sandbox(sandbox)
    func = tool.get_tool()
    result = await func(title="", overview="text", todos="[]")
    assert "Error" in result
    assert "empty" in result.lower()


@pytest.mark.asyncio
async def test_create_plan_invalid_todos(sandbox) -> None:
    tool = CreatePlanTool()
    tool.set_sandbox(sandbox)
    func = tool.get_tool()
    result = await func(title="Plan", overview="text", todos="not valid yaml [")
    assert "Error" in result


# ─── EditPlanTool ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_edit_plan(sandbox) -> None:
    # Create a plan first
    create_tool = CreatePlanTool()
    create_tool.set_sandbox(sandbox)
    create_fn = create_tool.get_tool()

    todos_yaml = yaml.dump([
        {"id": "s1", "content": "Step 1", "status": "pending", "activeForm": "Doing Step 1"},
    ])
    plan_id = await create_fn(title="Edit Test", overview="Original overview.", todos=todos_yaml)

    # Now edit it
    edit_tool = EditPlanTool()
    edit_tool.set_sandbox(sandbox)
    edit_fn = edit_tool.get_tool()

    overview_hunk = (
        "@@\n"
        "-overview: Original overview.\n"
        "+overview: Updated overview.\n"
    )

    result = await edit_fn(plan_id=plan_id, overview_patch=overview_hunk)
    assert "updated successfully" in result

    # Verify the change
    content = await sandbox.read_file(f"plans/{plan_id}.yaml")
    assert "Updated overview" in content


@pytest.mark.asyncio
async def test_edit_plan_not_found(sandbox) -> None:
    edit_tool = EditPlanTool()
    edit_tool.set_sandbox(sandbox)
    edit_fn = edit_tool.get_tool()
    result = await edit_fn(plan_id="nonexistent", overview_patch="@@ some patch")
    assert "Error" in result
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_edit_plan_no_patches(sandbox) -> None:
    # Create a plan first
    create_tool = CreatePlanTool()
    create_tool.set_sandbox(sandbox)
    create_fn = create_tool.get_tool()
    plan_id = await create_fn(title="NoEdit", overview="text", todos="[]")

    edit_tool = EditPlanTool()
    edit_tool.set_sandbox(sandbox)
    edit_fn = edit_tool.get_tool()
    result = await edit_fn(plan_id=plan_id)
    assert "Error" in result
    assert "No patches" in result
