"""Tests for agent_base.common_tools.apply_patch."""

import json

import pytest

from agent_base.common_tools.apply_patch import ApplyPatchTool
from agent_base.sandbox.local import LocalSandbox


@pytest.fixture()
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="test-patch", base_dir=str(tmp_path))
    async with sb:
        yield sb


@pytest.fixture()
def tool(sandbox):
    t = ApplyPatchTool()
    t.set_sandbox(sandbox)
    return t


# ─── Registration ──────────────────────────────────────────────────


def test_tool_has_schema(tool) -> None:
    func = tool.get_tool()
    assert hasattr(func, "__tool_schema__")
    assert func.__tool_schema__.name == "apply_patch"


def test_tool_has_instance(tool) -> None:
    func = tool.get_tool()
    assert func.__tool_instance__ is tool


# ─── Add File ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_file(sandbox, tool) -> None:
    func = tool.get_tool()
    patch = (
        "*** Begin Patch\n"
        "*** Add File: workspace/new.py\n"
        "+print('hello')\n"
        "*** End Patch"
    )
    result_str = await func(patch=patch)
    result = json.loads(result_str)
    assert result["status"] == "ok"
    assert result["op"] == "add"

    content = await sandbox.read_file("workspace/new.py")
    assert "print('hello')" in content


@pytest.mark.asyncio
async def test_add_file_already_exists(sandbox, tool) -> None:
    await sandbox.write_file("workspace/existing.py", "old content")
    func = tool.get_tool()
    patch = (
        "*** Begin Patch\n"
        "*** Add File: workspace/existing.py\n"
        "+new content\n"
        "*** End Patch"
    )
    result = json.loads(await func(patch=patch))
    assert result["status"] == "error"
    assert "already exists" in result["error"]


# ─── Update File ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_file(sandbox, tool) -> None:
    await sandbox.write_file("workspace/app.py", "def hello():\n    print('old')\n")
    func = tool.get_tool()
    patch = (
        "*** Begin Patch\n"
        "*** Update File: workspace/app.py\n"
        "@@\n"
        " def hello():\n"
        "-    print('old')\n"
        "+    print('new')\n"
        "*** End Patch"
    )
    result = json.loads(await func(patch=patch))
    assert result["status"] == "ok"
    assert result["op"] == "update"

    content = await sandbox.read_file("workspace/app.py")
    assert "print('new')" in content
    assert "print('old')" not in content


@pytest.mark.asyncio
async def test_update_file_not_found(sandbox, tool) -> None:
    func = tool.get_tool()
    patch = (
        "*** Begin Patch\n"
        "*** Update File: workspace/missing.py\n"
        "@@\n"
        " old\n"
        "+new\n"
        "*** End Patch"
    )
    result = json.loads(await func(patch=patch))
    assert result["status"] == "error"
    assert "does not exist" in result["error"]


# ─── Delete File ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_file(sandbox, tool) -> None:
    await sandbox.write_file("workspace/trash.py", "delete me\n")
    func = tool.get_tool()
    patch = (
        "*** Begin Patch\n"
        "*** Delete File: workspace/trash.py\n"
        "*** End Patch"
    )
    result = json.loads(await func(patch=patch))
    assert result["status"] == "ok"
    assert result["op"] == "delete"

    exists = await sandbox.file_exists("workspace/trash.py")
    assert not exists


@pytest.mark.asyncio
async def test_delete_file_not_found(sandbox, tool) -> None:
    func = tool.get_tool()
    patch = (
        "*** Begin Patch\n"
        "*** Delete File: workspace/ghost.py\n"
        "*** End Patch"
    )
    result = json.loads(await func(patch=patch))
    assert result["status"] == "error"
    assert "does not exist" in result["error"]


# ─── Dry Run ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dry_run_does_not_write(sandbox, tool) -> None:
    func = tool.get_tool()
    patch = (
        "*** Begin Patch\n"
        "*** Add File: workspace/dryrun.py\n"
        "+content\n"
        "*** End Patch"
    )
    result = json.loads(await func(patch=patch, dry_run=True))
    assert result["status"] == "ok"
    assert result["dry_run"] is True

    exists = await sandbox.file_exists("workspace/dryrun.py")
    assert not exists


# ─── Move File ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_move_file(sandbox, tool) -> None:
    await sandbox.write_file("workspace/old_name.py", "content\n")
    func = tool.get_tool()
    patch = (
        "*** Begin Patch\n"
        "*** Update File: workspace/old_name.py\n"
        "*** Move to: workspace/new_name.py\n"
        "@@\n"
        " content\n"
        "*** End Patch"
    )
    result = json.loads(await func(patch=patch))
    assert result["status"] == "ok"
    assert result["path"] == "workspace/new_name.py"

    old_exists = await sandbox.file_exists("workspace/old_name.py")
    new_exists = await sandbox.file_exists("workspace/new_name.py")
    assert not old_exists
    assert new_exists


# ─── Error Cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_patch_format(sandbox, tool) -> None:
    func = tool.get_tool()
    result = json.loads(await func(patch="not a valid patch"))
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_null_bytes_rejected(sandbox, tool) -> None:
    func = tool.get_tool()
    result = json.loads(await func(patch="*** Begin Patch\n\x00\n*** End Patch"))
    assert result["status"] == "error"
    assert "null bytes" in result["error"].lower()
