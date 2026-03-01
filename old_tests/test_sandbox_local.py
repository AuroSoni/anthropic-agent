"""Tests for agent_base.sandbox.LocalSandbox."""

import asyncio

import pytest

from agent_base.sandbox import ExecResult, FileEntry, LocalSandbox, Sandbox


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sandbox(tmp_path):
    """Return an un-setup LocalSandbox using tmp_path as base_dir."""
    return LocalSandbox(sandbox_id="test-agent-001", base_dir=tmp_path)


@pytest.fixture
async def live_sandbox(tmp_path):
    """Return a setup LocalSandbox, torn down after the test."""
    sb = LocalSandbox(sandbox_id="test-agent-001", base_dir=tmp_path)
    await sb.setup()
    yield sb
    await sb.teardown()


# ─── sandbox_id Validation ───────────────────────────────────────────


def test_empty_sandbox_id_raises(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        LocalSandbox(sandbox_id="", base_dir=tmp_path)


def test_sandbox_id_with_slash_raises(tmp_path):
    with pytest.raises(ValueError, match="path separators"):
        LocalSandbox(sandbox_id="abc/def", base_dir=tmp_path)


def test_sandbox_id_with_backslash_raises(tmp_path):
    with pytest.raises(ValueError, match="path separators"):
        LocalSandbox(sandbox_id="abc\\def", base_dir=tmp_path)


# ─── Lifecycle ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_setup_creates_root_dir(sandbox):
    assert not sandbox.root.exists()
    await sandbox.setup()
    assert sandbox.root.exists()
    assert sandbox.root.is_dir()
    await sandbox.teardown()


@pytest.mark.asyncio
async def test_teardown_removes_root_dir(sandbox):
    await sandbox.setup()
    (sandbox.root / "some_file.txt").write_text("data")
    await sandbox.teardown()
    assert not sandbox.root.exists()


@pytest.mark.asyncio
async def test_double_teardown_is_noop(sandbox):
    await sandbox.setup()
    await sandbox.teardown()
    await sandbox.teardown()  # should not raise


@pytest.mark.asyncio
async def test_setup_is_idempotent(sandbox):
    await sandbox.setup()
    (sandbox.root / "keep.txt").write_text("important")
    await sandbox.setup()  # should not wipe existing files
    assert (sandbox.root / "keep.txt").read_text() == "important"
    await sandbox.teardown()


@pytest.mark.asyncio
async def test_context_manager(tmp_path):
    sb = LocalSandbox(sandbox_id="ctx-test", base_dir=tmp_path)
    async with sb:
        assert sb.root.exists()
        await sb.write_file("hello.txt", "world")
    assert not sb.root.exists()


@pytest.mark.asyncio
async def test_is_sandbox_subclass():
    assert issubclass(LocalSandbox, Sandbox)


# ─── Read / Write Text ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_and_read_text(live_sandbox):
    await live_sandbox.write_file("greeting.txt", "hello world")
    content = await live_sandbox.read_file("greeting.txt")
    assert content == "hello world"


@pytest.mark.asyncio
async def test_write_creates_parent_dirs(live_sandbox):
    await live_sandbox.write_file("deep/nested/dir/file.txt", "content")
    content = await live_sandbox.read_file("deep/nested/dir/file.txt")
    assert content == "content"


@pytest.mark.asyncio
async def test_write_overwrites_existing(live_sandbox):
    await live_sandbox.write_file("file.txt", "version1")
    await live_sandbox.write_file("file.txt", "version2")
    assert await live_sandbox.read_file("file.txt") == "version2"


@pytest.mark.asyncio
async def test_read_nonexistent_raises(live_sandbox):
    with pytest.raises(FileNotFoundError):
        await live_sandbox.read_file("does_not_exist.txt")


@pytest.mark.asyncio
async def test_read_file_utf8_replacement(live_sandbox):
    # Write raw bytes that aren't valid UTF-8 via streaming, then read as text
    async def _data():
        yield b"hello \xff world"

    await live_sandbox.write_file_bytes("bad_utf8.txt", _data())
    content = await live_sandbox.read_file("bad_utf8.txt")
    assert "hello" in content
    assert "world" in content


# ─── Read / Write Bytes ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_and_read_bytes(live_sandbox):
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    async def _data():
        yield data

    await live_sandbox.write_file_bytes("image.png", _data())
    chunks = []
    async for chunk in live_sandbox.read_file_bytes("image.png"):
        chunks.append(chunk)
    result = b"".join(chunks)
    assert result == data


# ─── Path Containment ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_file_path_traversal_blocked(live_sandbox):
    with pytest.raises(ValueError, match="Path traversal blocked"):
        await live_sandbox.read_file("../../etc/passwd")


@pytest.mark.asyncio
async def test_write_file_path_traversal_blocked(live_sandbox):
    with pytest.raises(ValueError, match="Path traversal blocked"):
        await live_sandbox.write_file("../escape.txt", "data")


@pytest.mark.asyncio
async def test_delete_path_traversal_blocked(live_sandbox):
    with pytest.raises(ValueError, match="Path traversal blocked"):
        await live_sandbox.delete("../../important_file")


@pytest.mark.asyncio
async def test_list_dir_path_traversal_blocked(live_sandbox):
    with pytest.raises(ValueError, match="Path traversal blocked"):
        await live_sandbox.list_dir("../..")


@pytest.mark.asyncio
async def test_exec_cwd_traversal_blocked(live_sandbox):
    with pytest.raises(ValueError, match="Path traversal blocked"):
        await live_sandbox.exec("ls", cwd="../../")


@pytest.mark.asyncio
async def test_symlink_escape_blocked(live_sandbox):
    # Create a symlink inside the sandbox that points outside
    link_path = live_sandbox.root / "sneaky_link"
    link_path.symlink_to("/tmp")
    with pytest.raises(ValueError, match="Path traversal blocked"):
        await live_sandbox.read_file("sneaky_link/some_file")


# ─── file_exists ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_file_exists_true_for_file(live_sandbox):
    await live_sandbox.write_file("exists.txt", "yes")
    exists, entry = await live_sandbox.file_exists("exists.txt")
    assert exists is True
    assert entry is not None
    assert entry.name == "exists.txt"
    assert entry.is_dir is False
    assert entry.extension == ".txt"


@pytest.mark.asyncio
async def test_file_exists_true_for_dir(live_sandbox):
    await live_sandbox.write_file("subdir/file.txt", "data")
    exists, entry = await live_sandbox.file_exists("subdir")
    assert exists is True
    assert entry is not None
    assert entry.is_dir is True


@pytest.mark.asyncio
async def test_file_exists_false_for_missing(live_sandbox):
    exists, entry = await live_sandbox.file_exists("nope.txt")
    assert exists is False
    assert entry is None


@pytest.mark.asyncio
async def test_file_exists_false_for_escaped_path(live_sandbox):
    # Should return (False, None), not raise
    exists, entry = await live_sandbox.file_exists("../../etc/passwd")
    assert exists is False
    assert entry is None


# ─── delete ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_file(live_sandbox):
    await live_sandbox.write_file("doomed.txt", "bye")
    assert await live_sandbox.delete("doomed.txt") is True
    exists, _ = await live_sandbox.file_exists("doomed.txt")
    assert exists is False


@pytest.mark.asyncio
async def test_delete_directory_recursive(live_sandbox):
    await live_sandbox.write_file("dir/a.txt", "a")
    await live_sandbox.write_file("dir/b.txt", "b")
    assert await live_sandbox.delete("dir") is True
    exists, _ = await live_sandbox.file_exists("dir")
    assert exists is False


@pytest.mark.asyncio
async def test_delete_missing_returns_false(live_sandbox):
    assert await live_sandbox.delete("ghost.txt") is False


# ─── list_dir ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_dir_returns_sorted_entries(live_sandbox):
    await live_sandbox.write_file("b.txt", "bb")
    await live_sandbox.write_file("a.txt", "a")
    await live_sandbox.write_file("subdir/c.txt", "c")

    entries = await live_sandbox.list_dir(".")
    names = [e.name for e in entries]
    assert names == sorted(names)

    # Check types
    file_entry = next(e for e in entries if e.name == "a.txt")
    assert file_entry.is_dir is False
    assert file_entry.size_bytes == 1

    dir_entry = next(e for e in entries if e.name == "subdir")
    assert dir_entry.is_dir is True


@pytest.mark.asyncio
async def test_list_dir_nonexistent_raises(live_sandbox):
    with pytest.raises(FileNotFoundError):
        await live_sandbox.list_dir("no_such_dir")


@pytest.mark.asyncio
async def test_list_dir_on_file_raises(live_sandbox):
    await live_sandbox.write_file("file.txt", "data")
    with pytest.raises(NotADirectoryError):
        await live_sandbox.list_dir("file.txt")


# ─── exec ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exec_basic(live_sandbox):
    # Ensure workspace dir exists for default cwd
    live_sandbox.workspace.mkdir(parents=True, exist_ok=True)
    result = await live_sandbox.exec("echo hello")
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert result.timed_out is False
    assert result.duration_ms > 0


@pytest.mark.asyncio
async def test_exec_cwd(live_sandbox):
    live_sandbox.workspace.mkdir(parents=True, exist_ok=True)
    result = await live_sandbox.exec("pwd", cwd="workspace")
    assert result.stdout.strip().endswith("/workspace")


@pytest.mark.asyncio
async def test_exec_timeout(live_sandbox):
    live_sandbox.workspace.mkdir(parents=True, exist_ok=True)
    result = await live_sandbox.exec("sleep 10", timeout=0.1)
    assert result.timed_out is True
    assert result.exit_code == -1


@pytest.mark.asyncio
async def test_exec_env(live_sandbox):
    live_sandbox.workspace.mkdir(parents=True, exist_ok=True)
    result = await live_sandbox.exec(
        "echo $SANDBOX_TEST_VAR",
        env={"SANDBOX_TEST_VAR": "it_works"},
    )
    assert result.stdout.strip() == "it_works"


@pytest.mark.asyncio
async def test_exec_failure(live_sandbox):
    live_sandbox.workspace.mkdir(parents=True, exist_ok=True)
    result = await live_sandbox.exec("exit 42")
    assert result.exit_code == 42


@pytest.mark.asyncio
async def test_exec_stderr(live_sandbox):
    live_sandbox.workspace.mkdir(parents=True, exist_ok=True)
    result = await live_sandbox.exec("echo err >&2")
    assert result.stderr.strip() == "err"


@pytest.mark.asyncio
async def test_exec_pipes(live_sandbox):
    live_sandbox.workspace.mkdir(parents=True, exist_ok=True)
    result = await live_sandbox.exec("echo 'a\nb\nc' | wc -l")
    assert result.exit_code == 0
    assert result.stdout.strip() == "3"


# ─── exec_stream ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exec_stream_yields_lines(live_sandbox):
    live_sandbox.workspace.mkdir(parents=True, exist_ok=True)
    lines = []
    async for line in live_sandbox.exec_stream("printf 'line1\nline2\nline3\n'"):
        lines.append(line.strip())
    assert lines == ["line1", "line2", "line3"]


@pytest.mark.asyncio
async def test_exec_stream_early_break_reaps_process(live_sandbox):
    live_sandbox.workspace.mkdir(parents=True, exist_ok=True)
    count = 0
    async for line in live_sandbox.exec_stream("yes"):
        count += 1
        if count >= 3:
            break
    # If we get here without hanging, the process was reaped correctly
    assert count == 3


# ─── Concurrent exec ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_exec(live_sandbox):
    live_sandbox.workspace.mkdir(parents=True, exist_ok=True)
    results = await asyncio.gather(
        live_sandbox.exec("echo task1"),
        live_sandbox.exec("echo task2"),
    )
    outputs = {r.stdout.strip() for r in results}
    assert outputs == {"task1", "task2"}


# ─── Zone directory usage ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_zone_directories_accessible(live_sandbox):
    """Zones are siblings of workspace/ under the sandbox root. Tools must
    be able to read/write to all zones via relative paths."""
    await live_sandbox.write_file("workspace/src/main.py", "print('hi')")
    await live_sandbox.write_file(".exports/report.csv", "a,b,c")
    await live_sandbox.write_file(".ephemeral/logs/run.log", "log data")

    assert await live_sandbox.read_file("workspace/src/main.py") == "print('hi')"
    assert await live_sandbox.read_file(".exports/report.csv") == "a,b,c"
    assert await live_sandbox.read_file(".ephemeral/logs/run.log") == "log data"
