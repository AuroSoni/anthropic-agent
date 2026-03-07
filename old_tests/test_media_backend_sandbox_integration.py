"""Integration tests for MediaBackend.attach_sandbox, materialize, and flush_exports."""

import pytest

from agent_base.media_backend import LocalMediaBackend, MediaMetadata
from agent_base.sandbox import LocalSandbox


AGENT_UUID = "agent-integration-001"
SAMPLE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


@pytest.fixture
async def sandbox(tmp_path):
    sb = LocalSandbox(sandbox_id="integration-test", base_dir=tmp_path / "sandboxes")
    await sb.setup()
    yield sb
    if sb.root.exists():
        await sb.teardown()


@pytest.fixture
async def backend(tmp_path, sandbox):
    b = LocalMediaBackend(base_path=tmp_path / "media")
    await b.connect()
    b.attach_sandbox(sandbox)
    yield b
    await b.close()


# ─── attach_sandbox ─────────────────────────────────────────────


def test_attach_sandbox_stores_reference(sandbox):
    b = LocalMediaBackend()
    b.attach_sandbox(sandbox)
    assert b._sandbox is sandbox


def test_attach_sandbox_idempotent(sandbox):
    b = LocalMediaBackend()
    b.attach_sandbox(sandbox)
    b.attach_sandbox(sandbox)
    assert b._sandbox is sandbox


# ─── materialize ────────────────────────────────────────────────


async def test_materialize_without_sandbox_raises():
    b = LocalMediaBackend()
    await b.connect()
    with pytest.raises(RuntimeError, match="No sandbox attached"):
        await b.materialize("any-id", AGENT_UUID)
    await b.close()


async def test_materialize_downloads_to_sandbox(backend, sandbox):
    meta = await backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    path = await backend.materialize(meta.media_id, AGENT_UUID)
    content = await sandbox.read_file_bytes(path)
    assert content == SAMPLE_PNG


async def test_materialize_returns_path_string(backend):
    meta = await backend.store(b"data", "f.txt", "text/plain", AGENT_UUID)
    path = await backend.materialize(meta.media_id, AGENT_UUID)
    assert isinstance(path, str)
    assert meta.media_id in path
    assert path.endswith("f.txt")


async def test_materialize_idempotent(backend, sandbox):
    meta = await backend.store(b"data", "f.txt", "text/plain", AGENT_UUID)
    path1 = await backend.materialize(meta.media_id, AGENT_UUID)
    path2 = await backend.materialize(meta.media_id, AGENT_UUID)
    assert path1 == path2


async def test_materialize_missing_media_raises(backend):
    with pytest.raises(FileNotFoundError, match="not found"):
        await backend.materialize("nonexistent-id", AGENT_UUID)


async def test_materialize_binary_content(backend, sandbox):
    data = bytes(range(256))
    meta = await backend.store(data, "binary.bin", "application/octet-stream", AGENT_UUID)
    path = await backend.materialize(meta.media_id, AGENT_UUID)
    assert await sandbox.read_file_bytes(path) == data


async def test_materialize_preserves_filename(backend):
    meta = await backend.store(b"pdf", "document.pdf", "application/pdf", AGENT_UUID)
    path = await backend.materialize(meta.media_id, AGENT_UUID)
    assert path.endswith("document.pdf")


# ─── flush_exports ──────────────────────────────────────────────


async def test_flush_exports_without_sandbox_raises():
    b = LocalMediaBackend()
    await b.connect()
    with pytest.raises(RuntimeError, match="No sandbox attached"):
        await b.flush_exports(AGENT_UUID)
    await b.close()


async def test_flush_exports_empty(backend):
    result = await backend.flush_exports(AGENT_UUID)
    assert result == []


async def test_flush_exports_stores_files(backend, sandbox):
    await sandbox.write_file_bytes(".exports/report.csv", b"a,b,c")
    await sandbox.write_file_bytes(".exports/data.json", b'{"key": "val"}')
    result = await backend.flush_exports(AGENT_UUID)
    assert len(result) == 2
    assert all(isinstance(m, MediaMetadata) for m in result)


async def test_flush_exports_content_retrievable(backend, sandbox):
    await sandbox.write_file_bytes(".exports/out.txt", b"output data")
    result = await backend.flush_exports(AGENT_UUID)
    assert len(result) == 1
    content = await backend.retrieve(result[0].media_id, AGENT_UUID)
    assert content == b"output data"


async def test_flush_exports_infers_mime_type(backend, sandbox):
    await sandbox.write_file_bytes(".exports/photo.png", SAMPLE_PNG)
    result = await backend.flush_exports(AGENT_UUID)
    assert result[0].media_mime_type == "image/png"


async def test_flush_exports_nested_uses_basename(backend, sandbox):
    await sandbox.write_file_bytes(".exports/sub/deep/f.txt", b"deep")
    result = await backend.flush_exports(AGENT_UUID)
    assert len(result) == 1
    assert result[0].media_filename == "f.txt"
    assert result[0].extras.get("export_path") == "sub/deep/f.txt"


async def test_flush_exports_flat_file_no_extras(backend, sandbox):
    await sandbox.write_file_bytes(".exports/flat.txt", b"flat")
    result = await backend.flush_exports(AGENT_UUID)
    assert result[0].media_filename == "flat.txt"
    assert "export_path" not in result[0].extras


async def test_flush_exports_returns_metadata_list(backend, sandbox):
    await sandbox.write_file_bytes(".exports/f.txt", b"data")
    result = await backend.flush_exports(AGENT_UUID)
    meta = result[0]
    assert meta.media_id
    assert meta.media_size == 4
    assert meta.storage_type == "local"
