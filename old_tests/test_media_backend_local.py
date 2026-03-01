"""Tests for agent_base.media_backend.LocalMediaBackend."""

import base64

import pytest

from agent_base.media_backend import LocalMediaBackend, MediaBackend, MediaMetadata


AGENT_UUID = "agent-test-001"
SAMPLE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
SAMPLE_PDF = b"%PDF-1.4 fake pdf content"


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def backend(tmp_path):
    """Return an unconnected LocalMediaBackend using tmp_path."""
    return LocalMediaBackend(base_path=tmp_path / "media")


@pytest.fixture
async def live_backend(tmp_path):
    """Return a connected LocalMediaBackend, closed after the test."""
    b = LocalMediaBackend(base_path=tmp_path / "media")
    await b.connect()
    yield b
    await b.close()


# ─── Class hierarchy ─────────────────────────────────────────────────


def test_is_media_backend_subclass():
    assert issubclass(LocalMediaBackend, MediaBackend)


# ─── Constructor ─────────────────────────────────────────────────────


def test_default_base_path():
    b = LocalMediaBackend()
    assert str(b.base_path) == "agent-media"


def test_custom_base_path(tmp_path):
    b = LocalMediaBackend(base_path=tmp_path / "custom")
    assert b.base_path == tmp_path / "custom"


def test_url_prefix_stored():
    b = LocalMediaBackend(url_prefix="/api/media/")
    assert b.url_prefix == "/api/media"  # trailing slash stripped


def test_url_prefix_none_by_default():
    b = LocalMediaBackend()
    assert b.url_prefix is None


# ─── Lifecycle ───────────────────────────────────────────────────────


async def test_connect_creates_base_dir(backend):
    assert not backend.base_path.exists()
    await backend.connect()
    assert backend.base_path.exists()
    assert backend.base_path.is_dir()


async def test_connect_is_idempotent(live_backend):
    meta = await live_backend.store(b"data", "file.txt", "text/plain", AGENT_UUID)
    await live_backend.connect()  # should not wipe existing files
    content = await live_backend.retrieve(meta.media_id, AGENT_UUID)
    assert content == b"data"


async def test_context_manager(tmp_path):
    b = LocalMediaBackend(base_path=tmp_path / "media")
    async with b:
        assert b.base_path.exists()
        await b.store(b"hello", "hello.txt", "text/plain", AGENT_UUID)
    # close() is a no-op — files persist
    assert b.base_path.exists()


async def test_close_is_noop(live_backend):
    await live_backend.store(b"data", "file.txt", "text/plain", AGENT_UUID)
    await live_backend.close()
    # Files should still exist on disk
    assert live_backend.base_path.exists()


# ─── store() ─────────────────────────────────────────────────────────


async def test_store_returns_metadata_with_all_fields(live_backend):
    meta = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    assert isinstance(meta, MediaMetadata)
    assert meta.media_id  # non-empty
    assert meta.media_mime_type == "image/png"
    assert meta.media_filename == "photo.png"
    assert meta.media_extension == "png"
    assert meta.media_size == len(SAMPLE_PNG)
    assert meta.storage_type == "local"
    assert meta.storage_location  # non-empty absolute path


async def test_store_generates_unique_media_id(live_backend):
    meta1 = await live_backend.store(b"a", "a.txt", "text/plain", AGENT_UUID)
    meta2 = await live_backend.store(b"b", "b.txt", "text/plain", AGENT_UUID)
    assert meta1.media_id != meta2.media_id


async def test_store_media_id_is_hex_uuid4(live_backend):
    meta = await live_backend.store(b"data", "f.txt", "text/plain", AGENT_UUID)
    assert len(meta.media_id) == 32
    int(meta.media_id, 16)  # should not raise — valid hex


async def test_store_writes_file_to_disk(live_backend):
    meta = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    from pathlib import Path

    assert Path(meta.storage_location).exists()


async def test_store_file_content_matches(live_backend):
    meta = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    from pathlib import Path

    assert Path(meta.storage_location).read_bytes() == SAMPLE_PNG


async def test_store_creates_agent_dir(live_backend):
    await live_backend.store(b"data", "f.txt", "text/plain", AGENT_UUID)
    assert (live_backend.base_path / AGENT_UUID).is_dir()


async def test_store_multiple_files_same_agent(live_backend):
    m1 = await live_backend.store(b"a", "a.txt", "text/plain", AGENT_UUID)
    m2 = await live_backend.store(b"b", "b.txt", "text/plain", AGENT_UUID)
    assert await live_backend.retrieve(m1.media_id, AGENT_UUID) == b"a"
    assert await live_backend.retrieve(m2.media_id, AGENT_UUID) == b"b"


async def test_store_multiple_agents(live_backend):
    m1 = await live_backend.store(b"agent1", "f.txt", "text/plain", "agent-001")
    m2 = await live_backend.store(b"agent2", "f.txt", "text/plain", "agent-002")
    assert await live_backend.retrieve(m1.media_id, "agent-001") == b"agent1"
    assert await live_backend.retrieve(m2.media_id, "agent-002") == b"agent2"


# ─── retrieve() ──────────────────────────────────────────────────────


async def test_retrieve_returns_stored_bytes(live_backend):
    meta = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    content = await live_backend.retrieve(meta.media_id, AGENT_UUID)
    assert content == SAMPLE_PNG


async def test_retrieve_missing_returns_none(live_backend):
    assert await live_backend.retrieve("nonexistent", AGENT_UUID) is None


async def test_retrieve_wrong_agent_returns_none(live_backend):
    meta = await live_backend.store(b"data", "f.txt", "text/plain", AGENT_UUID)
    assert await live_backend.retrieve(meta.media_id, "wrong-agent") is None


async def test_retrieve_binary_content(live_backend):
    data = bytes(range(256))
    meta = await live_backend.store(data, "binary.bin", "application/octet-stream", AGENT_UUID)
    assert await live_backend.retrieve(meta.media_id, AGENT_UUID) == data


# ─── delete() ────────────────────────────────────────────────────────


async def test_delete_existing_returns_true(live_backend):
    meta = await live_backend.store(b"data", "f.txt", "text/plain", AGENT_UUID)
    assert await live_backend.delete(meta.media_id, AGENT_UUID) is True


async def test_delete_missing_returns_false(live_backend):
    assert await live_backend.delete("nonexistent", AGENT_UUID) is False


async def test_delete_then_retrieve_returns_none(live_backend):
    meta = await live_backend.store(b"data", "f.txt", "text/plain", AGENT_UUID)
    await live_backend.delete(meta.media_id, AGENT_UUID)
    assert await live_backend.retrieve(meta.media_id, AGENT_UUID) is None


async def test_delete_wrong_agent_returns_false(live_backend):
    meta = await live_backend.store(b"data", "f.txt", "text/plain", AGENT_UUID)
    assert await live_backend.delete(meta.media_id, "wrong-agent") is False


# ─── exists() ────────────────────────────────────────────────────────


async def test_exists_true_after_store(live_backend):
    meta = await live_backend.store(b"data", "f.txt", "text/plain", AGENT_UUID)
    assert await live_backend.exists(meta.media_id, AGENT_UUID) is True


async def test_exists_false_before_store(live_backend):
    assert await live_backend.exists("nonexistent", AGENT_UUID) is False


async def test_exists_false_after_delete(live_backend):
    meta = await live_backend.store(b"data", "f.txt", "text/plain", AGENT_UUID)
    await live_backend.delete(meta.media_id, AGENT_UUID)
    assert await live_backend.exists(meta.media_id, AGENT_UUID) is False


async def test_exists_wrong_agent_false(live_backend):
    meta = await live_backend.store(b"data", "f.txt", "text/plain", AGENT_UUID)
    assert await live_backend.exists(meta.media_id, "wrong-agent") is False


# ─── get_metadata() ──────────────────────────────────────────────────


async def test_get_metadata_returns_correct_fields(live_backend):
    meta_store = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    meta = await live_backend.get_metadata(meta_store.media_id, AGENT_UUID)
    assert meta is not None
    assert meta.media_id == meta_store.media_id
    assert meta.media_filename == "photo.png"
    assert meta.media_extension == "png"
    assert meta.media_size == len(SAMPLE_PNG)
    assert meta.storage_type == "local"


async def test_get_metadata_missing_returns_none(live_backend):
    assert await live_backend.get_metadata("nonexistent", AGENT_UUID) is None


async def test_get_metadata_infers_mime_type(live_backend):
    meta_store = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    meta = await live_backend.get_metadata(meta_store.media_id, AGENT_UUID)
    assert meta is not None
    assert meta.media_mime_type == "image/png"


async def test_get_metadata_unknown_extension(live_backend):
    meta_store = await live_backend.store(b"data", "file.xyz123", "application/x-custom", AGENT_UUID)
    meta = await live_backend.get_metadata(meta_store.media_id, AGENT_UUID)
    assert meta is not None
    assert meta.media_mime_type == "application/octet-stream"


async def test_get_metadata_size_matches(live_backend):
    data = b"hello world"
    meta_store = await live_backend.store(data, "hello.txt", "text/plain", AGENT_UUID)
    meta = await live_backend.get_metadata(meta_store.media_id, AGENT_UUID)
    assert meta is not None
    assert meta.media_size == len(data)


# ─── to_base64() ─────────────────────────────────────────────────────


async def test_to_base64_returns_correct_dict_shape(live_backend):
    meta = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    result = await live_backend.to_base64(meta.media_id, AGENT_UUID)
    assert "data" in result
    assert "media_type" in result


async def test_to_base64_data_decodes_to_original(live_backend):
    meta = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    result = await live_backend.to_base64(meta.media_id, AGENT_UUID)
    decoded = base64.standard_b64decode(result["data"])
    assert decoded == SAMPLE_PNG


async def test_to_base64_media_type_matches(live_backend):
    meta = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    result = await live_backend.to_base64(meta.media_id, AGENT_UUID)
    assert result["media_type"] == "image/png"


async def test_to_base64_missing_raises_file_not_found(live_backend):
    with pytest.raises(FileNotFoundError):
        await live_backend.to_base64("nonexistent", AGENT_UUID)


# ─── to_url() ────────────────────────────────────────────────────────


async def test_to_url_default_returns_file_uri(live_backend):
    meta = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    url = await live_backend.to_url(meta.media_id, AGENT_UUID)
    assert url.startswith("file:///")


async def test_to_url_file_uri_contains_media_id(live_backend):
    meta = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    url = await live_backend.to_url(meta.media_id, AGENT_UUID)
    assert meta.media_id in url


async def test_to_url_with_prefix_returns_api_path(tmp_path):
    b = LocalMediaBackend(base_path=tmp_path / "media", url_prefix="/api/media")
    await b.connect()
    meta = await b.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    url = await b.to_url(meta.media_id, AGENT_UUID)
    assert url == f"/api/media/{AGENT_UUID}/{meta.media_id}"
    await b.close()


async def test_to_url_missing_raises_file_not_found(live_backend):
    with pytest.raises(FileNotFoundError):
        await live_backend.to_url("nonexistent", AGENT_UUID)


# ─── to_reference() ──────────────────────────────────────────────────


async def test_to_reference_returns_metadata_dict(live_backend):
    meta = await live_backend.store(SAMPLE_PNG, "photo.png", "image/png", AGENT_UUID)
    ref = await live_backend.to_reference(meta.media_id, AGENT_UUID)
    assert ref["media_id"] == meta.media_id
    assert ref["media_mime_type"] == "image/png"
    assert ref["media_filename"] == "photo.png"
    assert ref["media_extension"] == "png"
    assert ref["media_size"] == len(SAMPLE_PNG)
    assert ref["storage_type"] == "local"
    assert "storage_location" in ref
    assert "extras" in ref


async def test_to_reference_missing_raises_file_not_found(live_backend):
    with pytest.raises(FileNotFoundError):
        await live_backend.to_reference("nonexistent", AGENT_UUID)


# ─── Edge cases ──────────────────────────────────────────────────────


async def test_filename_with_underscores(live_backend):
    """Filenames with underscores should not confuse _extract_filename."""
    meta = await live_backend.store(b"data", "my_photo_2024.png", "image/png", AGENT_UUID)
    content = await live_backend.retrieve(meta.media_id, AGENT_UUID)
    assert content == b"data"
    meta_back = await live_backend.get_metadata(meta.media_id, AGENT_UUID)
    assert meta_back is not None
    assert meta_back.media_filename == "my_photo_2024.png"


async def test_filename_with_no_extension(live_backend):
    meta = await live_backend.store(b"data", "Makefile", "text/plain", AGENT_UUID)
    assert meta.media_extension == ""
    content = await live_backend.retrieve(meta.media_id, AGENT_UUID)
    assert content == b"data"
    meta_back = await live_backend.get_metadata(meta.media_id, AGENT_UUID)
    assert meta_back is not None
    assert meta_back.media_filename == "Makefile"


async def test_empty_content(live_backend):
    meta = await live_backend.store(b"", "empty.txt", "text/plain", AGENT_UUID)
    assert meta.media_size == 0
    content = await live_backend.retrieve(meta.media_id, AGENT_UUID)
    assert content == b""
