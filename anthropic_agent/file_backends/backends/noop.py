"""No-operation file storage backend."""

from datetime import datetime
from typing import Any

from ..base import FileMetadata, FileStorageBackend


class NoOpBackend(FileStorageBackend):
    """No-operation backend that doesn't store files.

    Returns file_id as-is in metadata without downloading.
    Useful for testing or when file storage is handled externally.
    """

    async def store(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        agent_uuid: str,
    ) -> FileMetadata:
        """Return minimal metadata without storing."""
        return FileMetadata(
            file_id=file_id,
            filename=filename,
            storage_location=None,
            size=len(content),
            timestamp=datetime.now().isoformat(),
            is_update=False,
            storage_backend="noop",
        )

    async def update(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        existing_metadata: dict[str, Any],
        agent_uuid: str,
    ) -> FileMetadata:
        """Return minimal metadata without updating."""
        return FileMetadata(
            file_id=file_id,
            filename=filename,
            storage_location=None,
            size=len(content),
            timestamp=datetime.now().isoformat(),
            is_update=True,
            storage_backend="noop",
            previous_size=existing_metadata.get("size"),
        )

    async def retrieve(
        self,
        file_id: str,
        agent_uuid: str,
    ) -> bytes | None:
        """NoOp backend has nothing to retrieve."""
        return None

    async def delete(
        self,
        file_id: str,
        agent_uuid: str,
    ) -> bool:
        """NoOp backend always reports success."""
        return True
