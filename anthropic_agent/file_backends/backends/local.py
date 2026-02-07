"""Local filesystem file storage backend."""

from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from ...logging import get_logger
from ..base import FileMetadata, FileStorageBackend

logger = get_logger(__name__)


class LocalFilesystemBackend(FileStorageBackend):
    """Store files in the local filesystem.

    Storage structure::

        {base_path}/
            {agent_uuid}/
                {file_id}_{filename}
    """

    def __init__(self, base_path: str = "./agent-files"):
        """Initialize local filesystem backend.

        Args:
            base_path: Root directory for file storage
        """
        self.base_path = Path(base_path)

    def _file_path(self, agent_uuid: str, file_id: str, filename: str) -> Path:
        """Build the on-disk path for a file."""
        return self.base_path / agent_uuid / f"{file_id}_{filename}"

    async def store(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        agent_uuid: str,
    ) -> FileMetadata:
        """Store a new file in the local filesystem."""
        agent_dir = self.base_path / agent_uuid
        agent_dir.mkdir(parents=True, exist_ok=True)

        file_path = self._file_path(agent_uuid, file_id, filename)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        storage_location = str(file_path.absolute())
        logger.info("Stored file", filename=filename, file_id=file_id, backend="local")

        return FileMetadata(
            file_id=file_id,
            filename=filename,
            storage_location=storage_location,
            size=len(content),
            timestamp=datetime.now().isoformat(),
            is_update=False,
            storage_backend="local",
        )

    async def update(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        existing_metadata: dict[str, Any],
        agent_uuid: str,
    ) -> FileMetadata:
        """Update an existing file in the local filesystem."""
        agent_dir = self.base_path / agent_uuid
        agent_dir.mkdir(parents=True, exist_ok=True)

        file_path = self._file_path(agent_uuid, file_id, filename)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        storage_location = str(file_path.absolute())
        logger.info("Updated file", filename=filename, file_id=file_id, backend="local")

        return FileMetadata(
            file_id=file_id,
            filename=filename,
            storage_location=storage_location,
            size=len(content),
            timestamp=datetime.now().isoformat(),
            is_update=True,
            storage_backend="local",
            previous_size=existing_metadata.get("size"),
        )

    async def retrieve(
        self,
        file_id: str,
        agent_uuid: str,
    ) -> bytes | None:
        """Read file content from disk.

        Scans the agent directory for a file matching the given file_id prefix.
        """
        agent_dir = self.base_path / agent_uuid
        if not agent_dir.exists():
            return None

        # Find matching file (pattern: {file_id}_{filename})
        matches = list(agent_dir.glob(f"{file_id}_*"))
        if not matches:
            return None

        async with aiofiles.open(matches[0], "rb") as f:
            return await f.read()

    async def delete(
        self,
        file_id: str,
        agent_uuid: str,
    ) -> bool:
        """Remove a file from disk."""
        agent_dir = self.base_path / agent_uuid
        if not agent_dir.exists():
            return False

        matches = list(agent_dir.glob(f"{file_id}_*"))
        if not matches:
            return False

        matches[0].unlink()
        logger.info("Deleted file", file_id=file_id, backend="local")
        return True
