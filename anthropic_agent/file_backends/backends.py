"""File storage backend implementations for Anthropic agent."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol, Any

logger = logging.getLogger(__name__)


class FileStorageBackend(Protocol):
    """Protocol for file storage backends."""

    def store(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        agent_uuid: str,
    ) -> dict[str, Any]:
        """Store a new file.

        Args:
            file_id: Anthropic's file identifier
            filename: Original filename
            content: File content bytes
            agent_uuid: Agent session UUID

        Returns:
            Metadata dict with storage_url/path and other info
        """
        ...

    def update(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        existing_metadata: dict[str, Any],
        agent_uuid: str,
    ) -> dict[str, Any]:
        """Update an existing file.

        Args:
            file_id: Anthropic's file identifier
            filename: Filename (may have changed)
            content: New file content
            existing_metadata: Previous metadata for this file
            agent_uuid: Agent session UUID

        Returns:
            Updated metadata dict
        """
        ...


class NoOpBackend:
    """No-operation backend that doesn't store files.

    Returns file_id as-is in metadata without downloading.
    Useful for testing or when file storage is handled externally.
    """

    def store(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        agent_uuid: str,
    ) -> dict[str, Any]:
        """Return minimal metadata without storing."""
        return {
            "file_id": file_id,
            "filename": filename,
            "size": len(content),
            "timestamp": datetime.now().isoformat(),
            "is_update": False,
            "storage_backend": "noop",
        }

    def update(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        existing_metadata: dict[str, Any],
        agent_uuid: str,
    ) -> dict[str, Any]:
        """Return minimal metadata without updating."""
        return {
            "file_id": file_id,
            "filename": filename,
            "size": len(content),
            "timestamp": datetime.now().isoformat(),
            "is_update": True,
            "storage_backend": "noop",
        }


class LocalFilesystemBackend:
    """Store files in local filesystem.

    Storage structure:
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

    def store(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        agent_uuid: str,
    ) -> dict[str, Any]:
        """Store a new file in the local filesystem.

        Args:
            file_id: Anthropic's file identifier
            filename: Original filename
            content: File content bytes
            agent_uuid: Agent session UUID

        Returns:
            Metadata dict with storage_path and other info
        """
        # Create agent-specific directory
        agent_dir = self.base_path / agent_uuid
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Store file with pattern: {file_id}_{filename}
        file_path = agent_dir / f"{file_id}_{filename}"
        file_path.write_bytes(content)

        logger.info(f"Stored file {filename} ({file_id}) at {file_path}")

        return {
            "file_id": file_id,
            "filename": filename,
            "storage_path": str(file_path.absolute()),
            "size": len(content),
            "timestamp": datetime.now().isoformat(),
            "is_update": False,
            "storage_backend": "local",
        }

    def update(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        existing_metadata: dict[str, Any],
        agent_uuid: str,
    ) -> dict[str, Any]:
        """Update an existing file in the local filesystem.

        Args:
            file_id: Anthropic's file identifier
            filename: Filename (may have changed)
            content: New file content
            existing_metadata: Previous metadata for this file
            agent_uuid: Agent session UUID

        Returns:
            Updated metadata dict
        """
        # Create agent-specific directory
        agent_dir = self.base_path / agent_uuid
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Store file with pattern: {file_id}_{filename}
        file_path = agent_dir / f"{file_id}_{filename}"
        file_path.write_bytes(content)

        logger.info(f"Updated file {filename} ({file_id}) at {file_path}")

        return {
            "file_id": file_id,
            "filename": filename,
            "storage_path": str(file_path.absolute()),
            "size": len(content),
            "timestamp": datetime.now().isoformat(),
            "is_update": True,
            "storage_backend": "local",
            "previous_size": existing_metadata.get("size"),
        }


class S3Backend:
    """S3 storage backend (placeholder).

    Storage structure:
        s3://{bucket}/{prefix}/{agent_uuid}/{file_id}_{filename}
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "agent-files",
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ):
        """Initialize S3 backend.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all files
            region: AWS region
            endpoint_url: Custom endpoint for S3-compatible services
        """
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.endpoint_url = endpoint_url

    def store(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        agent_uuid: str,
    ) -> dict[str, Any]:
        """Store a new file in S3.

        Args:
            file_id: Anthropic's file identifier
            filename: Original filename
            content: File content bytes
            agent_uuid: Agent session UUID

        Returns:
            Metadata dict with storage_url and other info
        """
        raise NotImplementedError(
            "S3Backend is not yet implemented. Use LocalFilesystemBackend instead."
        )

    def update(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        existing_metadata: dict[str, Any],
        agent_uuid: str,
    ) -> dict[str, Any]:
        """Update an existing file in S3.

        Args:
            file_id: Anthropic's file identifier
            filename: Filename (may have changed)
            content: New file content
            existing_metadata: Previous metadata for this file
            agent_uuid: Agent session UUID

        Returns:
            Updated metadata dict
        """
        raise NotImplementedError(
            "S3Backend is not yet implemented. Use LocalFilesystemBackend instead."
        )

