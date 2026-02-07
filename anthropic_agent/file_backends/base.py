"""Base abstractions for file storage backends.

This module defines:
1. FileMetadata dataclass — typed return value from backend operations
2. FileStorageBackend ABC — async interface for file storage implementations

Users can extend FileMetadata via the `extras` dict field.
Users can implement custom backends by subclassing FileStorageBackend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Self


@dataclass
class FileMetadata:
    """Metadata returned by file storage operations.

    All backends return this dataclass instead of raw dicts,
    providing type safety and IDE autocomplete.

    The `extras` field allows backends to store custom data
    (e.g., S3 bucket/key) without subclassing.
    """
    file_id: str
    filename: str
    storage_location: str | None
    size: int
    timestamp: str
    is_update: bool
    storage_backend: str
    previous_size: int | None = None

    # Backend-specific extension point
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict for backward compatibility.

        Merges extras into the top-level dict so callers that
        previously consumed raw dicts still work.
        """
        result = {
            "file_id": self.file_id,
            "filename": self.filename,
            "storage_location": self.storage_location,
            "size": self.size,
            "timestamp": self.timestamp,
            "is_update": self.is_update,
            "storage_backend": self.storage_backend,
        }
        if self.previous_size is not None:
            result["previous_size"] = self.previous_size
        result.update(self.extras)
        return result


class FileStorageBackend(ABC):
    """Abstract base class for file storage backends.

    Provides async lifecycle management (connect/close) and
    four abstract methods: store, update, retrieve, delete.

    Subclass this to implement custom storage (Google Drive, Azure Blob, etc.).

    Example::

        class GoogleDriveBackend(FileStorageBackend):
            async def store(self, file_id, filename, content, agent_uuid):
                # ... upload to Drive ...
                return FileMetadata(...)
    """

    # -- Lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        """Initialize connections or resources. Override if needed."""
        pass

    async def close(self) -> None:
        """Cleanup connections or resources. Override if needed."""
        pass

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    # -- Abstract operations ------------------------------------------------

    @abstractmethod
    async def store(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        agent_uuid: str,
    ) -> FileMetadata:
        """Store a new file.

        Args:
            file_id: Anthropic's file identifier
            filename: Original filename
            content: File content bytes
            agent_uuid: Agent session UUID

        Returns:
            FileMetadata with storage location and other info
        """
        ...

    @abstractmethod
    async def update(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        existing_metadata: dict[str, Any],
        agent_uuid: str,
    ) -> FileMetadata:
        """Update an existing file.

        Args:
            file_id: Anthropic's file identifier
            filename: Filename (may have changed)
            content: New file content
            existing_metadata: Previous metadata for this file
            agent_uuid: Agent session UUID

        Returns:
            Updated FileMetadata
        """
        ...

    @abstractmethod
    async def retrieve(
        self,
        file_id: str,
        agent_uuid: str,
    ) -> bytes | None:
        """Retrieve file content by ID.

        Args:
            file_id: Anthropic's file identifier
            agent_uuid: Agent session UUID

        Returns:
            File content bytes, or None if not found
        """
        ...

    @abstractmethod
    async def delete(
        self,
        file_id: str,
        agent_uuid: str,
    ) -> bool:
        """Delete a file.

        Args:
            file_id: Anthropic's file identifier
            agent_uuid: Agent session UUID

        Returns:
            True if deleted, False if not found
        """
        ...
