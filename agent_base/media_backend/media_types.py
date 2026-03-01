from __future__ import annotations

import mimetypes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_base.sandbox.sandbox_types import Sandbox


@dataclass
class MediaMetadata:
    """Metadata describing a stored media file."""

    media_id: str
    media_mime_type: str  # MIME type (e.g. "image/png", "application/pdf")
    media_filename: str  # Original filename (e.g. "image.png", "document.pdf")
    media_extension: str  # Extension without dot (e.g. "png", "pdf")
    media_size: int  # Size in bytes
    storage_type: str  # Backend type (e.g. "local", "s3", "cloudflare_r2")
    storage_location: str  # Backend-specific location (file path, S3 URL, etc.)

    extras: dict[str, Any] = field(default_factory=dict)
    """Backend-specific extension point (must be JSON serializable)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_id": self.media_id,
            "media_mime_type": self.media_mime_type,
            "media_filename": self.media_filename,
            "media_extension": self.media_extension,
            "media_size": self.media_size,
            "storage_type": self.storage_type,
            "storage_location": self.storage_location,
            "extras": self.extras,
        }


class MediaBackend(ABC):
    """Abstract base class for media storage and resolution backends.

    A MediaBackend provides three capabilities:
      1. Storage — store, retrieve, and delete media files.
      2. Existence — check whether media exists without fetching bytes.
      3. Resolution — project stored media into different representations
         (base64, URL, metadata reference) for different consumers.

    All methods that operate on specific media take both media_id and
    agent_uuid. Media is namespaced to agent sessions for isolation,
    cleanup, and storage layout.

    media_id values are generated internally by store() — callers never
    supply their own. The canonical format is uuid4().hex (32-char hex).

    Lifecycle: use as an async context manager.

        async with backend:
            meta = await backend.store(content, "image.png", "image/png", agent_uuid)
            data = await backend.retrieve(meta.media_id, agent_uuid)

    Implementations:
      - LocalMediaBackend  (media_backend/local.py)  — filesystem storage
      - S3MediaBackend     (media_backend/s3.py)     — AWS S3 (future)
      - MemoryMediaBackend (media_backend/memory.py) — in-process (future)
    """

    # ─── Lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize connections or resources. Override if needed.

        Idempotent: calling connect() on an already-connected backend is safe.
        """

    async def close(self) -> None:
        """Release connections or resources. Override if needed.

        After close(), no other method should be called.
        """

    async def __aenter__(self) -> MediaBackend:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.close()

    # ─── Storage operations ───────────────────────────────────────────

    @abstractmethod
    async def store(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        agent_uuid: str,
    ) -> MediaMetadata:
        """Store media bytes and return metadata with a generated media_id.

        The backend generates a new media_id (uuid4 hex) and persists
        the bytes at a backend-specific location.

        Args:
            content: Raw file bytes.
            filename: Original filename (e.g. "photo.png").
            mime_type: MIME type (e.g. "image/png").
            agent_uuid: Agent session UUID for namespacing.

        Returns:
            MediaMetadata with all fields populated.
        """
        ...

    @abstractmethod
    async def retrieve(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> bytes | None:
        """Retrieve media bytes by media_id.

        Args:
            media_id: The media identifier returned by store().
            agent_uuid: Agent session UUID.

        Returns:
            Raw bytes if found, None if not found.
        """
        ...

    @abstractmethod
    async def delete(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> bool:
        """Delete a stored media file.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def exists(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> bool:
        """Check whether media exists without fetching bytes.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            True if the media exists, False otherwise.
        """
        ...

    @abstractmethod
    async def get_metadata(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> MediaMetadata | None:
        """Retrieve metadata for a stored media file without fetching bytes.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            MediaMetadata if found, None if not found.
        """
        ...

    @abstractmethod
    async def update_metadata(
        self,
        media_id: str,
        agent_uuid: str,
        extras: dict[str, Any],
    ) -> None:
        """Merge extra metadata for a stored media file.

        Updates the ``extras`` dict on the stored metadata by merging
        the provided keys. Existing keys not in ``extras`` are preserved.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.
            extras: Key-value pairs to merge into the metadata extras.

        Raises:
            FileNotFoundError: If the media_id does not exist.
        """
        ...

    # ─── Resolution (projections for different consumers) ─────────────

    @abstractmethod
    async def to_base64(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> dict[str, str]:
        """Project media as base64-encoded data for LLM provider adapters.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            {"data": "<base64-encoded-string>", "media_type": "<mime-type>"}

        Raises:
            FileNotFoundError: If the media_id does not exist.
        """
        ...

    @abstractmethod
    async def to_url(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> str:
        """Project media as a URL for frontend rendering.

        The URL format depends on the backend:
          - Local: file:// URI or configurable API route prefix
          - S3: presigned URL

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            A URL string suitable for the backend type.

        Raises:
            FileNotFoundError: If the media_id does not exist.
        """
        ...

    @abstractmethod
    async def to_reference(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> dict[str, Any]:
        """Project media as a lightweight metadata dict for conversation logs.

        Does not include file bytes or base64 data — just the metadata
        needed to later resolve or display the media.

        Args:
            media_id: The media identifier.
            agent_uuid: Agent session UUID.

        Returns:
            The result of MediaMetadata.to_dict(), or equivalent dict.

        Raises:
            FileNotFoundError: If the media_id does not exist.
        """
        ...

    # ─── Sandbox integration ─────────────────────────────────────────

    _sandbox: Sandbox | None = None

    def attach_sandbox(self, sandbox: Sandbox) -> None:
        """Store a reference to the active sandbox.

        Must be called before materialize() or flush_exports().

        Args:
            sandbox: The sandbox instance for this agent session.
        """
        self._sandbox = sandbox

    async def materialize(self, media_id: str, agent_uuid: str) -> str:
        """Retrieve a file from storage and import it into the sandbox.

        Repeated calls overwrite the imported file content for the same
        media_id + filename path.

        Args:
            media_id: The media identifier in the backend.
            agent_uuid: Agent session UUID.

        Returns:
            Sandbox-relative path where the file is accessible.

        Raises:
            RuntimeError: If no sandbox is attached.
            FileNotFoundError: If media_id does not exist in the backend.
        """
        if self._sandbox is None:
            raise RuntimeError(
                "No sandbox attached. Call attach_sandbox() before materialize()."
            )

        metadata = await self.get_metadata(media_id, agent_uuid)
        if metadata is None:
            raise FileNotFoundError(
                f"Cannot materialize: media_id={media_id!r} not found "
                f"for agent_uuid={agent_uuid!r}"
            )

        content = await self.retrieve(media_id, agent_uuid)
        if content is None:
            raise FileNotFoundError(
                f"Cannot materialize: media_id={media_id!r} not found "
                f"for agent_uuid={agent_uuid!r}"
            )

        return await self._sandbox.import_file(
            media_id, metadata.media_filename, content
        )

    async def flush_exports(self, agent_uuid: str) -> list[MediaMetadata]:
        """Collect files from the sandbox exports area and store them.

        Calls sandbox.get_exported_files() to discover all files tools
        have produced, then stores each one in the backend.

        Args:
            agent_uuid: Agent session UUID.

        Returns:
            List of MediaMetadata for each stored file.

        Raises:
            RuntimeError: If no sandbox is attached.
        """
        if self._sandbox is None:
            raise RuntimeError(
                "No sandbox attached. Call attach_sandbox() before flush_exports()."
            )

        exported = await self._sandbox.get_exported_files()
        results: list[MediaMetadata] = []

        for rel_path, content in exported:
            basename = os.path.basename(rel_path)
            mime_type = (
                mimetypes.guess_type(basename)[0] or "application/octet-stream"
            )
            metadata = await self.store(content, basename, mime_type, agent_uuid)
            if rel_path != basename:
                metadata.extras["export_path"] = rel_path
            results.append(metadata)

        return results
