from __future__ import annotations

import base64
import mimetypes
import uuid
from pathlib import Path
from typing import Any

import aiofiles

from .types import MediaBackend, MediaMetadata


class LocalMediaBackend(MediaBackend):
    """Filesystem-backed media storage.

    Storage structure::

        {base_path}/
            {agent_uuid}/
                {media_id}_{filename}

    This follows the same layout as the LocalFilesystemBackend
    in anthropic_agent/file_backends.
    """

    def __init__(
        self,
        base_path: str | Path = "./agent-media",
        url_prefix: str | None = None,
    ) -> None:
        """Initialize local media backend.

        Args:
            base_path: Root directory for media storage.
            url_prefix: Optional URL prefix for to_url(). When set,
                to_url() returns "{url_prefix}/{agent_uuid}/{media_id}".
                When None, returns a file:// URI.
        """
        self.base_path = Path(base_path)
        self.url_prefix = url_prefix.rstrip("/") if url_prefix else None

    async def connect(self) -> None:
        """Create the base directory if it does not exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ─── Internal helpers ─────────────────────────────────────────────

    def _file_path(self, agent_uuid: str, media_id: str, filename: str) -> Path:
        """Build the on-disk path for a media file."""
        return self.base_path / agent_uuid / f"{media_id}_{filename}"

    def _find_file(self, agent_uuid: str, media_id: str) -> Path | None:
        """Locate a media file by media_id using glob (filename unknown)."""
        agent_dir = self.base_path / agent_uuid
        if not agent_dir.exists():
            return None
        matches = list(agent_dir.glob(f"{media_id}_*"))
        return matches[0] if matches else None

    def _extract_filename(self, path: Path, media_id: str) -> str:
        """Extract the original filename from the stored path.

        Given path .../abc123_image.png and media_id abc123,
        returns "image.png".
        """
        return path.name[len(media_id) + 1:]  # skip "{media_id}_"

    @staticmethod
    def _extension_from_filename(filename: str) -> str:
        """Extract extension without the leading dot."""
        dot_pos = filename.rfind(".")
        return filename[dot_pos + 1:] if dot_pos >= 0 else ""

    def _build_metadata(
        self,
        media_id: str,
        filename: str,
        mime_type: str,
        size: int,
        path: Path,
    ) -> MediaMetadata:
        """Construct a MediaMetadata from components."""
        return MediaMetadata(
            media_id=media_id,
            media_mime_type=mime_type,
            media_filename=filename,
            media_extension=self._extension_from_filename(filename),
            media_size=size,
            storage_type="local",
            storage_location=str(path.absolute()),
        )

    # ─── Storage operations ───────────────────────────────────────────

    async def store(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        agent_uuid: str,
    ) -> MediaMetadata:
        media_id = uuid.uuid4().hex
        agent_dir = self.base_path / agent_uuid
        agent_dir.mkdir(parents=True, exist_ok=True)

        file_path = self._file_path(agent_uuid, media_id, filename)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        return self._build_metadata(
            media_id=media_id,
            filename=filename,
            mime_type=mime_type,
            size=len(content),
            path=file_path,
        )

    async def retrieve(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> bytes | None:
        path = self._find_file(agent_uuid, media_id)
        if path is None:
            return None

        async with aiofiles.open(path, "rb") as f:
            return await f.read()

    async def delete(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> bool:
        path = self._find_file(agent_uuid, media_id)
        if path is None:
            return False
        path.unlink()
        return True

    async def exists(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> bool:
        return self._find_file(agent_uuid, media_id) is not None

    async def get_metadata(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> MediaMetadata | None:
        path = self._find_file(agent_uuid, media_id)
        if path is None:
            return None

        filename = self._extract_filename(path, media_id)
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        size = path.stat().st_size

        return self._build_metadata(
            media_id=media_id,
            filename=filename,
            mime_type=mime_type,
            size=size,
            path=path,
        )

    # ─── Resolution ───────────────────────────────────────────────────

    async def to_base64(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> dict[str, str]:
        content = await self.retrieve(media_id, agent_uuid)
        if content is None:
            raise FileNotFoundError(
                f"Media not found: media_id={media_id!r}, agent_uuid={agent_uuid!r}"
            )

        metadata = await self.get_metadata(media_id, agent_uuid)
        assert metadata is not None  # retrieve succeeded, so file exists

        encoded = base64.standard_b64encode(content).decode("ascii")
        return {
            "data": encoded,
            "media_type": metadata.media_mime_type,
        }

    async def to_url(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> str:
        path = self._find_file(agent_uuid, media_id)
        if path is None:
            raise FileNotFoundError(
                f"Media not found: media_id={media_id!r}, agent_uuid={agent_uuid!r}"
            )

        if self.url_prefix is not None:
            return f"{self.url_prefix}/{agent_uuid}/{media_id}"

        return path.absolute().as_uri()

    async def to_reference(
        self,
        media_id: str,
        agent_uuid: str,
    ) -> dict[str, Any]:
        metadata = await self.get_metadata(media_id, agent_uuid)
        if metadata is None:
            raise FileNotFoundError(
                f"Media not found: media_id={media_id!r}, agent_uuid={agent_uuid!r}"
            )
        return metadata.to_dict()
