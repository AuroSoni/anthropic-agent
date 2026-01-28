"""File storage backend implementations for Anthropic agent."""

from datetime import datetime
from pathlib import Path
from typing import Protocol, Any

from ..logging import get_logger

logger = get_logger(__name__)


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
            "storage_location": None,
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
            "storage_location": None,
            "size": len(content),
            "timestamp": datetime.now().isoformat(),
            "is_update": True,
            "previous_size": existing_metadata.get("size"),
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

        storage_location = str(file_path.absolute())
        logger.info("Stored file", filename=filename, file_id=file_id, backend="local")

        return {
            "file_id": file_id,
            "filename": filename,
            "storage_location": storage_location,
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

        storage_location = str(file_path.absolute())
        logger.info("Updated file", filename=filename, file_id=file_id, backend="local")

        return {
            "file_id": file_id,
            "filename": filename,
            "storage_location": storage_location,
            "size": len(content),
            "timestamp": datetime.now().isoformat(),
            "is_update": True,
            "previous_size": existing_metadata.get("size"),
            "storage_backend": "local",
        }


class S3Backend:
    """S3 storage backend with lazy client initialization.

    Storage structure:
        s3://{bucket}/{prefix}/{agent_uuid}/{file_id}_{filename}
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "agent-files",
        region: str = "ap-south-1",
        endpoint_url: str | None = None,
    ):
        """Initialize S3 backend.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all files
            region: AWS region
            endpoint_url: Custom endpoint for S3-compatible services (e.g., MinIO, LocalStack)
        """
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self.endpoint_url = endpoint_url
        self._client = None

    @property
    def client(self):
        """Lazily initialize and return the S3 client.
        
        Automatically loads AWS credentials from .env file if present.
        """
        if self._client is None:
            from dotenv import load_dotenv
            import boto3

            load_dotenv()

            self._client = boto3.client(
                "s3",
                region_name=self.region,
                endpoint_url=self.endpoint_url,
            )
        return self._client

    def _build_key(self, agent_uuid: str, file_id: str, filename: str) -> str:
        """Build S3 object key."""
        return f"{self.prefix}/{agent_uuid}/{file_id}_{filename}"

    def _build_public_url(self, key: str) -> str:
        """Build publicly accessible URL for an S3 object."""
        if self.endpoint_url:
            # Custom endpoint (e.g., MinIO, LocalStack)
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{key}"
        # Standard S3 public URL
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"

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
            Metadata dict with storage_location (public URL) and other info
        """
        key = self._build_key(agent_uuid, file_id, filename)

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
        )

        storage_location = self._build_public_url(key)
        logger.info("Stored file", filename=filename, file_id=file_id, backend="s3", bucket=self.bucket)

        return {
            "file_id": file_id,
            "filename": filename,
            "storage_location": storage_location,
            "size": len(content),
            "timestamp": datetime.now().isoformat(),
            "is_update": False,
            "storage_backend": "s3",
            "s3_bucket": self.bucket,
            "s3_key": key,
        }

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
        key = self._build_key(agent_uuid, file_id, filename)

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
        )

        storage_location = self._build_public_url(key)
        logger.info("Updated file", filename=filename, file_id=file_id, backend="s3", bucket=self.bucket)

        return {
            "file_id": file_id,
            "filename": filename,
            "storage_location": storage_location,
            "size": len(content),
            "timestamp": datetime.now().isoformat(),
            "is_update": True,
            "previous_size": existing_metadata.get("size"),
            "storage_backend": "s3",
            "s3_bucket": self.bucket,
            "s3_key": key,
        }

