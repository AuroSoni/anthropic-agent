"""S3 file storage backend."""

import asyncio
from datetime import datetime
from typing import Any

from ...logging import get_logger
from ..base import FileMetadata, FileStorageBackend

logger = get_logger(__name__)


class S3Backend(FileStorageBackend):
    """S3 storage backend with lazy client initialization.

    Storage structure::

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
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{key}"
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"

    # -- Async operations (wrap sync boto3 via asyncio.to_thread) -----------

    async def store(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        agent_uuid: str,
    ) -> FileMetadata:
        """Store a new file in S3."""
        key = self._build_key(agent_uuid, file_id, filename)

        await asyncio.to_thread(
            self.client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=content,
        )

        storage_location = self._build_public_url(key)
        logger.info("Stored file", filename=filename, file_id=file_id, backend="s3", bucket=self.bucket)

        return FileMetadata(
            file_id=file_id,
            filename=filename,
            storage_location=storage_location,
            size=len(content),
            timestamp=datetime.now().isoformat(),
            is_update=False,
            storage_backend="s3",
            extras={"s3_bucket": self.bucket, "s3_key": key},
        )

    async def update(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        existing_metadata: dict[str, Any],
        agent_uuid: str,
    ) -> FileMetadata:
        """Update an existing file in S3."""
        key = self._build_key(agent_uuid, file_id, filename)

        await asyncio.to_thread(
            self.client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=content,
        )

        storage_location = self._build_public_url(key)
        logger.info("Updated file", filename=filename, file_id=file_id, backend="s3", bucket=self.bucket)

        return FileMetadata(
            file_id=file_id,
            filename=filename,
            storage_location=storage_location,
            size=len(content),
            timestamp=datetime.now().isoformat(),
            is_update=True,
            storage_backend="s3",
            previous_size=existing_metadata.get("size"),
            extras={"s3_bucket": self.bucket, "s3_key": key},
        )

    async def retrieve(
        self,
        file_id: str,
        agent_uuid: str,
    ) -> bytes | None:
        """Download file content from S3.

        Attempts to find the object by listing the prefix for the given file_id.
        """
        prefix = f"{self.prefix}/{agent_uuid}/{file_id}_"

        try:
            response = await asyncio.to_thread(
                self.client.list_objects_v2,
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=1,
            )
            contents = response.get("Contents", [])
            if not contents:
                return None

            key = contents[0]["Key"]
            obj = await asyncio.to_thread(
                self.client.get_object,
                Bucket=self.bucket,
                Key=key,
            )
            body = obj["Body"].read()
            return body

        except Exception:
            logger.warning("Failed to retrieve file from S3", file_id=file_id, exc_info=True)
            return None

    async def delete(
        self,
        file_id: str,
        agent_uuid: str,
    ) -> bool:
        """Delete a file from S3."""
        prefix = f"{self.prefix}/{agent_uuid}/{file_id}_"

        try:
            response = await asyncio.to_thread(
                self.client.list_objects_v2,
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=1,
            )
            contents = response.get("Contents", [])
            if not contents:
                return False

            key = contents[0]["Key"]
            await asyncio.to_thread(
                self.client.delete_object,
                Bucket=self.bucket,
                Key=key,
            )
            logger.info("Deleted file from S3", file_id=file_id, bucket=self.bucket)
            return True

        except Exception:
            logger.warning("Failed to delete file from S3", file_id=file_id, exc_info=True)
            return False
