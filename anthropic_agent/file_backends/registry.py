"""File backend registry and factory functions."""

from typing import Literal
from .backends import FileStorageBackend, NoOpBackend, LocalFilesystemBackend, S3Backend

# Type for built-in file backend names
FileBackendType = Literal["local", "s3", "none"]

# Registry mapping names to backend classes
FILE_BACKENDS: dict[str, type[FileStorageBackend]] = {
    "local": LocalFilesystemBackend,
    "s3": S3Backend,
    "none": NoOpBackend,
}


def get_file_backend(
    name: str,
    **kwargs
) -> FileStorageBackend:
    """Factory function to create file backend by name.
    
    Args:
        name: Backend name ("local", "s3", "none")
        **kwargs: Additional arguments passed to backend constructor
        
    Returns:
        Instantiated file backend
        
    Raises:
        ValueError: If backend name is not recognized
        
    Example:
        >>> backend = get_file_backend("local", base_path="/data/files")
        >>> backend = get_file_backend("s3", bucket="my-bucket", region="us-west-2")
    """
    if name not in FILE_BACKENDS:
        raise ValueError(
            f"Unknown file backend: {name}. "
            f"Available backends: {', '.join(FILE_BACKENDS.keys())}"
        )
    
    backend_class = FILE_BACKENDS[name]
    return backend_class(**kwargs)

