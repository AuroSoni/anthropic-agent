"""Database backend registry and factory function.

.. deprecated::
    This module is deprecated. Use :mod:`anthropic_agent.storage.registry` instead.
"""

import warnings
from typing import Literal
from .backends import DatabaseBackend, FilesystemBackend, SQLBackend


# Type alias for backend names
DBBackendType = Literal["filesystem", "sql"]

# Registry of available backends
DB_BACKENDS: dict[str, type[DatabaseBackend]] = {
    "filesystem": FilesystemBackend,
    "sql": SQLBackend,
}


def get_db_backend(name: str, **kwargs) -> DatabaseBackend:
    """Factory function to create database backend by name.
    
    .. deprecated::
        Use :func:`anthropic_agent.storage.create_adapters` instead.
    
    Args:
        name: Backend name ("filesystem" or "sql")
        **kwargs: Backend-specific configuration options
        
    Returns:
        Configured DatabaseBackend instance
        
    Raises:
        ValueError: If backend name is not recognized
        
    Examples:
        >>> backend = get_db_backend("filesystem")
        >>> backend = get_db_backend("filesystem", base_path="/var/data")
        >>> backend = get_db_backend("sql", connection_string="postgresql://...")
    """
    warnings.warn(
        "get_db_backend is deprecated. Use anthropic_agent.storage.create_adapters instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if name not in DB_BACKENDS:
        raise ValueError(
            f"Unknown database backend: {name}. "
            f"Available backends: {list(DB_BACKENDS.keys())}"
        )
    
    backend_class = DB_BACKENDS[name]
    return backend_class(**kwargs)
