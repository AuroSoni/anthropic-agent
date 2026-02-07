"""Database backends for persisting agent state, conversation history, and run logs.

.. deprecated::
    This module is deprecated. Use :mod:`anthropic_agent.storage` instead.
    
    Migration guide:
        # Old way (deprecated)
        from anthropic_agent.database import FilesystemBackend, get_db_backend
        backend = get_db_backend("filesystem")
        
        # New way (recommended)
        from anthropic_agent.storage import (
            create_adapters,
            FilesystemAgentConfigAdapter,
            FilesystemConversationAdapter,
            FilesystemAgentRunAdapter,
        )
        config_adapter, conv_adapter, run_adapter = create_adapters("filesystem")
        
    The new storage module provides:
    - Type-safe entity dataclasses (AgentConfig, Conversation, AgentRunLog)
    - Specialized adapters for each entity type
    - True async I/O with aiofiles
    - Better separation of concerns for custom implementations
"""

import warnings

from .backends import DatabaseBackend, FilesystemBackend, SQLBackend
from .registry import DBBackendType, get_db_backend, DB_BACKENDS

__all__ = [
    "DatabaseBackend",
    "FilesystemBackend",
    "SQLBackend",
    "DBBackendType",
    "get_db_backend",
    "DB_BACKENDS",
]


def __getattr__(name: str):
    """Emit deprecation warning when accessing module attributes."""
    if name in __all__:
        warnings.warn(
            f"anthropic_agent.database is deprecated. "
            f"Use anthropic_agent.storage instead. "
            f"See the module docstring for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )
        if name == "DatabaseBackend":
            return DatabaseBackend
        elif name == "FilesystemBackend":
            return FilesystemBackend
        elif name == "SQLBackend":
            return SQLBackend
        elif name == "DBBackendType":
            return DBBackendType
        elif name == "get_db_backend":
            return get_db_backend
        elif name == "DB_BACKENDS":
            return DB_BACKENDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
