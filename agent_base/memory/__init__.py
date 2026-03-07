"""Memory module for agent_base.

Cross-session knowledge stores that operate at run boundaries only.
Independent of context compaction.

Usage::

    from agent_base.memory import NoOpMemoryStore

    memory_store = NoOpMemoryStore()

    # Factory
    memory_store = get_memory_store("none")
"""
from typing import Any

from .base import MemoryStore, MemoryStoreType
from .stores import NoOpMemoryStore

# Registry mapping string names to store classes.
MEMORY_STORES: dict[str, type[MemoryStore]] = {
    "none": NoOpMemoryStore,
}


def get_memory_store(name: MemoryStoreType, **kwargs: Any) -> MemoryStore:
    """Get a memory store instance by name.

    Args:
        name: Memory store name (currently ``"none"``).
        **kwargs: Additional arguments to pass to the constructor.

    Returns:
        An instance of the requested memory store.

    Raises:
        ValueError: If memory store name is not recognized.
    """
    if name not in MEMORY_STORES:
        available = ", ".join(MEMORY_STORES.keys())
        raise ValueError(
            f"Unknown memory store '{name}'. Available: {available}"
        )

    return MEMORY_STORES[name](**kwargs)


__all__ = [
    # ABC
    "MemoryStore",
    "MemoryStoreType",
    # Implementations
    "NoOpMemoryStore",
    # Factory
    "MEMORY_STORES",
    "get_memory_store",
]
