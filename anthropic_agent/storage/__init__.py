"""Storage module for anthropic_agent.

This module provides a flexible storage abstraction layer with:
- Entity dataclasses (AgentConfig, Conversation, AgentRunLog)
- Abstract adapter interfaces (AgentConfigAdapter, ConversationAdapter, AgentRunAdapter)
- Default implementations for Memory, Filesystem, and PostgreSQL
- Factory functions for easy adapter creation

Usage:
    # Create adapters via factory
    from anthropic_agent.storage import create_adapters
    
    config_adapter, conv_adapter, run_adapter = create_adapters(
        "filesystem",
        base_path="./data"
    )
    
    # Or create individual adapters
    from anthropic_agent.storage import create_agent_config_adapter
    
    adapter = create_agent_config_adapter("postgres", connection_string="...")
    
    # Use adapters
    async with adapter:
        config = AgentConfig(agent_uuid="...", model="claude-sonnet-4-20250514")
        await adapter.save(config)
        
        loaded = await adapter.load(config.agent_uuid)

Extending:
    # Extend entity dataclasses
    from anthropic_agent.storage import AgentConfig
    
    @dataclass
    class MyConfig(AgentConfig):
        custom_field: str = ""
    
    # Or implement custom adapters
    from anthropic_agent.storage import AgentConfigAdapter
    
    class MyAdapter(AgentConfigAdapter):
        async def save(self, config): ...
        async def load(self, uuid): ...
        # ... implement other methods
"""

# Entity dataclasses
from .base import (
    AgentConfig,
    Conversation,
    AgentRunLog,
)

# Abstract base classes
from .base import (
    StorageAdapter,
    AgentConfigAdapter,
    ConversationAdapter,
    AgentRunAdapter,
)

# Exceptions
from .exceptions import (
    StorageError,
    StorageConnectionError,
    StorageNotFoundError,
    StorageValidationError,
    StorageOperationError,
)

# Factory functions
from .registry import (
    AdapterType,
    create_agent_config_adapter,
    create_conversation_adapter,
    create_agent_run_adapter,
    create_adapters,
    available_adapter_types,
)

# Adapter implementations
from .adapters import (
    # Memory
    MemoryAgentConfigAdapter,
    MemoryConversationAdapter,
    MemoryAgentRunAdapter,
    # Filesystem
    FilesystemAgentConfigAdapter,
    FilesystemConversationAdapter,
    FilesystemAgentRunAdapter,
    # PostgreSQL
    PostgresAgentConfigAdapter,
    PostgresConversationAdapter,
    PostgresAgentRunAdapter,
)

__all__ = [
    # Entities
    "AgentConfig",
    "Conversation",
    "AgentRunLog",
    # Abstract bases
    "StorageAdapter",
    "AgentConfigAdapter",
    "ConversationAdapter",
    "AgentRunAdapter",
    # Exceptions
    "StorageError",
    "StorageConnectionError",
    "StorageNotFoundError",
    "StorageValidationError",
    "StorageOperationError",
    # Factory
    "AdapterType",
    "create_agent_config_adapter",
    "create_conversation_adapter",
    "create_agent_run_adapter",
    "create_adapters",
    "available_adapter_types",
    # Memory adapters
    "MemoryAgentConfigAdapter",
    "MemoryConversationAdapter",
    "MemoryAgentRunAdapter",
    # Filesystem adapters
    "FilesystemAgentConfigAdapter",
    "FilesystemConversationAdapter",
    "FilesystemAgentRunAdapter",
    # PostgreSQL adapters
    "PostgresAgentConfigAdapter",
    "PostgresConversationAdapter",
    "PostgresAgentRunAdapter",
]
