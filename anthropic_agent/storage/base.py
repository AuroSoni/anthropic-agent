"""Base abstractions for storage adapters.

This module defines:
1. Entity dataclasses (AgentConfig, Conversation, AgentRunLog)
2. Abstract base classes for storage adapters

Users can extend entities via subclassing or the `extras` dict field.
Users can implement custom adapters by subclassing the adapter ABCs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar, Self

T = TypeVar("T")


# =============================================================================
# Entity Dataclasses
# =============================================================================

@dataclass
class AgentConfig:
    """Agent configuration and state for session resumption.
    
    All fields match the existing schema in database/schemas.md.
    The `extras` field allows users to store custom data without subclassing.
    """
    agent_uuid: str
    
    # Core configuration
    system_prompt: str | None = None
    description: str | None = None
    model: str = "claude-sonnet-4-20250514"
    max_steps: int = 50
    thinking_tokens: int = 0
    max_tokens: int = 2048
    
    # State for resumption
    container_id: str | None = None
    messages: list[dict] = field(default_factory=list)
    
    # Tools configuration
    tool_schemas: list[dict] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    server_tools: list[dict] = field(default_factory=list)
    
    # Beta features
    beta_headers: list[str] = field(default_factory=list)
    
    # API configuration
    api_kwargs: dict[str, Any] = field(default_factory=dict)
    
    # Component configuration
    formatter: str | None = None
    stream_meta_history_and_tool_results: bool = False
    compactor_type: str | None = None
    memory_store_type: str | None = None
    
    # File registry
    file_registry: dict[str, dict] = field(default_factory=dict)
    
    # Retry configuration
    max_retries: int = 5
    base_delay: float = 1.0
    
    # Token tracking
    last_known_input_tokens: int = 0
    last_known_output_tokens: int = 0
    
    # Frontend tool relay state
    pending_frontend_tools: list[dict] = field(default_factory=list)
    pending_backend_results: list[dict] = field(default_factory=list)
    awaiting_frontend_tools: bool = False
    current_step: int = 0
    conversation_history: list[dict] = field(default_factory=list)

    # Subagent hierarchy
    parent_agent_uuid: str | None = None
    subagent_schemas: list[dict] = field(default_factory=list)
    
    # UI metadata
    title: str | None = None
    
    # Timestamps
    created_at: str | None = None
    updated_at: str | None = None
    last_run_at: str | None = None
    total_runs: int = 0
    
    # User extension point - store custom fields here
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A single conversation/run record for UI display and pagination.
    
    All fields match the existing conversation_history schema.
    """
    conversation_id: str
    agent_uuid: str
    run_id: str
    
    # Run timing
    started_at: str | None = None
    completed_at: str | None = None
    
    # User interaction
    user_message: str = ""
    final_response: str | None = None
    
    # Full conversation for this run
    messages: list[dict] = field(default_factory=list)
    
    # Run outcome
    stop_reason: str | None = None
    total_steps: int | None = None
    
    # Token usage
    usage: dict[str, int] = field(default_factory=dict)
    
    # Files generated in this run
    generated_files: list[dict] = field(default_factory=list)

    # Cost breakdown for this run (CostBreakdown as dict)
    cost: dict[str, Any] = field(default_factory=dict)

    # Sequence for pagination (auto-assigned)
    sequence_number: int | None = None
    
    # Metadata
    created_at: str | None = None
    
    # User extension point
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunLog:
    """Container for agent run logs (one per run).
    
    Logs are stored as a list of log entries for a specific run.
    """
    agent_uuid: str
    run_id: str
    logs: list[dict] = field(default_factory=list)
    
    # User extension point
    extras: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Abstract Base Classes for Adapters
# =============================================================================

class StorageAdapter(ABC, Generic[T]):
    """Base class for all storage adapters with lifecycle management.
    
    Provides async context manager support for resource cleanup.
    """
    
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


class AgentConfigAdapter(StorageAdapter[AgentConfig]):
    """Abstract adapter for agent configuration storage.
    
    Implementations must handle:
    - Saving/loading full agent config
    - Updating specific fields (like title)
    - Listing sessions with pagination
    
    Default implementations are provided for filesystem and PostgreSQL
    that maintain backward compatibility with existing data structures.
    """
    
    @abstractmethod
    async def save(self, config: AgentConfig) -> None:
        """Save or update agent configuration.
        
        Args:
            config: AgentConfig instance to save
        """
        ...
    
    @abstractmethod
    async def load(self, agent_uuid: str) -> AgentConfig | None:
        """Load agent configuration by UUID.
        
        Args:
            agent_uuid: Agent session UUID
            
        Returns:
            AgentConfig if found, None otherwise
        """
        ...
    
    @abstractmethod
    async def delete(self, agent_uuid: str) -> bool:
        """Delete agent configuration.
        
        Args:
            agent_uuid: Agent session UUID
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    @abstractmethod
    async def update_title(self, agent_uuid: str, title: str) -> bool:
        """Update the title for an agent session.
        
        Args:
            agent_uuid: Agent session UUID
            title: New title
            
        Returns:
            True if updated, False if not found
        """
        ...
    
    @abstractmethod
    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict], int]:
        """List all agent sessions with metadata.
        
        Args:
            limit: Maximum sessions to return
            offset: Number of sessions to skip
            
        Returns:
            Tuple of (sessions list, total count)
            Each session dict contains: agent_uuid, title, created_at, updated_at, total_runs
        """
        ...


class ConversationAdapter(StorageAdapter[Conversation]):
    """Abstract adapter for conversation history storage.
    
    Implementations must handle:
    - Saving conversations with auto-incrementing sequence numbers
    - Loading paginated history (offset-based and cursor-based)
    
    Default implementations maintain backward compatibility with existing structures.
    """
    
    @abstractmethod
    async def save(self, conversation: Conversation) -> None:
        """Save a conversation record.
        
        The sequence_number should be auto-assigned by the adapter.
        
        Args:
            conversation: Conversation instance to save
        """
        ...
    
    @abstractmethod
    async def load_history(
        self,
        agent_uuid: str,
        limit: int = 20,
        offset: int = 0
    ) -> list[Conversation]:
        """Load paginated conversation history (newest first).
        
        Args:
            agent_uuid: Agent session UUID
            limit: Maximum conversations to return
            offset: Number of conversations to skip
            
        Returns:
            List of Conversation instances, sorted by sequence_number descending
        """
        ...
    
    @abstractmethod
    async def load_cursor(
        self,
        agent_uuid: str,
        before: int | None = None,
        limit: int = 20
    ) -> tuple[list[Conversation], bool]:
        """Load conversations with cursor-based pagination.
        
        Designed for infinite scroll UIs that load newest-to-oldest.
        
        Args:
            agent_uuid: Agent session UUID
            before: Load conversations with sequence_number < before (None = latest)
            limit: Maximum conversations to return
            
        Returns:
            Tuple of (conversations newest->oldest, has_more)
        """
        ...


class AgentRunAdapter(StorageAdapter[AgentRunLog]):
    """Abstract adapter for agent run logs storage.
    
    Implementations must handle:
    - Batch saving of run logs at end of run
    - Loading all logs for a specific run
    
    Default implementations maintain backward compatibility (JSONL for filesystem).
    """
    
    @abstractmethod
    async def save_logs(
        self,
        agent_uuid: str,
        run_id: str,
        logs: list[dict]
    ) -> None:
        """Save batched agent run logs.
        
        Called at the end of a run with all accumulated logs.
        
        Args:
            agent_uuid: Agent session UUID
            run_id: Unique run identifier
            logs: List of log entries
        """
        ...
    
    @abstractmethod
    async def load_logs(
        self,
        agent_uuid: str,
        run_id: str
    ) -> list[dict]:
        """Load all logs for a specific run.
        
        Args:
            agent_uuid: Agent session UUID
            run_id: Unique run identifier
            
        Returns:
            List of log entries in chronological order
        """
        ...
