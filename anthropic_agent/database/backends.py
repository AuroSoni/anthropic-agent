"""Database backend implementations for persisting agent state.

This module provides the DatabaseBackend protocol and concrete implementations
for storing agent configuration, conversation history, and detailed run logs.
"""

import json
import logging
from pathlib import Path
from typing import Protocol
from datetime import datetime


logger = logging.getLogger(__name__)


class DatabaseBackend(Protocol):
    """Protocol for database backend implementations.
    
    Backends persist three types of data:
    1. agent_config: Session state for resumption (messages, container_id, etc.)
    2. conversation_history: UI-displayable conversation records  
    3. agent_runs: Detailed execution logs for debugging and evaluation
    """
    
    async def save_agent_config(self, config: dict) -> None:
        """Save/update agent configuration.
        
        Args:
            config: Agent configuration including system_prompt, model, messages,
                   container_id, file_registry, component configs, etc.
        """
        ...
    
    async def load_agent_config(self, agent_uuid: str) -> dict | None:
        """Load agent configuration.
        
        Args:
            agent_uuid: Agent session UUID
            
        Returns:
            Agent configuration dict, or None if not found
        """
        ...
    
    async def save_conversation_history(self, conversation: dict) -> None:
        """Save a conversation history entry.
        
        Args:
            conversation: Conversation data including user_message, final_response,
                         messages, usage, generated_files, etc.
        """
        ...
    
    async def load_conversation_history(
        self, 
        agent_uuid: str, 
        limit: int = 20, 
        offset: int = 0
    ) -> list[dict]:
        """Load paginated conversation history (latest first).
        
        Args:
            agent_uuid: Agent session UUID
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            
        Returns:
            List of conversation dicts, sorted by sequence_number descending
        """
        ...
    
    async def save_agent_run_logs(
        self, 
        agent_uuid: str, 
        run_id: str, 
        logs: list[dict]
    ) -> None:
        """Save batched agent_run logs (called at end of run).
        
        Args:
            agent_uuid: Agent session UUID
            run_id: Unique run identifier
            logs: List of log entries with action_type, action_data, timestamps, etc.
        """
        ...
    
    async def load_agent_run_logs(
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


class FilesystemBackend:
    """Filesystem-based database backend.
    
    Stores data as JSON files in a hierarchical directory structure:
        {base_path}/
            agent_config/
                {agent_uuid}.json
            conversation_history/
                {agent_uuid}/
                    001.json
                    002.json
                    index.json
            agent_runs/
                {agent_uuid}/
                    {run_id}.jsonl
    """
    
    def __init__(self, base_path: str = "./data"):
        """Initialize filesystem backend.
        
        Args:
            base_path: Root directory for database storage (default: ./data)
        """
        self.base_path = Path(base_path)
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create directory structure if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "agent_config").mkdir(exist_ok=True)
        (self.base_path / "conversation_history").mkdir(exist_ok=True)
        (self.base_path / "agent_runs").mkdir(exist_ok=True)
    
    async def save_agent_config(self, config: dict) -> None:
        """Save agent configuration to JSON file."""
        agent_uuid = config["agent_uuid"]
        config_file = self.base_path / "agent_config" / f"{agent_uuid}.json"
        
        # Write atomically by writing to temp file then renaming
        temp_file = config_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(config, f, indent=2)
        temp_file.replace(config_file)
        
        logger.debug(f"Saved agent config for {agent_uuid}")
    
    async def load_agent_config(self, agent_uuid: str) -> dict | None:
        """Load agent configuration from JSON file."""
        config_file = self.base_path / "agent_config" / f"{agent_uuid}.json"
        
        if not config_file.exists():
            return None
        
        with open(config_file, "r") as f:
            config = json.load(f)
        
        logger.debug(f"Loaded agent config for {agent_uuid}")
        return config
    
    async def save_conversation_history(self, conversation: dict) -> None:
        """Save conversation history entry with automatic sequence numbering."""
        agent_uuid = conversation["agent_uuid"]
        conv_dir = self.base_path / "conversation_history" / agent_uuid
        conv_dir.mkdir(parents=True, exist_ok=True)
        
        # Get next sequence number from index
        index_file = conv_dir / "index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                index = json.load(f)
                last_sequence = index.get("last_sequence", 0)
        else:
            last_sequence = 0
        
        next_sequence = last_sequence + 1
        conversation["sequence_number"] = next_sequence
        
        # Save conversation with padded sequence number
        conv_file = conv_dir / f"{next_sequence:03d}.json"
        with open(conv_file, "w") as f:
            json.dump(conversation, f, indent=2)
        
        # Update index
        index = {
            "last_sequence": next_sequence,
            "total_conversations": next_sequence,
            "updated_at": datetime.now().isoformat()
        }
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)
        
        logger.debug(f"Saved conversation history for {agent_uuid}, sequence {next_sequence}")
    
    async def load_conversation_history(
        self, 
        agent_uuid: str, 
        limit: int = 20, 
        offset: int = 0
    ) -> list[dict]:
        """Load paginated conversation history (latest first)."""
        conv_dir = self.base_path / "conversation_history" / agent_uuid
        
        if not conv_dir.exists():
            return []
        
        # Get all conversation files
        conv_files = sorted(conv_dir.glob("[0-9]*.json"), reverse=True)
        
        # Apply pagination
        paginated_files = conv_files[offset:offset + limit]
        
        # Load conversations
        conversations = []
        for conv_file in paginated_files:
            with open(conv_file, "r") as f:
                conversations.append(json.load(f))
        
        logger.debug(f"Loaded {len(conversations)} conversations for {agent_uuid}")
        return conversations
    
    async def save_agent_run_logs(
        self, 
        agent_uuid: str, 
        run_id: str, 
        logs: list[dict]
    ) -> None:
        """Save agent run logs in JSONL format (one JSON object per line)."""
        runs_dir = self.base_path / "agent_runs" / agent_uuid
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = runs_dir / f"{run_id}.jsonl"
        
        # Write all logs in JSONL format
        with open(log_file, "w") as f:
            for log_entry in logs:
                f.write(json.dumps(log_entry) + "\n")
        
        logger.debug(f"Saved {len(logs)} log entries for run {run_id}")
    
    async def load_agent_run_logs(
        self, 
        agent_uuid: str, 
        run_id: str
    ) -> list[dict]:
        """Load agent run logs from JSONL file."""
        log_file = self.base_path / "agent_runs" / agent_uuid / f"{run_id}.jsonl"
        
        if not log_file.exists():
            return []
        
        # Read JSONL format
        logs = []
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        
        logger.debug(f"Loaded {len(logs)} log entries for run {run_id}")
        return logs


class SQLBackend:
    """Placeholder for PostgreSQL backend implementation.
    
    TODO: Implement using asyncpg or sqlalchemy with async support.
    Schema defined in anthropic_agent/src/database/schemas.md
    
    Planned features:
    - Connection pooling
    - Transactions for atomic writes
    - Indexes for efficient queries
    - Partitioning for large-scale deployments
    """
    
    def __init__(
        self, 
        connection_string: str,
        pool_size: int = 10
    ):
        """Initialize SQL backend.
        
        Args:
            connection_string: PostgreSQL connection string
            pool_size: Connection pool size
            
        Raises:
            NotImplementedError: SQLBackend is not yet implemented
        """
        raise NotImplementedError(
            "SQLBackend not yet implemented. "
            "Use FilesystemBackend for now. "
            "See anthropic_agent/src/database/schemas.md for planned schema."
        )
    
    async def save_agent_config(self, config: dict) -> None:
        """Save agent configuration to database."""
        raise NotImplementedError
    
    async def load_agent_config(self, agent_uuid: str) -> dict | None:
        """Load agent configuration from database."""
        raise NotImplementedError
    
    async def save_conversation_history(self, conversation: dict) -> None:
        """Save conversation history entry to database."""
        raise NotImplementedError
    
    async def load_conversation_history(
        self, 
        agent_uuid: str, 
        limit: int = 20, 
        offset: int = 0
    ) -> list[dict]:
        """Load paginated conversation history from database."""
        raise NotImplementedError
    
    async def save_agent_run_logs(
        self, 
        agent_uuid: str, 
        run_id: str, 
        logs: list[dict]
    ) -> None:
        """Save agent run logs to database."""
        raise NotImplementedError
    
    async def load_agent_run_logs(
        self, 
        agent_uuid: str, 
        run_id: str
    ) -> list[dict]:
        """Load agent run logs from database."""
        raise NotImplementedError

