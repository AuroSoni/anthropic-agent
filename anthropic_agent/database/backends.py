"""Database backend implementations for persisting agent state.

This module provides the DatabaseBackend protocol and concrete implementations
for storing agent configuration, conversation history, and detailed run logs.
"""

import json
from pathlib import Path
from typing import Protocol, Any
from datetime import datetime

import asyncpg

from ..logging import get_logger

logger = get_logger(__name__)


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
    
    async def load_conversation_history_cursor(
        self,
        agent_uuid: str,
        before: int | None = None,
        limit: int = 20
    ) -> tuple[list[dict], bool]:
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
    
    async def update_agent_title(self, agent_uuid: str, title: str) -> bool:
        """Update the title for an agent session.
        
        Args:
            agent_uuid: Agent session UUID
            title: New title for the conversation
            
        Returns:
            True if updated successfully, False if agent not found
        """
        ...
    
    async def list_agent_sessions(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict], int]:
        """List all agent sessions with metadata.
        
        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            Tuple of (sessions list, total count) where each session has:
            {agent_uuid, title, created_at, updated_at, total_runs}
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
    
    def _json_default(self, obj: Any) -> str:
        """JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    async def save_agent_config(self, config: dict) -> None:
        """Save agent configuration to JSON file."""
        agent_uuid = config["agent_uuid"]
        config_file = self.base_path / "agent_config" / f"{agent_uuid}.json"
        
        # Write atomically by writing to temp file then renaming
        temp_file = config_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(config, f, indent=2, default=self._json_default)
        temp_file.replace(config_file)
        logger.debug("Saved agent config", agent_uuid=agent_uuid, backend="filesystem", config=config)
    async def load_agent_config(self, agent_uuid: str) -> dict | None:
        """Load agent configuration from JSON file."""
        config_file = self.base_path / "agent_config" / f"{agent_uuid}.json"
        
        if not config_file.exists():
            return None
        
        with open(config_file, "r") as f:
            config = json.load(f)
        
        logger.debug("Loaded agent config", agent_uuid=agent_uuid, backend="filesystem", config=config)
        return config
    
    async def update_agent_title(self, agent_uuid: str, title: str) -> bool:
        """Update the title for an agent session."""
        config = await self.load_agent_config(agent_uuid)
        if config is None:
            return False
        
        config["title"] = title
        await self.save_agent_config(config)
        
        logger.debug("Updated agent title", agent_uuid=agent_uuid, backend="filesystem", title=title)
        return True
    
    async def list_agent_sessions(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict], int]:
        """List all agent sessions with metadata."""
        config_dir = self.base_path / "agent_config"
        
        if not config_dir.exists():
            return [], 0
        
        # Get all config files and their metadata
        sessions: list[dict] = []
        config_files = list(config_dir.glob("*.json"))
        
        for config_file in config_files:
            if config_file.name.endswith(".tmp"):
                continue
            
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                
                # Use file modification time as fallback for updated_at
                file_mtime = datetime.fromtimestamp(config_file.stat().st_mtime)
                
                sessions.append({
                    "agent_uuid": config.get("agent_uuid", config_file.stem),
                    "title": config.get("title"),
                    "created_at": config.get("created_at"),
                    "updated_at": config.get("updated_at") or file_mtime.isoformat(),
                    "total_runs": config.get("total_runs", 0),
                })
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load config file", file=str(config_file))
                continue
        
        # Sort by updated_at descending (newest first)
        sessions.sort(
            key=lambda x: x.get("updated_at") or "",
            reverse=True
        )
        
        total = len(sessions)
        # Apply pagination
        sessions = sessions[offset:offset + limit]
        
        logger.debug("Listed agent sessions", sessions=len(sessions), total=total, backend="filesystem")
        return sessions, total
    
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
            json.dump(conversation, f, indent=2, default=self._json_default)
        
        # Update index
        index = {
            "last_sequence": next_sequence,
            "total_conversations": next_sequence,
            "updated_at": datetime.now().isoformat()
        }
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)
        logger.debug("Saved conversation history", agent_uuid=agent_uuid, backend="filesystem", conversation=conversation)
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
        
        logger.debug("Loaded conversation history", agent_uuid=agent_uuid, backend="filesystem", conversations=len(conversations))
        return conversations
    
    async def load_conversation_history_cursor(
        self,
        agent_uuid: str,
        before: int | None = None,
        limit: int = 20
    ) -> tuple[list[dict], bool]:
        """Load conversations with cursor-based pagination (newest first).
        
        Args:
            agent_uuid: Agent session UUID
            before: Load conversations with sequence_number < before (None = latest)
            limit: Maximum conversations to return
            
        Returns:
            Tuple of (conversations newest->oldest, has_more)
        """
        conv_dir = self.base_path / "conversation_history" / agent_uuid
        
        if not conv_dir.exists():
            return [], False
        
        # Get all conversation files sorted by sequence number descending
        conv_files = sorted(conv_dir.glob("[0-9]*.json"), reverse=True)
        
        # Filter by cursor if provided
        if before is not None:
            # Extract sequence number from filename (e.g., "001.json" -> 1)
            conv_files = [
                f for f in conv_files
                if int(f.stem) < before
            ]
        
        # Take limit + 1 to determine has_more
        selected_files = conv_files[:limit + 1]
        has_more = len(selected_files) > limit
        files_to_load = selected_files[:limit]
        
        # Load conversations
        conversations = []
        for conv_file in files_to_load:
            with open(conv_file, "r") as f:
                conversations.append(json.load(f))
        
        logger.debug("Loaded conversation history cursor", agent_uuid=agent_uuid, backend="filesystem", conversations=len(conversations), has_more=has_more)
        return conversations, has_more
    
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
                f.write(json.dumps(log_entry, default=self._json_default) + "\n")
        logger.debug("Saved agent run logs", agent_uuid=agent_uuid, run_id=run_id, backend="filesystem", logs=len(logs))
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
        
        logger.debug("Loaded agent run logs", agent_uuid=agent_uuid, run_id=run_id, backend="filesystem", logs=len(logs))
        return logs


class SQLBackend:
    """PostgreSQL backend implementation using asyncpg.
    
    Features:
    - Connection pooling with lazy initialization
    - UPSERT for agent_config updates
    - Batch inserts for run logs
    - Proper JSONB handling
    
    Schema defined in anthropic_agent/database/schemas.md
    """
    
    def __init__(
        self, 
        connection_string: str,
        pool_size: int = 10
    ):
        """Initialize SQL backend.
        
        Args:
            connection_string: PostgreSQL connection string
                e.g., "postgresql://user:pass@host:5432/dbname"
            pool_size: Maximum connection pool size (default: 10)
            
        Raises:
            ImportError: If asyncpg is not installed
        """
        
        self._connection_string = connection_string
        self._pool_size = pool_size
        self._pool: "asyncpg.Pool | None" = None
    
    async def _get_pool(self) -> "asyncpg.Pool":
        """Get or create the connection pool (lazy initialization).
        
        Returns:
            asyncpg connection pool
        """
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=1,
                max_size=self._pool_size,
                server_settings={'timezone': 'Asia/Kolkata'}
            )
            logger.info("Created PostgreSQL connection pool", max_size=self._pool_size)
        return self._pool
    
    async def close(self) -> None:
        """Close the connection pool.
        
        Should be called before application shutdown.
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Closed PostgreSQL connection pool")
    
    def _parse_datetime(self, value: Any) -> str | None:
        """Parse datetime to ISO format string."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
    
    def _json_default(self, obj: Any) -> str:
        """JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    def _to_jsonb(self, value: Any) -> str | None:
        """Serialize a Python object to JSON string for JSONB columns.
        
        Asyncpg requires JSON strings for JSONB columns; it does not auto-serialize
        Python dicts/lists. This method handles datetime serialization within the data.
        """
        if value is None:
            return None
        return json.dumps(value, default=self._json_default)
    
    def _from_jsonb(self, value: Any) -> Any:
        """Deserialize JSONB column value to Python object.
        
        Handles both cases:
        - If asyncpg already decoded JSONB to dict/list, return as-is
        - If value is a string (legacy data or edge case), parse with json.loads
        """
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value  # Return as-is if not valid JSON
        return value  # Already a dict/list
    
    def _to_datetime(self, value: Any) -> datetime | None:
        """Coerce possible ISO string timestamps to datetime objects for SQL parameters.
        
        Accepts None, datetime, or ISO-like strings (with optional trailing 'Z').
        Returns a datetime or None if parsing fails.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            text = value.strip()
            # Handle common 'Z' suffix for UTC
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(text)
            except Exception:
                return None
        return None
    
    async def save_agent_config(self, config: dict) -> None:
        """Save/update agent configuration using UPSERT.
        
        Args:
            config: Agent configuration dict with agent_uuid as primary key
        """
        pool = await self._get_pool()
        
        query = """
            INSERT INTO agent_config (
                agent_uuid, system_prompt, model, max_steps, thinking_tokens,
                max_tokens, container_id, messages, tool_schemas, tool_names,
                server_tools, beta_headers, api_kwargs, formatter,
                stream_meta_history_and_tool_results, compactor_type,
                memory_store_type, file_registry, max_retries, base_delay,
                last_known_input_tokens, last_known_output_tokens,
                title, created_at, updated_at, last_run_at, total_runs,
                pending_frontend_tools, pending_backend_results,
                awaiting_frontend_tools, current_step, conversation_history,
                extras
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32,
                $33
            )
            ON CONFLICT (agent_uuid) DO UPDATE SET
                system_prompt = EXCLUDED.system_prompt,
                model = EXCLUDED.model,
                max_steps = EXCLUDED.max_steps,
                thinking_tokens = EXCLUDED.thinking_tokens,
                max_tokens = EXCLUDED.max_tokens,
                container_id = EXCLUDED.container_id,
                messages = EXCLUDED.messages,
                tool_schemas = EXCLUDED.tool_schemas,
                tool_names = EXCLUDED.tool_names,
                server_tools = EXCLUDED.server_tools,
                beta_headers = EXCLUDED.beta_headers,
                api_kwargs = EXCLUDED.api_kwargs,
                formatter = EXCLUDED.formatter,
                stream_meta_history_and_tool_results = EXCLUDED.stream_meta_history_and_tool_results,
                compactor_type = EXCLUDED.compactor_type,
                memory_store_type = EXCLUDED.memory_store_type,
                file_registry = EXCLUDED.file_registry,
                max_retries = EXCLUDED.max_retries,
                base_delay = EXCLUDED.base_delay,
                last_known_input_tokens = EXCLUDED.last_known_input_tokens,
                last_known_output_tokens = EXCLUDED.last_known_output_tokens,
                title = EXCLUDED.title,
                updated_at = EXCLUDED.updated_at,
                last_run_at = EXCLUDED.last_run_at,
                total_runs = EXCLUDED.total_runs,
                pending_frontend_tools = EXCLUDED.pending_frontend_tools,
                pending_backend_results = EXCLUDED.pending_backend_results,
                awaiting_frontend_tools = EXCLUDED.awaiting_frontend_tools,
                current_step = EXCLUDED.current_step,
                conversation_history = EXCLUDED.conversation_history,
                extras = EXCLUDED.extras
        """
        
        async with pool.acquire() as conn:
            await conn.execute(
                query,
                config.get("agent_uuid"),
                config.get("system_prompt"),
                config.get("model"),
                config.get("max_steps"),
                config.get("thinking_tokens"),
                config.get("max_tokens"),
                config.get("container_id"),
                self._to_jsonb(config.get("messages")),
                self._to_jsonb(config.get("tool_schemas")),
                config.get("tool_names"),
                self._to_jsonb(config.get("server_tools")),
                config.get("beta_headers"),
                self._to_jsonb(config.get("api_kwargs")),
                config.get("formatter"),
                config.get("stream_meta_history_and_tool_results", False),
                config.get("compactor_type"),
                config.get("memory_store_type"),
                self._to_jsonb(config.get("file_registry")),
                config.get("max_retries"),
                config.get("base_delay"),
                config.get("last_known_input_tokens"),
                config.get("last_known_output_tokens"),
                config.get("title"),
                self._to_datetime(config.get("created_at")),
                self._to_datetime(config.get("updated_at")),
                self._to_datetime(config.get("last_run_at")),
                config.get("total_runs"),
                # Frontend tool relay state
                self._to_jsonb(config.get("pending_frontend_tools", [])),
                self._to_jsonb(config.get("pending_backend_results", [])),
                config.get("awaiting_frontend_tools", False),
                config.get("current_step", 0),
                self._to_jsonb(config.get("conversation_history", [])),
                self._to_jsonb(config.get("extras", {})),
            )
        logger.debug("Saved agent config", agent_uuid=config.get("agent_uuid"), backend="sql", config=config)
        
    async def load_agent_config(self, agent_uuid: str) -> dict | None:
        """Load agent configuration from database.
        
        Args:
            agent_uuid: Agent session UUID
            
        Returns:
            Agent configuration dict, or None if not found
        """
        pool = await self._get_pool()
        
        query = """
            SELECT 
                agent_uuid, system_prompt, model, max_steps, thinking_tokens,
                max_tokens, container_id, messages, tool_schemas, tool_names,
                server_tools, beta_headers, api_kwargs, formatter,
                stream_meta_history_and_tool_results, compactor_type,
                memory_store_type, file_registry, max_retries, base_delay,
                last_known_input_tokens, last_known_output_tokens,
                created_at, updated_at, last_run_at, total_runs,
                pending_frontend_tools, pending_backend_results,
                awaiting_frontend_tools, current_step, conversation_history,
                title, extras
            FROM agent_config
            WHERE agent_uuid = $1
        """
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, agent_uuid)
        
        if row is None:
            return None
        
        # Convert row to dict with proper type handling
        config = {
            "agent_uuid": str(row["agent_uuid"]),
            "system_prompt": row["system_prompt"],
            "model": row["model"],
            "max_steps": row["max_steps"],
            "thinking_tokens": row["thinking_tokens"],
            "max_tokens": row["max_tokens"],
            "container_id": row["container_id"],
            "messages": self._from_jsonb(row["messages"]),
            "tool_schemas": self._from_jsonb(row["tool_schemas"]),
            "tool_names": list(row["tool_names"]) if row["tool_names"] else [],
            "server_tools": self._from_jsonb(row["server_tools"]),
            "beta_headers": list(row["beta_headers"]) if row["beta_headers"] else [],
            "api_kwargs": self._from_jsonb(row["api_kwargs"]),
            "formatter": row["formatter"],
            "stream_meta_history_and_tool_results": row["stream_meta_history_and_tool_results"],
            "compactor_type": row["compactor_type"],
            "memory_store_type": row["memory_store_type"],
            "file_registry": self._from_jsonb(row["file_registry"]),
            "max_retries": row["max_retries"],
            "base_delay": row["base_delay"],
            "last_known_input_tokens": row["last_known_input_tokens"],
            "last_known_output_tokens": row["last_known_output_tokens"],
            "created_at": self._parse_datetime(row["created_at"]),
            "updated_at": self._parse_datetime(row["updated_at"]),
            "last_run_at": self._parse_datetime(row["last_run_at"]),
            "total_runs": row["total_runs"],
            # Frontend tool relay state
            "pending_frontend_tools": self._from_jsonb(row["pending_frontend_tools"]) or [],
            "pending_backend_results": self._from_jsonb(row["pending_backend_results"]) or [],
            "awaiting_frontend_tools": row["awaiting_frontend_tools"] or False,
            "current_step": row["current_step"] or 0,
            "conversation_history": self._from_jsonb(row["conversation_history"]) or [],
            # Title for conversation display
            "title": row["title"],
            "extras": self._from_jsonb(row["extras"]) or {},
        }
        
        logger.debug("Loaded agent config", agent_uuid=agent_uuid, backend="sql", config=config)
        return config
    
    async def update_agent_title(self, agent_uuid: str, title: str) -> bool:
        """Update the title for an agent session.
        
        Args:
            agent_uuid: Agent session UUID
            title: New title for the conversation
            
        Returns:
            True if updated successfully, False if agent not found
        """
        pool = await self._get_pool()
        
        query = """
            UPDATE agent_config
            SET title = $2, updated_at = NOW()
            WHERE agent_uuid = $1
            RETURNING agent_uuid
        """
        
        async with pool.acquire() as conn:
            result = await conn.fetchrow(query, agent_uuid, title)
        
        if result is None:
            return False
        
        logger.debug("Updated agent title", agent_uuid=agent_uuid, backend="sql", title=title)
        return True
    
    async def list_agent_sessions(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict], int]:
        """List all agent sessions with metadata."""
        pool = await self._get_pool()
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM agent_config"
        
        # Get sessions with pagination
        list_query = """
            SELECT 
                agent_uuid, title, created_at, updated_at, total_runs
            FROM agent_config
            ORDER BY updated_at DESC NULLS LAST
            LIMIT $1 OFFSET $2
        """
        
        async with pool.acquire() as conn:
            total_row = await conn.fetchrow(count_query)
            total = total_row[0] if total_row else 0
            
            rows = await conn.fetch(list_query, limit, offset)
        
        sessions = [
            {
                "agent_uuid": str(row["agent_uuid"]),
                "title": row["title"],
                "created_at": self._parse_datetime(row["created_at"]),
                "updated_at": self._parse_datetime(row["updated_at"]),
                "total_runs": row["total_runs"] or 0,
            }
            for row in rows
        ]
        
        logger.debug("Listed agent sessions", sessions=len(sessions), total=total, backend="sql")
        return sessions, total
    
    async def save_conversation_history(self, conversation: dict) -> None:
        """Save conversation history entry.
        
        Note: sequence_number is auto-generated by database trigger.
        
        Args:
            conversation: Conversation data dict
        """
        pool = await self._get_pool()
        
        query = """
            INSERT INTO conversation_history (
                conversation_id, agent_uuid, run_id, started_at, completed_at,
                user_message, final_response, messages, stop_reason, total_steps,
                usage, generated_files, extras, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
            )
        """
        
        async with pool.acquire() as conn:
            await conn.execute(
                query,
                conversation.get("conversation_id"),
                conversation.get("agent_uuid"),
                conversation.get("run_id"),
                self._to_datetime(conversation.get("started_at")),
                self._to_datetime(conversation.get("completed_at")),
                conversation.get("user_message"),
                conversation.get("final_response"),
                self._to_jsonb(conversation.get("messages")),
                conversation.get("stop_reason"),
                conversation.get("total_steps"),
                self._to_jsonb(conversation.get("usage")),
                self._to_jsonb(conversation.get("generated_files")),
                self._to_jsonb(conversation.get("extras", {})),
                self._to_datetime(conversation.get("created_at")),
            )
        logger.debug("Saved conversation history", agent_uuid=conversation.get("agent_uuid"), backend="sql", conversation=conversation)
        
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
        pool = await self._get_pool()
        
        query = """
            SELECT 
                conversation_id, agent_uuid, run_id, started_at, completed_at,
                user_message, final_response, messages, stop_reason, total_steps,
                usage, generated_files, extras, sequence_number, created_at
            FROM conversation_history
            WHERE agent_uuid = $1
            ORDER BY sequence_number DESC
            LIMIT $2 OFFSET $3
        """
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, agent_uuid, limit, offset)
        
        conversations = []
        for row in rows:
            conversations.append({
                "conversation_id": str(row["conversation_id"]),
                "agent_uuid": str(row["agent_uuid"]),
                "run_id": str(row["run_id"]),
                "started_at": self._parse_datetime(row["started_at"]),
                "completed_at": self._parse_datetime(row["completed_at"]),
                "user_message": row["user_message"],
                "final_response": row["final_response"],
                "messages": self._from_jsonb(row["messages"]),
                "stop_reason": row["stop_reason"],
                "total_steps": row["total_steps"],
                "usage": self._from_jsonb(row["usage"]),
                "generated_files": self._from_jsonb(row["generated_files"]),
                "extras": self._from_jsonb(row["extras"]) or {},
                "sequence_number": row["sequence_number"],
                "created_at": self._parse_datetime(row["created_at"]),
            })
        
        logger.debug("Loaded conversation history", agent_uuid=agent_uuid, backend="sql", conversations=len(conversations))
        return conversations
    
    async def load_conversation_history_cursor(
        self,
        agent_uuid: str,
        before: int | None = None,
        limit: int = 20
    ) -> tuple[list[dict], bool]:
        """Load conversations with cursor-based pagination (newest first).
        
        Args:
            agent_uuid: Agent session UUID
            before: Load conversations with sequence_number < before (None = latest)
            limit: Maximum conversations to return
            
        Returns:
            Tuple of (conversations newest->oldest, has_more)
        """
        pool = await self._get_pool()
        
        # Build query with optional cursor filter
        # Fetch limit + 1 to determine has_more
        if before is not None:
            query = """
                SELECT 
                    conversation_id, agent_uuid, run_id, started_at, completed_at,
                    user_message, final_response, messages, stop_reason, total_steps,
                    usage, generated_files, extras, sequence_number, created_at
                FROM conversation_history
                WHERE agent_uuid = $1 AND sequence_number < $2
                ORDER BY sequence_number DESC
                LIMIT $3
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, agent_uuid, before, limit + 1)
        else:
            query = """
                SELECT 
                    conversation_id, agent_uuid, run_id, started_at, completed_at,
                    user_message, final_response, messages, stop_reason, total_steps,
                    usage, generated_files, extras, sequence_number, created_at
                FROM conversation_history
                WHERE agent_uuid = $1
                ORDER BY sequence_number DESC
                LIMIT $2
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, agent_uuid, limit + 1)
        
        # Determine has_more and slice to limit
        has_more = len(rows) > limit
        rows_to_process = rows[:limit]
        
        conversations = []
        for row in rows_to_process:
            conversations.append({
                "conversation_id": str(row["conversation_id"]),
                "agent_uuid": str(row["agent_uuid"]),
                "run_id": str(row["run_id"]),
                "started_at": self._parse_datetime(row["started_at"]),
                "completed_at": self._parse_datetime(row["completed_at"]),
                "user_message": row["user_message"],
                "final_response": row["final_response"],
                "messages": self._from_jsonb(row["messages"]),
                "stop_reason": row["stop_reason"],
                "total_steps": row["total_steps"],
                "usage": self._from_jsonb(row["usage"]),
                "generated_files": self._from_jsonb(row["generated_files"]),
                "extras": self._from_jsonb(row["extras"]) or {},
                "sequence_number": row["sequence_number"],
                "created_at": self._parse_datetime(row["created_at"]),
            })
        
        logger.debug("Loaded conversation history cursor", agent_uuid=agent_uuid, backend="sql", conversations=len(conversations), has_more=has_more)
        return conversations, has_more
    
    async def save_agent_run_logs(
        self, 
        agent_uuid: str, 
        run_id: str, 
        logs: list[dict]
    ) -> None:
        """Save batched agent run logs.
        
        Args:
            agent_uuid: Agent session UUID
            run_id: Unique run identifier
            logs: List of log entries to save
        """
        if not logs:
            return
        
        pool = await self._get_pool()
        
        query = """
            INSERT INTO agent_runs (
                agent_uuid, run_id, timestamp, step_number, action_type,
                action_data, messages_snapshot, messages_count, estimated_tokens,
                duration_ms
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
            )
        """
        
        # Prepare batch data
        records = [
            (
                agent_uuid,
                run_id,
                self._to_datetime(log.get("timestamp")),
                log.get("step_number"),
                log.get("action_type"),
                self._to_jsonb(log.get("action_data")),
                self._to_jsonb(log.get("messages_snapshot")),
                log.get("messages_count"),
                log.get("estimated_tokens"),
                log.get("duration_ms"),
            )
            for log in logs
        ]
        
        async with pool.acquire() as conn:
            await conn.executemany(query, records)
        logger.debug("Saved agent run logs", agent_uuid=agent_uuid, run_id=run_id, backend="sql", logs=len(logs))
        
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
        pool = await self._get_pool()
        
        query = """
            SELECT 
                log_id, agent_uuid, run_id, timestamp, step_number, action_type,
                action_data, messages_snapshot, messages_count, estimated_tokens,
                duration_ms
            FROM agent_runs
            WHERE agent_uuid = $1 AND run_id = $2
            ORDER BY timestamp ASC
        """
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, agent_uuid, run_id)
        
        logs = []
        for row in rows:
            logs.append({
                "log_id": row["log_id"],
                "agent_uuid": str(row["agent_uuid"]),
                "run_id": str(row["run_id"]),
                "timestamp": self._parse_datetime(row["timestamp"]),
                "step_number": row["step_number"],
                "action_type": row["action_type"],
                "action_data": self._from_jsonb(row["action_data"]),
                "messages_snapshot": self._from_jsonb(row["messages_snapshot"]),
                "messages_count": row["messages_count"],
                "estimated_tokens": row["estimated_tokens"],
                "duration_ms": row["duration_ms"],
            })
        
        logger.debug("Loaded agent run logs", agent_uuid=agent_uuid, run_id=run_id, backend="sql", logs=len(logs))
        return logs

