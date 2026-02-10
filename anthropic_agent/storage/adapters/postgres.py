"""PostgreSQL storage adapters - backward-compatible with existing typed schema.

These adapters use the existing table schemas:
- agent_config: 32 typed columns
- conversation_history: typed columns with auto-incrementing sequence
- agent_runs: typed columns for log entries

Maintains full compatibility with existing SQLBackend.
"""

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

import asyncpg

from ..base import (
    AgentConfig,
    AgentConfigAdapter,
    Conversation,
    ConversationAdapter,
    AgentRunAdapter,
)
from ..exceptions import StorageConnectionError, StorageOperationError
from ...logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_datetime(value: Any) -> str | None:
    """Parse datetime to ISO format string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _json_default(obj: Any) -> str:
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _to_jsonb(value: Any) -> str | None:
    """Serialize Python object to JSON string for JSONB columns."""
    if value is None:
        return None
    return json.dumps(value, default=_json_default)


def _from_jsonb(value: Any) -> Any:
    """Deserialize JSONB column value to Python object."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _to_datetime(value: Any) -> datetime | None:
    """Coerce ISO string timestamps to datetime objects."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except Exception:
            return None
    return None


def _config_to_row_values(config: AgentConfig) -> tuple:
    """Convert AgentConfig to tuple of values for SQL insert."""
    return (
        config.agent_uuid,
        config.system_prompt,
        config.description,
        config.model,
        config.max_steps,
        config.thinking_tokens,
        config.max_tokens,
        config.container_id,
        _to_jsonb(config.messages),
        _to_jsonb(config.tool_schemas),
        config.tool_names,
        _to_jsonb(config.server_tools),
        config.beta_headers,
        _to_jsonb(config.api_kwargs),
        config.formatter,
        config.stream_meta_history_and_tool_results,
        config.compactor_type,
        config.memory_store_type,
        _to_jsonb(config.file_registry),
        config.max_retries,
        config.base_delay,
        config.last_known_input_tokens,
        config.last_known_output_tokens,
        config.title,
        _to_datetime(config.created_at),
        _to_datetime(config.updated_at),
        _to_datetime(config.last_run_at),
        config.total_runs,
        _to_jsonb(config.pending_frontend_tools),
        _to_jsonb(config.pending_backend_results),
        config.awaiting_frontend_tools,
        config.current_step,
        _to_jsonb(config.conversation_history),
        config.parent_agent_uuid,
        _to_jsonb(config.subagent_schemas),
        _to_jsonb(config.extras),
    )


def _row_to_config(row: asyncpg.Record) -> AgentConfig:
    """Convert database row to AgentConfig."""
    return AgentConfig(
        agent_uuid=str(row["agent_uuid"]),
        system_prompt=row["system_prompt"],
        description=row.get("description"),
        model=row["model"],
        max_steps=row["max_steps"],
        thinking_tokens=row["thinking_tokens"],
        max_tokens=row["max_tokens"],
        container_id=row["container_id"],
        messages=_from_jsonb(row["messages"]) or [],
        tool_schemas=_from_jsonb(row["tool_schemas"]) or [],
        tool_names=list(row["tool_names"]) if row["tool_names"] else [],
        server_tools=_from_jsonb(row["server_tools"]) or [],
        beta_headers=list(row["beta_headers"]) if row["beta_headers"] else [],
        api_kwargs=_from_jsonb(row["api_kwargs"]) or {},
        formatter=row["formatter"],
        stream_meta_history_and_tool_results=row["stream_meta_history_and_tool_results"] or False,
        compactor_type=row["compactor_type"],
        memory_store_type=row["memory_store_type"],
        file_registry=_from_jsonb(row["file_registry"]) or {},
        max_retries=row["max_retries"],
        base_delay=row["base_delay"],
        last_known_input_tokens=row["last_known_input_tokens"] or 0,
        last_known_output_tokens=row["last_known_output_tokens"] or 0,
        pending_frontend_tools=_from_jsonb(row["pending_frontend_tools"]) or [],
        pending_backend_results=_from_jsonb(row["pending_backend_results"]) or [],
        awaiting_frontend_tools=row["awaiting_frontend_tools"] or False,
        current_step=row["current_step"] or 0,
        conversation_history=_from_jsonb(row["conversation_history"]) or [],
        parent_agent_uuid=row.get("parent_agent_uuid"),
        subagent_schemas=_from_jsonb(row.get("subagent_schemas")) or [],
        title=row["title"],
        extras=_from_jsonb(row["extras"]) or {},
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
        last_run_at=_parse_datetime(row["last_run_at"]),
        total_runs=row["total_runs"] or 0,
    )


def _row_to_conversation(row: asyncpg.Record) -> Conversation:
    """Convert database row to Conversation."""
    return Conversation(
        conversation_id=str(row["conversation_id"]),
        agent_uuid=str(row["agent_uuid"]),
        run_id=str(row["run_id"]),
        started_at=_parse_datetime(row["started_at"]),
        completed_at=_parse_datetime(row["completed_at"]),
        user_message=row["user_message"],
        final_response=row["final_response"],
        messages=_from_jsonb(row["messages"]) or [],
        stop_reason=row["stop_reason"],
        total_steps=row["total_steps"],
        usage=_from_jsonb(row["usage"]) or {},
        generated_files=_from_jsonb(row["generated_files"]) or [],
        cost=_from_jsonb(row["cost"]) or {},
        extras=_from_jsonb(row["extras"]) or {},
        sequence_number=row["sequence_number"],
        created_at=_parse_datetime(row["created_at"]),
    )


# =============================================================================
# Adapter Implementations
# =============================================================================

class PostgresAgentConfigAdapter(AgentConfigAdapter):
    """PostgreSQL adapter for agent configuration.
    
    Uses the existing agent_config table with 33 typed columns.
    Maintains full backward compatibility with existing SQLBackend.
    """
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        timezone: str = "UTC",
    ):
        """Initialize PostgreSQL agent config adapter.
        
        Args:
            connection_string: PostgreSQL connection string
            pool_size: Maximum connection pool size
            timezone: Server timezone setting
        """
        self._dsn = connection_string
        self._pool_size = pool_size
        self._timezone = timezone
        self._pool: asyncpg.Pool | None = None
    
    async def connect(self) -> None:
        """Create the connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=1,
                max_size=self._pool_size,
                server_settings={"timezone": self._timezone}
            )
            logger.info(
                "Created PostgreSQL connection pool",
                max_size=self._pool_size
            )
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Closed PostgreSQL connection pool")
    
    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool (lazy initialization)."""
        if self._pool is None:
            await self.connect()
        return self._pool  # type: ignore
    
    async def save(self, config: AgentConfig) -> None:
        """Save/update agent configuration using UPSERT."""
        pool = await self._get_pool()
        
        query = """
            INSERT INTO agent_config (
                agent_uuid, system_prompt, description, model, max_steps,
                thinking_tokens, max_tokens, container_id, messages,
                tool_schemas, tool_names, server_tools, beta_headers,
                api_kwargs, formatter,
                stream_meta_history_and_tool_results, compactor_type,
                memory_store_type, file_registry, max_retries, base_delay,
                last_known_input_tokens, last_known_output_tokens,
                title, created_at, updated_at, last_run_at, total_runs,
                pending_frontend_tools, pending_backend_results,
                awaiting_frontend_tools, current_step, conversation_history,
                parent_agent_uuid, subagent_schemas,
                extras
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
                $31, $32, $33, $34, $35, $36
            )
            ON CONFLICT (agent_uuid) DO UPDATE SET
                system_prompt = EXCLUDED.system_prompt,
                description = EXCLUDED.description,
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
                parent_agent_uuid = EXCLUDED.parent_agent_uuid,
                subagent_schemas = EXCLUDED.subagent_schemas,
                extras = EXCLUDED.extras
        """
        
        values = _config_to_row_values(config)
        
        async with pool.acquire() as conn:
            await conn.execute(query, *values)
        
        logger.debug(
            "Saved agent config",
            agent_uuid=config.agent_uuid,
            backend="postgres"
        )
    
    async def load(self, agent_uuid: str) -> AgentConfig | None:
        """Load agent configuration from database."""
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
        
        config = _row_to_config(row)
        logger.debug(
            "Loaded agent config",
            agent_uuid=agent_uuid,
            backend="postgres"
        )
        return config
    
    async def delete(self, agent_uuid: str) -> bool:
        """Delete agent configuration."""
        pool = await self._get_pool()
        
        query = "DELETE FROM agent_config WHERE agent_uuid = $1 RETURNING agent_uuid"
        
        async with pool.acquire() as conn:
            result = await conn.fetchrow(query, agent_uuid)
        
        if result is None:
            return False
        
        logger.debug(
            "Deleted agent config",
            agent_uuid=agent_uuid,
            backend="postgres"
        )
        return True
    
    async def update_title(self, agent_uuid: str, title: str) -> bool:
        """Update the title for an agent session."""
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
        
        logger.debug(
            "Updated agent title",
            agent_uuid=agent_uuid,
            title=title,
            backend="postgres"
        )
        return True
    
    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict], int]:
        """List all agent sessions with metadata."""
        pool = await self._get_pool()
        
        count_query = "SELECT COUNT(*) FROM agent_config"
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
                "created_at": _parse_datetime(row["created_at"]),
                "updated_at": _parse_datetime(row["updated_at"]),
                "total_runs": row["total_runs"] or 0,
            }
            for row in rows
        ]
        
        logger.debug(
            "Listed agent sessions",
            count=len(sessions),
            total=total,
            backend="postgres"
        )
        return sessions, total


class PostgresConversationAdapter(ConversationAdapter):
    """PostgreSQL adapter for conversation history.
    
    Uses the existing conversation_history table.
    sequence_number is auto-generated by database trigger.
    """
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        timezone: str = "UTC",
    ):
        """Initialize PostgreSQL conversation adapter.
        
        Args:
            connection_string: PostgreSQL connection string
            pool_size: Maximum connection pool size
            timezone: Server timezone setting
        """
        self._dsn = connection_string
        self._pool_size = pool_size
        self._timezone = timezone
        self._pool: asyncpg.Pool | None = None
    
    async def connect(self) -> None:
        """Create the connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=1,
                max_size=self._pool_size,
                server_settings={"timezone": self._timezone}
            )
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
    
    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool."""
        if self._pool is None:
            await self.connect()
        return self._pool  # type: ignore
    
    async def save(self, conversation: Conversation) -> None:
        """Save conversation history entry."""
        pool = await self._get_pool()
        
        query = """
            INSERT INTO conversation_history (
                conversation_id, agent_uuid, run_id, started_at, completed_at,
                user_message, final_response, messages, stop_reason, total_steps,
                usage, generated_files, cost, extras, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            )
        """

        async with pool.acquire() as conn:
            await conn.execute(
                query,
                conversation.conversation_id,
                conversation.agent_uuid,
                conversation.run_id,
                _to_datetime(conversation.started_at),
                _to_datetime(conversation.completed_at),
                conversation.user_message,
                conversation.final_response,
                _to_jsonb(conversation.messages),
                conversation.stop_reason,
                conversation.total_steps,
                _to_jsonb(conversation.usage),
                _to_jsonb(conversation.generated_files),
                _to_jsonb(conversation.cost),
                _to_jsonb(conversation.extras),
                _to_datetime(conversation.created_at),
            )
        
        logger.debug(
            "Saved conversation",
            agent_uuid=conversation.agent_uuid,
            conversation_id=conversation.conversation_id,
            backend="postgres"
        )
    
    async def load_history(
        self,
        agent_uuid: str,
        limit: int = 20,
        offset: int = 0
    ) -> list[Conversation]:
        """Load paginated conversation history (newest first)."""
        pool = await self._get_pool()
        
        query = """
            SELECT 
                conversation_id, agent_uuid, run_id, started_at, completed_at,
                user_message, final_response, messages, stop_reason, total_steps,
                usage, generated_files, cost, extras, sequence_number, created_at
            FROM conversation_history
            WHERE agent_uuid = $1
            ORDER BY sequence_number DESC
            LIMIT $2 OFFSET $3
        """
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, agent_uuid, limit, offset)
        
        conversations = [_row_to_conversation(row) for row in rows]
        
        logger.debug(
            "Loaded conversation history",
            agent_uuid=agent_uuid,
            count=len(conversations),
            backend="postgres"
        )
        return conversations
    
    async def load_cursor(
        self,
        agent_uuid: str,
        before: int | None = None,
        limit: int = 20
    ) -> tuple[list[Conversation], bool]:
        """Load conversations with cursor-based pagination."""
        pool = await self._get_pool()
        
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
        
        has_more = len(rows) > limit
        rows_to_process = rows[:limit]
        
        conversations = [_row_to_conversation(row) for row in rows_to_process]
        
        logger.debug(
            "Loaded conversation cursor",
            agent_uuid=agent_uuid,
            count=len(conversations),
            has_more=has_more,
            backend="postgres"
        )
        return conversations, has_more


class PostgresAgentRunAdapter(AgentRunAdapter):
    """PostgreSQL adapter for agent run logs.
    
    Uses the existing agent_runs table with typed columns.
    """
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        timezone: str = "UTC",
    ):
        """Initialize PostgreSQL agent run adapter.
        
        Args:
            connection_string: PostgreSQL connection string
            pool_size: Maximum connection pool size
            timezone: Server timezone setting
        """
        self._dsn = connection_string
        self._pool_size = pool_size
        self._timezone = timezone
        self._pool: asyncpg.Pool | None = None
    
    async def connect(self) -> None:
        """Create the connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=1,
                max_size=self._pool_size,
                server_settings={"timezone": self._timezone}
            )
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
    
    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool."""
        if self._pool is None:
            await self.connect()
        return self._pool  # type: ignore
    
    async def save_logs(
        self,
        agent_uuid: str,
        run_id: str,
        logs: list[dict]
    ) -> None:
        """Save batched agent run logs."""
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
        
        records = [
            (
                agent_uuid,
                run_id,
                _to_datetime(log.get("timestamp")),
                log.get("step_number"),
                log.get("action_type"),
                _to_jsonb(log.get("action_data")),
                _to_jsonb(log.get("messages_snapshot")),
                log.get("messages_count"),
                log.get("estimated_tokens"),
                log.get("duration_ms"),
            )
            for log in logs
        ]
        
        async with pool.acquire() as conn:
            await conn.executemany(query, records)
        
        logger.debug(
            "Saved agent run logs",
            agent_uuid=agent_uuid,
            run_id=run_id,
            count=len(logs),
            backend="postgres"
        )
    
    async def load_logs(
        self,
        agent_uuid: str,
        run_id: str
    ) -> list[dict]:
        """Load all logs for a specific run."""
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
        
        logs = [
            {
                "log_id": row["log_id"],
                "agent_uuid": str(row["agent_uuid"]),
                "run_id": str(row["run_id"]),
                "timestamp": _parse_datetime(row["timestamp"]),
                "step_number": row["step_number"],
                "action_type": row["action_type"],
                "action_data": _from_jsonb(row["action_data"]),
                "messages_snapshot": _from_jsonb(row["messages_snapshot"]),
                "messages_count": row["messages_count"],
                "estimated_tokens": row["estimated_tokens"],
                "duration_ms": row["duration_ms"],
            }
            for row in rows
        ]
        
        logger.debug(
            "Loaded agent run logs",
            agent_uuid=agent_uuid,
            run_id=run_id,
            count=len(logs),
            backend="postgres"
        )
        return logs
