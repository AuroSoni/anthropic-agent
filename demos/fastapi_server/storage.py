"""Shared storage adapters for the FastAPI demo.

Adapters are created once and shared across all agent configurations.
Connection lifecycle is managed in main.py startup/shutdown hooks.
"""
import os
from anthropic_agent.storage import (
    FilesystemAgentConfigAdapter,
    FilesystemConversationAdapter,
    FilesystemAgentRunAdapter,
    PostgresAgentConfigAdapter,
    PostgresConversationAdapter,
    PostgresAgentRunAdapter,
)

_base_path = os.getenv("STORAGE_BASE_PATH", "./data")

# Shared adapter instances - initialized once, connected in startup hook
config_adapter = FilesystemAgentConfigAdapter(base_path=_base_path)
conversation_adapter = FilesystemConversationAdapter(base_path=_base_path)
run_adapter = FilesystemAgentRunAdapter(base_path=_base_path)

_dsn = os.getenv("DATABASE_URL", "")

# Shared adapter instances - initialized once, connected in startup hook
# config_adapter = PostgresAgentConfigAdapter(connection_string=_dsn)
# conversation_adapter = PostgresConversationAdapter(connection_string=_dsn)
# run_adapter = PostgresAgentRunAdapter(connection_string=_dsn)