"""Provider-agnostic public types.

The original implementation of this project was Anthropic-specific and used
Anthropic SDK types directly (e.g., ``BetaMessage`` / ``BetaUsage``). In order
to support multiple providers (OpenAI, Gemini, Grok, LiteLLM, ...), these
dataclasses are intentionally provider-agnostic.

Provider-specific agents may store the raw provider response object in
:attr:`AgentResult.final_message`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AgentResult:
    """Result returned by an agent run.

    Attributes:
        final_message:
            Provider-specific raw response object for the final model call.
            (Example: Anthropic ``BetaMessage``, OpenAI ``ChatCompletion`` or
            ``Response`` object.)

        final_answer:
            The extracted assistant text returned to the user (post tool calls).

        conversation_history:
            Canonical message history for the *current run* (not necessarily the
            full persisted multi-run history). Messages are stored in a
            provider-agnostic content-block format.

        stop_reason:
            Canonical stop reason string (e.g., "end_turn", "tool_use",
            "max_tokens"). When the agent pauses for frontend tools, this is set
            to "awaiting_frontend_tools".

        model:
            Model identifier used for the final model call.

        usage:
            Provider-agnostic usage dictionary. Common keys include:
            - input_tokens
            - output_tokens
            Providers may include additional keys.

        container_id:
            Optional provider-specific container/session identifier.

        total_steps:
            Number of agent steps executed for this run.

        agent_logs:
            Additional structured logs captured during the run.

        generated_files:
            Optional list of generated file metadata captured during the run.
    """

    final_message: Any
    final_answer: str
    conversation_history: list[dict[str, Any]]
    stop_reason: str
    model: str
    usage: dict[str, Any]
    container_id: Optional[str] = None
    total_steps: int = 1
    agent_logs: Optional[list[dict[str, Any]]] = None
    generated_files: Optional[list[dict[str, Any]]] = None
