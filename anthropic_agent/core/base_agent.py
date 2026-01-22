"""Provider-agnostic base agent.

This module contains the core *workflow* for running a tool-using, multi-step agent:

- Maintains a canonical, provider-agnostic message history (role + list of content blocks).
- Streams model output via a provider-specific implementation hook.
- Detects tool calls, executes backend tools, and optionally pauses for frontend tools.
- Supports compaction and memory injection/update.
- Persists agent state and run artifacts via the configured DB backend.

Provider-specific agents (e.g., Anthropic, OpenAI, Gemini, Grok, LiteLLM) should subclass
:class:`BaseAgent` and implement :meth:`_stream_llm_response` (and optionally override
other hooks such as file processing).

Canonical message format
------------------------
The base agent stores messages internally in a canonical format that is intentionally
close to Anthropic's "content blocks" representation because it can express:

- normal text blocks: {"type": "text", "text": "..."}
- tool calls:         {"type": "tool_use", "id": "...", "name": "tool", "input": {...}}
- tool results:       {"type": "tool_result", "tool_use_id": "...", "content": "...", "is_error": bool?}

Other providers can be adapted to/from this structure in their respective agent
implementations:

- OpenAI / Grok (OpenAI-compatible): map `tool_calls[]` <-> `tool_use` blocks and
  `role=tool` messages <-> `tool_result` blocks.
- Gemini: map `function_call` parts <-> `tool_use` blocks and function responses
  <-> `tool_result` blocks (Gemini may not provide call IDs; provider adapters can
  synthesize stable IDs).
- LiteLLM: typically returns OpenAI-shaped responses; treat it similarly to OpenAI.

The base agent deliberately does *not* import or depend on any provider SDK.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import uuid
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from .types import AgentResult
from ..database import DBBackendType, DatabaseBackend, get_db_backend
from ..file_backends import FileBackendType, FileStorageBackend, get_file_backend
from ..memory import MemoryStore, MemoryStoreType, get_memory_store
from ..streaming import FormatterType
from ..tools.base import ToolRegistry, ToolResultContent
from .compaction import Compactor, CompactorType, get_compactor, get_default_compactor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults (provider-agnostic)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that should help the user with their questions."

# Subclasses should override DEFAULT_MODEL with a provider-appropriate default.
DEFAULT_MODEL = ""

DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_TOKENS = 2048
DEFAULT_STREAM_META = False
DEFAULT_FORMATTER: FormatterType = "xml"
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0


# ---------------------------------------------------------------------------
# Generic retry utility (kept provider-agnostic)
# ---------------------------------------------------------------------------

T = Any


def retry_with_backoff(max_retries: int, base_delay: float) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Retry async function with exponential backoff.

    This is intentionally lightweight and provider-agnostic.

    Args:
        max_retries: Maximum number of attempts (including the first).
        base_delay: Base delay in seconds for exponential backoff.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    last_exc = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Retryable error in {func.__name__} "
                            f"(attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}",
                            exc_info=True,
                        )
                        raise last_exc

            # Unreachable, but keeps type-checkers happy
            raise RuntimeError(f"retry_with_backoff: exhausted retries for {func.__name__}")  # pragma: no cover

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Provider response wrapper
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Provider response normalized for the BaseAgent workflow.

    Provider-specific agents should build and return this from `_stream_llm_response`.

    Attributes:
        raw: The raw provider SDK response object (kept for debugging / external access).
        assistant_message: Canonical assistant message dict with keys {role, content}.
        stop_reason: Canonical stop reason string. Providers should map their finish
            reasons to one of: "tool_use", "end_turn", "max_tokens", "content_filter", etc.
        model: Model identifier returned by the provider (or the configured model).
        usage: Dict-like usage info. Should include "input_tokens" and "output_tokens" when available.
        container_id: Optional provider session/container identifier (e.g. Anthropic container).
    """

    raw: Any
    assistant_message: dict[str, Any]
    stop_reason: str
    model: str | None
    usage: dict[str, Any] | None = None
    container_id: str | None = None


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Provider-agnostic agent core.

    Subclasses must implement `_stream_llm_response` to:
    1) call the provider SDK
    2) stream chunks into `queue` (if provided)
    3) return a fully-assembled :class:`LLMResponse`

    The rest of the agent lifecycle (tool loop, compaction, persistence, etc.) is handled here.
    """

    # Subclasses can override these defaults
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
    DEFAULT_MODEL = DEFAULT_MODEL
    DEFAULT_MAX_STEPS = DEFAULT_MAX_STEPS
    DEFAULT_MAX_TOKENS = DEFAULT_MAX_TOKENS
    DEFAULT_STREAM_META = DEFAULT_STREAM_META
    DEFAULT_FORMATTER = DEFAULT_FORMATTER
    DEFAULT_MAX_RETRIES = DEFAULT_MAX_RETRIES
    DEFAULT_BASE_DELAY = DEFAULT_BASE_DELAY

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_steps: Optional[int] = None,
        max_tokens: Optional[int] = None,   # To be removed. Can be provider specific.
        stream_meta_history_and_tool_results: Optional[bool] = None,
        tools: list[Callable[..., Any]] | None = None,
        frontend_tools: list[Callable[..., Any]] | None = None,
        server_tools: list[dict[str, Any]] | None = None,   # To be removed. It is provider specific.
        messages: list[dict] | None = None, # Generic message list.
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        formatter: FormatterType | None = None,
        compactor: CompactorType | Compactor | None = None,
        memory_store: MemoryStoreType | MemoryStore | None = None,
        final_answer_check: Optional[Callable[[str], tuple[bool, str]]] = None,
        agent_uuid: str | None = None,
        db_backend: DBBackendType | DatabaseBackend = "filesystem",
        file_backend: FileBackendType | FileStorageBackend | None = None,
        title_generator: Optional[Callable[[str], Awaitable[str]]] = None,
        **api_kwargs: Any,
    ):
        # ------------------------------------------------------------------
        # Items to take from the constructor or set a default value.
        # ------------------------------------------------------------------

        # Core config
        self.system_prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT
        self.model = model if model is not None else self.DEFAULT_MODEL
        self.max_steps = max_steps if max_steps is not None else self.DEFAULT_MAX_STEPS
        self.max_tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS
        self.stream_meta_history_and_tool_results = (
            stream_meta_history_and_tool_results
            if stream_meta_history_and_tool_results is not None
            else self.DEFAULT_STREAM_META
        )
        self.formatter: FormatterType = formatter if formatter is not None else self.DEFAULT_FORMATTER
        self.max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        self.base_delay = base_delay if base_delay is not None else self.DEFAULT_BASE_DELAY

        # Provider/adapter knobs (kept generic; subclasses may interpret)
        self.server_tools = server_tools or []
        self.api_kwargs = api_kwargs or {}

        # Session identity
        self.agent_uuid = agent_uuid or str(uuid.uuid4())

        # DB backend
        if isinstance(db_backend, str):
            self.db_backend: DatabaseBackend = get_db_backend(db_backend)
        else:
            self.db_backend = db_backend

        # File backend (optional)
        if file_backend is None:
            self.file_backend: FileStorageBackend | None = None
        elif isinstance(file_backend, str):
            self.file_backend = get_file_backend(file_backend)
        else:
            self.file_backend = file_backend

        if self.file_backend is not None:
            self._on_file_backend_configured()

        # ------------------------------------------------------------------
        # - Constructing the consolidated tool schema.
        # - Initializing the tool registry and injecting agent_uuid into tools that accept it.
        # - Initializing agent context modulators (memory store, compactor).
        # - Initializing the final answer check callback.
        # - Initializing the title generator callback.
        # ------------------------------------------------------------------

        # Tools
        self.tool_registry: ToolRegistry | None = None
        self.tool_schemas: list[dict[str, Any]] = []

        if tools:
            self.tool_registry = ToolRegistry()
            self.tool_registry.register_tools(tools)
            self.tool_schemas = self.tool_registry.get_schemas()

        # Frontend tools are schema-only here.
        self.frontend_tool_schemas: list[dict[str, Any]] = []
        self.frontend_tool_names: list[str] = []
        if frontend_tools:
            frontend_registry = ToolRegistry()
            frontend_registry.register_tools(frontend_tools)
            self.frontend_tool_schemas = frontend_registry.get_schemas()
            self.frontend_tool_names = [t.get("name") for t in self.frontend_tool_schemas if t.get("name")]

        # Inject agent_uuid into tools that accept it
        self._inject_agent_uuid_to_tools()

        # Compaction
        if isinstance(compactor, str):
            if compactor.lower() == "none":
                self.compactor = None
            else:
                self.compactor = get_compactor(compactor)
        elif compactor is None:
            self.compactor = get_default_compactor()
        else:
            self.compactor = compactor

        # Memory store
        if isinstance(memory_store, str):
            if memory_store.lower() == "none":
                self.memory_store = None
            else:
                self.memory_store = get_memory_store(memory_store)
        else:
            self.memory_store = memory_store

        # Validation hook
        self.final_answer_check = final_answer_check

        # Optional title generator callback (provider-agnostic)
        self._title_generator = title_generator

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------

        # Runtime state
        self._initialized = False

        self.messages: list[dict[str, Any]] = messages or []
        self.file_registry: dict[str, dict[str, Any]] = {}

        self._last_known_input_tokens: int = 0
        self._last_known_output_tokens: int = 0
        self._token_usage_history: list[dict[str, Any]] = []

        # Frontend tool pause/resume state
        self._pending_frontend_tools: list[dict[str, Any]] = []
        self._pending_backend_results: list[dict[str, Any]] = []
        self._awaiting_frontend_tools: bool = False
        self._current_step: int = 0
        self._loaded_conversation_history: list[dict[str, Any]] = []

        # Per-run state (set in run())
        self._run_id: str = ""
        self._run_start_time: datetime | None = None
        self._run_logs_buffer: list[dict[str, Any]] = []

        # Background persistence tasks
        self._background_tasks: set[asyncio.Task[Any]] = set()

    # ------------------------------------------------------------------
    # Provider hooks
    # ------------------------------------------------------------------

    @abstractmethod
    async def _stream_llm_response(
        self,
        *,
        queue: Optional[asyncio.Queue],
        formatter: FormatterType,
        tools_enabled: bool = True,
        system_prompt_override: Optional[str] = None,
    ) -> LLMResponse:
        """Call provider, stream chunks to queue, and return the assembled response."""

    def _on_file_backend_configured(self) -> None:
        """Hook for provider-specific adjustments when file backend is enabled."""

    async def _finalize_file_processing(self, queue: Optional[asyncio.Queue] = None) -> None:
        """Optional hook for provider-specific file processing at end of run."""
        return

    def _get_provider_type(self) -> str:
        """Return provider identifier for DB storage. Override in subclasses."""
        return "base"

    def _get_provider_specific_config(self) -> dict[str, Any]:
        """Return provider-specific fields to persist. Override in subclasses."""
        return {}

    def _restore_provider_specific_state(
        self,
        provider_config: dict[str, Any],
        full_config: dict[str, Any],
    ) -> None:
        """Restore provider-specific state from config. Override in subclasses.

        Args:
            provider_config: The nested provider_config dict (v2 format)
            full_config: The full config dict (for v1 backward compatibility fallback)
        """
        pass

    def _get_provider_specific_usage_fields(self, usage: dict[str, Any]) -> dict[str, Any]:
        """Extract provider-specific token usage fields. Override in subclasses."""
        return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Load persisted state from the DB backend (if any)."""
        if self._initialized:
            return

        try:
            config = await self.db_backend.load_agent_config(self.agent_uuid)
            if config:
                self._restore_state_from_config(config)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load agent state for {self.agent_uuid}: {e}")

        self._initialized = True

    def execute_tool_call(self, tool_name: str, tool_input: dict) -> tuple[ToolResultContent, list[dict[str, Any]]]:
        """Execute a registered backend tool through the ToolRegistry."""
        if not self.tool_registry:
            return "No tools have been registered for this agent.", []

        return self.tool_registry.execute(
            tool_name,
            tool_input,
            file_backend=self.file_backend,
            agent_uuid=self.agent_uuid,
        )

    async def run(
        self,
        prompt: str | list[dict[str, Any]] | dict[str, Any],
        queue: Optional[asyncio.Queue] = None,
        formatter: Optional[FormatterType] = None,
    ) -> AgentResult:
        """Run the agent on a new user prompt."""
        
        # ------------------------------------------------------------------
        # Load agent state from DB backend if the uuid exists.
        # ------------------------------------------------------------------

        if not self._initialized:
            await self.initialize()

        # ------------------------------------------------------------------
        # Initialize run state variables.
        # ------------------------------------------------------------------
        effective_formatter: FormatterType = formatter if formatter is not None else self.formatter

        # Fresh per-run state
        self._run_id = str(uuid.uuid4())
        self._run_start_time = datetime.now()
        self._run_logs_buffer = []
        self.conversation_history: list[dict[str, Any]] = []
        self.agent_logs: list[dict[str, Any]] = []

        # Reset per-run token history
        self._token_usage_history = []

        # Clear any leftover pause state
        self._pending_frontend_tools = []
        self._pending_backend_results = []
        self._awaiting_frontend_tools = False
        self._current_step = 0
        self._loaded_conversation_history = []

        # Build canonical user message
        user_message = self._build_user_message(prompt)

        # Add to context and run history
        self.messages.append(user_message)
        self.conversation_history.append(user_message)

        # Log: run started
        self._log_action(
            "run_started",
            {
                "user_message": prompt if isinstance(prompt, str) else str(prompt)[:200],
                "queue_present": queue is not None,
                "formatter": effective_formatter,
            },
            step_number=0,
        )

        # Emit meta_init
        if queue is not None:
            meta_init: dict[str, Any] = {
                "format": effective_formatter,
                "user_query": prompt if isinstance(prompt, str) else json.dumps(prompt, default=str),
                "agent_uuid": self.agent_uuid,
                "model": self.model,
            }
            if self.stream_meta_history_and_tool_results:
                meta_init["message_history"] = self.conversation_history
            escaped_json = html.escape(json.dumps(meta_init, default=str), quote=True)
            await queue.put(f'<meta_init data="{escaped_json}"></meta_init>')

        # Retrieve and inject semantic memories (not added to conversation_history)
        if self.memory_store:
            self.messages = self.memory_store.retrieve(
                tools=self.tool_schemas,
                user_message=user_message,
                messages=self.messages,
                model=self.model,
            )

        # Initialize heuristic token estimate for current context.
        combined_tools: list[dict[str, Any]] = []
        if self.tool_schemas:
            combined_tools.extend(self.tool_schemas)
        if self.server_tools:
            combined_tools.extend(self.server_tools)

        self._last_known_input_tokens = await self._estimate_tokens(
            messages=self.messages,
            system=self.system_prompt,
            tools=combined_tools or None,
        )

        return await self._run_loop(start_step=0, queue=queue, formatter=effective_formatter)

    async def continue_with_tool_results(
        self,
        frontend_results: list[dict[str, Any]],
        queue: Optional[asyncio.Queue] = None,
        formatter: Optional[FormatterType] = None,
    ) -> AgentResult:
        """Resume agent execution after frontend tools have been executed."""
        # Ensure agent is initialized (loads state from DB if needed)
        if not self._initialized:
            await self.initialize()

        if not self._awaiting_frontend_tools:
            raise ValueError("Agent is not awaiting frontend tools")

        if not self._pending_frontend_tools:
            raise ValueError("No pending frontend tools found - state may not have been loaded from DB")

        effective_formatter: FormatterType = formatter if formatter is not None else self.formatter

        # Initialize run state if resuming from DB (normally set in run())
        if not getattr(self, "conversation_history", None):
            self.conversation_history = getattr(self, "_loaded_conversation_history", []).copy()
        if not getattr(self, "agent_logs", None):
            self.agent_logs = []
        if not getattr(self, "_run_logs_buffer", None):
            self._run_logs_buffer = []
        if not getattr(self, "_run_id", None):
            self._run_id = str(uuid.uuid4())
        if not getattr(self, "_run_start_time", None):
            self._run_start_time = datetime.now()

        pending_ids = {t["tool_use_id"] for t in self._pending_frontend_tools}
        result_ids = {r["tool_use_id"] for r in frontend_results}
        if pending_ids != result_ids:
            raise ValueError(f"Tool result mismatch. Expected tool_use_ids: {pending_ids}, got: {result_ids}")

        # Combine backend + frontend results (backend first)
        all_results: list[dict[str, Any]] = list(self._pending_backend_results)
        for r in frontend_results:
            all_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": r["tool_use_id"],
                    "content": r.get("content", ""),
                    **({"is_error": True} if r.get("is_error") else {}),
                }
            )

        # Stream frontend tool results only when meta streaming is enabled
        if queue is not None and self.stream_meta_history_and_tool_results:
            for r in frontend_results:
                tool_use_id = html.escape(str(r["tool_use_id"]), quote=True)
                tool_name = next((t["name"] for t in self._pending_frontend_tools if t["tool_use_id"] == r["tool_use_id"]), "unknown")
                tool_name_escaped = html.escape(str(tool_name), quote=True)
                content_val = r.get("content", "")
                content_str = content_val if isinstance(content_val, str) else json.dumps(content_val, default=str)
                await queue.put(
                    f'<content-block-tool_result id="{tool_use_id}" name="{tool_name_escaped}"><![CDATA[{content_str}]]></content-block-tool_result>'
                )

        # Append combined tool results
        tool_result_message = {"role": "user", "content": all_results}
        self.messages.append(tool_result_message)
        self.conversation_history.append(tool_result_message)

        self._log_action(
            "frontend_tools_completed",
            {
                "frontend_results_count": len(frontend_results),
                "backend_results_count": len(self._pending_backend_results),
                "total_results": len(all_results),
            },
            step_number=self._current_step,
        )

        # Clear pause state
        self._pending_frontend_tools = []
        self._pending_backend_results = []
        self._awaiting_frontend_tools = False

        # Update token estimate (heuristic delta)
        delta_tokens = await self._estimate_tokens(messages=[tool_result_message])
        self._last_known_input_tokens += delta_tokens

        return await self._run_loop(start_step=self._current_step, queue=queue, formatter=effective_formatter)

    # ------------------------------------------------------------------
    # Core loop + helpers
    # ------------------------------------------------------------------

    async def _run_loop(
        self,
        *,
        start_step: int,
        queue: Optional[asyncio.Queue],
        formatter: FormatterType,
    ) -> AgentResult:
        step = start_step

        while step < self.max_steps:
            step += 1
            self._current_step = step

            # Compaction hook (always called; compactor decides whether to compact)
            if self.compactor:
                self._apply_compaction(step_number=step)

            # Provider call
            response = await self._stream_llm_response(
                queue=queue,
                formatter=formatter,
                tools_enabled=True,
                system_prompt_override=None,
            )

            # Track token usage
            self._track_usage(step=step, response=response)

            # Log: API response received
            self._log_action(
                "api_response_received",
                {
                    "stop_reason": response.stop_reason,
                    "input_tokens": (response.usage or {}).get("input_tokens"),
                    "output_tokens": (response.usage or {}).get("output_tokens"),
                },
                step_number=step,
            )

            # Add assistant response to history
            self.messages.append(response.assistant_message)
            self.conversation_history.append(response.assistant_message)
            logger.debug("Assistant message: %s", response.assistant_message)

            if response.container_id:
                self.container_id = response.container_id

            # Tool call handling
            tool_calls = self._extract_tool_calls(response.assistant_message)
            if tool_calls:
                backend_tool_calls = [t for t in tool_calls if t["name"] not in self.frontend_tool_names]
                frontend_tool_calls = [t for t in tool_calls if t["name"] in self.frontend_tool_names]

                tool_results = await self._execute_backend_tools(
                    backend_tool_calls=backend_tool_calls,
                    queue=queue,
                    step=step,
                )

                if frontend_tool_calls:
                    # Pause for frontend execution
                    self._pending_backend_results = tool_results
                    self._pending_frontend_tools = [
                        {"tool_use_id": t["tool_use_id"], "name": t["name"], "input": t["input"]}
                        for t in frontend_tool_calls
                    ]
                    self._awaiting_frontend_tools = True

                    self._log_action(
                        "awaiting_frontend_tools",
                        {
                            "frontend_tools": [t["name"] for t in frontend_tool_calls],
                            "backend_results_count": len(tool_results),
                        },
                        step_number=step,
                    )

                    if queue is not None:
                        tools_json = html.escape(json.dumps(self._pending_frontend_tools, default=str), quote=True)
                        await queue.put(f'<awaiting_frontend_tools data="{tools_json}"></awaiting_frontend_tools>')

                    # Persist state before returning so the agent can be re-hydrated
                    await self._save_agent_config()

                    return AgentResult(
                        final_message=response.raw,
                        final_answer="",
                        conversation_history=self.conversation_history.copy(),
                        stop_reason="awaiting_frontend_tools",
                        model=response.model or self.model,
                        usage=response.usage or {},
                        container_id=self.container_id,
                        total_steps=step,
                        agent_logs=self.agent_logs.copy(),
                        generated_files=None,
                    )

                # No frontend tools: append tool results and continue
                tool_result_message = {"role": "user", "content": tool_results}
                self.messages.append(tool_result_message)
                self.conversation_history.append(tool_result_message)

                delta_tokens = await self._estimate_tokens(messages=[tool_result_message])
                self._last_known_input_tokens += delta_tokens
                continue

            # Final answer validation (only if this step produced a final answer)
            extracted_final_answer = self._extract_final_answer(response.assistant_message)
            if self.final_answer_check:
                success, error_message = self.final_answer_check(extracted_final_answer)
                if not success:
                    self.agent_logs.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "action": "final_answer_validation_failed",
                            "details": {"error": error_message, "step": step},
                        }
                    )
                    logger.warning(f"Final answer validation failed at step {step}: {error_message}")

                    error_user_message = {
                        "role": "user",
                        "content": [{"type": "text", "text": error_message}],
                    }
                    self.messages.append(error_user_message)
                    self.conversation_history.append(error_user_message)
                    continue

            # Update memory store
            if self.memory_store:
                memory_metadata = self.memory_store.update(
                    messages=self.messages,
                    conversation_history=self.conversation_history,
                    tools=self.tool_schemas,
                    model=self.model,
                )
                self.agent_logs.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "action": "memory_update",
                        "details": memory_metadata,
                    }
                )
                logger.info(f"Memory updated: {memory_metadata}")

            # Build final result
            result = AgentResult(
                final_message=response.raw,
                final_answer=extracted_final_answer,
                conversation_history=self.conversation_history.copy(),
                stop_reason=response.stop_reason,
                model=response.model or self.model,
                usage=response.usage or {},
                container_id=self.container_id,
                total_steps=step,
                agent_logs=self.agent_logs.copy(),
                generated_files=None,
            )

            self._log_action(
                "run_completed",
                {
                    "stop_reason": response.stop_reason,
                    "total_steps": step,
                    "total_input_tokens": (response.usage or {}).get("input_tokens"),
                    "total_output_tokens": (response.usage or {}).get("output_tokens"),
                },
                step_number=step,
            )

            # Finalize files
            await self._finalize_file_processing(queue)
            all_files_metadata: list[dict[str, Any]] = list(self.file_registry.values())
            result.generated_files = all_files_metadata

            # Persist run data asynchronously
            self._save_run_data_async(result, all_files_metadata)

            # Stream meta_final
            await self._emit_meta_final(queue, result)

            return result

        # Max steps reached
        self._log_action(
            "max_steps_reached",
            {"steps": self.max_steps, "max_steps": self.max_steps},
            step_number=self.max_steps,
        )
        logger.warning(f"Max steps ({self.max_steps}) reached, generating final summary")
        return await self._generate_final_summary(queue=queue, formatter=formatter)

    async def _generate_final_summary(
        self,
        *,
        queue: Optional[asyncio.Queue],
        formatter: FormatterType,
    ) -> AgentResult:
        """Generate a final response when max_steps is exhausted."""
        logger.info("Generating final summary due to max_steps reached")

        if self.compactor:
            self._apply_compaction(step_number=self.max_steps)

        summary_system_prompt = (
            f"{self.system_prompt}\n\n"
            "IMPORTANT: You have reached the maximum number of steps. "
            "Please provide a final summary or response based on the work completed so far."
        )

        self.agent_logs.append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": "max_steps_summary",
                "details": {"reason": "max_steps_reached", "max_steps": self.max_steps, "tools_disabled": True},
            }
        )

        response = await self._stream_llm_response(
            queue=queue,
            formatter=formatter,
            tools_enabled=False,
            system_prompt_override=summary_system_prompt,
        )

        # Track token usage
        self._track_usage(step=self.max_steps, response=response)

        # Append assistant summary
        self.messages.append(response.assistant_message)
        self.conversation_history.append(response.assistant_message)

        if response.container_id:
            self.container_id = response.container_id

        extracted_final_answer = self._extract_final_answer(response.assistant_message)

        # Update memory store
        if self.memory_store:
            memory_metadata = self.memory_store.update(
                messages=self.messages,
                conversation_history=self.conversation_history,
                tools=self.tool_schemas,
                model=self.model,
            )
            self.agent_logs.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "memory_update",
                    "details": memory_metadata,
                }
            )
            logger.info(f"Memory updated after final summary: {memory_metadata}")

        result = AgentResult(
            final_message=response.raw,
            final_answer=extracted_final_answer,
            conversation_history=self.conversation_history.copy(),
            stop_reason=response.stop_reason,
            model=response.model or self.model,
            usage=response.usage or {},
            container_id=self.container_id,
            total_steps=self.max_steps,
            agent_logs=self.agent_logs.copy(),
            generated_files=None,
        )

        self._log_action(
            "final_summary_generation",
            {
                "stop_reason": response.stop_reason,
                "total_steps": self.max_steps,
                "total_input_tokens": (response.usage or {}).get("input_tokens"),
                "total_output_tokens": (response.usage or {}).get("output_tokens"),
            },
            step_number=self.max_steps,
        )

        await self._finalize_file_processing(queue)
        all_files_metadata = list(self.file_registry.values())
        result.generated_files = all_files_metadata

        self._save_run_data_async(result, all_files_metadata)
        await self._emit_meta_final(queue, result)

        return result

    async def _execute_backend_tools(
        self,
        *,
        backend_tool_calls: list[dict[str, Any]],
        queue: Optional[asyncio.Queue],
        step: int,
    ) -> list[dict[str, Any]]:
        tool_results: list[dict[str, Any]] = []

        for tool_call in backend_tool_calls:
            is_error = False
            result_content: ToolResultContent = ""
            image_refs: list[dict[str, Any]] = []

            try:
                result = self.execute_tool_call(tool_call["name"], tool_call["input"])
                if asyncio.iscoroutine(result):
                    result = await result
                result_content, image_refs = result
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call["tool_use_id"],
                        "content": result_content,
                    }
                )
            except Exception as e:  # noqa: BLE001
                is_error = True
                result_content = f"Error executing tool: {str(e)}"
                image_refs = []
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call["tool_use_id"],
                        "content": result_content,
                        "is_error": True,
                    }
                )

            # Stream tool result to queue (kept compatible with existing agent.py behavior)
            if queue is not None:
                tool_use_id = html.escape(str(tool_call["tool_use_id"]), quote=True)
                tool_name_escaped = html.escape(str(tool_call["name"]), quote=True)

                if image_refs:
                    text_parts: list[str] = []
                    if isinstance(result_content, list):
                        for block in result_content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(str(block.get("text", "")))
                    text_content = "\n".join(text_parts) if text_parts else ""

                    image_tags = "".join(
                        f'<image src="{html.escape(ref.get("src", ""), quote=True)}" '
                        f'media_type="{html.escape(ref.get("media_type", ""), quote=True)}" />'
                        for ref in image_refs
                    )

                    await queue.put(
                        f'<content-block-tool_result id="{tool_use_id}" name="{tool_name_escaped}">'
                        f'<text><![CDATA[{text_content}]]></text>'
                        f"{image_tags}"
                        f"</content-block-tool_result>"
                    )
                else:
                    if result_content is None:
                        content_str = ""
                    elif isinstance(result_content, str):
                        content_str = result_content
                    else:
                        content_str = json.dumps(result_content, default=str)

                    await queue.put(
                        f'<content-block-tool_result id="{tool_use_id}" name="{tool_name_escaped}"><![CDATA[{content_str}]]></content-block-tool_result>'
                    )

            self._log_action(
                "tool_execution",
                {
                    "tool_name": tool_call["name"],
                    "tool_use_id": tool_call["tool_use_id"],
                    "success": not is_error,
                    "has_images": len(image_refs) > 0,
                },
                step_number=step,
            )

        return tool_results

    # ------------------------------------------------------------------
    # Compaction + final answer helpers
    # ------------------------------------------------------------------

    def _apply_compaction(self, step_number: int = 0) -> None:
        if not self.compactor:
            return

        if self.memory_store:
            self.memory_store.before_compact(self.messages, self.model)

        original_messages = self.messages.copy() if self.memory_store else None
        estimated_tokens = self._last_known_input_tokens

        compacted, metadata = self.compactor.compact(self.messages, self.model, estimated_tokens=estimated_tokens)

        if metadata.get("compaction_applied", False):
            self.messages = compacted

            if self.memory_store:
                self.messages, after_meta = self.memory_store.after_compact(
                    original_messages=original_messages,
                    compacted_messages=self.messages,
                    model=self.model,
                )
                metadata["memory"] = after_meta

            self.agent_logs.append({"timestamp": datetime.now().isoformat(), "action": "compaction", "details": metadata})
            self._log_action("compaction", metadata, step_number=step_number)
            logger.info(f"Compaction applied: {metadata}")

    def _extract_final_answer(self, assistant_message: dict[str, Any]) -> str:
        """Extract concatenated text after the last tool-related block."""
        if not assistant_message:
            return ""

        content = assistant_message.get("content")
        if content is None:
            return ""

        # If content is a plain string (some providers), return it.
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            try:
                return json.dumps(content, default=str)
            except Exception:
                return str(content)

        start_index = 0
        for i, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") in {
                "tool_use",
                "tool_result",
                "server_tool_use",
                "server_tool_result",
                "web_search_tool_result",
                "web_search_tool_use",
                "function_call",
                "function_result",
            }:
                start_index = i + 1

        full_text: list[str] = []
        for block in content[start_index:]:
            if isinstance(block, dict):
                if block.get("type") in {"text", "output_text"}:
                    full_text.append(str(block.get("text", "")))

        return "".join(full_text)

    def compact_messages(self) -> dict[str, Any]:
        """Explicitly request compaction."""
        if not self.compactor:
            return {"error": "No compactor configured"}

        estimated_tokens = self._last_known_input_tokens
        compacted, metadata = self.compactor.compact(self.messages, self.model, estimated_tokens)

        if metadata.get("compaction_applied", False):
            self.messages = compacted
            self.agent_logs.append({"timestamp": datetime.now().isoformat(), "action": "manual_compaction", "details": metadata})
            logger.info(f"Manual compaction applied: {metadata}")

        return metadata

    def _get_estimated_tokens(self) -> int:
        return self._last_known_input_tokens

    async def _estimate_tokens(
        self,
        *,
        messages: Optional[list[dict[str, Any]]] = None,
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> int:
        """Cheap heuristic token estimator (character-count based)."""
        text_parts: list[str] = []

        if system:
            text_parts.append(system)

        if tools:
            try:
                text_parts.append(json.dumps(tools, separators=(",", ":")))
            except TypeError:
                text_parts.append(str(tools))

        if messages:
            for message in messages:
                content = message.get("content")
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text" and "text" in block:
                            text_parts.append(str(block["text"]))
                        else:
                            try:
                                text_parts.append(json.dumps(block, separators=(",", ":"), ensure_ascii=False, default=str))
                            except TypeError:
                                text_parts.append(str(block))
                elif isinstance(content, dict):
                    try:
                        text_parts.append(json.dumps(content, separators=(",", ":"), ensure_ascii=False, default=str))
                    except TypeError:
                        text_parts.append(str(content))

        full_text = " ".join(text_parts)
        approx_tokens = max(0, len(full_text) // 4)
        return approx_tokens

    # ------------------------------------------------------------------
    # Tool call parsing
    # ------------------------------------------------------------------

    def _extract_tool_calls(self, assistant_message: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract canonical tool calls from an assistant message."""
        content = assistant_message.get("content") if assistant_message else None
        if not isinstance(content, list):
            return []

        tool_calls: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue

            tool_use_id = block.get("id") or block.get("tool_use_id") or block.get("call_id")
            name = block.get("name") or block.get("tool_name")
            raw_input = block.get("input") if "input" in block else block.get("arguments")

            # Normalize arguments to dict
            tool_input: dict[str, Any]
            if isinstance(raw_input, dict):
                tool_input = raw_input
            elif isinstance(raw_input, str):
                try:
                    tool_input = json.loads(raw_input)
                except Exception:
                    tool_input = {"_raw": raw_input}
            elif raw_input is None:
                tool_input = {}
            else:
                tool_input = {"_raw": raw_input}

            if not tool_use_id:
                # Provide a deterministic-but-unique ID if provider didn't supply one.
                tool_use_id = f"tool_{uuid.uuid4().hex}"

            if not name:
                name = "unknown"

            tool_calls.append({"tool_use_id": str(tool_use_id), "name": str(name), "input": tool_input})

        return tool_calls

    # ------------------------------------------------------------------
    # Logging + persistence
    # ------------------------------------------------------------------

    def _log_action(
        self,
        action_type: str,
        action_data: dict[str, Any],
        step_number: int = 0,
        messages_snapshot: list[dict[str, Any]] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        log_entry: dict[str, Any] = {
            "log_id": len(self._run_logs_buffer) + 1,
            "agent_uuid": self.agent_uuid,
            "run_id": self._run_id,
            "timestamp": datetime.now(),
            "step_number": step_number,
            "action_type": action_type,
            "action_data": action_data,
            "messages_count": len(self.messages),
            "estimated_tokens": self._get_estimated_tokens(),
        }

        if messages_snapshot is not None:
            log_entry["messages_snapshot"] = messages_snapshot

        if duration_ms is not None:
            log_entry["duration_ms"] = duration_ms

        self._run_logs_buffer.append(log_entry)

    def _track_usage(self, *, step: int, response: LLMResponse) -> None:
        """Update token tracking and token usage history from a provider response."""
        usage = response.usage or {}

        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")

        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            # Keep existing semantics: output becomes input in next call
            self._last_known_input_tokens = input_tokens + output_tokens
            self._last_known_output_tokens = output_tokens
        else:
            # Fallback: keep heuristic estimate, but don't regress
            pass

        usage_entry: dict[str, Any] = {
            "step": step,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        # Add provider-specific fields via hook
        usage_entry.update(self._get_provider_specific_usage_fields(usage))

        self._token_usage_history.append(usage_entry)

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _save_agent_config(self) -> None:
        existing_config = await self.db_backend.load_agent_config(self.agent_uuid)
        existing_config = existing_config or {}

        config: dict[str, Any] = {
            "agent_uuid": self.agent_uuid,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "tool_schemas": self.tool_schemas,
            "tool_names": [t.get("name") for t in self.tool_schemas] if self.tool_schemas else [],
            "server_tools": self.server_tools,
            "formatter": self.formatter,
            "stream_meta_history_and_tool_results": self.stream_meta_history_and_tool_results,
            "file_registry": self.file_registry,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "api_kwargs": self.api_kwargs,
            "last_known_input_tokens": self._last_known_input_tokens,
            "last_known_output_tokens": self._last_known_output_tokens,
            "pending_frontend_tools": self._pending_frontend_tools,
            "pending_backend_results": self._pending_backend_results,
            "awaiting_frontend_tools": self._awaiting_frontend_tools,
            "current_step": self._current_step,
            "conversation_history": getattr(self, "conversation_history", []),
            "created_at": (
                datetime.fromisoformat(existing_config["created_at"].replace("Z", "+00:00"))
                if isinstance(existing_config.get("created_at"), str)
                else (existing_config.get("created_at") or datetime.now())
            ),
            "updated_at": datetime.now(),
            "last_run_at": getattr(self, "_run_start_time", None),
            "total_runs": existing_config.get("total_runs", 0) + 1,
            # Provider identification (top-level for easy querying)
            "provider_type": self._get_provider_type(),
            # Provider-specific config (nested)
            "provider_config": self._get_provider_specific_config(),
        }

        if self.compactor:
            config["compactor_type"] = self.compactor.__class__.__name__
        if self.memory_store:
            config["memory_store_type"] = self.memory_store.__class__.__name__

        await self.db_backend.save_agent_config(config)

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _save_conversation_entry(self, result: AgentResult, files_metadata: list[dict[str, Any]]) -> None:
        # Extract first user message
        user_message = ""
        for msg in result.conversation_history:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                user_message = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            break

        usage = result.usage or {}

        conversation: dict[str, Any] = {
            "conversation_id": str(uuid.uuid4()),
            "agent_uuid": self.agent_uuid,
            "run_id": self._run_id,
            "started_at": self._run_start_time,
            "completed_at": datetime.now(),
            "user_message": user_message,
            "final_response": result.final_answer,
            "messages": result.conversation_history,
            "stop_reason": result.stop_reason,
            "total_steps": result.total_steps,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "usage": usage,
            "generated_files": files_metadata,
            "created_at": datetime.now(),
        }

        await self.db_backend.save_conversation_history(conversation)

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _save_run_logs(self) -> None:
        await self.db_backend.save_agent_run_logs(self.agent_uuid, self._run_id, self._run_logs_buffer)

    def _extract_first_user_message(self) -> str | None:
        for msg in self.messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
        return None

    async def _generate_and_save_title(self, user_message: str) -> None:
        if not self._title_generator:
            return

        existing_config = await self.db_backend.load_agent_config(self.agent_uuid)
        if existing_config and existing_config.get("title"):
            return

        title = await self._title_generator(user_message)

        try:
            await self.db_backend.update_agent_title(self.agent_uuid, title)
            logger.debug(f"Generated title for {self.agent_uuid}: {title}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to save title for {self.agent_uuid}: {e}")

    async def _save_run_data_with_retry(self, result: AgentResult, files_metadata: list[dict[str, Any]]) -> None:
        operations: list[tuple[str, Callable[[], Awaitable[None]]]] = [
            ("agent_config", self._save_agent_config),
            ("conversation_history", lambda: self._save_conversation_entry(result, files_metadata)),
            ("agent_runs", self._save_run_logs),
        ]

        for operation_type, op in operations:
            try:
                await op()
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to persist {operation_type} for run {self._run_id}: {e}", exc_info=True)
                failure_metadata = {
                    "run_id": self._run_id,
                    "agent_uuid": self.agent_uuid,
                    "retry_count": 3,
                    "operation_type": operation_type,
                    "timestamp": datetime.now().isoformat(),
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                }
                self._on_persistence_failure(e, failure_metadata)

        # Schedule title generation
        user_message = self._extract_first_user_message()
        if user_message and self._title_generator:
            title_task = asyncio.create_task(self._generate_and_save_title(user_message))
            self._background_tasks.add(title_task)
            title_task.add_done_callback(self._background_tasks.discard)

    def _save_run_data_async(self, result: AgentResult, files_metadata: list[dict[str, Any]]) -> None:
        task = asyncio.create_task(self._save_run_data_with_retry(result, files_metadata))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def drain_background_tasks(self, timeout: float = 30.0) -> dict[str, Any]:
        if not self._background_tasks:
            return {"total_tasks": 0, "completed": 0, "timed_out": 0, "task_ids": []}

        total_tasks = len(self._background_tasks)
        logger.info(f"Draining {total_tasks} background task(s) with {timeout}s timeout")

        try:
            await asyncio.wait_for(asyncio.gather(*self._background_tasks, return_exceptions=True), timeout=timeout)
            completed = total_tasks
            timed_out = 0
            incomplete_ids: list[str] = []
        except asyncio.TimeoutError:
            incomplete_tasks = [t for t in self._background_tasks if not t.done()]
            completed = total_tasks - len(incomplete_tasks)
            timed_out = len(incomplete_tasks)
            logger.warning(
                f"Timeout after {timeout}s: {completed}/{total_tasks} tasks completed, {timed_out} tasks still pending"
            )
            incomplete_ids = [f"task_{id(t)}" for t in incomplete_tasks]

        return {"total_tasks": total_tasks, "completed": completed, "timed_out": timed_out, "task_ids": incomplete_ids}

    def _on_persistence_failure(self, exception: Exception, metadata: dict[str, Any]) -> None:
        """Hook for subclasses to report persistence failures."""
        return

    async def _emit_meta_final(self, queue: Optional[asyncio.Queue], result: AgentResult) -> None:
        if queue is None or not self.stream_meta_history_and_tool_results:
            return

        meta_final: dict[str, Any] = {
            "conversation_history": result.conversation_history,
            "stop_reason": result.stop_reason,
            "total_steps": result.total_steps,
            "generated_files": result.generated_files,
        }
        escaped_json = html.escape(json.dumps(meta_final, default=str), quote=True)
        await queue.put(f'<meta_final data="{escaped_json}"></meta_final>')

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def export_agent_view_yaml(self) -> str:
        """Export the agent's system prompt and consolidated tool schemas as YAML."""
        combined_tools: list[dict[str, Any]] = []
        if self.tool_schemas:
            combined_tools.extend(self.tool_schemas)
        if self.frontend_tool_schemas:
            combined_tools.extend(self.frontend_tool_schemas)
        if self.server_tools:
            combined_tools.extend(self.server_tools)

        agent_view = {"system_prompt": self.system_prompt, "tools": combined_tools}
        return yaml.safe_dump(agent_view, sort_keys=False, allow_unicode=True, default_flow_style=False)

    def _restore_state_from_config(self, config: dict[str, Any]) -> None:
        # Core resumable state
        self.messages = config.get("messages", self.messages)
        self.file_registry = config.get("file_registry", self.file_registry)
        self._last_known_input_tokens = config.get("last_known_input_tokens", self._last_known_input_tokens)
        self._last_known_output_tokens = config.get("last_known_output_tokens", self._last_known_output_tokens)

        # Pause/resume state
        self._pending_frontend_tools = config.get("pending_frontend_tools", [])
        self._pending_backend_results = config.get("pending_backend_results", [])
        self._awaiting_frontend_tools = config.get("awaiting_frontend_tools", False)
        self._current_step = config.get("current_step", 0)

        # Run metadata
        self._run_id = config.get("run_id", self._run_id)
        self._run_start_time = config.get("run_start_time", self._run_start_time)
        self._run_logs_buffer = config.get("run_logs", self._run_logs_buffer)

        if self._awaiting_frontend_tools:
            self._loaded_conversation_history = config.get("conversation_history", [])
        else:
            self._loaded_conversation_history = []

        # Provider-specific restoration via hook (pass full_config for v1 fallback)
        provider_config = config.get("provider_config", {})
        self._restore_provider_specific_state(provider_config, config)

    def _build_user_message(self, prompt: str | list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
        if isinstance(prompt, str):
            return {"role": "user", "content": [{"type": "text", "text": prompt}]}
        if isinstance(prompt, list):
            return {"role": "user", "content": prompt}
        # Assume dict-like
        return prompt

    def _inject_agent_uuid_to_tools(self) -> None:
        tool_functions = []
        if self.tool_registry:
            tool_functions.extend(self.tool_registry.tools.values())
        # Frontend tools are only schemas; no execution registry here.

        for tool_func in tool_functions:
            if hasattr(tool_func, "agent_uuid"):
                try:
                    setattr(tool_func, "agent_uuid", self.agent_uuid)
                except Exception:
                    pass

    def __str__(self) -> str:
        tool_names = [schema.get("name", "<unnamed>") for schema in self.tool_schemas]
        config_snapshot = {
            "agent_uuid": self.agent_uuid,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "stream_meta_history_and_tool_results": self.stream_meta_history_and_tool_results,
            "messages_count": len(self.messages),
            "conversation_history_count": len(getattr(self, "conversation_history", [])),
            "agent_logs_count": len(getattr(self, "agent_logs", [])),
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "api_kwargs": self.api_kwargs,
            "formatter": self.formatter,
            "compactor": self.compactor.__class__.__name__ if self.compactor else None,
            "memory_store": self.memory_store.__class__.__name__ if self.memory_store else None,
            "final_answer_check": self.final_answer_check is not None,
            "db_backend": self.db_backend.__class__.__name__ if self.db_backend else None,
            "tools": tool_names,
            "provider_type": self._get_provider_type(),
            "provider_config": self._get_provider_specific_config(),
        }
        return json.dumps(config_snapshot, indent=2, default=str)
