# Move to anthropic_agent/core/agent.py
import asyncio
import json
import html
import anthropic
import logging
import uuid
import warnings
from datetime import datetime
from typing import Optional, Callable, Any, Awaitable
from collections.abc import Mapping, Sequence
from anthropic.types.beta import BetaMessage, FileMetadata

from .types import AgentResult
from .retry import anthropic_stream_with_backoff, retry_with_backoff
from .title_generator import generate_title
from ..tools.base import ToolRegistry, ToolResultContent
from ..streaming import FormatterType
from .compaction import CompactorType, get_compactor, Compactor
from ..memory import MemoryStoreType, get_memory_store, MemoryStore
from ..database import DBBackendType, get_db_backend, DatabaseBackend
from ..file_backends import FileBackendType, get_file_backend, FileStorageBackend

logger = logging.getLogger(__name__)


# Default configuration values for agents
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that should help the user with their questions."
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_STEPS = 50
DEFAULT_THINKING_TOKENS = 0
DEFAULT_MAX_TOKENS = 2048  # TODO: Model specific limits per model
DEFAULT_STREAM_META = False
DEFAULT_FORMATTER: FormatterType = "xml"
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0

# Cache control configuration (Anthropic limits)
MAX_CACHE_BLOCKS = 4
MIN_CACHE_TOKENS_SONNET = 1024  # Claude Sonnet/Opus minimum
MIN_CACHE_TOKENS_HAIKU = 2048   # Claude Haiku minimum

class AnthropicAgent:
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_steps: Optional[int] = None,
        thinking_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stream_meta_history_and_tool_results: Optional[bool] = None,
        tools: list[Callable[..., Any]] | None = None,
        frontend_tools: list[Callable[..., Any]] | None = None,
        server_tools: list[dict[str, Any]] | None = None,
        beta_headers: list[str] | None = None,
        container_id: str | None = None,
        messages: list[dict] | None = None,    # TODO: Add Message Type.
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        formatter: FormatterType | None = None,
        enable_cache_control: Optional[bool] = None,
        compactor: CompactorType | Compactor | None = None,
        memory_store: MemoryStoreType | MemoryStore | None = None,
        final_answer_check: Optional[Callable[[str], tuple[bool, str]]] = None,
        agent_uuid: str | None = None,
        db_backend: DBBackendType | DatabaseBackend = "filesystem",
        file_backend: FileBackendType | FileStorageBackend | None = None,
        **api_kwargs: Any,
    ):
        """Initialize AnthropicAgent with configuration.
        
        Args:
            system_prompt: System prompt to guide the agent's behavior
            model: Anthropic model name (default: "claude-sonnet-4-5")
            max_steps: Maximum conversation turns before stopping (default: 50)
            thinking_tokens: Budget for extended thinking tokens (default: 0 / disabled)
            max_tokens: Maximum tokens in response (default: 2048)
            stream_meta_history_and_tool_results: Include metadata in stream (default: False)
            tools: List of functions decorated with @tool for backend execution (default: None)
            frontend_tools: List of functions decorated with @tool(executor="frontend") for
                browser-side execution. These tools are schema-only on the server; when Claude
                calls them, the agent pauses and emits an awaiting_frontend_tools event. The
                browser executes the tool and POSTs the result back via /agent/tool_results.
            beta_headers: Beta feature headers for Anthropic API (default: None)
            container_id: Container ID for multi-turn conversations (default: None)
            messages: Initial message history (default: None)
            max_retries: Maximum retry attempts for API calls (default: 5)
            base_delay: Base delay in seconds for exponential backoff (default: 5.0)
            formatter: Default formatter for stream output ("xml" or "raw", default: "xml")
            enable_cache_control: Enable cache_control injection for message content blocks
                (default: True). When enabled, adds cache_control to supported content block
                types (text, image, document) in both user and assistant messages.
            compactor: Either a compactor name ("tool_result_removal", "none") or a pre-configured
                Compactor instance. If a string is provided, a compactor is created with default
                settings (no threshold). For custom threshold, create and pass a Compactor instance.
                Example: get_compactor("tool_result_removal", threshold=50000)
            memory_store: Either a memory store name ("placeholder", "none") or a pre-configured
                MemoryStore instance. Memory stores retrieve relevant context for injection and
                integrate with the compaction lifecycle to preserve important information.
            final_answer_check: Optional callable that validates the extracted final answer text.
                Signature: check(final_answer: str) -> (success: bool, error_message: str)
                If validation fails, the error message is injected as a user message and the agent
                continues until validation passes or max_steps is reached.
            agent_uuid: Optional agent session UUID for resuming previous sessions. If provided,
                agent state (messages, container_id, file_registry) will be automatically loaded
                from the database. If not provided, a new UUID is generated.
            db_backend: Database backend for persisting agent state ("filesystem" or "sql"), or
                a pre-configured DatabaseBackend instance. Default is "filesystem" with ./data path.
                All agents persist state by default.
            file_backend: Optional file storage backend for generated files ("local", "s3", "none"),
                or a pre-configured FileStorageBackend instance. If None, files are not stored.
                When configured, enables Files API beta header automatically.
                Example: "local" or LocalFilesystemBackend(base_path="/data/agent-files")
            **api_kwargs: Arbitrary keyword arguments to pass to the Anthropic API.
                These are persisted in agent config and merged into request parameters.
                Common options: temperature, top_p, top_k, stop_sequences, metadata.
        
        Background Task Management:
            Agent state is persisted asynchronously after each run. Use drain_background_tasks()
            before shutdown to ensure all persistence operations complete. Override the
            _on_persistence_failure() hook method to implement custom failure handling
            (e.g., metrics reporting, alerting).
        """
        
        #################################################################### 
        # DB Backend is mandatory and always present by default.
        # If agent_uuid is provided, state is loaded from database.
        # If agent_uuid is provided, but db_backend is not, an error is raised.
        #################################################################### 
        # Database backend for persistence (always present by default)
        if isinstance(db_backend, str):
            self.db_backend = get_db_backend(db_backend)
        else:
            self.db_backend = db_backend
        
        # Initialization state - tracks whether state has been loaded from DB
        # Use initialize() to load state, or run() will call it automatically
        self._initialized = False
        
        #################################################################### 
        # Non serializable params that are not loaded from database.
        # These are initialized per Agent instance.
        # Take special care to provide exact same params on agent initialization
        # if you want to resume an agent from a previous session.
        #################################################################### 
        # Final answer validation checker. Cannot be loaded from database.
        self.final_answer_check = final_answer_check
        
        # File storage backend (optional)
        if file_backend is None:
            self.file_backend: Optional[FileStorageBackend] = None
        elif isinstance(file_backend, str):
            # String name provided - create file backend with default settings
            self.file_backend = get_file_backend(file_backend)
        else:
            # Pre-configured FileStorageBackend instance provided
            self.file_backend = file_backend
        
        self.tool_registry: Optional[ToolRegistry] = None
        self.tool_schemas: list[dict[str, Any]] = []
        if tools:
            self.tool_registry = ToolRegistry()
            self.tool_registry.register_tools(tools)
            self.tool_schemas = self.tool_registry.get_schemas()
        
        # Frontend tools (executed in browser, schema-only on server)
        self.frontend_tool_schemas: list[dict[str, Any]] = []
        self.frontend_tool_names: set[str] = set()
        if frontend_tools:
            for fn in frontend_tools:
                if hasattr(fn, "__tool_schema__"):
                    schema = fn.__tool_schema__
                    self.frontend_tool_schemas.append(schema)
                    self.frontend_tool_names.add(schema["name"])
                else:
                    raise ValueError(
                        f"Frontend tool '{fn.__name__}' must be decorated with @tool(executor='frontend')"
                    )

        # Agent UUID for session tracking
        # If agent_uuid provided, state will be loaded from DB via initialize() 
        # (called automatically in run() or explicitly by caller)
        self.agent_uuid = agent_uuid or str(uuid.uuid4())
        
        # Inject agent_uuid into tools that support it (have __tool_instance__ with set_agent_uuid)
        if tools:
            for tool_fn in tools:
                if hasattr(tool_fn, '__tool_instance__'):
                    tool_instance = tool_fn.__tool_instance__
                    if hasattr(tool_instance, 'set_agent_uuid'):
                        tool_instance.set_agent_uuid(self.agent_uuid)
        
        # db_config is empty at construction - state is loaded asynchronously via initialize()
        db_config: dict[str, Any] = {}
        
        #################################################################### 
        # Resolve core configuration and state from (constructor args, DB, defaults)
        #################################################################### 
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else db_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        )
        self.model = (
            model
            if model is not None
            else db_config.get("model", DEFAULT_MODEL)
        )
        self.max_steps = (
            max_steps
            if max_steps is not None
            else db_config.get("max_steps", DEFAULT_MAX_STEPS)
        )
        self.thinking_tokens = (
            thinking_tokens
            if thinking_tokens is not None
            else db_config.get("thinking_tokens", DEFAULT_THINKING_TOKENS)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else db_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        )

        self.beta_headers = (
            beta_headers
            if beta_headers is not None
            else db_config.get("beta_headers", [])
        )
        # Anthropic server tools configuration (e.g., code_execution, web_search).
        # This is a pass-through list of tool dicts that will be merged with
        # client-side tools (self.tool_schemas) when building requests.
        self.server_tools: list[dict[str, Any]] = (
            server_tools
            if server_tools is not None
            else db_config.get("server_tools", [])
        )
        
        # Messages and container state - initialized to defaults or constructor args.
        # For resumed agents (with agent_uuid), these will be overwritten by
        # _restore_state_from_config() when initialize() is called.
        self.messages = messages or []
        self.container_id = container_id
        self.file_registry: dict[str, dict] = {}
        self._last_known_input_tokens = 0
        self._last_known_output_tokens = 0
        
        # Frontend tool relay state (for resume after browser execution)
        # Will be restored from DB via initialize() for resumed agents
        self._pending_frontend_tools: list = []
        self._pending_backend_results: list = []
        self._awaiting_frontend_tools = False
        self._current_step = 0
        self._loaded_conversation_history: list = []
        
        # Runtime configuration that may be persisted but can always be overridden
        # by explicit (non-None) constructor arguments.
        self.stream_meta_history_and_tool_results = (
            stream_meta_history_and_tool_results
            if stream_meta_history_and_tool_results is not None
            else db_config.get(
                "stream_meta_history_and_tool_results",
                DEFAULT_STREAM_META,
            )
        )
        self.formatter = (
            formatter
            if formatter is not None
            else db_config.get("formatter", DEFAULT_FORMATTER)
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else db_config.get("max_retries", DEFAULT_MAX_RETRIES)
        )
        self.base_delay = (
            base_delay
            if base_delay is not None
            else db_config.get("base_delay", DEFAULT_BASE_DELAY)
        )
        self.enable_cache_control = (
            enable_cache_control
            if enable_cache_control is not None
            else db_config.get("enable_cache_control", True)
        )
        # Arbitrary API kwargs (e.g., temperature, top_p, stop_sequences)
        self.api_kwargs: dict[str, Any] = (
            api_kwargs
            if api_kwargs
            else db_config.get("api_kwargs", {})
        )
        # Context compaction
        if compactor is None:
            self.compactor: Optional[Compactor] = None
        elif isinstance(compactor, str):
            # String name provided - create compactor with default settings (no threshold)
            self.compactor = get_compactor(compactor)
        else:
            # Pre-configured Compactor instance provided
            self.compactor = compactor
        
        # Semantic memory
        if memory_store is None:
            self.memory_store: Optional[MemoryStore] = None
        elif isinstance(memory_store, str):
            # String name provided - create memory store with default settings
            self.memory_store = get_memory_store(memory_store)
        else:
            # Pre-configured MemoryStore instance provided
            self.memory_store = memory_store
            
        #################################################################### 
        # The following params are used to track the agent's run state.
        # They are initialized per Agent instance.
        #################################################################### 
        
        self._background_tasks: set = set()
        
        # Token tracking (persisted across runs, reset per-run for history)
        self._token_usage_history: list[dict] = []
        
        # Three-state tracking
        # messages: compacted messages for API (existing)
        # conversation_history: per-run uncompacted history (reset in run())
        # agent_logs: meta-information about compactions and actions (reset in run())
        
        # Note: Frontend tool relay state (_pending_frontend_tools, _pending_backend_results,
        # _awaiting_frontend_tools, _current_step) is initialized in the agent_uuid block above
        # to support loading from DB for resumed agents.
        
        # Initialize the Anthropic async client for proper async streaming
        self.client = anthropic.AsyncAnthropic()
    
    @property
    def is_initialized(self) -> bool:
        """Check if agent has been initialized (state loaded from DB).
        
        Returns True if initialize() has been called or if there's no agent_uuid
        to load state for.
        """
        return self._initialized
    
    async def initialize(self) -> dict[str, Any]:
        """Initialize agent by loading state from database.
        
        Can be called explicitly to access agent state before run(),
        or will be called automatically at the start of run().
        
        This method is idempotent - calling it multiple times has no effect
        after the first successful initialization.
        
        Returns:
            Loaded configuration dict, or empty dict if:
            - Already initialized
            - No db_backend configured
            - No agent_uuid (new agent)
            - Agent not found in database
            - Error during loading
        """
        if self._initialized:
            return {}  # Already initialized
        
        if not self.db_backend:
            self._initialized = True
            return {}
        
        try:
            config = await self.db_backend.load_agent_config(self.agent_uuid)
            
            if config is None:
                # New agent - will be created on first run
                logger.info(f"No existing state for agent {self.agent_uuid}, creating new")
                self._initialized = True
                return {}
            
            # Restore state from loaded configuration
            self._restore_state_from_config(config)
            
            logger.info(
                f"Loaded state for agent {self.agent_uuid}: "
                f"{len(config.get('messages', []))} messages, "
                f"container_id={config.get('container_id')}"
            )
            self._initialized = True
            return config
            
        except Exception as e:
            logger.error(f"Failed to load state for agent {self.agent_uuid}: {e}", exc_info=True)
            self._initialized = True
            return {}
    
    async def run(
        self,
        prompt: str | list[dict],
        queue: Optional[asyncio.Queue] = None,
        formatter: Optional[FormatterType] = None,
    ) -> AgentResult:
        """
        Execute an agent run with the given user message.
        
        Args:
            prompt: Either a string or a list of content blocks (for multimodal input)
            queue: Optional async queue to stream formatted output chunks
            formatter: Formatter to use for stream output ("xml" or "raw"). 
                      If None, uses the default formatter set at agent initialization.
        
        Returns:
            AgentResult object containing:
                - final_message: The last assistant message (BetaMessage)
                - conversation_history: Full list of all messages
                - stop_reason: Why the model stopped
                - model: Model used
                - usage: Token usage statistics
                - container_id: Container ID (if applicable)
                - total_steps: Number of agent steps taken
        """
        # Ensure agent is initialized (loads state from DB if needed)
        if not self._initialized:
            await self.initialize()
        
        #################################################################### 
        # The following params are used to track the agent's run state.
        # They are initialized per run.
        #################################################################### 
        # Initialize run tracking
        self._run_id = str(uuid.uuid4())
        self._run_start_time = datetime.now()
        self._run_logs_buffer = []
        self.conversation_history: list[dict] = []
        self.agent_logs: list[dict] = []
        
        # Reset per-run token tracking (preserve _last_known_input/output_tokens from previous runs)
        self._token_usage_history = []
        
        # Build initial message
        if isinstance(prompt, str):
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        elif isinstance(prompt, list):
            # List of content blocks (e.g., for file uploads)
            user_message = {
                "role": "user",
                "content": prompt
            }
        else:
            # Assume it's already a properly formatted message dict
            user_message = prompt

        # Add user message to live context and conversation history
        self.messages.append(user_message)
        self.conversation_history.append(user_message)
        
        # Log: run started
        self._log_action("run_started", {
            "user_message": prompt if isinstance(prompt, str) else str(prompt)[:200],
            "queue_present": queue is not None,
            "formatter": formatter or self.formatter
        }, step_number=0)
        
        # Emit meta_init tag to signal stream format and provide metadata
        if queue is not None:
            meta_init = {
                "format": formatter if formatter is not None else self.formatter,
                "user_query": prompt if isinstance(prompt, str) else json.dumps(prompt),
                "message_history": self.conversation_history,
                "agent_uuid": self.agent_uuid,
                "model": self.model,
            }
            escaped_json = html.escape(json.dumps(meta_init), quote=True)
            await queue.put(f'<meta_init data="{escaped_json}"></meta_init>')
        
        # Retrieve and inject semantic memories
        if self.memory_store:
            self.messages = self.memory_store.retrieve(
                tools=self.tool_schemas,
                user_message=user_message,
                messages=self.messages,
                model=self.model
            )
            # Note: Memory-injected messages are NOT added to conversation_history
        
        # Initialize token estimate for current context using heuristic estimation.
        # Combine client tools (tool_schemas) and server tools for token counting.
        combined_tools: list[dict[str, Any]] = []
        if self.tool_schemas:
            combined_tools.extend(self.tool_schemas)
        if self.server_tools:
            combined_tools.extend(self.server_tools)
        estimated_tokens: int = await self._estimate_tokens(
            messages=self.messages,
            system=self.system_prompt,
            tools=combined_tools or None,
            thinking=(
                {"type": "enabled", "budget_tokens": self.thinking_tokens}
                if self.thinking_tokens and self.thinking_tokens > 0
                else None
            ),
            betas=self.beta_headers or None,
            container=self.container_id,
        )
        self._last_known_input_tokens = estimated_tokens
        
        step = 0
        while step < self.max_steps:
            step += 1
            
            # Always pass through compactor (it decides whether to compact)
            if self.compactor:
                self._apply_compaction(step_number=step)  # self._last_known_input_tokens is used here.
            
            # Prepare request parameters
            request_params = self._prepare_request_params()
            
            # Stream the response with retry logic
            accumulated_message = await anthropic_stream_with_backoff(
                client=self.client,
                request_params=request_params,
                queue=queue,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                formatter=formatter if formatter is not None else self.formatter,
            )
            
            # Track token usage from API response
            input_tokens = accumulated_message.usage.input_tokens
            output_tokens = accumulated_message.usage.output_tokens
            self._last_known_input_tokens = input_tokens + output_tokens # output becomes input in next call
            self._last_known_output_tokens = output_tokens
            self._token_usage_history.append({
                "step": step,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": accumulated_message.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": accumulated_message.usage.cache_read_input_tokens,
            })
            
            # Log: API response received
            self._log_action("api_response_received", {
                "stop_reason": accumulated_message.stop_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }, step_number=step)
            
            # Add assistant's response to live context and conversation history
            # assistant_message = {
            #     "role": accumulated_message.role,
            #     "content": accumulated_message.content
            # }
            assistant_message = accumulated_message.model_dump(
                mode="json",
                include=["role", "content"],
                exclude_unset=True,
                exclude=getattr(accumulated_message, "__api_exclude__", None),
                warnings=False
            )
            self.messages.append(assistant_message)
            self.conversation_history.append(assistant_message)
            logger.debug("Assistant message: %s", assistant_message)
            if accumulated_message.container != None:
                self.container_id = accumulated_message.container.id
            
            # Check if there are tool calls to execute
            # Only process tool calls if stop_reason is "tool_use"
            if accumulated_message.stop_reason == "tool_use":
                tool_calls = [
                    block for block in accumulated_message.content 
                    if block.type == 'tool_use'
                ]
                
                if tool_calls:
                    # Separate tool calls into backend vs frontend
                    backend_tool_calls = [t for t in tool_calls if t.name not in self.frontend_tool_names]
                    frontend_tool_calls = [t for t in tool_calls if t.name in self.frontend_tool_names]
                    
                    # Execute ALL backend tools first
                    tool_results = []
                    for tool_call in backend_tool_calls:
                        is_error = False
                        result_content: ToolResultContent = ""
                        image_refs: list[dict[str, Any]] = []
                        try:
                            # Execute the tool (support both sync and async executors)
                            result = self.execute_tool_call(tool_call.name, tool_call.input)
                            # Check if result is a coroutine (async function)
                            if asyncio.iscoroutine(result):
                                result = await result
                            # Unpack result tuple (content, image_refs)
                            result_content, image_refs = result
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": result_content
                            })
                        except Exception as e:
                            # Handle tool execution errors
                            is_error = True
                            result_content = f"Error executing tool: {str(e)}"
                            image_refs = []
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": result_content,
                                "is_error": True
                            })
                        
                        # Stream tool result to queue in XML format
                        if queue is not None:
                            tool_use_id = html.escape(str(tool_call.id), quote=True)
                            tool_name_escaped = html.escape(str(tool_call.name), quote=True)
                            
                            # Build streaming output based on content type
                            if image_refs:
                                # Multimodal result: stream text + image references
                                text_parts = []
                                if isinstance(result_content, list):
                                    for block in result_content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            text_parts.append(block.get("text", ""))
                                text_content = "\n".join(text_parts) if text_parts else ""
                                
                                # Build image reference tags
                                image_tags = "".join(
                                    f'<image src="{html.escape(ref["src"], quote=True)}" media_type="{html.escape(ref["media_type"], quote=True)}" />'
                                    for ref in image_refs
                                )
                                
                                await queue.put(
                                    f'<content-block-tool_result id="{tool_use_id}" name="{tool_name_escaped}">'
                                    f'<text><![CDATA[{text_content}]]></text>'
                                    f'{image_tags}'
                                    f'</content-block-tool_result>'
                                )
                            else:
                                # Text-only result: serialize as before
                                if result_content is None:
                                    content_str = ""
                                elif isinstance(result_content, str):
                                    content_str = result_content
                                else:
                                    content_str = json.dumps(result_content, default=str)
                                await queue.put(
                                    f'<content-block-tool_result id="{tool_use_id}" name="{tool_name_escaped}"><![CDATA[{content_str}]]></content-block-tool_result>'
                                )
                        
                        # Log: tool execution
                        self._log_action("tool_execution", {
                            "tool_name": tool_call.name,
                            "tool_use_id": tool_call.id,
                            "success": not is_error,
                            "has_images": len(image_refs) > 0,
                        }, step_number=step)
                    
                    # If frontend tools exist, pause and wait for browser execution
                    if frontend_tool_calls:
                        # Store backend results for later (will be combined with frontend results)
                        self._pending_backend_results = tool_results
                        self._pending_frontend_tools = [
                            {"tool_use_id": t.id, "name": t.name, "input": t.input}
                            for t in frontend_tool_calls
                        ]
                        self._awaiting_frontend_tools = True
                        self._current_step = step
                        
                        # Log: awaiting frontend tools
                        self._log_action("awaiting_frontend_tools", {
                            "frontend_tools": [t.name for t in frontend_tool_calls],
                            "backend_results_count": len(tool_results),
                        }, step_number=step)
                        
                        # Emit all pending frontend tools to client
                        if queue is not None:
                            tools_json = html.escape(json.dumps(self._pending_frontend_tools), quote=True)
                            await queue.put(f'<awaiting_frontend_tools data="{tools_json}"></awaiting_frontend_tools>')
                        
                        # Persist state to DB before returning (required for re-hydration)
                        await self._save_agent_config()
                        
                        # Return early with partial result (agent is paused)
                        return AgentResult(
                            final_message=accumulated_message,
                            final_answer="",
                            conversation_history=self.conversation_history.copy(),
                            stop_reason="awaiting_frontend_tools",
                            model=accumulated_message.model,
                            usage=accumulated_message.usage,
                            container_id=self.container_id,
                            total_steps=step,
                            agent_logs=self.agent_logs.copy(),
                            generated_files=None
                        )
                    
                    # No frontend tools - add all tool results and continue
                    tool_result_message = {
                        "role": "user",
                        "content": tool_results
                    }
                    self.messages.append(tool_result_message)
                    self.conversation_history.append(tool_result_message)

                    # Update token estimate using a lightweight heuristic on just the new tool results.
                    delta_tokens: int = await self._estimate_tokens(
                        messages=[tool_result_message],
                        system=None,
                        tools=None,
                        thinking=None,
                        betas=None,
                        container=None,
                    )
                    # api_token_count: Optional[int] = await self._count_tokens_api(
                    #     messages=self.messages,
                    #     system=self.system_prompt,
                    #     tools=self.tool_schemas or None,
                    #     thinking=(
                    #         {"type": "enabled", "budget_tokens": self.thinking_tokens}
                    #         if self.thinking_tokens and self.thinking_tokens > 0
                    #         else None
                    #     ),
                    #     betas=self.beta_headers or None,
                    #     container=self.container_id,
                    # )
                    self._last_known_input_tokens += delta_tokens
                    
                    # Continue the loop to get the next response
                    continue
            
            # If stop_reason is not "tool_use", validate final answer format
            if accumulated_message.stop_reason != "tool_use":
                # Validate final answer format if checker is configured
                if self.final_answer_check:
                    extracted_final_answer = self._extract_final_answer(accumulated_message)
                    success, error_message = self.final_answer_check(extracted_final_answer)
                    if not success:
                        # Log validation failure
                        self.agent_logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "action": "final_answer_validation_failed",
                            "details": {"error": error_message, "step": step}
                        })
                        logger.warning(f"Final answer validation failed at step {step}: {error_message}")
                        
                        # Inject error as user message and continue loop
                        error_user_message = {
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": error_message
                            }]
                        }
                        self.messages.append(error_user_message)
                        self.conversation_history.append(error_user_message)
                        continue
                
                # Validation passed or no checker - proceed with memory update and return
            
            # Update memory store with conversation results
            if self.memory_store:
                memory_metadata = self.memory_store.update(
                    messages=self.messages,
                    conversation_history=self.conversation_history,
                    tools=self.tool_schemas,
                    model=self.model
                )
                self.agent_logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "memory_update",
                    "details": memory_metadata
                })
                logger.info(f"Memory updated: {memory_metadata}")

            # Build AgentResult (generated_files populated after file processing/registry aggregation)
            result = AgentResult(
                final_message=accumulated_message,
                final_answer=self._extract_final_answer(accumulated_message),
                conversation_history=self.conversation_history.copy(),
                stop_reason=accumulated_message.stop_reason,
                model=accumulated_message.model,
                usage=accumulated_message.usage,
                container_id=self.container_id,
                total_steps=step,
                agent_logs=self.agent_logs.copy(),
                generated_files=None  # Will be populated from file_registry below
            )
            
            # Log: run completed
            self._log_action("run_completed", {
                "stop_reason": accumulated_message.stop_reason,
                "total_steps": step,
                "total_input_tokens": accumulated_message.usage.input_tokens,
                "total_output_tokens": accumulated_message.usage.output_tokens,
            }, step_number=step)

            # Finalize file processing (extract, store, stream)
            await self._finalize_file_processing(queue)

            # Get updated metadata
            all_files_metadata: list[dict[str, Any]] = list(self.file_registry.values())

            # Update result with all known files for this agent
            result.generated_files = all_files_metadata

            # Save run data asynchronously with complete file registry snapshot
            self._save_run_data_async(result, all_files_metadata)
            
            return result

        # Max steps reached - generate final summary
        logger.warning(f"Max steps ({self.max_steps}) reached, generating final summary")
        return await self._generate_final_summary(queue=queue, formatter=formatter)
    
    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | str | None]:
        """Apply cache_control to message content blocks with prioritized slot allocation.
        
        Anthropic limits cache_control to 4 blocks maximum. This method applies
        caching in priority order:
        1. System prompt (if large enough)
        2. Document/image blocks (high value, typically large)
        3. Large text blocks (sorted by size)
        4. Recent message blocks (fallback)
        
        Args:
            messages: List of message dicts to process
            system: Optional system prompt string
            
        Returns:
            Tuple of (processed_messages, processed_system) where:
            - processed_messages: Messages with cache_control applied to priority blocks
            - processed_system: Either block-format list with cache_control, or original string
        """
        if not self.enable_cache_control:
            return messages, system
        
        # Determine minimum token threshold based on model
        min_tokens = (
            MIN_CACHE_TOKENS_HAIKU 
            if "haiku" in self.model.lower() 
            else MIN_CACHE_TOKENS_SONNET
        )
        remaining_slots = MAX_CACHE_BLOCKS
        
        # Block types that support cache_control
        supported_types = {"text", "image", "document"}
        
        # Track which blocks to cache: list of (msg_idx, block_idx) tuples
        blocks_to_cache: list[tuple[int, int]] = []
        
        # ----------------------------------------------------------------
        # Priority 1: System prompt
        # ----------------------------------------------------------------
        processed_system: list[dict[str, Any]] | str | None = system
        if system and remaining_slots > 0:
            system_tokens = len(system) // 4  # Reuse existing heuristic
            if system_tokens >= min_tokens:
                # Convert to block format with cache_control
                processed_system = [{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"}
                }]
                remaining_slots -= 1
        
        # ----------------------------------------------------------------
        # Priority 2: Document/Image blocks (scan all messages)
        # ----------------------------------------------------------------
        doc_image_blocks: list[tuple[int, int]] = []
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block_idx, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type in ("document", "image"):
                    doc_image_blocks.append((msg_idx, block_idx))
        
        # Add document/image blocks up to remaining slots
        for loc in doc_image_blocks:
            if remaining_slots <= 0:
                break
            blocks_to_cache.append(loc)
            remaining_slots -= 1
        
        # ----------------------------------------------------------------
        # Priority 3: Large text blocks (sorted by size descending)
        # ----------------------------------------------------------------
        if remaining_slots > 0:
            large_text_blocks: list[tuple[int, int, int]] = []  # (msg_idx, block_idx, size)
            for msg_idx, msg in enumerate(messages):
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for block_idx, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and "text" in block:
                        text_len = len(block["text"])
                        text_tokens = text_len // 4
                        if text_tokens >= min_tokens:
                            # Skip if already marked for caching (shouldn't happen but safe)
                            if (msg_idx, block_idx) not in blocks_to_cache:
                                large_text_blocks.append((msg_idx, block_idx, text_len))
            
            # Sort by size descending and add up to remaining slots
            large_text_blocks.sort(key=lambda x: x[2], reverse=True)
            for msg_idx, block_idx, _ in large_text_blocks:
                if remaining_slots <= 0:
                    break
                blocks_to_cache.append((msg_idx, block_idx))
                remaining_slots -= 1
        
        # ----------------------------------------------------------------
        # Priority 4: Recent message blocks (fallback)
        # ----------------------------------------------------------------
        if remaining_slots > 0:
            # Iterate messages in reverse, then blocks in reverse
            for msg_idx in range(len(messages) - 1, -1, -1):
                if remaining_slots <= 0:
                    break
                msg = messages[msg_idx]
                role = msg.get("role")
                content = msg.get("content", [])
                
                # Only process user/assistant messages with list content
                if role not in ("user", "assistant") or not isinstance(content, list):
                    continue
                
                for block_idx in range(len(content) - 1, -1, -1):
                    if remaining_slots <= 0:
                        break
                    block = content[block_idx]
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") in supported_types:
                        # Skip if already marked for caching
                        if (msg_idx, block_idx) not in blocks_to_cache:
                            blocks_to_cache.append((msg_idx, block_idx))
                            remaining_slots -= 1
        
        # ----------------------------------------------------------------
        # Build result messages with cache_control applied to selected blocks
        # ----------------------------------------------------------------
        blocks_to_cache_set = set(blocks_to_cache)
        result: list[dict[str, Any]] = []
        
        for msg_idx, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", [])
            
            # Pass through messages without list content unchanged
            if not isinstance(content, list):
                result.append(msg)
                continue
            
            # Deep copy message and inject cache_control into selected blocks
            new_msg = {"role": role, "content": []}
            for block_idx, block in enumerate(content):
                new_block = dict(block) if isinstance(block, dict) else block
                if (msg_idx, block_idx) in blocks_to_cache_set:
                    new_block["cache_control"] = {"type": "ephemeral"}
                new_msg["content"].append(new_block)
            result.append(new_msg)
        
        return result, processed_system
    
    def _prepare_request_params(self) -> dict:
        # Prepare request parameters with prioritized cache_control applied
        messages, system = self._apply_cache_control(self.messages, self.system_prompt)
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        
        # Add system prompt (may be block format with cache_control or original string)
        if system:
            request_params["system"] = system
        
        if self.thinking_tokens > 0:
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_tokens
            }
        
        # Merge client tools (tool_schemas), frontend tools, and Anthropic server tools for this call.
        combined_tools: list[dict[str, Any]] = []
        if self.tool_schemas:
            combined_tools.extend(self.tool_schemas)
        if self.frontend_tool_schemas:
            combined_tools.extend(self.frontend_tool_schemas)
        if self.server_tools:
            combined_tools.extend(self.server_tools)
        if combined_tools:
            request_params["tools"] = combined_tools
        
        if self.beta_headers:
            request_params["betas"] = self.beta_headers
        
        if self.container_id:
            request_params["container"] = self.container_id
        
        # Merge arbitrary API kwargs (e.g., temperature, top_p, stop_sequences)
        if self.api_kwargs:
            request_params.update(self.api_kwargs)
        
        return request_params
    
    def execute_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
    ) -> tuple[ToolResultContent, list[dict[str, Any]]]:
        """Execute a registered tool function through the ToolRegistry.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of input parameters for the tool
            
        Returns:
            Tuple of (content, image_refs) where:
            - content: String or list of content blocks for Anthropic API
            - image_refs: List of image reference dicts for streaming
        """
        if not self.tool_registry:
            return "No tools have been registered for this agent.", []
        
        return self.tool_registry.execute(
            tool_name,
            tool_input,
            file_backend=self.file_backend,
            agent_uuid=self.agent_uuid,
        )
    
    async def continue_with_tool_results(
        self,
        frontend_results: list[dict],
        queue: Optional[asyncio.Queue] = None,
        formatter: Optional[FormatterType] = None,
    ) -> AgentResult:
        """Resume agent execution after frontend tools have been executed.
        
        This method is called after the browser executes frontend tools and POSTs
        the results back. It combines backend results (already executed) with
        frontend results, adds them to the conversation, and continues the agent loop.
        
        Args:
            frontend_results: List of frontend tool results, each containing:
                - tool_use_id: ID of the tool call being responded to
                - content: String result from frontend execution
                - is_error: Optional boolean indicating if execution failed
            queue: Optional async queue to stream formatted output chunks
            formatter: Formatter to use for stream output ("xml" or "raw")
            
        Returns:
            AgentResult from continuing the agent run
            
        Raises:
            ValueError: If agent is not awaiting frontend tools, or if tool_use_ids
                don't match the pending frontend tools
        """
        if not self._awaiting_frontend_tools:
            raise ValueError("Agent is not awaiting frontend tools")
        
        if not self._pending_frontend_tools:
            raise ValueError("No pending frontend tools found - state may not have been loaded from DB")
        
        # Initialize run state if resuming from DB (these are normally set in run())
        if not hasattr(self, 'conversation_history') or not self.conversation_history:
            self.conversation_history = getattr(self, '_loaded_conversation_history', []).copy()
        if not hasattr(self, 'agent_logs'):
            self.agent_logs = []
        if not hasattr(self, '_run_logs_buffer'):
            self._run_logs_buffer = []
        if not hasattr(self, '_run_id'):
            self._run_id = str(uuid.uuid4())
        if not hasattr(self, '_run_start_time'):
            self._run_start_time = datetime.now()
        
        # Validate all tool_use_ids match pending tools
        pending_ids = {t["tool_use_id"] for t in self._pending_frontend_tools}
        result_ids = {r["tool_use_id"] for r in frontend_results}
        if pending_ids != result_ids:
            raise ValueError(
                f"Tool result mismatch. Expected tool_use_ids: {pending_ids}, got: {result_ids}"
            )
        
        # Combine backend + frontend results (backend results come first)
        all_results = self._pending_backend_results + [
            {
                "type": "tool_result",
                "tool_use_id": r["tool_use_id"],
                "content": r["content"],
                **({"is_error": True} if r.get("is_error") else {})
            }
            for r in frontend_results
        ]
        
        # Stream frontend tool results to queue
        if queue is not None:
            for r in frontend_results:
                tool_use_id = html.escape(str(r["tool_use_id"]), quote=True)
                # Find the tool name from pending tools
                tool_name = next(
                    (t["name"] for t in self._pending_frontend_tools if t["tool_use_id"] == r["tool_use_id"]),
                    "unknown"
                )
                tool_name = html.escape(tool_name, quote=True)
                content_str = r["content"] if isinstance(r["content"], str) else json.dumps(r["content"], default=str)
                await queue.put(
                    f'<content-block-tool_result id="{tool_use_id}" name="{tool_name}"><![CDATA[{content_str}]]></content-block-tool_result>'
                )
        
        # Add combined tool results to messages and conversation history
        tool_result_message = {
            "role": "user",
            "content": all_results
        }
        self.messages.append(tool_result_message)
        self.conversation_history.append(tool_result_message)
        
        # Log: frontend tools completed
        self._log_action("frontend_tools_completed", {
            "frontend_results_count": len(frontend_results),
            "backend_results_count": len(self._pending_backend_results),
            "total_results": len(all_results),
        }, step_number=self._current_step)
        
        # Clear pending state
        self._pending_frontend_tools = []
        self._pending_backend_results = []
        self._awaiting_frontend_tools = False
        
        # Update token estimate
        delta_tokens: int = await self._estimate_tokens(
            messages=[tool_result_message],
            system=None,
            tools=None,
            thinking=None,
            betas=None,
            container=None,
        )
        self._last_known_input_tokens += delta_tokens
        
        # Resume agent loop from current step
        return await self._resume_run(queue=queue, formatter=formatter)
    
    async def _resume_run(
        self,
        queue: Optional[asyncio.Queue] = None,
        formatter: Optional[FormatterType] = None,
    ) -> AgentResult:
        """Internal method to resume the agent loop after frontend tool completion.
        
        This continues from the current step, streaming responses and handling
        any subsequent tool calls.
        """
        step = self._current_step
        
        while step < self.max_steps:
            step += 1
            self._current_step = step
            
            # Always pass through compactor (it decides whether to compact)
            if self.compactor:
                self._apply_compaction(step_number=step)
            
            # Prepare request parameters
            request_params = self._prepare_request_params()
            
            # Stream the response with retry logic
            accumulated_message = await anthropic_stream_with_backoff(
                client=self.client,
                request_params=request_params,
                queue=queue,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                formatter=formatter if formatter is not None else self.formatter,
            )
            
            # Track token usage from API response
            input_tokens = accumulated_message.usage.input_tokens
            output_tokens = accumulated_message.usage.output_tokens
            self._last_known_input_tokens = input_tokens + output_tokens
            self._last_known_output_tokens = output_tokens
            self._token_usage_history.append({
                "step": step,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": accumulated_message.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": accumulated_message.usage.cache_read_input_tokens,
            })
            
            # Log: API response received
            self._log_action("api_response_received", {
                "stop_reason": accumulated_message.stop_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }, step_number=step)
            
            # Add assistant's response to messages and conversation history
            assistant_message = accumulated_message.model_dump(
                mode="json",
                include=["role", "content"],
                exclude_unset=True,
                exclude=getattr(accumulated_message, "__api_exclude__", None),
                warnings=False
            )
            self.messages.append(assistant_message)
            self.conversation_history.append(assistant_message)
            logger.debug("Assistant message: %s", assistant_message)
            if accumulated_message.container is not None:
                self.container_id = accumulated_message.container.id
            
            # Check if there are tool calls to execute
            if accumulated_message.stop_reason == "tool_use":
                tool_calls = [
                    block for block in accumulated_message.content 
                    if block.type == 'tool_use'
                ]
                
                if tool_calls:
                    # Separate backend vs frontend tools
                    backend_tool_calls = [t for t in tool_calls if t.name not in self.frontend_tool_names]
                    frontend_tool_calls = [t for t in tool_calls if t.name in self.frontend_tool_names]
                    
                    # Execute ALL backend tools first
                    tool_results = []
                    for tool_call in backend_tool_calls:
                        is_error = False
                        result_content: ToolResultContent = ""
                        image_refs: list[dict[str, Any]] = []
                        try:
                            result = self.execute_tool_call(tool_call.name, tool_call.input)
                            if asyncio.iscoroutine(result):
                                result = await result
                            # Unpack result tuple (content, image_refs)
                            result_content, image_refs = result
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": result_content
                            })
                        except Exception as e:
                            is_error = True
                            result_content = f"Error executing tool: {str(e)}"
                            image_refs = []
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": result_content,
                                "is_error": True
                            })
                        
                        # Stream tool result to queue
                        if queue is not None:
                            tool_use_id = html.escape(str(tool_call.id), quote=True)
                            tool_name_escaped = html.escape(str(tool_call.name), quote=True)
                            
                            # Build streaming output based on content type
                            if image_refs:
                                # Multimodal result: stream text + image references
                                text_parts = []
                                if isinstance(result_content, list):
                                    for block in result_content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            text_parts.append(block.get("text", ""))
                                text_content = "\n".join(text_parts) if text_parts else ""
                                
                                # Build image reference tags
                                image_tags = "".join(
                                    f'<image src="{html.escape(ref["src"], quote=True)}" media_type="{html.escape(ref["media_type"], quote=True)}" />'
                                    for ref in image_refs
                                )
                                
                                await queue.put(
                                    f'<content-block-tool_result id="{tool_use_id}" name="{tool_name_escaped}">'
                                    f'<text><![CDATA[{text_content}]]></text>'
                                    f'{image_tags}'
                                    f'</content-block-tool_result>'
                                )
                            else:
                                # Text-only result: serialize as before
                                if result_content is None:
                                    content_str = ""
                                elif isinstance(result_content, str):
                                    content_str = result_content
                                else:
                                    content_str = json.dumps(result_content, default=str)
                                await queue.put(
                                    f'<content-block-tool_result id="{tool_use_id}" name="{tool_name_escaped}"><![CDATA[{content_str}]]></content-block-tool_result>'
                                )
                        
                        self._log_action("tool_execution", {
                            "tool_name": tool_call.name,
                            "tool_use_id": tool_call.id,
                            "success": not is_error,
                            "has_images": len(image_refs) > 0,
                        }, step_number=step)
                    
                    # If frontend tools exist, pause and wait
                    if frontend_tool_calls:
                        self._pending_backend_results = tool_results
                        self._pending_frontend_tools = [
                            {"tool_use_id": t.id, "name": t.name, "input": t.input}
                            for t in frontend_tool_calls
                        ]
                        self._awaiting_frontend_tools = True
                        self._current_step = step
                        
                        self._log_action("awaiting_frontend_tools", {
                            "frontend_tools": [t.name for t in frontend_tool_calls],
                            "backend_results_count": len(tool_results),
                        }, step_number=step)
                        
                        if queue is not None:
                            tools_json = html.escape(json.dumps(self._pending_frontend_tools), quote=True)
                            await queue.put(f'<awaiting_frontend_tools data="{tools_json}"></awaiting_frontend_tools>')
                        
                        # Persist state to DB before returning (required for re-hydration)
                        await self._save_agent_config()
                        
                        return AgentResult(
                            final_message=accumulated_message,
                            final_answer="",
                            conversation_history=self.conversation_history.copy(),
                            stop_reason="awaiting_frontend_tools",
                            model=accumulated_message.model,
                            usage=accumulated_message.usage,
                            container_id=self.container_id,
                            total_steps=step,
                            agent_logs=self.agent_logs.copy(),
                            generated_files=None
                        )
                    
                    # No frontend tools - add results and continue
                    tool_result_message = {
                        "role": "user",
                        "content": tool_results
                    }
                    self.messages.append(tool_result_message)
                    self.conversation_history.append(tool_result_message)
                    
                    delta_tokens = await self._estimate_tokens(
                        messages=[tool_result_message],
                        system=None,
                        tools=None,
                        thinking=None,
                        betas=None,
                        container=None,
                    )
                    self._last_known_input_tokens += delta_tokens
                    continue
            
            # If stop_reason is not "tool_use", validate final answer format
            if accumulated_message.stop_reason != "tool_use":
                if self.final_answer_check:
                    extracted_final_answer = self._extract_final_answer(accumulated_message)
                    success, error_message = self.final_answer_check(extracted_final_answer)
                    if not success:
                        self.agent_logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "action": "final_answer_validation_failed",
                            "details": {"error": error_message, "step": step}
                        })
                        logger.warning(f"Final answer validation failed at step {step}: {error_message}")
                        
                        error_user_message = {
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": error_message
                            }]
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
                    model=self.model
                )
                self.agent_logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "memory_update",
                    "details": memory_metadata
                })
                logger.info(f"Memory updated: {memory_metadata}")

            # Build AgentResult
            result = AgentResult(
                final_message=accumulated_message,
                final_answer=self._extract_final_answer(accumulated_message),
                conversation_history=self.conversation_history.copy(),
                stop_reason=accumulated_message.stop_reason,
                model=accumulated_message.model,
                usage=accumulated_message.usage,
                container_id=self.container_id,
                total_steps=step,
                agent_logs=self.agent_logs.copy(),
                generated_files=None
            )
            
            self._log_action("run_completed", {
                "stop_reason": accumulated_message.stop_reason,
                "total_steps": step,
                "total_input_tokens": accumulated_message.usage.input_tokens,
                "total_output_tokens": accumulated_message.usage.output_tokens,
            }, step_number=step)

            # Finalize file processing
            await self._finalize_file_processing(queue)
            all_files_metadata: list[dict[str, Any]] = list(self.file_registry.values())
            result.generated_files = all_files_metadata
            self._save_run_data_async(result, all_files_metadata)
            
            return result

        # Max steps reached
        logger.warning(f"Max steps ({self.max_steps}) reached in _resume_run")
        return await self._generate_final_summary(queue=queue, formatter=formatter)
    
    def _apply_compaction(self, step_number: int = 0) -> None:
        """Apply compaction to self.messages and log the event.
        
        This method is called before each API call. The compactor itself decides
        whether compaction is needed based on its internal threshold and logic.
        If compaction is applied, the event is logged to agent_logs with metadata.
        
        Memory integration: If a memory store is configured, it is called before
        and after compaction to extract and preserve important information.
        
        Args:
            step_number: Current step number in the agent loop (for logging)
        """
        if not self.compactor:
            return
        
        # Before compaction: Let memory store extract info
        if self.memory_store:
            self.memory_store.before_compact(self.messages, self.model)
        
        # Preserve original for after_compact
        original_messages = self.messages.copy() if self.memory_store else None
        
        # Estimate current context: last_input
        estimated_tokens = self._last_known_input_tokens
        
        # Compact
        compacted, metadata = self.compactor.compact(self.messages, self.model, estimated_tokens=estimated_tokens)
        
        # Only update messages if compaction was actually applied
        if metadata.get("compaction_applied", False):
            self.messages = compacted
            
            # After compaction: Let memory store update compacted messages
            if self.memory_store:
                self.messages, after_meta = self.memory_store.after_compact(
                    original_messages=original_messages,
                    compacted_messages=self.messages,
                    model=self.model
                )
                metadata["memory"] = after_meta
            
            # Log compaction with memory metadata
            self.agent_logs.append({
                "timestamp": datetime.now().isoformat(),
                "action": "compaction",
                "details": metadata
            })
            
            # Log: compaction applied (to run logs buffer)
            self._log_action("compaction", metadata, step_number=step_number)
            
            logger.info(f"Compaction applied: {metadata}")
    
    def _extract_final_answer(self, message: BetaMessage) -> str:
        """Extract and concatenate text from all content blocks in the message.
        
        Text blocks are collected only from the blocks appearing after the last
        tool use block. If no tool use block is present, all text blocks are returned.
        
        Args:
            message: The assistant message object
            
        Returns:
            Concatenated text from the relevant text blocks
        """
        if not message or not message.content:
            return ""

        start_index = 0
        # Find the index of the last tool_use block, if any
        for i, block in enumerate(message.content):
            if getattr(block, 'type', '') in ['server_tool_use', 'web_search_tool_result', 'tool_use', 'tool_result']:
                start_index = i + 1
        
        full_text = []
        for block in message.content[start_index:]:
            # Check if block is a text block (standard or beta)
            if hasattr(block, 'text') and getattr(block, 'type', '') == 'text':
                full_text.append(block.text)
        
        return "".join(full_text)

    async def _generate_final_summary(
        self,
        queue: Optional[asyncio.Queue] = None,
        formatter: Optional[FormatterType] = None
    ) -> AgentResult:
        """Generate a final summary when max steps is reached.
        
        This method is called when the agent exhausts max_steps without reaching
        a natural completion. It makes one final API call with tools disabled
        and a modified system prompt requesting a summary.
        
        Args:
            queue: Optional async queue to stream formatted output chunks
            formatter: Formatter to use for stream output
        
        Returns:
            AgentResult with the final summary response
        """
        logger.info("Generating final summary due to max_steps reached")
        
        # Apply compaction before final call
        if self.compactor:
            self._apply_compaction(step_number=self.max_steps)
        
        # Create temporary system prompt for summary
        summary_system_prompt = (
            f"{self.system_prompt}\n\n"
            "IMPORTANT: You have reached the maximum number of steps. "
            "Please provide a final summary or response based on the work completed so far."
        )
        
        # Prepare request parameters (tools disabled) with prioritized cache_control applied
        messages, system = self._apply_cache_control(self.messages, summary_system_prompt)
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "system": system or summary_system_prompt,  # Fallback if caching returned None
        }
        
        # Add thinking tokens if configured
        if self.thinking_tokens > 0:
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_tokens
            }
        
        # Add beta headers if configured
        if self.beta_headers:
            request_params["betas"] = self.beta_headers
        
        # Add container if present
        if self.container_id:
            request_params["container"] = self.container_id
        
        # Log the final summary attempt
        self.agent_logs.append({
            "timestamp": datetime.now().isoformat(),
            "action": "max_steps_summary",
            "details": {
                "reason": "max_steps_reached",
                "max_steps": self.max_steps,
                "tools_disabled": True
            }
        })
        
        # Make API call with retry logic (stream final summary to user)
        accumulated_message = await anthropic_stream_with_backoff(
            client=self.client,
            request_params=request_params,
            queue=queue,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            formatter=formatter if formatter is not None else self.formatter,
        )
        
        # Track token usage from API response
        input_tokens = accumulated_message.usage.input_tokens
        output_tokens = accumulated_message.usage.output_tokens
        self._last_known_input_tokens = input_tokens + output_tokens # output becomes input in next call
        self._last_known_output_tokens = output_tokens
        self._token_usage_history.append({
            "step": self.max_steps,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": accumulated_message.usage.cache_creation_input_tokens,
            "cache_read_input_tokens": accumulated_message.usage.cache_read_input_tokens,
        })
        
        # Add assistant's final summary to live context and conversation history
        # assistant_message = {
        #     "role": accumulated_message.role,
        #     "content": accumulated_message.content
        # }
        assistant_message = accumulated_message.model_dump(
                mode="json",
                include=["role", "content"],
                exclude_unset=True,
                exclude=getattr(accumulated_message, "__api_exclude__", None),
                warnings=False
            )
        self.messages.append(assistant_message)
        self.conversation_history.append(assistant_message)
        
        # Update container ID if present
        if accumulated_message.container is not None:
            self.container_id = accumulated_message.container.id

        # Update memory store with final conversation results
        if self.memory_store:
            memory_metadata = self.memory_store.update(
                messages=self.messages,
                conversation_history=self.conversation_history,
                tools=self.tool_schemas,
                model=self.model
            )
            self.agent_logs.append({
                "timestamp": datetime.now().isoformat(),
                "action": "memory_update",
                "details": memory_metadata
            })
            logger.info(f"Memory updated after final summary: {memory_metadata}")
        
        # Build AgentResult (generated_files populated from file_registry below)
        result = AgentResult(
            final_message=accumulated_message,
            final_answer=self._extract_final_answer(accumulated_message),
            conversation_history=self.conversation_history.copy(),
            stop_reason=accumulated_message.stop_reason,
            model=accumulated_message.model,
            usage=accumulated_message.usage,
            container_id=self.container_id,
            total_steps=self.max_steps,
            agent_logs=self.agent_logs.copy(),
            generated_files=None  # Will be populated from file_registry below
        )
        
        # Log: final summary generated
        self._log_action("final_summary_generation", {
            "stop_reason": accumulated_message.stop_reason,
            "total_steps": self.max_steps,
            "total_input_tokens": accumulated_message.usage.input_tokens,
            "total_output_tokens": accumulated_message.usage.output_tokens,
        }, step_number=self.max_steps)

        # Finalize file processing (extract, store, stream)
        await self._finalize_file_processing(queue)

        # Get updated metadata
        all_files_metadata: list[dict[str, Any]] = list(self.file_registry.values())

        # Update result with all known files for this agent
        result.generated_files = all_files_metadata

        # Save run data asynchronously with complete file registry snapshot
        self._save_run_data_async(result, all_files_metadata)
        
        # Return AgentResult with final summary
        return result
    
    def compact_messages(self) -> dict[str, Any]:
        """Explicitly trigger compaction on message history.
        
        This method can be called manually to request compaction of the message
        history. The compactor will decide whether to actually compact based on
        its internal logic and threshold.
        
        Returns:
            Dictionary containing metadata about the compaction operation,
            including messages_removed, tool_results_modified, and tokens_saved.
            Returns error dict if no compactor is configured.
        """
        if not self.compactor:
            return {"error": "No compactor configured"}
        
        estimated_tokens = self._last_known_input_tokens
        compacted, metadata = self.compactor.compact(self.messages, self.model, estimated_tokens)
        
        # Update messages if compaction was applied
        if metadata.get("compaction_applied", False):
            self.messages = compacted
            self.agent_logs.append({
                "timestamp": datetime.now().isoformat(),
                "action": "manual_compaction",
                "details": metadata
            })
            logger.info(f"Manual compaction applied: {metadata}")
        
        return metadata
    
    def _get_estimated_tokens(self) -> int:
        """Get estimated token count for current context.
        
        Returns last known input tokens from API response, or 0 if no API call made yet.
        This reflects the actual context size sent to the API, accounting for
        compaction and memory injection.
        
        Returns:
            Estimated token count for current context
        """
        return self._last_known_input_tokens
    
    def _filter_messages_for_token_count(
        self,
        messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter messages to remove content types unsupported by token counting API.
        
        The count_tokens endpoint doesn't support URL-based document sources
        (especially PDFs). This method removes such blocks to prevent 400 errors.
        
        Args:
            messages: List of message dicts to filter
            
        Returns:
            Filtered messages list with unsupported content removed
        """
        filtered: list[dict[str, Any]] = []
        
        for msg in messages:
            content = msg.get("content")
            
            # Pass through messages without list content unchanged
            if not isinstance(content, list):
                filtered.append(msg)
                continue
            
            # Filter content blocks
            filtered_content: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    filtered_content.append(block)
                    continue
                
                # Check for document blocks with URL sources
                if block.get("type") == "document":
                    source = block.get("source", {})
                    if isinstance(source, dict) and source.get("type") == "url":
                        # Skip URL-based documents (not supported by count_tokens)
                        continue
                
                filtered_content.append(block)
            
            # Only include message if it still has content
            if filtered_content:
                filtered.append({"role": msg.get("role"), "content": filtered_content})
        
        return filtered
    
    async def _count_tokens_api(
        self,
        *,
        messages: list[dict[str, Any]],
        system: Optional[str],
        tools: Optional[list[dict[str, Any]]],
        thinking: Optional[dict[str, Any]],
        betas: Optional[list[str]],
        container: Optional[str],
    ) -> Optional[int]:
        """Best-effort token counting via Anthropic count_tokens API.
        
        .. deprecated::
            This method is deprecated and will be removed in a future version.
            Use :meth:`_estimate_tokens` instead.
        
        Returns:
            input_tokens from the API response, or None if the call fails.
        """
        warnings.warn(
            "_count_tokens_api is deprecated and will be removed in a future version. "
            "Use _estimate_tokens instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Filter tools to only those supported by the count_tokens API.
        # The endpoint currently only accepts a limited set of server tool
        # types; passing unsupported types like "code_execution_20250825"
        # results in 400 errors. We keep client tools (which typically
        # have no "type" field) and allow only known server tool tags.
        allowed_server_tool_types = {
            "bash_20250124",
            "custom",
            "text_editor_20250124",
            "text_editor_20250429",
            "text_editor_20250728",
            "web_search_20250305",
        }

        filtered_tools: list[dict[str, Any]] = []
        if tools:
            for tool in tools:
                tool_type = tool.get("type")
                # Client tools (no explicit type) are always allowed.
                if tool_type is None or tool_type in allowed_server_tool_types:
                    filtered_tools.append(tool)

        # Filter messages to remove unsupported content types (e.g., URL PDF sources)
        filtered_messages = self._filter_messages_for_token_count(messages)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": filtered_messages,
        }
        if system:
            params["system"] = system
        if filtered_tools:
            params["tools"] = filtered_tools
        if thinking:
            params["thinking"] = thinking
            
        # Pass betas as extra headers since the count_tokens API doesn't support the 'betas' parameter directly
        if betas:
            params["extra_headers"] = {"anthropic-beta": ",".join(betas)}
            
        # if container:
        #     params["container"] = container   # This is not supported by the count_tokens API
        
        logger.debug("Anthropic count_tokens params: %s", params)
        try:
            response = await self.client.messages.count_tokens(**params)
            return getattr(response, "input_tokens", None)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Token count API call failed: {e}")
            return None
    
    async def _estimate_tokens(
        self,
        *,
        messages: Optional[list[dict[str, Any]]] = None,
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[dict[str, Any]] = None,  # kept for signature compatibility
        betas: Optional[list[str]] = None,  # kept for signature compatibility
        container: Optional[str] = None,  # kept for signature compatibility
    ) -> int:
        """Heuristically estimate token count for the given delta/context.
        
        Uses a simple character-count heuristic so it can be used cheaply on
        small deltas like tool_result messages, while sharing I/O shape with
        _count_tokens_api.
        """
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
                                text_parts.append(json.dumps(block, separators=(",", ":"), ensure_ascii=False))
                            except TypeError:
                                text_parts.append(str(block))
                elif isinstance(content, dict):
                    try:
                        text_parts.append(json.dumps(content, separators=(",", ":"), ensure_ascii=False))
                    except TypeError:
                        text_parts.append(str(content))
        
        full_text = " ".join(text_parts)
        approx_tokens = max(0, len(full_text) // 4)
        return approx_tokens
    
    def _on_persistence_failure(
        self,
        exception: Exception,
        metadata: dict[str, Any]
    ) -> None:
        """Hook called when persistence fails after all retries.
        
        Override this method in subclasses to implement custom failure handling
        (e.g., metrics reporting, alerting, fallback storage).
        
        Args:
            exception: The exception that caused the failure
            metadata: Detailed information about the failure including:
                - run_id: Run identifier
                - agent_uuid: Agent session identifier
                - retry_count: Number of retries attempted
                - operation_type: Type of operation that failed
                - timestamp: When the failure occurred
                - error_message: String representation of the error
                - error_type: Exception class name
        """
        # Default implementation: log only (already done in _save_run_data_with_retry)
        pass

    def __str__(self) -> str:
        """Return the current configuration and runtime state for this agent."""
        tool_names = [
            schema.get("name", "<unnamed>")
            for schema in self.tool_schemas
        ]
        config_snapshot = {
            "agent_uuid": self.agent_uuid,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "max_steps": self.max_steps,
            "thinking_tokens": self.thinking_tokens,
            "max_tokens": self.max_tokens,
            "stream_meta_history_and_tool_results": self.stream_meta_history_and_tool_results,
            "beta_headers": self.beta_headers,
            "container_id": self.container_id,
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
            "db_backend": self.db_backend.__class__.__name__,
            "tools": tool_names,
        }
        return json.dumps(config_snapshot, indent=2)
    
    def _restore_state_from_config(self, config: dict[str, Any]) -> None:
        """Restore agent state from a loaded configuration dict.
        
        Called by initialize() after loading config from database.
        Updates agent instance variables with persisted state.
        
        Args:
            config: Configuration dict loaded from database
        """
        # Restore messages and container state
        self.messages = config.get("messages", [])
        self.container_id = config.get("container_id")
        self.file_registry = config.get("file_registry", {})
        self._last_known_input_tokens = config.get("last_known_input_tokens", 0)
        self._last_known_output_tokens = config.get("last_known_output_tokens", 0)
        
        # Frontend tool relay state (for resume after browser execution)
        self._pending_frontend_tools = config.get("pending_frontend_tools", [])
        self._pending_backend_results = config.get("pending_backend_results", [])
        self._awaiting_frontend_tools = config.get("awaiting_frontend_tools", False)
        self._current_step = config.get("current_step", 0)
        
        # Conversation history - only load when resuming due to frontend tools
        # Regular agent resumptions should start fresh with a new conversation_history.
        # NOTE: This is the per-run history used to populate AgentResult, NOT the
        # conversation_history TABLE which stores completed runs across user turns.
        # When frontend tools pause the agent, we preserve this so AgentResult
        # contains the complete run history after continuation.
        if self._awaiting_frontend_tools:
            self._loaded_conversation_history = config.get("conversation_history", [])
        else:
            self._loaded_conversation_history = []
    
    def _log_action(
        self,
        action_type: str,
        action_data: dict,
        step_number: int = 0,
        messages_snapshot: list[dict] | None = None,
        duration_ms: int | None = None
    ) -> None:
        """Add an action log entry to the buffer.
        
        Logs are batched and written to database at end of run.
        
        Args:
            action_type: Type of action (e.g., "run_started", "api_call", "tool_execution")
            action_data: Action-specific data
            step_number: Step number in the agent loop
            messages_snapshot: Optional snapshot of messages at this point
            duration_ms: Optional duration in milliseconds
        """
        log_entry = {
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
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _save_agent_config(self) -> None:
        """
        Save current agent configuration to database.
        
        This method saves the resumable agent configuration to the database.
        Preserves created_at from existing config and increments total_runs.
        """
        # Load existing config to preserve created_at and increment total_runs
        existing_config = await self.db_backend.load_agent_config(self.agent_uuid)
        existing_config = existing_config or {}
        
        config = {
            "agent_uuid": self.agent_uuid,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "max_steps": self.max_steps,
            "thinking_tokens": self.thinking_tokens,
            "max_tokens": self.max_tokens,
            "container_id": self.container_id,
            "messages": self.messages,
            "tool_schemas": self.tool_schemas,
            "tool_names": [t["name"] for t in self.tool_schemas] if self.tool_schemas else [],
            "beta_headers": self.beta_headers,
            "server_tools": self.server_tools,
            "formatter": self.formatter,
            "stream_meta_history_and_tool_results": self.stream_meta_history_and_tool_results,
            "file_registry": self.file_registry,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "api_kwargs": self.api_kwargs,
            "last_known_input_tokens": self._last_known_input_tokens,
            "last_known_output_tokens": self._last_known_output_tokens,
            # Frontend tool relay state (for resume after browser execution)
            "pending_frontend_tools": self._pending_frontend_tools,
            "pending_backend_results": self._pending_backend_results,
            "awaiting_frontend_tools": self._awaiting_frontend_tools,
            "current_step": self._current_step,
            # NOTE: This conversation_history is the per-run history used to populate AgentResult.
            # When frontend tools cause a pause, this preserves the current run's history so it
            # can be returned in the AgentResult after continuation. This is distinct from the
            # conversation_history TABLE which stores completed runs across multiple user turns.
            "conversation_history": getattr(self, "conversation_history", []),
            # Preserve created_at from existing config, or set to now (as datetime)
            "created_at": (
                datetime.fromisoformat(existing_config["created_at"].replace("Z", "+00:00"))
                if isinstance(existing_config.get("created_at"), str)
                else (existing_config.get("created_at") or datetime.now())
            ),
            "updated_at": datetime.now(),
            "last_run_at": getattr(self, "_run_start_time", None),
            # Increment total_runs counter
            "total_runs": existing_config.get("total_runs", 0) + 1,
        }
        
        # Add component configs
        if self.compactor:
            config["compactor_type"] = self.compactor.__class__.__name__
        
        if self.memory_store:
            config["memory_store_type"] = self.memory_store.__class__.__name__
        
        await self.db_backend.save_agent_config(config)
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _save_conversation_entry(
        self,
        result: AgentResult,
        files_metadata: list[dict]
    ) -> None:
        """Save conversation history entry to database."""
        
        # Extract user message (first user message in conversation_history)
        user_message = ""
        for msg in result.conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_message = content
                elif isinstance(content, list):
                    # Extract text from content blocks
                    user_message = " ".join(
                        block.get("text", "") 
                        for block in content 
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                break
        
        # Extract final response
        final_response = ""
        if result.final_message and result.final_message.content:
            for block in result.final_message.content:
                if hasattr(block, 'text'):
                    final_response = block.text
                    break
        
        conversation = {
            "conversation_id": str(uuid.uuid4()),
            "agent_uuid": self.agent_uuid,
            "run_id": self._run_id,
            "started_at": self._run_start_time,
            "completed_at": datetime.now(),
            "user_message": user_message,
            "final_response": final_response,
            "messages": result.conversation_history,
            "stop_reason": result.stop_reason,
            "total_steps": result.total_steps,
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cache_creation_input_tokens": result.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": result.usage.cache_read_input_tokens,
            },
            # Persist full file metadata snapshot associated with this run
            "generated_files": files_metadata,
            "created_at": datetime.now(),
        }
        
        await self.db_backend.save_conversation_history(conversation)
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _save_run_logs(self) -> None:
        """Persist batched agent run logs to the database."""
        await self.db_backend.save_agent_run_logs(
            self.agent_uuid,
            self._run_id,
            self._run_logs_buffer
        )
    
    def _extract_first_user_message(self) -> str | None:
        """Extract the first user message from conversation history."""
        for msg in self.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    # Handle content blocks
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            return block.get("text", "")
        return None
    
    async def _generate_and_save_title(self, user_message: str) -> None:
        """Background task to generate and persist conversation title.
        
        Only runs for new conversations (no existing title).
        """
        if not self.db_backend:
            return
        
        # Check if title already exists
        existing_config = await self.db_backend.load_agent_config(self.agent_uuid)
        if existing_config and existing_config.get("title"):
            return  # Don't overwrite existing title
        
        title = await generate_title(user_message)
        
        try:
            await self.db_backend.update_agent_title(self.agent_uuid, title)
            logger.debug(f"Generated title for {self.agent_uuid}: {title}")
        except Exception as e:
            logger.error(f"Failed to save title for {self.agent_uuid}: {e}")
    
    async def _save_run_data_with_retry(
        self,
        result: AgentResult,
        files_metadata: list[dict],
    ) -> None:
        """Orchestrate saving run data using per-operation retry decorators.
        
        Each persistence operation (_save_agent_config, _save_conversation_entry,
        _save_run_logs) has its own retry_with_backoff decorator. Partial success
        is acceptable: failures in one operation do not prevent the others from
        attempting to save. On ultimate failure of any operation, the
        _on_persistence_failure hook is invoked with operation-specific metadata.
        """
        operations: list[tuple[str, Callable[[], Awaitable[None]]]] = [
            ("agent_config", self._save_agent_config),
            ("conversation_history", lambda: self._save_conversation_entry(result, files_metadata)),
            ("agent_runs", self._save_run_logs),
        ]
        
        for operation_type, op in operations:
            try:
                await op()
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"Failed to persist {operation_type} for run {self._run_id}: {e}",
                    exc_info=True,
                )
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
        
        # Schedule title generation as background task
        # The _generate_and_save_title method checks if title already exists before generating
        if self.db_backend:
            user_message = self._extract_first_user_message()
            if user_message:
                title_task = asyncio.create_task(
                    self._generate_and_save_title(user_message)
                )
                self._background_tasks.add(title_task)
                title_task.add_done_callback(self._background_tasks.discard)
    
    def _save_run_data_async(
        self,
        result: AgentResult,
        files_metadata: list[dict]
    ) -> None:
        """Save all run data asynchronously with retry logic.
        
        Creates background tasks for:
        - agent_config update
        - conversation_history entry
        - agent_runs logs (batched)
        """
        # Create background task
        task = asyncio.create_task(
            self._save_run_data_with_retry(result, files_metadata)
        )
        
        # Store task reference to prevent garbage collection
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def drain_background_tasks(self, timeout: float = 30.0) -> dict[str, Any]:
        """Wait for all background persistence tasks to complete.
        
        This method should be called before shutting down the agent or process
        to ensure all run data is persisted. Tasks that don't complete within
        the timeout will be logged as warnings.
        
        Args:
            timeout: Maximum time in seconds to wait for tasks (default: 30.0)
            
        Returns:
            Dictionary with completion statistics:
                - total_tasks: Number of tasks tracked
                - completed: Number of tasks that completed
                - timed_out: Number of tasks that didn't complete in time
                - task_ids: List of run_ids for incomplete tasks (if any)
        """
        if not self._background_tasks:
            return {
                "total_tasks": 0,
                "completed": 0,
                "timed_out": 0,
                "task_ids": []
            }
        
        total_tasks = len(self._background_tasks)
        logger.info(f"Draining {total_tasks} background task(s) with {timeout}s timeout")
        
        try:
            # Wait for all tasks with timeout
            await asyncio.wait_for(
                asyncio.gather(*self._background_tasks, return_exceptions=True),
                timeout=timeout
            )
            completed = total_tasks
            timed_out = 0
            incomplete_ids = []
            
        except asyncio.TimeoutError:
            # Some tasks didn't complete in time
            incomplete_tasks = [t for t in self._background_tasks if not t.done()]
            completed = total_tasks - len(incomplete_tasks)
            timed_out = len(incomplete_tasks)
            
            logger.warning(
                f"Timeout after {timeout}s: {completed}/{total_tasks} tasks completed, "
                f"{timed_out} tasks still pending"
            )
            
            # Try to extract run_ids from incomplete tasks (best effort)
            incomplete_ids = [f"task_{id(t)}" for t in incomplete_tasks]
        
        return {
            "total_tasks": total_tasks,
            "completed": completed,
            "timed_out": timed_out,
            "task_ids": incomplete_ids
        }
    
    def _upsert_file_registry_entry(
        self,
        *,
        file_id: str,
        filename: str,
        step: int,
        raw_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create or update a single file entry in the in-memory registry.
        
        Ensures there is at most one entry per file_id, while preserving
        existing metadata such as storage paths from previous runs.
        """
        now = datetime.now().isoformat()
        existing: dict[str, Any] = self.file_registry.get(file_id, {})

        # Start from existing metadata to preserve backend-specific fields
        updated: dict[str, Any] = dict(existing)
        updated["file_id"] = file_id
        updated["filename"] = filename
        updated.setdefault("first_seen_step", step)
        updated.setdefault("created_at", now)
        updated["last_seen_step"] = step
        updated["updated_at"] = now

        if raw_metadata is not None:
            # Store lightweight raw metadata snapshot under a dedicated key
            updated.setdefault("raw", raw_metadata)

        self.file_registry[file_id] = updated

    def _register_files_from_message(self, message: BetaMessage | dict[str, Any], step: int) -> None:
        """Scan a message or content structure and register any file references.
        
        This is called incrementally for each assistant response and for
        tool-result messages, so that file_ids are discovered per-step
        instead of by scanning self.messages at the end of a run.
        """
        file_ids = self.extract_file_ids(message)
        
        for file_id in file_ids:
            self._upsert_file_registry_entry(
                file_id=file_id,
                filename=f"file_{file_id}",  # Placeholder until filename is available
                step=step,
            )
            
    async def _finalize_file_processing(self, queue: Optional[asyncio.Queue] = None) -> None:
        """
        Finalize file processing at the end of a run.
        1. Extract all file IDs from conversation history.
        2. Process (download/store) files via backend.
        3. Stream file metadata to the client.
        """
        # 1. Extract and register all files from history
        # We use the final step count for simplicity, or 0 if unknown
        current_step = getattr(self, "total_steps", 0) # total_steps might not be set on self yet
        # Actually run() sets 'step' variable. We can just pass 0 or use a counter if we iterate.
        # Since we are doing this at the end, we can just scan everything.
        
        for i, message in enumerate(self.conversation_history):
            # rough step approximation or just use 0. 
            # ideally we would know the step for each message, but history is flat list.
            self._register_files_from_message(message, step=0)

        # 2. Process files via backend (download & store)
        if self.file_backend:
            await self._process_generated_files(step=0)

        # 3. Stream metadata
        all_files_metadata: list[dict[str, Any]] = list(self.file_registry.values())
        if queue and all_files_metadata:
            await self._stream_file_metadata(queue, all_files_metadata)

    async def _download_file(self, file_id: str) -> tuple[FileMetadata, bytes]:
        """Download file content from Anthropic Files API.
        
        Args:
            file_id: Anthropic's file identifier
            
        Returns:
            File content as bytes
            
        Raises:
            Exception: If download fails
        """
        try:
            # Download file using Files API
            response = await self.client.beta.files.download(file_id)
            file_content = await response.read()
            file_metadata = await self.client.beta.files.retrieve_metadata(file_id)
            return file_metadata, file_content
        
        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}", exc_info=True)
            raise
    
    def extract_file_ids(self, message: BetaMessage | dict[str, Any]) -> list[str]:
        file_ids = []
        
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
            
        if not content:
            return file_ids
            
        # Ensure content is iterable
        if not isinstance(content, list):
            logger.warning(f"Message content is not a list: {type(content)}")
            return file_ids

        for item in content:
            # Handle both object and dict access for item
            if isinstance(item, dict):
                item_type = item.get("type")
                item_content = item.get("content")
            else:
                item_type = getattr(item, "type", "")
                item_content = getattr(item, "content", None)
            
            # logger.debug(f"Scanning content item type: {item_type}")

            # Check for both specific beta type and generic tool_result that might contain bash result
            if item_type == 'bash_code_execution_tool_result' or item_type == 'tool_result':
                # content_item is the nested bash_code_execution_result
                
                if not item_content:
                    continue

                if isinstance(item_content, dict):
                    inner_type = item_content.get("type")
                    files = item_content.get("content", [])
                elif hasattr(item_content, "type"):
                    inner_type = getattr(item_content, "type", "")
                    files = getattr(item_content, "content", [])
                else:
                    # Content is likely a string or list, not the expected nested structure
                    continue
                
                # logger.debug(f"Found potentially relevant result with inner type: {inner_type}")

                if inner_type == 'bash_code_execution_result':
                    if isinstance(files, list):
                        for file in files:
                            # Handle both object and dict access for file
                            if isinstance(file, dict):
                                file_id = file.get("file_id")
                            else:
                                file_id = getattr(file, "file_id", None)
                                
                            if file_id:
                                logger.info(f"Found file_id: {file_id}")
                                file_ids.append(file_id)
        return file_ids

    async def _process_generated_files(self, step: int) -> list[dict]:
        """Download, store, and track all files via the configured file backend.
        
        Args:
            step: Current agent step number
            
        Returns:
            List of file metadata dicts for all successfully processed files
        """
        if not self.file_backend:
            return []

        files_metadata: list[dict[str, Any]] = []

        if not self.file_registry:
            return []

        logger.info(f"Processing {len(self.file_registry)} file(s) via file backend")

        for file_id, registry_entry in self.file_registry.items():
            filename = registry_entry.get("filename") or str(file_id)

            try:
                # Download file content from Anthropic Files API
                logger.info(f"Downloading file {filename} ({file_id}) for backend storage")
                # content is a tuple (FileMetadata, bytes) - we need index 1 for content bytes
                file_metadata_api, content_bytes = await self._download_file(file_id)
                # Update filename from metadata if available
                if hasattr(file_metadata_api, 'filename') and file_metadata_api.filename:
                    filename = file_metadata_api.filename
                    # Update registry entry with new filename
                    registry_entry["filename"] = filename

                # Decide whether to store or update based on existing backend metadata
                has_backend_metadata = "storage_backend" in registry_entry
                if has_backend_metadata:
                    metadata = self.file_backend.update(
                        file_id=file_id,
                        filename=filename,
                        content=content_bytes,
                        existing_metadata=registry_entry,
                        agent_uuid=self.agent_uuid,
                    )
                else:
                    metadata = self.file_backend.store(
                        file_id=file_id,
                        filename=filename,
                        content=content_bytes,
                        agent_uuid=self.agent_uuid,
                    )

                # Attach step information for this processing pass
                metadata["processed_at_step"] = step

                # Merge backend metadata back into the registry entry
                merged: dict[str, Any] = dict(registry_entry)
                merged.update(metadata)
                self.file_registry[file_id] = merged

                files_metadata.append(merged)

                logger.info(f"Successfully processed file {filename} ({file_id}) via backend")

            except Exception as e:
                # Log error but continue processing other files
                logger.warning(
                    f"Failed to process file {filename} ({file_id}) via backend: {e}",
                    exc_info=True,
                )
                continue

        return files_metadata
    
    async def _stream_file_metadata(
        self,
        queue: asyncio.Queue,
        metadata: list[dict]
    ) -> None:
        """Stream file metadata to queue in meta tag format.
        
        Args:
            queue: Async queue to send formatted output
            metadata: List of file metadata dicts to stream
        """
        if not metadata:
            return
        
        # Format as JSON inside custom content block
        files_json = json.dumps({"files": metadata}, indent=2)
        meta_tag = f'<content-block-meta_files><![CDATA[{files_json}]]></content-block-meta_files>'
        
        await queue.put(meta_tag)
