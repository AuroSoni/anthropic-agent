# Move to anthropic_agent/core/agent.py
import asyncio
import json
import html
import anthropic
import logging
import uuid
from datetime import datetime
from typing import Optional, Callable, Any, Awaitable
from collections.abc import Mapping, Sequence
from anthropic.types.beta import BetaMessage, FileMetadata

from .types import AgentResult
from .retry import anthropic_stream_with_backoff, retry_with_backoff
from ..tools.base import ToolRegistry
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
        server_tools: list[dict[str, Any]] | None = None,
        beta_headers: list[str] | None = None,
        container_id: str | None = None,
        messages: list[dict] | None = None,    # TODO: Add Message Type.
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        formatter: FormatterType | None = None,
        compactor: CompactorType | Compactor | None = None,
        memory_store: MemoryStoreType | MemoryStore | None = None,
        final_answer_check: Optional[Callable[[dict], tuple[bool, str]]] = None,
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
            tools: List of functions decorated with @tool (default: None)
            beta_headers: Beta feature headers for Anthropic API (default: None)
            container_id: Container ID for multi-turn conversations (default: None)
            messages: Initial message history (default: None)
            max_retries: Maximum retry attempts for API calls (default: 5)
            base_delay: Base delay in seconds for exponential backoff (default: 5.0)
            formatter: Default formatter for stream output ("xml" or "raw", default: "xml")
            compactor: Either a compactor name ("tool_result_removal", "none") or a pre-configured
                Compactor instance. If a string is provided, a compactor is created with default
                settings (no threshold). For custom threshold, create and pass a Compactor instance.
                Example: get_compactor("tool_result_removal", threshold=50000)
            memory_store: Either a memory store name ("placeholder", "none") or a pre-configured
                MemoryStore instance. Memory stores retrieve relevant context for injection and
                integrate with the compaction lifecycle to preserve important information.
            final_answer_check: Optional callable that validates the final assistant message format.
                Signature: check(assistant_message: dict) -> (success: bool, error_message: str)
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

        # Load persisted configuration if agent_uuid provided
        db_config: dict[str, Any] = {}
        if agent_uuid:
            self.agent_uuid = agent_uuid
            if not self.db_backend:
                raise ValueError("Database backend is required to load state from database")
            db_config = self._load_state_from_db()
        else:
            # Agent UUID for session tracking
            self.agent_uuid = agent_uuid or str(uuid.uuid4())
        
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
        
        # Messages and container_id are strictly from DB for resumed agents,
        # and from constructor for new agents.
        if agent_uuid:
            self.messages = db_config.get("messages", [])
            self.container_id = db_config.get("container_id")
            self.file_registry = db_config.get("file_registry", {})
            self._last_known_input_tokens = db_config.get("last_known_input_tokens", 0)
            self._last_known_output_tokens = db_config.get("last_known_output_tokens", 0)
        else:
            self.messages = messages or []
            self.container_id = container_id
            self.file_registry: dict[str, dict] = {}
            self._last_known_input_tokens = 0
            self._last_known_output_tokens = 0
        
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
        
        # Initialize the Anthropic async client for proper async streaming
        self.client = anthropic.AsyncAnthropic()
    
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
        
        # Retrieve and inject semantic memories
        if self.memory_store:
            self.messages = self.memory_store.retrieve(
                tools=self.tool_schemas,
                user_message=user_message,
                messages=self.messages,
                model=self.model
            )
            # Note: Memory-injected messages are NOT added to conversation_history
        
        # Initialize token estimate for current context using API-based counting.
        # Combine client tools (tool_schemas) and server tools for token counting.
        combined_tools: list[dict[str, Any]] = []
        if self.tool_schemas:
            combined_tools.extend(self.tool_schemas)
        if self.server_tools:
            combined_tools.extend(self.server_tools)
        api_token_count: Optional[int] = await self._count_tokens_api(
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
        if api_token_count is not None:
            self._last_known_input_tokens = api_token_count
        
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
                    # Build tool results message
                    # TODO: Execute tool calls parallelly.
                    tool_results = []
                    for tool_call in tool_calls:
                        is_error = False
                        result_content = None
                        try:
                            # Execute the tool (support both sync and async executors)
                            result = self.execute_tool_call(tool_call.name, tool_call.input)
                            # Check if result is a coroutine (async function)
                            if asyncio.iscoroutine(result):
                                result = await result
                            result_content = result
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": result
                            })
                        except Exception as e:
                            # Handle tool execution errors
                            is_error = True
                            result_content = f"Error executing tool: {str(e)}"
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": result_content,
                                "is_error": True
                            })
                        
                        # Stream tool result to queue in XML format
                        if queue is not None:
                            tool_use_id = html.escape(str(tool_call.id), quote=True)
                            tool_name = html.escape(str(tool_call.name), quote=True)
                            # Serialize result content to string
                            if result_content is None:
                                content_str = ""
                            elif isinstance(result_content, str):
                                content_str = result_content
                            else:
                                content_str = json.dumps(result_content, default=str)
                            await queue.put(
                                f'<content-block-tool_result id="{tool_use_id}" name="{tool_name}"><![CDATA[{content_str}]]></content-block-tool_result>'
                            )
                        
                        # Log: tool execution
                        self._log_action("tool_execution", {
                            "tool_name": tool_call.name,
                            "tool_use_id": tool_call.id,
                            "success": not is_error,
                        }, step_number=step)
                    
                    # Add tool results to live context and conversation history
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
                    success, error_message = self.final_answer_check(assistant_message)
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
    
    def _prepare_request_params(self) -> dict:
        # Prepare request parameters
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
        }
        
        # Add optional parameters
        if self.system_prompt:
            request_params["system"] = self.system_prompt
        
        if self.thinking_tokens > 0:
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_tokens
            }
        
        # Merge client tools (tool_schemas) and Anthropic server tools for this call.
        combined_tools: list[dict[str, Any]] = []
        if self.tool_schemas:
            combined_tools.extend(self.tool_schemas)
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
    
    def execute_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """Execute a registered tool function through the ToolRegistry.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of input parameters for the tool
            
        Returns:
            String result from the tool execution
        """
        if not self.tool_registry:
            return "No tools have been registered for this agent."
        
        return self.tool_registry.execute(tool_name, tool_input)
    
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
        
        # Prepare request parameters (tools disabled)
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "system": summary_system_prompt,
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
        
        Returns:
            input_tokens from the API response, or None if the call fails.
        """
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

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
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
    
    def _load_state_from_db(self) -> dict[str, Any]:
        """Synchronously load agent configuration from database.
        
        Called during __init__ if agent_uuid is provided.
        
        Returns:
            Configuration dict loaded from the backend, or {} if not found or on error.
        """
        try:
            # Use asyncio.run since we're in __init__ (sync context)
            config = asyncio.run(self.db_backend.load_agent_config(self.agent_uuid))
            
            if config is None:
                # New agent - will be created on first run
                logger.info(f"No existing state for agent {self.agent_uuid}, creating new")
                return {}
            
            logger.info(
                f"Loaded state for agent {self.agent_uuid}: "
                f"{len(config.get('messages', []))} messages, "
                f"container_id={config.get('container_id')}"
            )
            return config
        except Exception as e:
            logger.error(f"Failed to load state for agent {self.agent_uuid}: {e}", exc_info=True)
            return {}
    
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
            # Preserve created_at from existing config, or set to now (as datetime)
            "created_at": (
                datetime.fromisoformat(existing_config["created_at"].replace("Z", "+00:00"))
                if isinstance(existing_config.get("created_at"), str)
                else (existing_config.get("created_at") or datetime.now())
            ),
            "updated_at": datetime.now(),
            "last_run_at": self._run_start_time if self._run_start_time else None,
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
        meta_tag = f'<content-block-meta_files>\n{files_json}\n</content-block-meta_files>'
        
        await queue.put(meta_tag)
