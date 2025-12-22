"""Base agent orchestrator for multi-provider support.

This module provides the BaseAgent class that handles all provider-agnostic
agent orchestration logic including tool execution, memory management,
compaction, persistence, and the main run loop.
"""

import asyncio
import json
import html
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Callable, Any, Awaitable

from .types import GenericUsage, GenericAgentResult
from .protocols import LLMClient, LLMResponse
from .retry import retry_with_backoff
from .compaction import CompactorType, get_compactor, Compactor
from ..tools.base import ToolRegistry
from ..memory import MemoryStoreType, get_memory_store, MemoryStore
from ..database import DBBackendType, get_db_backend, DatabaseBackend
from ..file_backends import FileBackendType, get_file_backend, FileStorageBackend

logger = logging.getLogger(__name__)


# Default configuration values
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that should help the user with their questions."
DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_TOKENS = 2048
DEFAULT_STREAM_META = False
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0


class BaseAgent(ABC):
    """Provider-agnostic agent orchestrator.
    
    BaseAgent handles all the common agent logic that is independent of the
    specific LLM provider being used:
    
    - Message management (append, history tracking)
    - Tool execution via ToolRegistry
    - Context compaction via Compactor
    - Memory retrieval/update via MemoryStore
    - State persistence via DatabaseBackend
    - File handling via FileBackend
    - Agent run loop orchestration
    
    Subclasses must implement provider-specific methods:
    - _stream_completion(): Stream from the LLM
    - _count_tokens(): Count tokens for context
    - _extract_final_answer(): Extract text from provider response
    - _process_response(): Handle provider-specific response processing
    
    Example:
        >>> from anthropic_agent.providers.anthropic import AnthropicClient
        >>> 
        >>> class MyAgent(BaseAgent):
        ...     def __init__(self, **kwargs):
        ...         client = AnthropicClient()
        ...         super().__init__(client=client, **kwargs)
        ...     # ... implement abstract methods
        >>> 
        >>> agent = MyAgent(model="claude-sonnet-4-5")
        >>> result = await agent.run("Hello!")
    """
    
    def __init__(
        self,
        client: LLMClient,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_steps: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stream_meta_history_and_tool_results: Optional[bool] = None,
        tools: list[Callable[..., Any]] | None = None,
        messages: list[dict] | None = None,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        compactor: CompactorType | Compactor | None = None,
        memory_store: MemoryStoreType | MemoryStore | None = None,
        final_answer_check: Optional[Callable[[str], tuple[bool, str]]] = None,
        agent_uuid: str | None = None,
        db_backend: DBBackendType | DatabaseBackend = "filesystem",
        file_backend: FileBackendType | FileStorageBackend | None = None,
        **api_kwargs: Any,
    ):
        """Initialize BaseAgent with configuration.
        
        Args:
            client: LLM client implementing the LLMClient protocol
            system_prompt: System prompt to guide the agent's behavior
            model: Model name/identifier for the LLM
            max_steps: Maximum conversation turns before stopping
            max_tokens: Maximum tokens in response
            stream_meta_history_and_tool_results: Include metadata in stream
            tools: List of functions decorated with @tool
            messages: Initial message history
            max_retries: Maximum retry attempts for API calls
            base_delay: Base delay for exponential backoff
            compactor: Compactor for context management
            memory_store: Memory store for semantic context
            final_answer_check: Validation function for final answers
            agent_uuid: Session UUID for resuming previous sessions
            db_backend: Database backend for persistence
            file_backend: File storage backend for generated files
            **api_kwargs: Additional provider-specific API arguments
        """
        self.client = client
        
        # Database backend for persistence
        if isinstance(db_backend, str):
            self.db_backend = get_db_backend(db_backend)
        else:
            self.db_backend = db_backend
        
        # Non-serializable params
        self.final_answer_check = final_answer_check
        
        # File storage backend
        if file_backend is None:
            self.file_backend: Optional[FileStorageBackend] = None
        elif isinstance(file_backend, str):
            self.file_backend = get_file_backend(file_backend)
        else:
            self.file_backend = file_backend
        
        # Tool registry
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
            db_config = self._load_state_from_db()
        else:
            self.agent_uuid = str(uuid.uuid4())
        
        # Resolve configuration from (constructor args, DB, defaults)
        self.system_prompt = (
            system_prompt if system_prompt is not None
            else db_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        )
        self.model = (
            model if model is not None
            else db_config.get("model", self._get_default_model())
        )
        self.max_steps = (
            max_steps if max_steps is not None
            else db_config.get("max_steps", DEFAULT_MAX_STEPS)
        )
        self.max_tokens = (
            max_tokens if max_tokens is not None
            else db_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        )
        
        # Messages and state
        if agent_uuid:
            self.messages = db_config.get("messages", [])
            self.file_registry = db_config.get("file_registry", {})
            self._last_known_input_tokens = db_config.get("last_known_input_tokens", 0)
            self._last_known_output_tokens = db_config.get("last_known_output_tokens", 0)
        else:
            self.messages = messages or []
            self.file_registry: dict[str, dict] = {}
            self._last_known_input_tokens = 0
            self._last_known_output_tokens = 0
        
        # Runtime configuration
        self.stream_meta_history_and_tool_results = (
            stream_meta_history_and_tool_results
            if stream_meta_history_and_tool_results is not None
            else db_config.get("stream_meta_history_and_tool_results", DEFAULT_STREAM_META)
        )
        self.max_retries = (
            max_retries if max_retries is not None
            else db_config.get("max_retries", DEFAULT_MAX_RETRIES)
        )
        self.base_delay = (
            base_delay if base_delay is not None
            else db_config.get("base_delay", DEFAULT_BASE_DELAY)
        )
        self.api_kwargs: dict[str, Any] = (
            api_kwargs if api_kwargs
            else db_config.get("api_kwargs", {})
        )
        
        # Compactor
        if compactor is None:
            self.compactor: Optional[Compactor] = None
        elif isinstance(compactor, str):
            self.compactor = get_compactor(compactor)
        else:
            self.compactor = compactor
        
        # Memory store
        if memory_store is None:
            self.memory_store: Optional[MemoryStore] = None
        elif isinstance(memory_store, str):
            self.memory_store = get_memory_store(memory_store)
        else:
            self.memory_store = memory_store
        
        # Run state tracking
        self._background_tasks: set = set()
        self._token_usage_history: list[dict] = []
    
    @abstractmethod
    def _get_default_model(self) -> str:
        """Get the default model for this provider.
        
        Returns:
            Default model identifier string
        """
        ...
    
    @abstractmethod
    async def _stream_completion(
        self,
        messages: list[dict],
        queue: Optional[asyncio.Queue] = None,
    ) -> Any:
        """Stream a completion from the LLM.
        
        Args:
            messages: List of message dictionaries
            queue: Optional async queue for streaming output
            
        Returns:
            Provider-specific response object
        """
        ...
    
    @abstractmethod
    async def _count_tokens(
        self,
        messages: list[dict],
    ) -> int | None:
        """Count tokens for the given messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Token count or None if counting fails
        """
        ...
    
    @abstractmethod
    def _extract_final_answer(self, response: Any) -> str:
        """Extract the final answer text from a response.
        
        Args:
            response: Provider-specific response object
            
        Returns:
            Extracted text answer
        """
        ...
    
    @abstractmethod
    def _response_to_message(self, response: Any) -> dict:
        """Convert a provider response to a message dict.
        
        Args:
            response: Provider-specific response object
            
        Returns:
            Message dictionary with 'role' and 'content'
        """
        ...
    
    @abstractmethod
    def _get_stop_reason(self, response: Any) -> str:
        """Get the stop reason from a response.
        
        Args:
            response: Provider-specific response object
            
        Returns:
            Stop reason string (e.g., "end_turn", "tool_use")
        """
        ...
    
    @abstractmethod
    def _get_usage(self, response: Any) -> GenericUsage:
        """Get token usage from a response.
        
        Args:
            response: Provider-specific response object
            
        Returns:
            GenericUsage object with token counts
        """
        ...
    
    @abstractmethod
    def _get_tool_calls(self, response: Any) -> list[dict]:
        """Extract tool calls from a response.
        
        Args:
            response: Provider-specific response object
            
        Returns:
            List of tool call dictionaries with 'id', 'name', 'input'
        """
        ...
    
    @abstractmethod
    def _build_tool_result_message(self, tool_results: list[dict]) -> dict:
        """Build a tool result message in provider format.
        
        Args:
            tool_results: List of tool result dictionaries
            
        Returns:
            Message dictionary with tool results
        """
        ...
    
    @abstractmethod
    def _to_generic_result(
        self,
        response: Any,
        conversation_history: list[dict],
        total_steps: int,
        agent_logs: list[dict],
        generated_files: list[dict] | None,
    ) -> GenericAgentResult:
        """Convert final response to GenericAgentResult.
        
        Args:
            response: Final provider-specific response
            conversation_history: Full message history
            total_steps: Number of steps taken
            agent_logs: Log entries
            generated_files: File metadata
            
        Returns:
            GenericAgentResult
        """
        ...
    
    async def run(
        self,
        prompt: str | list[dict],
        queue: Optional[asyncio.Queue] = None,
    ) -> GenericAgentResult:
        """Execute an agent run with the given user message.
        
        Args:
            prompt: User message (string or content blocks)
            queue: Optional async queue for streaming output
            
        Returns:
            GenericAgentResult containing the full execution context
        """
        # Initialize run tracking
        self._run_id = str(uuid.uuid4())
        self._run_start_time = datetime.now()
        self._run_logs_buffer: list[dict] = []
        self.conversation_history: list[dict] = []
        self.agent_logs: list[dict] = []
        self._token_usage_history = []
        
        # Build user message
        if isinstance(prompt, str):
            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        elif isinstance(prompt, list):
            user_message = {"role": "user", "content": prompt}
        else:
            user_message = prompt
        
        # Add to context and history
        self.messages.append(user_message)
        self.conversation_history.append(user_message)
        
        # Log run start
        self._log_action("run_started", {
            "user_message": prompt if isinstance(prompt, str) else str(prompt)[:200],
            "queue_present": queue is not None,
        }, step_number=0)
        
        # Emit meta_init if queue provided
        if queue is not None:
            meta_init = {
                "user_query": prompt if isinstance(prompt, str) else json.dumps(prompt),
                "message_history": self.conversation_history,
                "agent_uuid": self.agent_uuid,
                "model": self.model,
            }
            escaped_json = html.escape(json.dumps(meta_init), quote=True)
            await queue.put(f'<meta_init data="{escaped_json}"></meta_init>')
        
        # Memory retrieval
        if self.memory_store:
            self.messages = self.memory_store.retrieve(
                tools=self.tool_schemas,
                user_message=user_message,
                messages=self.messages,
                model=self.model
            )
        
        # Initial token count
        api_token_count = await self._count_tokens(self.messages)
        if api_token_count is not None:
            self._last_known_input_tokens = api_token_count
        
        step = 0
        while step < self.max_steps:
            step += 1
            
            # Apply compaction if configured
            if self.compactor:
                self._apply_compaction(step_number=step)
            
            # Stream completion
            response = await self._stream_completion(self.messages, queue)
            
            # Track usage
            usage = self._get_usage(response)
            self._last_known_input_tokens = usage.input_tokens + usage.output_tokens
            self._last_known_output_tokens = usage.output_tokens
            self._token_usage_history.append({
                "step": step,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            })
            
            # Log response
            self._log_action("api_response_received", {
                "stop_reason": self._get_stop_reason(response),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }, step_number=step)
            
            # Add assistant response to context
            assistant_message = self._response_to_message(response)
            self.messages.append(assistant_message)
            self.conversation_history.append(assistant_message)
            
            # Check for tool calls
            stop_reason = self._get_stop_reason(response)
            if stop_reason == "tool_use":
                tool_calls = self._get_tool_calls(response)
                
                if tool_calls:
                    tool_results = []
                    for tool_call in tool_calls:
                        is_error = False
                        result_content = None
                        
                        try:
                            result = self.execute_tool_call(
                                tool_call["name"],
                                tool_call["input"]
                            )
                            if asyncio.iscoroutine(result):
                                result = await result
                            result_content = result
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call["id"],
                                "content": result
                            })
                        except Exception as e:
                            is_error = True
                            result_content = f"Error executing tool: {str(e)}"
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call["id"],
                                "content": result_content,
                                "is_error": True
                            })
                        
                        # Stream tool result
                        if queue is not None:
                            tool_use_id = html.escape(str(tool_call["id"]), quote=True)
                            tool_name = html.escape(str(tool_call["name"]), quote=True)
                            content_str = result_content if isinstance(result_content, str) else json.dumps(result_content, default=str)
                            await queue.put(
                                f'<content-block-tool_result id="{tool_use_id}" name="{tool_name}"><![CDATA[{content_str}]]></content-block-tool_result>'
                            )
                        
                        # Log tool execution
                        self._log_action("tool_execution", {
                            "tool_name": tool_call["name"],
                            "tool_use_id": tool_call["id"],
                            "success": not is_error,
                        }, step_number=step)
                    
                    # Add tool results to context
                    tool_result_message = self._build_tool_result_message(tool_results)
                    self.messages.append(tool_result_message)
                    self.conversation_history.append(tool_result_message)
                    
                    continue
            
            # Validate final answer if checker configured
            if stop_reason != "tool_use" and self.final_answer_check:
                extracted = self._extract_final_answer(response)
                success, error_message = self.final_answer_check(extracted)
                if not success:
                    self.agent_logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "action": "final_answer_validation_failed",
                        "details": {"error": error_message, "step": step}
                    })
                    logger.warning(f"Final answer validation failed at step {step}: {error_message}")
                    
                    error_user_message = {
                        "role": "user",
                        "content": [{"type": "text", "text": error_message}]
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
            
            # Build result
            result = self._to_generic_result(
                response=response,
                conversation_history=self.conversation_history.copy(),
                total_steps=step,
                agent_logs=self.agent_logs.copy(),
                generated_files=None,
            )
            
            # Log completion
            self._log_action("run_completed", {
                "stop_reason": stop_reason,
                "total_steps": step,
                "total_input_tokens": usage.input_tokens,
                "total_output_tokens": usage.output_tokens,
            }, step_number=step)
            
            # Finalize files
            await self._finalize_file_processing(queue)
            
            all_files_metadata = list(self.file_registry.values())
            result.generated_files = all_files_metadata
            
            # Save asynchronously
            self._save_run_data_async(result, all_files_metadata)
            
            return result
        
        # Max steps reached
        logger.warning(f"Max steps ({self.max_steps}) reached")
        return await self._generate_final_summary(queue=queue)
    
    def execute_tool_call(self, tool_name: str, tool_input: dict) -> Any:
        """Execute a registered tool function.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            
        Returns:
            Tool execution result
        """
        if not self.tool_registry:
            return "No tools have been registered for this agent."
        
        return self.tool_registry.execute(tool_name, tool_input)
    
    def _apply_compaction(self, step_number: int = 0) -> None:
        """Apply compaction to self.messages and log the event."""
        if not self.compactor:
            return
        
        if self.memory_store:
            self.memory_store.before_compact(self.messages, self.model)
        
        original_messages = self.messages.copy() if self.memory_store else None
        estimated_tokens = self._last_known_input_tokens
        
        compacted, metadata = self.compactor.compact(
            self.messages, self.model, estimated_tokens=estimated_tokens
        )
        
        if metadata.get("compaction_applied", False):
            self.messages = compacted
            
            if self.memory_store and original_messages:
                self.messages, after_meta = self.memory_store.after_compact(
                    original_messages=original_messages,
                    compacted_messages=self.messages,
                    model=self.model
                )
                metadata["memory"] = after_meta
            
            self.agent_logs.append({
                "timestamp": datetime.now().isoformat(),
                "action": "compaction",
                "details": metadata
            })
            
            self._log_action("compaction", metadata, step_number=step_number)
            logger.info(f"Compaction applied: {metadata}")
    
    async def _generate_final_summary(
        self,
        queue: Optional[asyncio.Queue] = None,
    ) -> GenericAgentResult:
        """Generate a final summary when max steps is reached."""
        logger.info("Generating final summary due to max_steps reached")
        
        if self.compactor:
            self._apply_compaction(step_number=self.max_steps)
        
        # Make final call with modified system prompt
        original_system = self.system_prompt
        self.system_prompt = (
            f"{original_system}\n\n"
            "IMPORTANT: You have reached the maximum number of steps. "
            "Please provide a final summary or response based on the work completed so far."
        )
        
        # Temporarily disable tools for summary
        original_tools = self.tool_schemas
        self.tool_schemas = []
        
        try:
            response = await self._stream_completion(self.messages, queue)
        finally:
            self.system_prompt = original_system
            self.tool_schemas = original_tools
        
        # Track usage
        usage = self._get_usage(response)
        self._token_usage_history.append({
            "step": self.max_steps,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        })
        
        # Add to history
        assistant_message = self._response_to_message(response)
        self.messages.append(assistant_message)
        self.conversation_history.append(assistant_message)
        
        # Memory update
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
        
        result = self._to_generic_result(
            response=response,
            conversation_history=self.conversation_history.copy(),
            total_steps=self.max_steps,
            agent_logs=self.agent_logs.copy(),
            generated_files=None,
        )
        
        self._log_action("final_summary_generation", {
            "stop_reason": self._get_stop_reason(response),
            "total_steps": self.max_steps,
        }, step_number=self.max_steps)
        
        await self._finalize_file_processing(queue)
        
        all_files_metadata = list(self.file_registry.values())
        result.generated_files = all_files_metadata
        
        self._save_run_data_async(result, all_files_metadata)
        
        return result
    
    def _log_action(
        self,
        action_type: str,
        action_data: dict,
        step_number: int = 0,
    ) -> None:
        """Add an action log entry."""
        log_entry = {
            "log_id": len(self._run_logs_buffer) + 1,
            "agent_uuid": self.agent_uuid,
            "run_id": self._run_id,
            "timestamp": datetime.now(),
            "step_number": step_number,
            "action_type": action_type,
            "action_data": action_data,
            "messages_count": len(self.messages),
            "estimated_tokens": self._last_known_input_tokens,
        }
        self._run_logs_buffer.append(log_entry)
    
    def _load_state_from_db(self) -> dict[str, Any]:
        """Load agent configuration from database."""
        try:
            config = asyncio.run(self.db_backend.load_agent_config(self.agent_uuid))
            
            if config is None:
                logger.info(f"No existing state for agent {self.agent_uuid}, creating new")
                return {}
            
            logger.info(
                f"Loaded state for agent {self.agent_uuid}: "
                f"{len(config.get('messages', []))} messages"
            )
            return config
        except Exception as e:
            logger.error(f"Failed to load state for agent {self.agent_uuid}: {e}", exc_info=True)
            return {}
    
    async def _finalize_file_processing(self, queue: Optional[asyncio.Queue] = None) -> None:
        """Finalize file processing at end of run."""
        # Process files via backend if configured
        if self.file_backend:
            await self._process_generated_files()
        
        # Stream metadata
        all_files_metadata = list(self.file_registry.values())
        if queue and all_files_metadata:
            await self._stream_file_metadata(queue, all_files_metadata)
    
    async def _process_generated_files(self) -> list[dict]:
        """Process files via configured backend."""
        if not self.file_backend or not self.file_registry:
            return []
        
        files_metadata: list[dict] = []
        
        for file_id, registry_entry in self.file_registry.items():
            filename = registry_entry.get("filename") or str(file_id)
            
            try:
                # Subclasses should implement file download
                content_bytes = await self._download_file(file_id)
                
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
                
                merged = dict(registry_entry)
                merged.update(metadata)
                self.file_registry[file_id] = merged
                files_metadata.append(merged)
                
            except Exception as e:
                logger.warning(f"Failed to process file {file_id}: {e}")
                continue
        
        return files_metadata
    
    async def _download_file(self, file_id: str) -> bytes:
        """Download a file by ID. Subclasses should override for provider-specific logic."""
        raise NotImplementedError("Subclasses must implement _download_file")
    
    async def _stream_file_metadata(self, queue: asyncio.Queue, metadata: list[dict]) -> None:
        """Stream file metadata to queue."""
        if not metadata:
            return
        
        files_json = json.dumps({"files": metadata}, indent=2)
        meta_tag = f'<content-block-meta_files><![CDATA[{files_json}]]></content-block-meta_files>'
        await queue.put(meta_tag)
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _save_agent_config(self) -> None:
        """Save agent configuration to database."""
        existing_config = await self.db_backend.load_agent_config(self.agent_uuid)
        existing_config = existing_config or {}
        
        config = {
            "agent_uuid": self.agent_uuid,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "tool_schemas": self.tool_schemas,
            "tool_names": [t["name"] for t in self.tool_schemas] if self.tool_schemas else [],
            "api_kwargs": self.api_kwargs,
            "stream_meta_history_and_tool_results": self.stream_meta_history_and_tool_results,
            "file_registry": self.file_registry,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "last_known_input_tokens": self._last_known_input_tokens,
            "last_known_output_tokens": self._last_known_output_tokens,
            "created_at": existing_config.get("created_at") or datetime.now(),
            "updated_at": datetime.now(),
            "last_run_at": self._run_start_time,
            "total_runs": existing_config.get("total_runs", 0) + 1,
        }
        
        if self.compactor:
            config["compactor_type"] = self.compactor.__class__.__name__
        if self.memory_store:
            config["memory_store_type"] = self.memory_store.__class__.__name__
        
        await self.db_backend.save_agent_config(config)
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _save_conversation_entry(
        self,
        result: GenericAgentResult,
        files_metadata: list[dict],
    ) -> None:
        """Save conversation history entry to database."""
        user_message = ""
        for msg in result.conversation_history:
            if msg.get("role") == "user":
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
        
        conversation = {
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
            "usage": result.usage.to_dict(),
            "generated_files": files_metadata,
            "created_at": datetime.now(),
        }
        
        await self.db_backend.save_conversation_history(conversation)
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _save_run_logs(self) -> None:
        """Save run logs to database."""
        await self.db_backend.save_agent_run_logs(
            self.agent_uuid,
            self._run_id,
            self._run_logs_buffer
        )
    
    async def _save_run_data_with_retry(
        self,
        result: GenericAgentResult,
        files_metadata: list[dict],
    ) -> None:
        """Save all run data with retry logic."""
        operations = [
            ("agent_config", self._save_agent_config),
            ("conversation_history", lambda: self._save_conversation_entry(result, files_metadata)),
            ("agent_runs", self._save_run_logs),
        ]
        
        for operation_type, op in operations:
            try:
                await op()
            except Exception as e:
                logger.error(f"Failed to persist {operation_type}: {e}", exc_info=True)
    
    def _save_run_data_async(
        self,
        result: GenericAgentResult,
        files_metadata: list[dict],
    ) -> None:
        """Save run data asynchronously."""
        task = asyncio.create_task(
            self._save_run_data_with_retry(result, files_metadata)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def drain_background_tasks(self, timeout: float = 30.0) -> dict[str, Any]:
        """Wait for all background persistence tasks to complete."""
        if not self._background_tasks:
            return {"total_tasks": 0, "completed": 0, "timed_out": 0}
        
        total_tasks = len(self._background_tasks)
        logger.info(f"Draining {total_tasks} background task(s)")
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._background_tasks, return_exceptions=True),
                timeout=timeout
            )
            return {"total_tasks": total_tasks, "completed": total_tasks, "timed_out": 0}
        except asyncio.TimeoutError:
            incomplete = len([t for t in self._background_tasks if not t.done()])
            return {
                "total_tasks": total_tasks,
                "completed": total_tasks - incomplete,
                "timed_out": incomplete,
            }
    
    def __str__(self) -> str:
        """Return current configuration as JSON."""
        tool_names = [schema.get("name", "<unnamed>") for schema in self.tool_schemas]
        config_snapshot = {
            "agent_uuid": self.agent_uuid,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "messages_count": len(self.messages),
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "compactor": self.compactor.__class__.__name__ if self.compactor else None,
            "memory_store": self.memory_store.__class__.__name__ if self.memory_store else None,
            "db_backend": self.db_backend.__class__.__name__,
            "tools": tool_names,
        }
        return json.dumps(config_snapshot, indent=2)

