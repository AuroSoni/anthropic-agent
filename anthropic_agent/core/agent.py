"""Anthropic-specific agent implementation.

This module provides the AnthropicAgent class that extends BaseAgent with
Anthropic-specific functionality for Claude models.
"""

import asyncio
import logging
from typing import Optional, Callable, Any

from anthropic.types.beta import BetaMessage

from .base_agent import BaseAgent
from .types import GenericUsage, GenericAgentResult
from ..providers.anthropic import AnthropicClient
from ..providers.anthropic.types import (
    convert_to_generic_usage,
    convert_to_generic_result,
    extract_text_from_message,
    extract_file_ids_from_message,
)
from ..providers.anthropic.streaming import FormatterType

logger = logging.getLogger(__name__)


# Default Anthropic model
DEFAULT_MODEL = "claude-sonnet-4-5"


class AnthropicAgent(BaseAgent):
    """Anthropic-specific agent for Claude models.
    
    This class extends BaseAgent with Anthropic-specific implementations for:
    - Streaming completions via Claude API
    - Token counting
    - Response processing
    - File handling via Anthropic Files API
    
    AnthropicAgent maintains full backward compatibility with the previous
    API while leveraging the new BaseAgent architecture.
    
    Example:
        >>> agent = AnthropicAgent(
        ...     system_prompt="You are a helpful assistant",
        ...     model="claude-sonnet-4-5",
        ... )
        >>> result = await agent.run("Hello!")
    """
    
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
        messages: list[dict] | None = None,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        formatter: FormatterType | None = None,
        compactor: Any = None,
        memory_store: Any = None,
        final_answer_check: Optional[Callable[[str], tuple[bool, str]]] = None,
        agent_uuid: str | None = None,
        db_backend: Any = "filesystem",
        file_backend: Any = None,
        **api_kwargs: Any,
    ):
        """Initialize AnthropicAgent with configuration.
        
        Args:
            system_prompt: System prompt to guide the agent's behavior
            model: Anthropic model name (default: "claude-sonnet-4-5")
            max_steps: Maximum conversation turns before stopping
            thinking_tokens: Budget for extended thinking tokens (0 = disabled)
            max_tokens: Maximum tokens in response
            stream_meta_history_and_tool_results: Include metadata in stream
            tools: List of functions decorated with @tool
            server_tools: Anthropic server-side tools (code_execution, web_search)
            beta_headers: Beta feature headers for Anthropic API
            container_id: Container ID for multi-turn conversations
            messages: Initial message history
            max_retries: Maximum retry attempts for API calls
            base_delay: Base delay for exponential backoff
            formatter: Default stream output formatter ("xml" or "raw")
            compactor: Context compaction strategy
            memory_store: Semantic memory store
            final_answer_check: Validation function for final answers
            agent_uuid: Session UUID for resuming previous sessions
            db_backend: Database backend for persistence
            file_backend: File storage backend
            **api_kwargs: Additional Anthropic API arguments (temperature, etc.)
        """
        # Create Anthropic client
        self._anthropic_client = AnthropicClient()
        
        # Anthropic-specific options
        self.thinking_tokens = thinking_tokens or 0
        self.beta_headers = beta_headers or []
        self.server_tools: list[dict[str, Any]] = server_tools or []
        self.container_id = container_id
        self.formatter: FormatterType = formatter or "xml"
        
        # Store api_kwargs for Anthropic-specific options
        self._api_kwargs = api_kwargs
        
        # Initialize base agent
        super().__init__(
            client=self._anthropic_client,
            system_prompt=system_prompt,
            model=model,
            max_steps=max_steps,
            max_tokens=max_tokens,
            stream_meta_history_and_tool_results=stream_meta_history_and_tool_results,
            tools=tools,
            messages=messages,
            max_retries=max_retries,
            base_delay=base_delay,
            compactor=compactor,
            memory_store=memory_store,
            final_answer_check=final_answer_check,
            agent_uuid=agent_uuid,
            db_backend=db_backend,
            file_backend=file_backend,
            **api_kwargs,
        )
        
        # Load Anthropic-specific state from DB if resuming
        if agent_uuid:
            db_config = self._load_state_from_db()
            if not thinking_tokens:
                self.thinking_tokens = db_config.get("thinking_tokens", 0)
            if not beta_headers:
                self.beta_headers = db_config.get("beta_headers", [])
            if not server_tools:
                self.server_tools = db_config.get("server_tools", [])
            if not container_id:
                self.container_id = db_config.get("container_id")
            if not formatter:
                self.formatter = db_config.get("formatter", "xml")
    
    def _get_default_model(self) -> str:
        """Get the default Anthropic model."""
        return DEFAULT_MODEL
    
    async def _stream_completion(
        self,
        messages: list[dict],
        queue: Optional[asyncio.Queue] = None,
    ) -> BetaMessage:
        """Stream a completion from Anthropic's API.
        
        Args:
            messages: List of message dictionaries
            queue: Optional async queue for streaming output
            
        Returns:
            BetaMessage containing the response
        """
        # Build combined tools list
        combined_tools: list[dict[str, Any]] = []
        if self.tool_schemas:
            combined_tools.extend(self.tool_schemas)
        if self.server_tools:
            combined_tools.extend(self.server_tools)
        
        # Build kwargs for Anthropic client
        kwargs: dict[str, Any] = {}
        
        if self.thinking_tokens > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_tokens
            }
        
        if self.beta_headers:
            kwargs["betas"] = self.beta_headers
        
        if self.container_id:
            kwargs["container"] = self.container_id
        
        # Merge API kwargs
        kwargs.update(self._api_kwargs)
        
        # Stream with retry
        response = await self._anthropic_client.stream_with_retry(
            messages=messages,
            model=self.model,
            system=self.system_prompt,
            tools=combined_tools if combined_tools else None,
            max_tokens=self.max_tokens,
            queue=queue,
            formatter=self.formatter,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            **kwargs,
        )
        
        # Update container ID if returned
        if hasattr(response, 'container') and response.container:
            self.container_id = response.container.id
        
        return response
    
    async def _count_tokens(
        self,
        messages: list[dict],
    ) -> int | None:
        """Count tokens using Anthropic's API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Token count or None if counting fails
        """
        combined_tools: list[dict[str, Any]] = []
        if self.tool_schemas:
            combined_tools.extend(self.tool_schemas)
        if self.server_tools:
            combined_tools.extend(self.server_tools)
        
        return await self._anthropic_client.count_tokens(
            messages=messages,
            model=self.model,
            system=self.system_prompt,
            tools=combined_tools if combined_tools else None,
        )
    
    def _extract_final_answer(self, response: BetaMessage) -> str:
        """Extract text from Anthropic response."""
        return extract_text_from_message(response)
    
    def _response_to_message(self, response: BetaMessage) -> dict:
        """Convert BetaMessage to message dict."""
        if hasattr(response, 'model_dump'):
            return response.model_dump(
                mode="json",
                include=["role", "content"],
                exclude_unset=True,
                warnings=False
            )
        return {
            "role": getattr(response, 'role', 'assistant'),
            "content": getattr(response, 'content', []),
        }
    
    def _get_stop_reason(self, response: BetaMessage) -> str:
        """Get stop reason from BetaMessage."""
        return response.stop_reason or "unknown"
    
    def _get_usage(self, response: BetaMessage) -> GenericUsage:
        """Get token usage from BetaMessage."""
        return convert_to_generic_usage(response.usage)
    
    def _get_tool_calls(self, response: BetaMessage) -> list[dict]:
        """Extract tool calls from BetaMessage."""
        tool_calls = []
        
        if not response.content:
            return tool_calls
        
        for block in response.content:
            if getattr(block, 'type', '') == 'tool_use':
                tool_calls.append({
                    "id": getattr(block, 'id', ''),
                    "name": getattr(block, 'name', ''),
                    "input": getattr(block, 'input', {}),
                })
        
        return tool_calls
    
    def _build_tool_result_message(self, tool_results: list[dict]) -> dict:
        """Build tool result message in Anthropic format."""
        return {
            "role": "user",
            "content": tool_results
        }
    
    def _to_generic_result(
        self,
        response: BetaMessage,
        conversation_history: list[dict],
        total_steps: int,
        agent_logs: list[dict],
        generated_files: list[dict] | None,
    ) -> GenericAgentResult:
        """Convert Anthropic response to GenericAgentResult."""
        return convert_to_generic_result(
            final_message=response,
            final_answer=self._extract_final_answer(response),
            conversation_history=conversation_history,
            stop_reason=self._get_stop_reason(response),
            model=response.model,
            total_steps=total_steps,
            agent_logs=agent_logs,
            generated_files=generated_files,
            container_id=self.container_id,
        )
    
    async def _download_file(self, file_id: str) -> bytes:
        """Download file from Anthropic Files API."""
        _, content = await self._anthropic_client.download_file(file_id)
        return content
    
    def _register_files_from_message(self, message: Any, step: int) -> None:
        """Scan message for file references and register them."""
        file_ids = extract_file_ids_from_message(message)
        
        for file_id in file_ids:
            if file_id not in self.file_registry:
                self.file_registry[file_id] = {
                    "file_id": file_id,
                    "filename": f"file_{file_id}",
                    "first_seen_step": step,
                }
    
    async def run(
        self,
        prompt: str | list[dict],
        queue: Optional[asyncio.Queue] = None,
        formatter: Optional[FormatterType] = None,
    ) -> GenericAgentResult:
        """Execute an agent run with the given user message.
        
        Args:
            prompt: User message (string or content blocks)
            queue: Optional async queue for streaming output
            formatter: Override default formatter for this run
            
        Returns:
            GenericAgentResult containing the full execution context
        """
        # Temporarily override formatter if provided
        original_formatter = self.formatter
        if formatter:
            self.formatter = formatter
        
        try:
            return await super().run(prompt, queue)
        finally:
            self.formatter = original_formatter
    
    # Backward compatibility properties
    @property
    def agent_logs(self) -> list[dict]:
        """Get agent logs (for backward compatibility)."""
        return getattr(self, '_agent_logs', [])
    
    @agent_logs.setter
    def agent_logs(self, value: list[dict]) -> None:
        self._agent_logs = value
    
    @property
    def conversation_history(self) -> list[dict]:
        """Get conversation history (for backward compatibility)."""
        return getattr(self, '_conversation_history', [])
    
    @conversation_history.setter
    def conversation_history(self, value: list[dict]) -> None:
        self._conversation_history = value
