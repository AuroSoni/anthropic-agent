"""Anthropic provider implementation built on the provider-agnostic BaseAgent.

This file keeps only Anthropic-specific concerns:
- Request parameter preparation for `anthropic` SDK.
- Prompt caching (cache_control) injection for supported Claude models.
- Anthropic streaming via `client.beta.messages.stream(...)`.
- Optional Anthropic Files API integration for code-execution generated files.

All provider-agnostic logic (tool loop, compaction, memory, persistence, meta
streaming) lives in `base_agent.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional, Callable

import anthropic
from anthropic.types.beta import BetaMessage, FileMetadata

from .base_agent import BaseAgent, LLMResponse
from .retry import anthropic_stream_with_backoff
from .title_generator import generate_title

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anthropic defaults / cache-control tuning
# ---------------------------------------------------------------------------

# Prompt caching appears to require minimum token thresholds, which vary by model.
# (These values were inherited from the original AnthropicAgent implementation.)
MIN_CACHE_TOKENS_SONNET = 1024
MIN_CACHE_TOKENS_HAIKU = 2048

# Claude 4.5 models have larger context; use larger thresholds for caching.
MIN_CACHE_TOKENS_SONNET_4_5 = 2048

# Default Anthropic model (subclasses of BaseAgent can override via class attrs)
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_BETA_HEADERS = ["prompt-caching-2024-07-31"]

# These defaults match the original AnthropicAgent.
DEFAULT_MAX_STEPS = 50
DEFAULT_THINKING_TOKENS = 4096
DEFAULT_MAX_TOKENS = 16384


class AnthropicAgent(BaseAgent):
    """Anthropic agent implementation.

    The public API and run loop come from :class:`~.base_agent.BaseAgent`.
    """

    # Override BaseAgent defaults
    DEFAULT_MODEL = DEFAULT_MODEL
    DEFAULT_BETA_HEADERS = DEFAULT_BETA_HEADERS
    DEFAULT_MAX_STEPS = DEFAULT_MAX_STEPS
    DEFAULT_THINKING_TOKENS = DEFAULT_THINKING_TOKENS
    DEFAULT_MAX_TOKENS = DEFAULT_MAX_TOKENS

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
        messages: list[dict] | None = None,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        formatter: Optional[str] = None,
        enable_cache_control: Optional[bool] = None,
        compactor: Any = None,
        memory_store: Any = None,
        final_answer_check: Optional[Callable[[str], tuple[bool, str]]] = None,
        agent_uuid: str | None = None,
        db_backend: Any = "filesystem",
        file_backend: Any = None,
        **api_kwargs: Any,
    ):
        # Anthropic-specific instance variables (set BEFORE super().__init__)
        self.thinking_tokens = thinking_tokens if thinking_tokens is not None else self.DEFAULT_THINKING_TOKENS
        self.beta_headers = beta_headers if beta_headers is not None else list(self.DEFAULT_BETA_HEADERS)
        self.container_id = container_id
        self.enable_cache_control = enable_cache_control if enable_cache_control is not None else True

        super().__init__(
            system_prompt=system_prompt,
            model=model,
            max_steps=max_steps,
            max_tokens=max_tokens,
            stream_meta_history_and_tool_results=stream_meta_history_and_tool_results,
            tools=tools,
            frontend_tools=frontend_tools,
            server_tools=server_tools,
            messages=messages,
            max_retries=max_retries,
            base_delay=base_delay,
            formatter=formatter,  # type: ignore[arg-type]
            compactor=compactor,
            memory_store=memory_store,
            final_answer_check=final_answer_check,
            agent_uuid=agent_uuid,
            db_backend=db_backend,
            file_backend=file_backend,
            title_generator=generate_title,
            **api_kwargs,
        )

        # Provider client
        self.client = anthropic.AsyncAnthropic()

    # ------------------------------------------------------------------
    # Provider hooks required by BaseAgent
    # ------------------------------------------------------------------

    def _on_file_backend_configured(self) -> None:
        """Anthropic Files API requires a beta header."""
        if self.file_backend and "files-api-2025-04-14" not in self.beta_headers:
            self.beta_headers.append("files-api-2025-04-14")

    def _get_provider_type(self) -> str:
        return "anthropic"

    def _get_provider_specific_config(self) -> dict[str, Any]:
        return {
            "thinking_tokens": self.thinking_tokens,
            "beta_headers": self.beta_headers,
            "container_id": self.container_id,
            "enable_cache_control": self.enable_cache_control,
        }

    def _restore_provider_specific_state(
        self,
        provider_config: dict[str, Any],
        full_config: dict[str, Any],
    ) -> None:
        """Restore Anthropic-specific state with backward compatibility for v1 format."""
        # Try v2 format first, fall back to v1 (top-level) if not found
        self.thinking_tokens = provider_config.get(
            "thinking_tokens",
            full_config.get("thinking_tokens", self.thinking_tokens),
        )
        self.beta_headers = provider_config.get(
            "beta_headers",
            full_config.get("beta_headers", self.beta_headers),
        )
        self.container_id = provider_config.get(
            "container_id",
            full_config.get("container_id", self.container_id),
        )
        self.enable_cache_control = provider_config.get(
            "enable_cache_control",
            full_config.get("enable_cache_control", self.enable_cache_control),
        )

    def _get_provider_specific_usage_fields(self, usage: dict[str, Any]) -> dict[str, Any]:
        return {
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        }

    async def _stream_llm_response(
        self,
        *,
        queue: Optional[asyncio.Queue],
        formatter: str,
        tools_enabled: bool = True,
        system_prompt_override: Optional[str] = None,
    ) -> LLMResponse:
        request_params = self._prepare_request_params(
            tools_enabled=tools_enabled,
            system_prompt_override=system_prompt_override,
        )

        accumulated: BetaMessage = await anthropic_stream_with_backoff(
            client=self.client,
            request_params=request_params,
            queue=queue,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            formatter=formatter,  # type: ignore[arg-type]
            stream_tool_results=self.stream_meta_history_and_tool_results,
        )

        # Convert provider message to canonical dict (role/content blocks)
        assistant_message = accumulated.model_dump(
            mode="json",
            include=["role", "content"],
            exclude_unset=True,
            exclude=getattr(accumulated, "__api_exclude__", None),
            warnings=False,
        )

        usage: dict[str, Any] = {}
        try:
            if getattr(accumulated, "usage", None) is not None:
                # BetaUsage is a pydantic model
                usage = accumulated.usage.model_dump(mode="json")  # type: ignore[union-attr]
        except Exception:
            usage = {}

        container_id: Optional[str] = None
        try:
            if getattr(accumulated, "container", None) is not None:
                container_id = accumulated.container.id  # type: ignore[union-attr]
        except Exception:
            container_id = None

        return LLMResponse(
            raw=accumulated,
            assistant_message=assistant_message,
            stop_reason=getattr(accumulated, "stop_reason", "end_turn"),
            model=getattr(accumulated, "model", None),
            usage=usage,
            container_id=container_id,
        )

    # ------------------------------------------------------------------
    # Anthropic request preparation / cache control
    # ------------------------------------------------------------------

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Apply Anthropic prompt caching controls.

        Returns:
            (processed_messages, processed_system)

        `processed_system` may be a string, a list of system blocks, or None.
        """

        if not self.enable_cache_control:
            return messages, system_prompt

        processed_messages = [dict(m) for m in messages]
        processed_system = system_prompt

        # Determine per-model cache thresholds
        model_name = (self.model or "").lower()
        if "sonnet" in model_name:
            min_cache_tokens = MIN_CACHE_TOKENS_SONNET_4_5 if "4-5" in model_name or "4.5" in model_name else MIN_CACHE_TOKENS_SONNET
        elif "haiku" in model_name:
            min_cache_tokens = MIN_CACHE_TOKENS_HAIKU
        else:
            min_cache_tokens = MIN_CACHE_TOKENS_SONNET

        if system_prompt:
            system_tokens = len(system_prompt) // 4
            if system_tokens >= min_cache_tokens:
                processed_system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

        # Apply cache control to last user message if large enough
        if processed_messages:
            last_msg = processed_messages[-1]
            content = last_msg.get("content")
            if isinstance(content, list):
                # Estimate size
                approx_tokens = 0
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        approx_tokens += len(str(block.get("text", ""))) // 4

                if approx_tokens >= min_cache_tokens:
                    # Add cache_control to last content block
                    if content and isinstance(content[-1], dict):
                        content[-1] = dict(content[-1])
                        content[-1]["cache_control"] = {"type": "ephemeral"}

        return processed_messages, processed_system

    def _prepare_request_params(
        self,
        *,
        tools_enabled: bool = True,
        system_prompt_override: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build Anthropic `client.beta.messages.stream(...)` parameters."""

        system_prompt = system_prompt_override if system_prompt_override is not None else self.system_prompt

        messages, system = self._apply_cache_control(self.messages, system_prompt)

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        if system is not None:
            params["system"] = system
        elif system_prompt is not None:
            params["system"] = system_prompt

        if self.thinking_tokens and self.thinking_tokens > 0:
            params["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_tokens}

        if tools_enabled:
            tools: list[dict[str, Any]] = []
            if self.tool_schemas:
                tools.extend(self.tool_schemas)
            if self.frontend_tool_schemas:
                tools.extend(self.frontend_tool_schemas)
            if self.server_tools:
                tools.extend(self.server_tools)
            if tools:
                params["tools"] = tools

        if self.beta_headers:
            params["betas"] = self.beta_headers

        if self.container_id:
            params["container"] = self.container_id

        if self.api_kwargs:
            params.update(self.api_kwargs)

        return params

    # ------------------------------------------------------------------
    # Optional: token counting using Anthropic API
    # ------------------------------------------------------------------

    def _filter_messages_for_token_count(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out message content blocks not supported by token counting endpoint."""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
                filtered_messages.append(msg)
                continue

            filtered_content = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") in {"text", "tool_use", "tool_result", "thinking"}:
                    filtered_content.append(block)
            filtered_msg = dict(msg)
            filtered_msg["content"] = filtered_content
            filtered_messages.append(filtered_msg)

        return filtered_messages

    async def _count_tokens_api(self, messages: Optional[list[dict[str, Any]]] = None) -> dict[str, int]:
        """Use Anthropic token counting endpoint (best-effort)."""
        if messages is None:
            messages = self.messages

        filtered_messages = self._filter_messages_for_token_count(messages)

        tools: list[dict[str, Any]] = []
        if self.tool_schemas:
            tools.extend(self.tool_schemas)
        if self.frontend_tool_schemas:
            tools.extend(self.frontend_tool_schemas)
        if self.server_tools:
            tools.extend(self.server_tools)

        system_prompt = self.system_prompt
        processed_messages, processed_system = self._apply_cache_control(filtered_messages, system_prompt)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": processed_messages,
            "system": processed_system or system_prompt,
        }

        if tools:
            params["tools"] = tools
        if self.thinking_tokens and self.thinking_tokens > 0:
            params["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_tokens}
        if self.beta_headers:
            params["betas"] = self.beta_headers
        if self.container_id:
            params["container"] = self.container_id
        if self.api_kwargs:
            params.update(self.api_kwargs)

        resp = await self.client.beta.messages.count_tokens(**params)
        return {"input_tokens": int(resp.input_tokens)}

    # ------------------------------------------------------------------
    # Files API integration (optional)
    # ------------------------------------------------------------------

    def _upsert_file_registry_entry(
        self,
        *,
        file_id: str,
        filename: str,
        step: int,
        raw_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create or update a single file entry in the in-memory registry."""
        now = datetime.now().isoformat()
        existing: dict[str, Any] = self.file_registry.get(file_id, {})

        updated: dict[str, Any] = dict(existing)
        updated["file_id"] = file_id
        updated["filename"] = filename
        updated.setdefault("first_seen_step", step)
        updated.setdefault("created_at", existing.get("created_at") or now)
        updated["last_seen_step"] = step
        updated["updated_at"] = now

        if raw_metadata is not None:
            updated.setdefault("raw", raw_metadata)

        self.file_registry[file_id] = updated

    def _register_files_from_message(self, message: BetaMessage | dict[str, Any], step: int) -> None:
        file_ids = self.extract_file_ids(message)
        for file_id in file_ids:
            self._upsert_file_registry_entry(file_id=file_id, filename=f"file_{file_id}", step=step)

    async def _finalize_file_processing(self, queue: Optional[asyncio.Queue] = None) -> None:
        """Finalize file processing for Anthropic Files API + local backend."""

        # Discover file ids from the run history
        for message in getattr(self, "conversation_history", []):
            self._register_files_from_message(message, step=0)

        # Store via backend
        if self.file_backend:
            await self._process_generated_files(step=0)

        # Stream metadata
        all_files_metadata: list[dict[str, Any]] = list(self.file_registry.values())
        if queue and all_files_metadata:
            await self._stream_file_metadata(queue, all_files_metadata)

    async def _download_file(self, file_id: str) -> tuple[FileMetadata, bytes]:
        """Download file content from Anthropic Files API."""
        response = await self.client.beta.files.download(file_id)
        file_content = await response.read()
        file_metadata = await self.client.beta.files.retrieve_metadata(file_id)
        return file_metadata, file_content

    def extract_file_ids(self, message: BetaMessage | dict[str, Any]) -> list[str]:
        file_ids: list[str] = []

        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        if not content:
            return file_ids
        if not isinstance(content, list):
            logger.warning("Message content is not a list: %s", type(content))
            return file_ids

        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                item_content = item.get("content")
            else:
                item_type = getattr(item, "type", "")
                item_content = getattr(item, "content", None)

            if item_type == "bash_code_execution_tool_result" or item_type == "tool_result":
                if not item_content:
                    continue

                if isinstance(item_content, dict):
                    inner_type = item_content.get("type")
                    files = item_content.get("content", [])
                elif hasattr(item_content, "type"):
                    inner_type = getattr(item_content, "type", "")
                    files = getattr(item_content, "content", [])
                else:
                    continue

                if inner_type == "bash_code_execution_result" and isinstance(files, list):
                    for f in files:
                        if isinstance(f, dict):
                            file_id = f.get("file_id")
                        else:
                            file_id = getattr(f, "file_id", None)

                        if file_id:
                            logger.info("Found file_id: %s", file_id)
                            file_ids.append(str(file_id))

        return file_ids

    async def _process_generated_files(self, step: int) -> list[dict[str, Any]]:
        if not self.file_backend:
            return []
        if not self.file_registry:
            return []

        files_metadata: list[dict[str, Any]] = []

        for file_id, registry_entry in self.file_registry.items():
            filename = registry_entry.get("filename") or str(file_id)

            try:
                file_metadata_api, content_bytes = await self._download_file(file_id)
                if getattr(file_metadata_api, "filename", None):
                    filename = file_metadata_api.filename  # type: ignore[assignment]
                    registry_entry["filename"] = filename

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

                metadata["processed_at_step"] = step

                merged = dict(registry_entry)
                merged.update(metadata)
                self.file_registry[file_id] = merged

                files_metadata.append(merged)

            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to process file %s (%s): %s", filename, file_id, e, exc_info=True)
                continue

        return files_metadata

    async def _stream_file_metadata(self, queue: asyncio.Queue, metadata: list[dict[str, Any]]) -> None:
        if not metadata:
            return

        files_json = json.dumps({"files": metadata}, indent=2)
        meta_tag = f'<content-block-meta_files><![CDATA[{files_json}]]></content-block-meta_files>'
        await queue.put(meta_tag)
