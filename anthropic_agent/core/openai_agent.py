"""OpenAI provider implementation.

This module provides :class:`OpenAIAgent`, an agent implementation that uses
OpenAI's official Python SDK as the LLM provider.

Only provider-specific logic lives here (request building, response normalization).
The agent orchestration (tool loop, persistence, compaction, memory injection, etc.)
is implemented in :class:`.BaseAgent`.

Streaming and retry logic are delegated to:
- :mod:`anthropic_agent.streaming.openai_formatters` for XML/raw formatting
- :mod:`anthropic_agent.core.openai_retry` for exponential backoff
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from .base_agent import BaseAgent, LLMResponse
from .openai_retry import openai_stream_with_backoff
from .title_generator import generate_title
from ..streaming import FormatterType
from ..streaming.openai_formatters import OpenAIFormatterType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_STEPS = 50
DEFAULT_THINKING_TOKENS = 0  # OpenAI uses reasoning_effort, not token budgets
DEFAULT_MAX_TOKENS = 16384


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _message_role_for_system(model: str) -> str:
    """Choose the instruction role for newer model families.

    The Chat Completions API recommends using `developer` messages for o1 and
    newer model families.
    """
    m = (model or "").lower()
    if m.startswith("o1") or m.startswith("o3") or m.startswith("gpt-5"):
        return "developer"
    return "system"


def _to_openai_function_tool(schema: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Convert a tool schema (possibly Anthropic-shaped) to OpenAI function-tool shape."""
    if not isinstance(schema, dict):
        return None

    # Already OpenAI function tool
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        return schema

    # Some registries might return {"type": "function", "name":..., "parameters":...}
    if schema.get("type") == "function" and "name" in schema and "function" not in schema:
        fn: dict[str, Any] = {
            "name": schema.get("name"),
            "description": schema.get("description"),
        }
        if "parameters" in schema:
            fn["parameters"] = schema["parameters"]
        if "strict" in schema:
            fn["strict"] = schema["strict"]
        return {"type": "function", "function": fn}

    # Anthropic-ish: {name, description, input_schema}
    if "name" in schema and ("input_schema" in schema or "parameters" in schema):
        fn = {
            "name": schema.get("name"),
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters") or schema.get("input_schema") or {"type": "object", "properties": {}},
        }
        # Carry strict if present
        if "strict" in schema:
            fn["strict"] = schema["strict"]
        return {"type": "function", "function": fn}

    return None


def _stringify_tool_output(content: Any) -> str:
    """Convert tool output to the `content` format expected by OpenAI tool messages."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _canonical_messages_to_openai_messages(
    *,
    canonical_messages: list[dict[str, Any]],
    system_prompt: str,
    model: str,
) -> list[dict[str, Any]]:
    """Convert internal canonical messages into OpenAI Chat Completions messages."""

    messages: list[dict[str, Any]] = []

    sys_role = _message_role_for_system(model)
    if system_prompt:
        messages.append({"role": sys_role, "content": system_prompt})

    for msg in canonical_messages:
        role = msg.get("role")
        content = msg.get("content")

        # Plain string content
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            # Best-effort fallback
            messages.append({"role": role, "content": _stringify_tool_output(content)})
            continue

        if role == "user":
            # Split tool results into separate tool-role messages
            text_parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    tool_call_id = block.get("tool_use_id")
                    tool_content = _stringify_tool_output(block.get("content"))
                    tool_msg: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                    messages.append(tool_msg)
                else:
                    # Ignore other block types for now (images, etc.)
                    pass

            if text_parts:
                messages.append({"role": "user", "content": "".join(text_parts)})
            continue

        if role == "assistant":
            # Convert tool_use blocks into tool_calls
            text_parts = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    call_id = block.get("id")
                    name = block.get("name")
                    args = block.get("input") or {}
                    try:
                        args_str = json.dumps(args, ensure_ascii=False)
                    except Exception:
                        args_str = _stringify_tool_output(args)

                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args_str},
                        }
                    )

            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                assistant_msg["content"] = "".join(text_parts)
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls

            # OpenAI requires either content or tool_calls
            if "content" in assistant_msg or "tool_calls" in assistant_msg:
                messages.append(assistant_msg)
            continue

        # Fallback for other roles
        messages.append({"role": role, "content": _stringify_tool_output(content)})

    return messages


def _normalize_usage(usage_obj: Any) -> dict[str, Any]:
    """Normalize OpenAI usage fields to the agent's common shape."""
    if usage_obj is None:
        return {}

    def _get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    prompt_tokens = _get(usage_obj, "prompt_tokens")
    completion_tokens = _get(usage_obj, "completion_tokens")
    total_tokens = _get(usage_obj, "total_tokens")

    out: dict[str, Any] = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw": usage_obj,
    }

    # prompt_tokens_details.cached_tokens
    ptd = _get(usage_obj, "prompt_tokens_details")
    if ptd is not None:
        cached = _get(ptd, "cached_tokens")
        if cached is not None:
            out["cache_read_input_tokens"] = cached

    # completion_tokens_details.reasoning_tokens
    ctd = _get(usage_obj, "completion_tokens_details")
    if ctd is not None:
        reasoning = _get(ctd, "reasoning_tokens")
        if reasoning is not None:
            out["reasoning_tokens"] = reasoning

    return out


# ---------------------------------------------------------------------------
# OpenAI Agent
# ---------------------------------------------------------------------------


class OpenAIAgent(BaseAgent):
    """An agent implementation backed by OpenAI's official Python SDK.

    The public API and run loop come from :class:`~.base_agent.BaseAgent`.
    """

    # Override BaseAgent defaults
    DEFAULT_MODEL = DEFAULT_MODEL
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
        formatter: FormatterType | None = None,
        enable_cache_control: Optional[bool] = None,
        compactor: Any = None,
        memory_store: Any = None,
        final_answer_check: Optional[Callable[[str], tuple[bool, str]]] = None,
        agent_uuid: str | None = None,
        db_backend: Any = "filesystem",
        file_backend: Any = None,
        openai_client: Any = None,
        openai_client_kwargs: Optional[dict[str, Any]] = None,
        **api_kwargs: Any,
    ):
        # OpenAI does not support Anthropic-style cache_control blocks; keep disabled by default.
        if enable_cache_control is None:
            enable_cache_control = False

        super().__init__(
            system_prompt=system_prompt,
            model=model,
            max_steps=max_steps,
            thinking_tokens=thinking_tokens,
            max_tokens=max_tokens,
            stream_meta_history_and_tool_results=stream_meta_history_and_tool_results,
            tools=tools,
            frontend_tools=frontend_tools,
            server_tools=server_tools,
            beta_headers=beta_headers,
            container_id=container_id,
            messages=messages,
            max_retries=max_retries,
            base_delay=base_delay,
            formatter=formatter,
            enable_cache_control=enable_cache_control,
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
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK is not installed. Install with `pip install openai`."
            ) from e

        if openai_client is not None:
            self.client = openai_client
        else:
            self.client = AsyncOpenAI(**(openai_client_kwargs or {}))

    # ------------------------------------------------------------------
    # Provider hooks required by BaseAgent
    # ------------------------------------------------------------------

    async def _stream_llm_response(
        self,
        *,
        queue: Optional[asyncio.Queue],
        formatter: FormatterType,
        tools_enabled: bool = True,
        system_prompt_override: Optional[str] = None,
    ) -> LLMResponse:
        system_prompt = system_prompt_override if system_prompt_override is not None else self.system_prompt

        openai_messages = _canonical_messages_to_openai_messages(
            canonical_messages=self.messages,
            system_prompt=system_prompt,
            model=self.model,
        )

        # Build tool payload (backend tools + frontend tools)
        tools: list[dict[str, Any]] = []
        if tools_enabled:
            if self.tool_registry:
                try:
                    backend_schemas = self.tool_registry.get_schemas(schema_type="openai")
                except TypeError:
                    backend_schemas = self.tool_registry.get_schemas()
                for s in backend_schemas or []:
                    converted = _to_openai_function_tool(s)
                    if converted:
                        tools.append(converted)

            for s in self.frontend_tool_schemas or []:
                converted = _to_openai_function_tool(s)
                if converted:
                    tools.append(converted)

        # Map thinking_tokens -> reasoning_effort when not explicitly set
        request_kwargs: dict[str, Any] = dict(self.api_kwargs or {})
        reasoning_effort = request_kwargs.pop("reasoning_effort", None)
        if reasoning_effort is None and self.thinking_tokens and self.thinking_tokens > 0:
            # Heuristic mapping: larger budgets -> higher effort
            reasoning_effort = "high" if self.thinking_tokens >= 4096 else "medium"

        # gpt-5.*-pro models only support high reasoning effort
        if (self.model or "").lower().endswith("pro") and reasoning_effort is None:
            reasoning_effort = "high"

        if reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = reasoning_effort

        request: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "max_completion_tokens": self.max_tokens,
            **request_kwargs,
        }

        if tools:
            request["tools"] = tools
            # Let the model decide; callers can override via api_kwargs/tool_choice
            request.setdefault("tool_choice", "auto")

        # Streaming path with retry/backoff
        if queue is not None:
            # Map FormatterType to OpenAIFormatterType (they're compatible)
            openai_formatter: OpenAIFormatterType = formatter if formatter in ("xml", "raw") else "xml"

            result = await openai_stream_with_backoff(
                client=self.client,
                request_params=request,
                queue=queue,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                formatter=openai_formatter,
                stream_tool_results=self.stream_meta_history_and_tool_results,
            )

            stop_reason = self._map_finish_reason(result.finish_reason, result.assistant_message)

            return LLMResponse(
                raw=None,
                assistant_message=result.assistant_message,
                stop_reason=stop_reason,
                model=result.model or self.model,
                usage=result.usage or {},
                container_id=None,
            )

        # Non-streaming path
        resp = await self.client.chat.completions.create(**request)
        assistant_message, stop_reason = self._response_to_canonical_message(resp)
        usage = _normalize_usage(getattr(resp, "usage", None))

        return LLMResponse(
            raw=resp,
            assistant_message=assistant_message,
            stop_reason=stop_reason,
            model=getattr(resp, "model", None) or self.model,
            usage=usage,
            container_id=None,
        )

    # ------------------------------------------------------------------
    # Response normalization helpers
    # ------------------------------------------------------------------

    def _map_finish_reason(self, finish_reason: Optional[str], assistant_message: dict[str, Any]) -> str:
        """Map OpenAI finish_reason to canonical stop_reason."""
        # Check if there are tool calls in the message
        content = assistant_message.get("content", [])
        has_tool_use = any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in (content if isinstance(content, list) else [])
        )

        if has_tool_use:
            return "tool_use"
        if finish_reason == "stop" or finish_reason is None:
            return "end_turn"
        if finish_reason == "length":
            return "max_tokens"
        if finish_reason == "tool_calls":
            return "tool_use"
        return str(finish_reason)

    def _response_to_canonical_message(self, resp: Any) -> tuple[dict[str, Any], str]:
        """Convert a non-streaming OpenAI response to canonical format."""
        choice = resp.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = choice.message

        blocks: list[dict[str, Any]] = []
        if getattr(message, "content", None):
            blocks.append({"type": "text", "text": message.content})

        tool_use_blocks: list[dict[str, Any]] = []
        if getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                call_id = getattr(tc, "id", None)
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn is not None else None
                arg_str = getattr(fn, "arguments", "{}") if fn is not None else "{}"
                try:
                    args = json.loads(arg_str) if isinstance(arg_str, str) else arg_str
                except Exception:
                    args = {"_raw": arg_str}
                tool_use_blocks.append({"type": "tool_use", "id": call_id, "name": name, "input": args})

        blocks.extend(tool_use_blocks)

        assistant_message: dict[str, Any] = {"role": "assistant", "content": blocks}
        stop_reason = self._map_finish_reason(finish_reason, assistant_message)
        return assistant_message, stop_reason
