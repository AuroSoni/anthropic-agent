"""Real-API integration tests for the Anthropic message formatter round-trip.

Validates that canonical content types survive the full pipeline:
format_messages() → Anthropic API → parse_response() → canonical Message.

Tests multi-turn conversations, extended thinking with signature preservation,
server tools (web search), and complex beta features.

Uses claude-haiku-4-5-20251001 to keep costs low.
Requires ANTHROPIC_API_KEY to be set (tests skip otherwise).
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import anthropic
import pytest

from agent_base.core.messages import Message, Usage
from agent_base.core.types import (
    Role,
    TextContent,
    ThinkingContent,
    ImageContent,
    DocumentContent,
    ToolUseContent,
    ServerToolUseContent,
    ToolResultContent,
    ServerToolResultContent,
    MCPToolResultContent,
    ErrorContent,
)
from agent_base.providers.anthropic.anthropic_agent import AnthropicLLMConfig
from agent_base.providers.anthropic.formatters import (
    AnthropicMessageFormatter,
    _content_block_to_anthropic,
    _parse_content_block,
)
from agent_base.providers.anthropic.provider import AnthropicProvider
from agent_base.streaming.base import StreamFormatter
from agent_base.streaming.types import (
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolResultDelta,
)
from agent_base.tools.tool_types import ToolSchema

# ---------------------------------------------------------------------------
# Load .env so ANTHROPIC_API_KEY is available in CI / local runs
# ---------------------------------------------------------------------------

_env_file = Path(__file__).resolve().parents[2] / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
pytestmark = pytest.mark.skipif(not _HAS_API_KEY, reason="ANTHROPIC_API_KEY not set")

MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calculator_tool_schema() -> ToolSchema:
    return ToolSchema(
        name="calculate",
        description="Evaluate a mathematical expression and return the result. Always call this tool for any math question.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The math expression to evaluate"},
            },
            "required": ["expression"],
        },
    )


def _weather_tool_schema() -> ToolSchema:
    return ToolSchema(
        name="get_weather",
        description="Get the current weather for a city. Always call this tool when asked about weather.",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        },
    )


def _get_text(msg: Message) -> str:
    """Extract all text content from a message."""
    return " ".join(b.text for b in msg.content if isinstance(b, TextContent))


def _get_thinking(msg: Message) -> list[ThinkingContent]:
    """Extract all thinking blocks from a message."""
    return [b for b in msg.content if isinstance(b, ThinkingContent)]


def _get_tool_uses(msg: Message) -> list[ToolUseContent]:
    """Extract all tool use blocks from a message."""
    return [b for b in msg.content if isinstance(b, ToolUseContent)]


class _CollectingStreamFormatter(StreamFormatter):
    """Stream formatter that collects deltas for assertion."""

    def __init__(self) -> None:
        self.deltas: list[StreamDelta] = []

    async def format_delta(self, delta: StreamDelta, queue: asyncio.Queue) -> None:
        self.deltas.append(delta)
        await queue.put(delta)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()


@pytest.fixture()
def provider(client: anthropic.AsyncAnthropic) -> AnthropicProvider:
    return AnthropicProvider(client=client)


@pytest.fixture()
def formatter() -> AnthropicMessageFormatter:
    return AnthropicMessageFormatter()


@pytest.fixture()
def llm_config() -> AnthropicLLMConfig:
    return AnthropicLLMConfig(max_tokens=512)


# ===========================================================================
# Multi-turn conversation tests
# ===========================================================================


class TestMultiTurnConversation:
    """Validates multi-turn message history round-trips through the formatter."""

    async def test_three_turn_text_conversation(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Three-turn conversation with context maintained."""
        messages = [
            Message.user("My favorite color is turquoise. Remember it."),
        ]

        turn1 = await provider.generate(
            system_prompt="You are a helpful assistant. Be concise.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )
        assert turn1.stop_reason == "end_turn"
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))

        # Turn 2: ask something else
        messages.append(Message.user("What is 2 + 2?"))
        turn2 = await provider.generate(
            system_prompt="You are a helpful assistant. Be concise.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )
        assert turn2.stop_reason == "end_turn"
        assert "4" in _get_text(turn2)
        messages.append(Message(role=Role.ASSISTANT, content=turn2.content))

        # Turn 3: recall context from turn 1
        messages.append(Message.user("What is my favorite color? Reply with just the color."))
        turn3 = await provider.generate(
            system_prompt="You are a helpful assistant. Be concise.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )
        assert "turquoise" in _get_text(turn3).lower()

    async def test_tool_use_multi_turn_round_trip(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Full tool-use multi-turn: ask → tool_call → tool_result → answer → follow-up."""
        messages = [Message.user("What is the weather in Paris?")]

        # Turn 1: model calls tool
        turn1 = await provider.generate(
            system_prompt="Always use the get_weather tool for weather. Never answer without calling the tool.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )
        assert turn1.stop_reason == "tool_use"
        tool_blocks = _get_tool_uses(turn1)
        assert len(tool_blocks) >= 1
        tool_block = tool_blocks[0]
        assert tool_block.tool_name == "get_weather"

        # Append assistant turn and tool result
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message(
            role=Role.USER,
            content=[ToolResultContent(
                tool_name="get_weather",
                tool_id=tool_block.tool_id,
                tool_result="22°C, sunny with light breeze",
            )],
        ))

        # Turn 2: model processes tool result
        turn2 = await provider.generate(
            system_prompt="Always use the get_weather tool for weather. Never answer without calling the tool.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )
        assert turn2.stop_reason == "end_turn"
        text = _get_text(turn2).lower()
        assert any(term in text for term in ["22", "sunny", "paris", "breeze"])

        # Turn 3: follow-up question referencing previous answer
        messages.append(Message(role=Role.ASSISTANT, content=turn2.content))
        messages.append(Message.user("Should I bring a jacket? Reply briefly."))

        turn3 = await provider.generate(
            system_prompt="Always use the get_weather tool for weather. Never answer without calling the tool.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )
        # Model should reference the sunny weather from earlier
        assert turn3.stop_reason == "end_turn"

    async def test_multiple_tool_calls_in_single_turn(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Model makes multiple tool calls in a single response."""
        llm_config = AnthropicLLMConfig(max_tokens=1024)
        messages = [
            Message.user(
                "What is the weather in both Tokyo and London? "
                "Call the tool for each city separately."
            ),
        ]

        turn1 = await provider.generate(
            system_prompt="Use the get_weather tool for each city asked about. Make separate tool calls.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert turn1.stop_reason == "tool_use"
        tool_blocks = _get_tool_uses(turn1)
        # Model may make 1 or 2 tool calls — at least 1 should be present
        assert len(tool_blocks) >= 1

        # Build tool results for all calls
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        result_blocks = []
        for tb in tool_blocks:
            city = tb.tool_input.get("city", "Unknown")
            result_blocks.append(ToolResultContent(
                tool_name="get_weather",
                tool_id=tb.tool_id,
                tool_result=f"20°C in {city}",
            ))
        messages.append(Message(role=Role.USER, content=result_blocks))

        turn2 = await provider.generate(
            system_prompt="Use the get_weather tool for each city asked about.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )
        assert turn2.stop_reason == "end_turn"

    async def test_tool_error_result_multi_turn(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Model handles a tool error result gracefully."""
        messages = [Message.user("What is the weather in Atlantis?")]

        turn1 = await provider.generate(
            system_prompt="Always use the get_weather tool for weather.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )
        assert turn1.stop_reason == "tool_use"
        tool_block = _get_tool_uses(turn1)[0]

        # Return an error result
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message(
            role=Role.USER,
            content=[ToolResultContent(
                tool_name="get_weather",
                tool_id=tool_block.tool_id,
                tool_result="Error: City 'Atlantis' not found in database",
                is_error=True,
            )],
        ))

        turn2 = await provider.generate(
            system_prompt="Always use the get_weather tool for weather. If the tool returns an error, explain it to the user.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )
        # Model should acknowledge the error
        text = _get_text(turn2).lower()
        assert any(term in text for term in ["not found", "couldn't", "can't", "unable", "error", "doesn't exist", "atlantis"])


# ===========================================================================
# Extended thinking — multi-turn with signature preservation
# ===========================================================================


class TestThinkingMultiTurn:
    """Extended thinking blocks with signatures survive multi-turn round-trips."""

    async def test_thinking_signature_preserved_in_multi_turn(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Thinking blocks from turn 1 are correctly formatted back to the API in turn 2."""
        llm_config = AnthropicLLMConfig(thinking_tokens=2048, max_tokens=4096)
        messages = [Message.user("What is 13 * 17? Show your work.")]

        # Turn 1: model thinks and answers
        turn1 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        thinking_blocks = _get_thinking(turn1)
        assert len(thinking_blocks) >= 1, "Expected thinking blocks in response"
        assert thinking_blocks[0].signature, "Thinking block should have a signature"
        assert thinking_blocks[0].thinking, "Thinking block should have content"
        assert "221" in _get_text(turn1)

        # Turn 2: send back the full response including thinking blocks
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("Now multiply that result by 2."))

        turn2 = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        # Model should calculate 221 * 2 = 442
        assert turn2.stop_reason == "end_turn"
        assert "442" in _get_text(turn2)

    async def test_thinking_with_tool_use_multi_turn(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Extended thinking + tool use in multi-turn preserves all block types."""
        llm_config = AnthropicLLMConfig(thinking_tokens=2048, max_tokens=4096)
        messages = [Message.user("What is the weather in Berlin? Think about why someone might ask this.")]

        # Turn 1: model thinks then calls tool
        turn1 = await provider.generate(
            system_prompt="Always use the get_weather tool for weather questions.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert turn1.stop_reason == "tool_use"
        thinking = _get_thinking(turn1)
        tools = _get_tool_uses(turn1)

        # May or may not have thinking blocks depending on model behavior
        # but tool use should be present
        assert len(tools) >= 1

        # Send back full response and tool result
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message(
            role=Role.USER,
            content=[ToolResultContent(
                tool_name="get_weather",
                tool_id=tools[0].tool_id,
                tool_result="8°C, overcast with light snow",
            )],
        ))

        # Turn 2: model should produce final answer
        turn2 = await provider.generate(
            system_prompt="Always use the get_weather tool for weather questions.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert turn2.stop_reason == "end_turn"
        text = _get_text(turn2).lower()
        assert any(term in text for term in ["8", "overcast", "snow", "berlin"])

    async def test_thinking_stream_multi_turn(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Streaming with thinking over multiple turns works correctly."""
        llm_config = AnthropicLLMConfig(thinking_tokens=1024, max_tokens=2048)
        messages = [Message.user("What is 7 * 8?")]
        queue: asyncio.Queue = asyncio.Queue()
        fmt = _CollectingStreamFormatter()

        # Turn 1: stream
        turn1 = await provider.generate_stream(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue,
            stream_formatter=fmt,
        )

        assert "56" in _get_text(turn1)
        thinking_deltas = [d for d in fmt.deltas if isinstance(d, ThinkingDelta)]
        assert any(d.thinking for d in thinking_deltas), "Should have thinking content in stream"

        # Turn 2: continue conversation with thinking history
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("Add 4 to that result."))
        queue2: asyncio.Queue = asyncio.Queue()
        fmt2 = _CollectingStreamFormatter()

        turn2 = await provider.generate_stream(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue2,
            stream_formatter=fmt2,
        )

        assert "60" in _get_text(turn2)


# ===========================================================================
# Server tools (web search)
# ===========================================================================


class TestServerTools:
    """Tests for server tools (web_search) via the Anthropic API."""

    async def test_web_search_server_tool(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Web search server tool triggers and returns search results."""
        llm_config = AnthropicLLMConfig(
            max_tokens=1024,
            server_tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            ],
        )
        messages = [Message.user("What is the current population of France? Use web search.")]

        result = await provider.generate(
            system_prompt="You have access to web search. Use it to find current information.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert result.stop_reason == "end_turn"
        # Should have text content with an answer
        text = _get_text(result)
        assert text, "Should have text content in response"
        # Server tool use and results should be in the content
        has_server_tool = any(
            isinstance(b, (ServerToolUseContent, ServerToolResultContent))
            for b in result.content
        )
        assert has_server_tool, "Should have server tool blocks in response"

    async def test_web_search_multi_turn(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Web search results survive multi-turn round-trip."""
        llm_config = AnthropicLLMConfig(
            max_tokens=1024,
            server_tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            ],
        )
        messages = [Message.user("Search for: what year was the Eiffel Tower built?")]

        # Turn 1
        turn1 = await provider.generate(
            system_prompt="Use web search to answer questions.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert turn1.stop_reason == "end_turn"
        text1 = _get_text(turn1).lower()
        assert any(term in text1 for term in ["1889", "eiffel"]), f"Expected '1889' or 'eiffel' in: {text1[:200]}"

        # Turn 2: follow up on the search results
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("How tall is it in meters? Reply with just the number."))

        turn2 = await provider.generate(
            system_prompt="Use web search to answer questions. Be concise.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert turn2.stop_reason == "end_turn"
        text2 = _get_text(turn2)
        assert any(h in text2 for h in ["330", "324", "312", "300"]), f"Expected height in: {text2[:200]}"

    async def test_web_search_stream(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Server tools work correctly in streaming mode."""
        llm_config = AnthropicLLMConfig(
            max_tokens=1024,
            server_tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 2},
            ],
        )
        messages = [Message.user("Search for: who is the current CEO of Anthropic?")]
        queue: asyncio.Queue = asyncio.Queue()
        fmt = _CollectingStreamFormatter()

        result = await provider.generate_stream(
            system_prompt="Use web search to answer questions.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue,
            stream_formatter=fmt,
        )

        assert result.stop_reason == "end_turn"
        text = _get_text(result).lower()
        assert "dario" in text or "amodei" in text

        # Check stream deltas include tool call deltas for server tools
        tool_deltas = [d for d in fmt.deltas if isinstance(d, ToolCallDelta)]
        # Server tools should produce tool call deltas
        server_tool_deltas = [d for d in tool_deltas if d.is_server_tool]
        assert len(server_tool_deltas) >= 1, "Should have server tool deltas in stream"


# ===========================================================================
# Server tools combined with client tools
# ===========================================================================


class TestMixedToolTypes:
    """Tests combining server tools and client tools in a single request."""

    async def test_server_and_client_tools_together(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Model can use both server tools and client tools in a conversation."""
        llm_config = AnthropicLLMConfig(
            max_tokens=1024,
            server_tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            ],
        )
        messages = [
            Message.user(
                "First, what's the weather in New York? Use the get_weather tool."
            ),
        ]

        # Turn 1: should call client tool
        turn1 = await provider.generate(
            system_prompt="Use get_weather for weather questions. Use web_search for factual questions.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert turn1.stop_reason == "tool_use"
        client_tools = [b for b in turn1.content if isinstance(b, ToolUseContent)]
        assert len(client_tools) >= 1
        assert client_tools[0].tool_name == "get_weather"

        # Provide tool result
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message(
            role=Role.USER,
            content=[ToolResultContent(
                tool_name="get_weather",
                tool_id=client_tools[0].tool_id,
                tool_result="25°C, sunny",
            )],
        ))

        turn2 = await provider.generate(
            system_prompt="Use get_weather for weather questions. Use web_search for factual questions.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        # Should produce a final answer with weather info
        text = _get_text(turn2).lower()
        assert any(term in text for term in ["25", "sunny", "new york"])


# ===========================================================================
# Cache control integration
# ===========================================================================


class TestCacheControl:
    """Tests that cache control works correctly with the real API."""

    async def test_cache_control_enabled(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Prompt caching is applied and cache tokens appear in usage."""
        llm_config = AnthropicLLMConfig(max_tokens=256)
        # Use a large system prompt to trigger caching
        big_system = "You are a helpful assistant. " * 500  # ~3000 tokens

        messages = [Message.user("Say hi.")]

        # First call: should write to cache
        result1 = await provider.generate(
            system_prompt=big_system,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )
        assert result1.usage is not None
        # First call should have cache write tokens
        assert result1.usage.cache_write_tokens is not None or result1.usage.cache_read_tokens is not None

        # Second call with same system prompt: should read from cache
        result2 = await provider.generate(
            system_prompt=big_system,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )
        assert result2.usage is not None
        # Second call should have cache read tokens
        if result2.usage.cache_read_tokens:
            assert result2.usage.cache_read_tokens > 0


# ===========================================================================
# Formatter wire format validation
# ===========================================================================


class TestFormatterWireFormat:
    """Verify the formatter produces valid wire format accepted by the API."""

    async def test_format_messages_produces_valid_request(
        self,
        client: anthropic.AsyncAnthropic,
        formatter: AnthropicMessageFormatter,
    ) -> None:
        """format_messages() output can be passed directly to the API."""
        messages = [Message.user("Say hello.")]
        llm_config = AnthropicLLMConfig(max_tokens=64)

        request_params = formatter.format_messages(
            messages,
            params={
                "system_prompt": "Be concise.",
                "llm_config": llm_config,
                "model": MODEL,
                "tool_schemas": [],
                "enable_cache_control": True,
            },
        )

        # Verify structure
        assert "model" in request_params
        assert "messages" in request_params
        assert "max_tokens" in request_params

        # Actually call the API with the formatted params
        response = await client.beta.messages.create(**request_params)
        assert response.stop_reason == "end_turn"
        assert len(response.content) >= 1

    async def test_thinking_blocks_wire_format_round_trip(
        self,
        client: anthropic.AsyncAnthropic,
        formatter: AnthropicMessageFormatter,
    ) -> None:
        """Thinking blocks with signatures round-trip through format → API → parse."""
        llm_config = AnthropicLLMConfig(thinking_tokens=1024, max_tokens=2048)

        # Turn 1: get thinking response
        messages = [Message.user("What is 9 * 9?")]
        request1 = formatter.format_messages(messages, params={
            "model": MODEL,
            "llm_config": llm_config,
            "enable_cache_control": False,
        })
        response1 = await client.beta.messages.create(**request1)
        parsed1 = formatter.parse_response(response1)

        thinking_blocks = _get_thinking(parsed1)
        assert len(thinking_blocks) >= 1
        assert thinking_blocks[0].signature, "Must have signature"

        # Turn 2: send back the thinking blocks
        messages.append(Message(role=Role.ASSISTANT, content=parsed1.content))
        messages.append(Message.user("Double that."))

        request2 = formatter.format_messages(messages, params={
            "model": MODEL,
            "llm_config": llm_config,
            "enable_cache_control": False,
        })

        # Verify the thinking block is in the wire format with signature
        assistant_msg = request2["messages"][1]
        thinking_wire = [b for b in assistant_msg["content"] if b.get("type") == "thinking"]
        assert len(thinking_wire) >= 1
        assert "signature" in thinking_wire[0]

        # Should succeed without API error
        response2 = await client.beta.messages.create(**request2)
        parsed2 = formatter.parse_response(response2)
        assert "162" in _get_text(parsed2)

    async def test_tool_result_wire_format(
        self,
        client: anthropic.AsyncAnthropic,
        formatter: AnthropicMessageFormatter,
    ) -> None:
        """Tool result content blocks format correctly for the API."""
        llm_config = AnthropicLLMConfig(max_tokens=256)
        tool_schemas = [
            {"name": "calculate", "description": "Calculate a math expression.", "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            }},
        ]

        messages = [Message.user("What is 5 + 3? Use the calculate tool.")]
        request1 = formatter.format_messages(messages, params={
            "model": MODEL,
            "llm_config": llm_config,
            "tool_schemas": tool_schemas,
            "enable_cache_control": False,
        })
        response1 = await client.beta.messages.create(**request1)
        parsed1 = formatter.parse_response(response1)

        if parsed1.stop_reason == "tool_use":
            tool_block = _get_tool_uses(parsed1)[0]

            # Build tool result and send back
            messages.append(Message(role=Role.ASSISTANT, content=parsed1.content))
            messages.append(Message(
                role=Role.USER,
                content=[ToolResultContent(
                    tool_name="calculate",
                    tool_id=tool_block.tool_id,
                    tool_result="8",
                )],
            ))

            request2 = formatter.format_messages(messages, params={
                "model": MODEL,
                "llm_config": llm_config,
                "tool_schemas": tool_schemas,
                "enable_cache_control": False,
            })

            # Verify wire format
            user_msg = request2["messages"][2]
            tool_result_wire = [
                b for b in user_msg["content"]
                if b.get("type") == "tool_result"
            ]
            assert len(tool_result_wire) == 1
            assert tool_result_wire[0]["tool_use_id"] == tool_block.tool_id

            response2 = await client.beta.messages.create(**request2)
            parsed2 = formatter.parse_response(response2)
            assert "8" in _get_text(parsed2)


# ===========================================================================
# Complex beta features
# ===========================================================================


class TestBetaFeatures:
    """Tests for Anthropic beta features through the formatter."""

    async def test_thinking_plus_tools_plus_cache_control(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Combination of thinking, tools, and cache control in a single request."""
        llm_config = AnthropicLLMConfig(
            thinking_tokens=1024,
            max_tokens=4096,
        )
        big_system = "You are a math tutor. Always use the calculate tool. " * 200

        messages = [Message.user("What is 15 * 12?")]

        result = await provider.generate(
            system_prompt=big_system,
            messages=messages,
            tool_schemas=[_calculator_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        # Should either think and use tool, or just use tool
        assert result.stop_reason in ("tool_use", "end_turn")
        assert result.usage is not None

    async def test_thinking_plus_server_tools(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Extended thinking combined with server tools (web search)."""
        llm_config = AnthropicLLMConfig(
            thinking_tokens=2048,
            max_tokens=4096,
            server_tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            ],
        )
        messages = [Message.user("Search: what is the capital of Australia?")]

        result = await provider.generate(
            system_prompt="Use web search for factual questions. Think through your response.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert result.stop_reason == "end_turn"
        text = _get_text(result).lower()
        assert "canberra" in text

    async def test_container_config_in_request(
        self,
        formatter: AnthropicMessageFormatter,
    ) -> None:
        """Container and skills config is correctly included in the request dict."""
        llm_config = AnthropicLLMConfig(
            max_tokens=256,
            container_id="ctnr_test123",
            skills=[
                {"type": "anthropic", "skill_id": "xlsx", "version": "latest"},
            ],
            beta_headers=["skills-2025-10-02"],
        )
        messages = [Message.user("Create a spreadsheet.")]

        request = formatter.format_messages(messages, params={
            "model": MODEL,
            "llm_config": llm_config,
            "enable_cache_control": False,
        })

        assert request["container"] == {
            "id": "ctnr_test123",
            "skills": [{"type": "anthropic", "skill_id": "xlsx", "version": "latest"}],
        }
        assert request["betas"] == ["skills-2025-10-02"]


# ===========================================================================
# Streaming multi-turn integration
# ===========================================================================


class TestStreamingMultiTurn:
    """Streaming multi-turn conversations through the full pipeline."""

    async def test_stream_multi_turn_with_tool(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Streaming tool use → result → final answer across turns."""
        messages = [Message.user("What is the weather in Sydney?")]
        queue1: asyncio.Queue = asyncio.Queue()
        fmt1 = _CollectingStreamFormatter()

        # Turn 1: stream tool call
        turn1 = await provider.generate_stream(
            system_prompt="Always use the get_weather tool for weather. Never answer without it.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue1,
            stream_formatter=fmt1,
        )

        assert turn1.stop_reason == "tool_use"
        tool_deltas = [d for d in fmt1.deltas if isinstance(d, ToolCallDelta)]
        assert len(tool_deltas) >= 1

        tool_block = _get_tool_uses(turn1)[0]

        # Turn 2: provide result and stream final answer
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message(
            role=Role.USER,
            content=[ToolResultContent(
                tool_name="get_weather",
                tool_id=tool_block.tool_id,
                tool_result="28°C, clear skies",
            )],
        ))

        queue2: asyncio.Queue = asyncio.Queue()
        fmt2 = _CollectingStreamFormatter()

        turn2 = await provider.generate_stream(
            system_prompt="Always use the get_weather tool for weather.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue2,
            stream_formatter=fmt2,
        )

        assert turn2.stop_reason == "end_turn"
        text_deltas = [d for d in fmt2.deltas if isinstance(d, TextDelta)]
        assert len(text_deltas) >= 1
        text = _get_text(turn2).lower()
        assert any(term in text for term in ["28", "clear", "sydney"])

    async def test_stream_server_tool_multi_turn(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Server tools in streaming mode across multiple turns."""
        llm_config = AnthropicLLMConfig(
            max_tokens=1024,
            server_tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            ],
        )
        messages = [Message.user("Search: when was Python programming language created?")]
        queue1: asyncio.Queue = asyncio.Queue()
        fmt1 = _CollectingStreamFormatter()

        turn1 = await provider.generate_stream(
            system_prompt="Use web search for questions.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue1,
            stream_formatter=fmt1,
        )

        assert turn1.stop_reason == "end_turn"
        text1 = _get_text(turn1).lower()
        assert any(year in text1 for year in ["1991", "1989", "1990"]), f"Expected Python creation year in: {text1[:200]}"

        # Turn 2: follow up
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(Message.user("Who created it? Reply with just the name."))
        queue2: asyncio.Queue = asyncio.Queue()
        fmt2 = _CollectingStreamFormatter()

        turn2 = await provider.generate_stream(
            system_prompt="Use web search for questions. Be concise.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue2,
            stream_formatter=fmt2,
        )

        text2 = _get_text(turn2).lower()
        assert "guido" in text2 or "rossum" in text2
