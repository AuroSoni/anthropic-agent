"""Integration tests for AnthropicProvider.

Tests the full provider pipeline against the real Anthropic API:
provider → formatter → retry → API call → response parsing.

Uses claude-haiku-4-5-20251001 with minimal prompts to keep costs low.
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

from agent_base.core.messages import Message
from agent_base.core.types import (
    Role,
    TextContent,
    ThinkingContent,
    ToolUseContent,
    ToolResultContent,
)
from agent_base.providers.anthropic.anthropic_agent import AnthropicLLMConfig
from agent_base.providers.anthropic.formatters import AnthropicMessageFormatter
from agent_base.providers.anthropic.provider import AnthropicProvider
from agent_base.streaming.base import StreamFormatter
from agent_base.streaming.types import (
    StreamDelta,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
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

# Use the cheapest / fastest model for integration tests.
MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
def llm_config() -> AnthropicLLMConfig:
    return AnthropicLLMConfig(max_tokens=256)


# ===========================================================================
# Non-streaming generate() tests
# ===========================================================================


class TestGenerate:
    """Real API calls through AnthropicProvider.generate()."""

    async def test_simple_text_response(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Provider returns a well-formed canonical Message from the real API."""
        messages = [Message.user("Reply with exactly: PONG")]

        result = await provider.generate(
            system_prompt="You are a test bot. Follow instructions exactly.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert result.role == Role.ASSISTANT
        assert len(result.content) >= 1
        assert isinstance(result.content[0], TextContent)
        assert "PONG" in result.content[0].text
        assert result.usage is not None
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0
        assert result.provider == "anthropic"
        assert MODEL in result.model
        assert result.stop_reason == "end_turn"

    async def test_tool_use_response(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Provider triggers a tool call and parses the tool_use block."""
        messages = [Message.user("What is the weather in Tokyo?")]

        result = await provider.generate(
            system_prompt="Always use the get_weather tool when asked about weather. Never answer without calling the tool first.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert result.stop_reason == "tool_use"
        tool_blocks = [b for b in result.content if isinstance(b, ToolUseContent)]
        assert len(tool_blocks) >= 1
        tool = tool_blocks[0]
        assert tool.tool_name == "get_weather"
        assert tool.tool_id  # non-empty
        assert isinstance(tool.tool_input, dict)
        assert "city" in tool.tool_input or "Tokyo" in json.dumps(tool.tool_input)

    async def test_multi_turn_with_tool_result(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Provider handles a full tool-use round-trip: call → result → final answer."""
        # Turn 1: user asks, model should call tool
        messages = [Message.user("What is the weather in London?")]

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
        tool_block = next(b for b in turn1.content if isinstance(b, ToolUseContent))

        # Turn 2: supply tool result, model should produce text
        messages.append(Message(role=Role.ASSISTANT, content=turn1.content))
        messages.append(
            Message(
                role=Role.USER,
                content=[
                    ToolResultContent(
                        tool_name="get_weather",
                        tool_id=tool_block.tool_id,
                        tool_result="15°C, cloudy with light rain",
                    )
                ],
            )
        )

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
        text_blocks = [b for b in turn2.content if isinstance(b, TextContent)]
        assert len(text_blocks) >= 1
        combined_text = " ".join(b.text for b in text_blocks).lower()
        # The model should reference the weather data we provided
        assert any(
            term in combined_text
            for term in ["15", "cloudy", "rain", "london", "weather"]
        )

    async def test_multi_turn_conversation(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Provider correctly handles a multi-turn conversation."""
        messages = [
            Message.user("My name is Zephyr. Remember it."),
            Message.assistant("Got it! Your name is Zephyr."),
            Message.user("What is my name? Reply with just the name."),
        ]

        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert isinstance(result.content[0], TextContent)
        assert "Zephyr" in result.content[0].text

    async def test_system_prompt_influences_response(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """System prompt actually affects the model's behavior."""
        messages = [Message.user("What do you do?")]

        result = await provider.generate(
            system_prompt="You are a pirate. Always respond in pirate speak. Use 'arrr' in every response.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        text = result.content[0].text.lower()
        assert any(word in text for word in ["arr", "pirate", "mate", "ahoy", "ye", "sea"])

    async def test_usage_tokens_populated(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Usage object has valid token counts from the real API."""
        messages = [Message.user("Say hi.")]

        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        assert result.usage is not None
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0
        assert result.usage.raw_usage  # non-empty dict from model_dump


# ===========================================================================
# Streaming generate_stream() tests
# ===========================================================================


class TestGenerateStream:
    """Real streaming API calls through AnthropicProvider.generate_stream()."""

    async def test_stream_text_response(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Streaming returns text deltas and a well-formed final Message."""
        messages = [Message.user("Count from 1 to 5, separated by commas.")]
        queue: asyncio.Queue = asyncio.Queue()
        formatter = _CollectingStreamFormatter()

        result = await provider.generate_stream(
            system_prompt="Be concise.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue,
            stream_formatter=formatter,
        )

        # Final message is valid
        assert result.role == Role.ASSISTANT
        assert len(result.content) >= 1
        assert isinstance(result.content[0], TextContent)
        assert result.usage is not None
        assert result.stop_reason == "end_turn"

        # Stream deltas were collected
        text_deltas = [d for d in formatter.deltas if isinstance(d, TextDelta)]
        assert len(text_deltas) >= 2  # at least some chunks + final marker
        # Concatenated streamed text should contain the numbers
        streamed_text = "".join(d.text for d in text_deltas)
        for n in ["1", "2", "3", "4", "5"]:
            assert n in streamed_text

        # Final marker present
        assert any(d.is_final for d in text_deltas)

    async def test_stream_tool_call(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Streaming correctly buffers and emits a tool call delta."""
        messages = [Message.user("What is the weather in Berlin?")]
        queue: asyncio.Queue = asyncio.Queue()
        formatter = _CollectingStreamFormatter()

        result = await provider.generate_stream(
            system_prompt="Always use the get_weather tool. Never answer weather questions without it.",
            messages=messages,
            tool_schemas=[_weather_tool_schema()],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue,
            stream_formatter=formatter,
        )

        assert result.stop_reason == "tool_use"

        # Should have a ToolCallDelta
        tool_deltas = [d for d in formatter.deltas if isinstance(d, ToolCallDelta)]
        assert len(tool_deltas) >= 1
        td = tool_deltas[0]
        assert td.tool_name == "get_weather"
        assert td.tool_id  # non-empty
        parsed = json.loads(td.arguments_json)
        assert isinstance(parsed, dict)

    async def test_stream_deltas_match_final_message(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Streamed text content matches the final parsed message content."""
        messages = [Message.user("Reply with exactly: HELLO WORLD")]
        queue: asyncio.Queue = asyncio.Queue()
        formatter = _CollectingStreamFormatter()

        result = await provider.generate_stream(
            system_prompt="Follow instructions exactly.",
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue,
            stream_formatter=formatter,
        )

        # Final message text
        final_text = "".join(
            b.text for b in result.content if isinstance(b, TextContent)
        )

        # Streamed text (excluding final markers)
        streamed_text = "".join(
            d.text for d in formatter.deltas
            if isinstance(d, TextDelta) and d.text
        )

        assert "HELLO WORLD" in final_text
        assert "HELLO WORLD" in streamed_text

    async def test_stream_queue_receives_deltas(
        self,
        provider: AnthropicProvider,
        llm_config: AnthropicLLMConfig,
    ) -> None:
        """Items are actually placed on the asyncio.Queue during streaming."""
        messages = [Message.user("Say OK.")]
        queue: asyncio.Queue = asyncio.Queue()
        formatter = _CollectingStreamFormatter()

        await provider.generate_stream(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue,
            stream_formatter=formatter,
        )

        # Queue should have items (formatter puts each delta on the queue)
        assert not queue.empty()
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert len(items) >= 1
        assert all(isinstance(i, StreamDelta) for i in items)


# ===========================================================================
# Extended thinking tests
# ===========================================================================


class TestExtendedThinking:
    """Tests for extended thinking (requires a model that supports it)."""

    async def test_thinking_produces_thinking_content(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Extended thinking returns ThinkingContent + TextContent blocks."""
        llm_config = AnthropicLLMConfig(thinking_tokens=1024, max_tokens=2048)
        messages = [Message.user("What is 17 * 23?")]

        result = await provider.generate(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
        )

        thinking_blocks = [b for b in result.content if isinstance(b, ThinkingContent)]
        text_blocks = [b for b in result.content if isinstance(b, TextContent)]

        assert len(thinking_blocks) >= 1
        assert thinking_blocks[0].thinking  # non-empty thinking
        assert thinking_blocks[0].signature  # has a signature

        assert len(text_blocks) >= 1
        assert "391" in text_blocks[0].text

    async def test_thinking_stream_produces_thinking_deltas(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Streaming with extended thinking emits ThinkingDelta objects."""
        llm_config = AnthropicLLMConfig(thinking_tokens=1024, max_tokens=2048)
        messages = [Message.user("What is 7 + 8?")]
        queue: asyncio.Queue = asyncio.Queue()
        formatter = _CollectingStreamFormatter()

        result = await provider.generate_stream(
            system_prompt=None,
            messages=messages,
            tool_schemas=[],
            llm_config=llm_config,
            model=MODEL,
            max_retries=3,
            base_delay=1.0,
            queue=queue,
            stream_formatter=formatter,
        )

        thinking_deltas = [d for d in formatter.deltas if isinstance(d, ThinkingDelta)]
        text_deltas = [d for d in formatter.deltas if isinstance(d, TextDelta)]

        # Should have thinking deltas with content
        assert any(d.thinking for d in thinking_deltas)
        # Should have a thinking final marker
        assert any(d.is_final for d in thinking_deltas)
        # Should have text deltas
        assert any(d.text for d in text_deltas)

        # Final message should contain the answer
        text_blocks = [b for b in result.content if isinstance(b, TextContent)]
        assert any("15" in b.text for b in text_blocks)


# ===========================================================================
# Provider construction (no API calls needed)
# ===========================================================================


class TestProviderConstruction:
    """Tests for AnthropicProvider initialization."""

    def test_custom_client_used(self) -> None:
        """When a client is provided, it is used directly."""
        custom_client = anthropic.AsyncAnthropic()
        p = AnthropicProvider(client=custom_client)
        assert p.client is custom_client

    def test_custom_formatter_used(self) -> None:
        """When a formatter is provided, it is used directly."""
        custom_formatter = AnthropicMessageFormatter()
        p = AnthropicProvider(client=anthropic.AsyncAnthropic(), formatter=custom_formatter)
        assert p.formatter is custom_formatter

    def test_default_formatter_created(self) -> None:
        """When no formatter is provided, a default one is created."""
        p = AnthropicProvider(client=anthropic.AsyncAnthropic())
        assert isinstance(p.formatter, AnthropicMessageFormatter)
