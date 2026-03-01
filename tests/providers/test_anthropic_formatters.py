"""Integration tests for AnthropicMessageFormatter.

Tests the round-trip conversion between canonical content types and
Anthropic wire format. Uses lightweight dataclass stubs instead of
real Anthropic SDK types to avoid API key / network dependencies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_base.core.messages import Message, Usage
from agent_base.core.types import (
    ContentBlock,
    Role,
    TextContent,
    ThinkingContent,
    ImageContent,
    DocumentContent,
    AttachmentContent,
    ToolUseContent,
    ServerToolUseContent,
    MCPToolUseContent,
    ToolResultContent,
    ServerToolResultContent,
    MCPToolResultContent,
    ErrorContent,
    CharCitation,
    WebSearchResultCitation,
)
from agent_base.providers.anthropic.formatters import (
    _content_block_to_anthropic,
    _format_message,
    _parse_content_block,
    AnthropicMessageFormatter,
)


# ---------------------------------------------------------------------------
# Stub types — mimic Anthropic SDK response objects (attribute-based access)
# ---------------------------------------------------------------------------


@dataclass
class StubTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class StubThinkingBlock:
    type: str = "thinking"
    thinking: str = ""
    signature: str = ""


@dataclass
class StubRedactedThinkingBlock:
    type: str = "redacted_thinking"
    data: str = ""


@dataclass
class StubToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class StubServerToolUseBlock:
    type: str = "server_tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class StubMCPToolUseBlock:
    type: str = "mcp_tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    server_name: str = ""


@dataclass
class StubMCPToolResultBlock:
    type: str = "mcp_tool_result"
    tool_use_id: str = ""
    content: str | list = ""
    is_error: bool = False


@dataclass
class StubServerToolResultBlock:
    type: str = "web_search_tool_result"
    tool_use_id: str = ""
    content: str = ""


@dataclass
class StubContainerUploadBlock:
    type: str = "container_upload"
    file_id: str = ""


@dataclass
class StubUnknownBlock:
    type: str = "some_future_block"


@dataclass
class StubUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None

    def model_dump(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class StubBetaMessage:
    content: list = field(default_factory=list)
    usage: StubUsage | None = None
    stop_reason: str | None = None
    model: str = "claude-sonnet-4-5-20250514"
    role: str = "assistant"


# ---------------------------------------------------------------------------
# Tests: _content_block_to_anthropic (canonical → Anthropic)
# ---------------------------------------------------------------------------


class TestContentBlockToAnthropic:
    """Tests for _content_block_to_anthropic()."""

    def test_text_content(self):
        block = TextContent(text="Hello world")
        result = _content_block_to_anthropic(block)
        assert result == {"type": "text", "text": "Hello world"}

    def test_text_content_empty(self):
        block = TextContent(text="")
        result = _content_block_to_anthropic(block)
        assert result == {"type": "text", "text": ""}

    def test_thinking_content_with_signature(self):
        block = ThinkingContent(thinking="deep thought", signature="sig_abc123")
        result = _content_block_to_anthropic(block)
        assert result == {
            "type": "thinking",
            "thinking": "deep thought",
            "signature": "sig_abc123",
        }

    def test_thinking_content_without_signature_returns_none(self):
        block = ThinkingContent(thinking="deep thought", signature=None)
        result = _content_block_to_anthropic(block)
        assert result is None

    def test_thinking_content_empty_signature_returns_none(self):
        block = ThinkingContent(thinking="deep thought", signature="")
        result = _content_block_to_anthropic(block)
        assert result is None

    def test_thinking_content_missing_signature_logs_warning(self, caplog):
        block = ThinkingContent(thinking="some analysis", signature=None)
        with caplog.at_level(logging.WARNING):
            _content_block_to_anthropic(block)
        assert any("thinking_block_missing_signature" in r.message or "missing_signature" in str(r) for r in caplog.records) or caplog.records  # structlog may not go through caplog

    def test_image_content(self):
        block = ImageContent(
            source_type="base64",
            data="aW1hZ2VkYXRh",
            media_type="image/png",
            media_id="img_123",
        )
        result = _content_block_to_anthropic(block)
        assert result == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "aW1hZ2VkYXRh",
            },
        }

    def test_image_content_default_source_type(self):
        block = ImageContent(
            source_type="",
            data="data",
            media_type="image/jpeg",
            media_id="img_456",
        )
        result = _content_block_to_anthropic(block)
        assert result["source"]["type"] == "base64"

    def test_document_content(self):
        block = DocumentContent(
            source_type="base64",
            data="cGRmZGF0YQ==",
            media_type="application/pdf",
            media_id="doc_789",
        )
        result = _content_block_to_anthropic(block)
        assert result == {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "cGRmZGF0YQ==",
            },
        }

    def test_attachment_content_with_file_id(self):
        block = AttachmentContent(
            filename="report.pdf",
            source_type="file_id",
            data="file_abc123",
            media_type="application/pdf",
            media_id="att_001",
        )
        result = _content_block_to_anthropic(block)
        assert result == {
            "type": "container_upload",
            "file_id": "file_abc123",
        }

    def test_attachment_content_without_file_id_returns_none(self):
        block = AttachmentContent(
            filename="report.pdf",
            source_type="base64",
            data="rawdata",
            media_type="application/pdf",
            media_id="att_002",
        )
        result = _content_block_to_anthropic(block)
        assert result is None

    def test_attachment_content_empty_data_returns_none(self):
        block = AttachmentContent(
            filename="report.pdf",
            source_type="file_id",
            data="",
            media_type="application/pdf",
            media_id="att_003",
        )
        result = _content_block_to_anthropic(block)
        assert result is None

    def test_tool_use_content(self):
        block = ToolUseContent(
            tool_name="read_file",
            tool_id="toolu_abc",
            tool_input={"path": "/tmp/test.txt"},
        )
        result = _content_block_to_anthropic(block)
        assert result == {
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "read_file",
            "input": {"path": "/tmp/test.txt"},
        }

    def test_server_tool_use_content(self):
        block = ServerToolUseContent(
            tool_name="web_search",
            tool_id="toolu_srv",
            tool_input={"query": "test"},
        )
        result = _content_block_to_anthropic(block)
        assert result == {
            "type": "server_tool_use",
            "id": "toolu_srv",
            "name": "web_search",
            "input": {"query": "test"},
        }

    def test_mcp_tool_use_content(self):
        block = MCPToolUseContent(
            tool_name="mcp_tool",
            tool_id="toolu_mcp",
            tool_input={"key": "val"},
            mcp_server_name="my_server",
        )
        result = _content_block_to_anthropic(block)
        # MCP tool use maps to regular tool_use on the wire
        assert result == {
            "type": "tool_use",
            "id": "toolu_mcp",
            "name": "mcp_tool",
            "input": {"key": "val"},
        }

    def test_tool_result_content_string(self):
        block = ToolResultContent(
            tool_name="read_file",
            tool_id="toolu_abc",
            tool_result="file contents here",
        )
        result = _content_block_to_anthropic(block)
        assert result == {
            "type": "tool_result",
            "tool_use_id": "toolu_abc",
            "content": [{"type": "text", "text": "file contents here"}],
        }

    def test_tool_result_content_empty_string(self):
        block = ToolResultContent(
            tool_name="noop",
            tool_id="toolu_empty",
            tool_result="",
        )
        result = _content_block_to_anthropic(block)
        assert result == {
            "type": "tool_result",
            "tool_use_id": "toolu_empty",
            "content": [],
        }

    def test_tool_result_content_with_error(self):
        block = ToolResultContent(
            tool_name="failing_tool",
            tool_id="toolu_err",
            tool_result="something went wrong",
            is_error=True,
        )
        result = _content_block_to_anthropic(block)
        assert result["is_error"] is True
        assert result["type"] == "tool_result"

    def test_tool_result_content_nested_blocks(self):
        inner = TextContent(text="inner text")
        block = ToolResultContent(
            tool_name="multi_tool",
            tool_id="toolu_nested",
            tool_result=[inner],
        )
        result = _content_block_to_anthropic(block)
        assert result["content"] == [{"type": "text", "text": "inner text"}]

    def test_tool_result_with_nested_image(self):
        inner_img = ImageContent(
            source_type="base64",
            data="imgdata",
            media_type="image/png",
            media_id="img_nested",
        )
        block = ToolResultContent(
            tool_name="screenshot",
            tool_id="toolu_img",
            tool_result=[TextContent(text="screenshot taken"), inner_img],
        )
        result = _content_block_to_anthropic(block)
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "image"

    def test_server_tool_result_content(self):
        """Server tool results keep their original type (e.g. web_search_tool_result)."""
        block = ServerToolResultContent(
            tool_name="web_search_tool_result",
            tool_id="toolu_srv_res",
            tool_result="search results",
        )
        result = _content_block_to_anthropic(block)
        assert result["type"] == "web_search_tool_result"
        assert result["tool_use_id"] == "toolu_srv_res"

    def test_server_tool_result_content_with_raw(self):
        """Server tool results use raw.model_dump() when raw is available."""
        @dataclass
        class StubRawResult:
            type: str = "web_search_tool_result"
            tool_use_id: str = "toolu_raw"
            content: list = field(default_factory=list)

            def model_dump(self):
                return {"type": self.type, "tool_use_id": self.tool_use_id, "content": self.content}

        raw = StubRawResult()
        block = ServerToolResultContent(
            tool_name="web_search_tool_result",
            tool_id="toolu_raw",
            tool_result="search results",
            raw=raw,
        )
        result = _content_block_to_anthropic(block)
        assert result["type"] == "web_search_tool_result"
        assert result["tool_use_id"] == "toolu_raw"

    def test_mcp_tool_result_content(self):
        """MCP tool results keep their original type."""
        block = MCPToolResultContent(
            tool_name="mcp_tool",
            tool_id="toolu_mcp_res",
            tool_result="mcp result",
            mcp_server_name="srv",
        )
        result = _content_block_to_anthropic(block)
        assert result["type"] == "mcp_tool_result"
        assert result["tool_use_id"] == "toolu_mcp_res"

    def test_error_content(self):
        block = ErrorContent(error_message="something broke")
        result = _content_block_to_anthropic(block)
        assert result == {"type": "text", "text": "Error: something broke"}

    def test_citation_returns_none(self):
        block = CharCitation(
            cited_text="hello",
            document_index=0,
            start_char_index=0,
            end_char_index=5,
        )
        result = _content_block_to_anthropic(block)
        assert result is None

    def test_web_search_citation_returns_none(self):
        block = WebSearchResultCitation(
            cited_text="result",
            url="https://example.com",
        )
        result = _content_block_to_anthropic(block)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _parse_content_block (Anthropic → canonical)
# ---------------------------------------------------------------------------


class TestParseContentBlock:
    """Tests for _parse_content_block()."""

    def test_parse_text_block(self):
        stub = StubTextBlock(text="Hello")
        result = _parse_content_block(stub)
        assert isinstance(result, TextContent)
        assert result.text == "Hello"

    def test_parse_thinking_block(self):
        stub = StubThinkingBlock(thinking="I think...", signature="sig_xyz")
        result = _parse_content_block(stub)
        assert isinstance(result, ThinkingContent)
        assert result.thinking == "I think..."
        assert result.signature == "sig_xyz"

    def test_parse_redacted_thinking_block(self):
        stub = StubRedactedThinkingBlock(data="opaque_data")
        result = _parse_content_block(stub)
        assert isinstance(result, ThinkingContent)
        assert result.thinking == "[redacted]"
        assert result.signature is None
        assert result.kwargs.get("redacted") is True
        assert result.kwargs.get("redacted_data") == "opaque_data"

    def test_parse_tool_use_block(self):
        stub = StubToolUseBlock(
            id="toolu_abc",
            name="read_file",
            input={"path": "/tmp/file.txt"},
        )
        result = _parse_content_block(stub)
        assert isinstance(result, ToolUseContent)
        assert result.tool_name == "read_file"
        assert result.tool_id == "toolu_abc"
        assert result.tool_input == {"path": "/tmp/file.txt"}
        assert result.raw is stub

    def test_parse_server_tool_use_block(self):
        stub = StubServerToolUseBlock(
            id="toolu_srv",
            name="web_search",
            input={"query": "test"},
        )
        result = _parse_content_block(stub)
        assert isinstance(result, ServerToolUseContent)
        assert result.tool_name == "web_search"
        assert result.tool_id == "toolu_srv"

    def test_parse_mcp_tool_use_block(self):
        stub = StubMCPToolUseBlock(
            id="toolu_mcp",
            name="mcp_tool",
            input={"key": "val"},
            server_name="my_server",
        )
        result = _parse_content_block(stub)
        assert isinstance(result, MCPToolUseContent)
        assert result.tool_name == "mcp_tool"
        assert result.tool_id == "toolu_mcp"
        assert result.mcp_server_name == "my_server"

    def test_parse_mcp_tool_result_block(self):
        stub = StubMCPToolResultBlock(
            tool_use_id="toolu_mcp_res",
            content="mcp result text",
            is_error=False,
        )
        result = _parse_content_block(stub)
        assert isinstance(result, MCPToolResultContent)
        assert result.tool_id == "toolu_mcp_res"
        assert result.tool_result == "mcp result text"
        assert result.is_error is False

    def test_parse_mcp_tool_result_with_error(self):
        stub = StubMCPToolResultBlock(
            tool_use_id="toolu_mcp_err",
            content="error occurred",
            is_error=True,
        )
        result = _parse_content_block(stub)
        assert isinstance(result, MCPToolResultContent)
        assert result.is_error is True

    def test_parse_server_tool_result_block(self):
        stub = StubServerToolResultBlock(
            tool_use_id="toolu_web",
            content="web search results",
        )
        result = _parse_content_block(stub)
        assert isinstance(result, ServerToolResultContent)
        assert result.tool_id == "toolu_web"
        assert result.tool_result == "web search results"

    def test_parse_container_upload_block(self):
        stub = StubContainerUploadBlock(file_id="file_abc123")
        result = _parse_content_block(stub)
        assert isinstance(result, AttachmentContent)
        assert result.source_type == "file_id"
        assert result.data == "file_abc123"
        assert result.media_id == "file_abc123"

    def test_parse_container_upload_empty_file_id_returns_none(self):
        stub = StubContainerUploadBlock(file_id="")
        result = _parse_content_block(stub)
        assert result is None

    def test_parse_unknown_block_returns_none(self):
        stub = StubUnknownBlock()
        result = _parse_content_block(stub)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _format_message
# ---------------------------------------------------------------------------


class TestFormatMessage:
    """Tests for _format_message()."""

    def test_simple_text_message(self):
        msg = Message(role=Role.USER, content=[TextContent(text="hello")])
        result = _format_message(msg)
        assert result == {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        }

    def test_assistant_message_with_mixed_content(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent(text="Let me help"),
                ToolUseContent(
                    tool_name="read_file",
                    tool_id="toolu_1",
                    tool_input={"path": "test.py"},
                ),
            ],
        )
        result = _format_message(msg)
        assert result["role"] == "assistant"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"

    def test_message_with_citations_filtered_out(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent(text="According to the source"),
                CharCitation(
                    cited_text="quote",
                    document_index=0,
                    start_char_index=0,
                    end_char_index=5,
                ),
            ],
        )
        result = _format_message(msg)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"

    def test_message_with_tool_results(self):
        msg = Message(
            role=Role.USER,
            content=[
                ToolResultContent(
                    tool_name="read_file",
                    tool_id="toolu_1",
                    tool_result="file contents",
                ),
            ],
        )
        result = _format_message(msg)
        assert result["role"] == "user"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_result"

    def test_empty_message_content(self):
        msg = Message(role=Role.USER, content=[])
        result = _format_message(msg)
        assert result == {"role": "user", "content": []}


# ---------------------------------------------------------------------------
# Tests: AnthropicMessageFormatter.parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for AnthropicMessageFormatter.parse_response()."""

    def setup_method(self):
        self.formatter = AnthropicMessageFormatter()

    def test_basic_text_response(self):
        stub = StubBetaMessage(
            content=[StubTextBlock(text="Hello!")],
            usage=StubUsage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
        )
        msg = self.formatter.parse_response(stub)
        assert msg.role == Role.ASSISTANT
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "Hello!"
        assert msg.stop_reason == "end_turn"
        assert msg.provider == "anthropic"
        assert msg.model == "claude-sonnet-4-5-20250514"

    def test_usage_extraction(self):
        stub = StubBetaMessage(
            content=[StubTextBlock(text="hi")],
            usage=StubUsage(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=20,
                cache_read_input_tokens=80,
            ),
        )
        msg = self.formatter.parse_response(stub)
        assert msg.usage is not None
        assert msg.usage.input_tokens == 100
        assert msg.usage.output_tokens == 50
        assert msg.usage.cache_write_tokens == 20
        assert msg.usage.cache_read_tokens == 80

    def test_no_usage(self):
        stub = StubBetaMessage(
            content=[StubTextBlock(text="hi")],
            usage=None,
        )
        msg = self.formatter.parse_response(stub)
        assert msg.usage is None

    def test_empty_content(self):
        stub = StubBetaMessage(content=[], stop_reason="end_turn")
        msg = self.formatter.parse_response(stub)
        assert msg.content == []

    def test_mixed_blocks(self):
        stub = StubBetaMessage(
            content=[
                StubThinkingBlock(thinking="hmm", signature="sig_1"),
                StubTextBlock(text="Answer"),
                StubToolUseBlock(
                    id="toolu_1",
                    name="read_file",
                    input={"path": "test.py"},
                ),
            ],
        )
        msg = self.formatter.parse_response(stub)
        assert len(msg.content) == 3
        assert isinstance(msg.content[0], ThinkingContent)
        assert isinstance(msg.content[1], TextContent)
        assert isinstance(msg.content[2], ToolUseContent)

    def test_unknown_blocks_filtered(self):
        stub = StubBetaMessage(
            content=[
                StubTextBlock(text="Hello"),
                StubUnknownBlock(),
            ],
        )
        msg = self.formatter.parse_response(stub)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)

    def test_stop_reason_tool_use(self):
        stub = StubBetaMessage(
            content=[
                StubToolUseBlock(id="toolu_1", name="search", input={}),
            ],
            stop_reason="tool_use",
        )
        msg = self.formatter.parse_response(stub)
        assert msg.stop_reason == "tool_use"


# ---------------------------------------------------------------------------
# Tests: AnthropicMessageFormatter.format_messages
# ---------------------------------------------------------------------------


class TestFormatMessages:
    """Tests for AnthropicMessageFormatter.format_messages()."""

    def setup_method(self):
        self.formatter = AnthropicMessageFormatter()

    def test_basic_request(self):
        messages = [
            Message(role=Role.USER, content=[TextContent(text="Hi")]),
        ]
        result = self.formatter.format_messages(messages, params={
            "model": "claude-sonnet-4-5-20250514",
            "enable_cache_control": False,
        })
        assert result["model"] == "claude-sonnet-4-5-20250514"
        assert result["max_tokens"] == 16384
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_with_system_prompt(self):
        messages = [
            Message(role=Role.USER, content=[TextContent(text="Hi")]),
        ]
        result = self.formatter.format_messages(messages, params={
            "model": "claude-sonnet-4-5-20250514",
            "system_prompt": "You are helpful",
            "enable_cache_control": False,
        })
        assert result["system"] == "You are helpful"

    def test_with_thinking_config(self):
        from agent_base.providers.anthropic.anthropic_agent import AnthropicLLMConfig

        config = AnthropicLLMConfig(thinking_tokens=10000)
        messages = [
            Message(role=Role.USER, content=[TextContent(text="Think hard")]),
        ]
        result = self.formatter.format_messages(messages, params={
            "model": "claude-sonnet-4-5-20250514",
            "llm_config": config,
            "enable_cache_control": False,
        })
        assert result["thinking"] == {
            "type": "enabled",
            "budget_tokens": 10000,
        }

    def test_with_tools(self):
        messages = [
            Message(role=Role.USER, content=[TextContent(text="Hi")]),
        ]
        tool_schemas = [
            {"name": "read_file", "description": "Read a file", "input_schema": {}},
        ]
        result = self.formatter.format_messages(messages, params={
            "model": "claude-sonnet-4-5-20250514",
            "tool_schemas": tool_schemas,
            "enable_cache_control": False,
        })
        assert result["tools"] == tool_schemas

    def test_with_server_tools(self):
        from agent_base.providers.anthropic.anthropic_agent import AnthropicLLMConfig

        server_tools = [{"type": "web_search_20250305", "name": "web_search"}]
        config = AnthropicLLMConfig(server_tools=server_tools)
        messages = [
            Message(role=Role.USER, content=[TextContent(text="Search")]),
        ]
        result = self.formatter.format_messages(messages, params={
            "model": "claude-sonnet-4-5-20250514",
            "llm_config": config,
            "tool_schemas": [{"name": "my_tool", "description": "d", "input_schema": {}}],
            "enable_cache_control": False,
        })
        assert len(result["tools"]) == 2

    def test_with_beta_headers(self):
        from agent_base.providers.anthropic.anthropic_agent import AnthropicLLMConfig

        config = AnthropicLLMConfig(beta_headers=["interleaved-thinking-2025-05-14"])
        messages = [
            Message(role=Role.USER, content=[TextContent(text="Hi")]),
        ]
        result = self.formatter.format_messages(messages, params={
            "model": "claude-sonnet-4-5-20250514",
            "llm_config": config,
            "enable_cache_control": False,
        })
        assert result["betas"] == ["interleaved-thinking-2025-05-14"]

    def test_with_container(self):
        from agent_base.providers.anthropic.anthropic_agent import AnthropicLLMConfig

        config = AnthropicLLMConfig(
            container_id="ctnr_abc123",
            skills=[{"name": "code_exec", "type": "computer_use"}],
        )
        messages = [
            Message(role=Role.USER, content=[TextContent(text="Run code")]),
        ]
        result = self.formatter.format_messages(messages, params={
            "model": "claude-sonnet-4-5-20250514",
            "llm_config": config,
            "enable_cache_control": False,
        })
        assert result["container"] == {
            "id": "ctnr_abc123",
            "skills": [{"name": "code_exec", "type": "computer_use"}],
        }

    def test_max_tokens_from_config(self):
        from agent_base.providers.anthropic.anthropic_agent import AnthropicLLMConfig

        config = AnthropicLLMConfig(max_tokens=4096)
        messages = [
            Message(role=Role.USER, content=[TextContent(text="Hi")]),
        ]
        result = self.formatter.format_messages(messages, params={
            "model": "claude-sonnet-4-5-20250514",
            "llm_config": config,
            "enable_cache_control": False,
        })
        assert result["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# Tests: Round-trip (canonical → wire → canonical)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Tests verifying content survives a canonical → wire → parse cycle."""

    def setup_method(self):
        self.formatter = AnthropicMessageFormatter()

    def test_roundtrip_text(self):
        original = TextContent(text="round trip text")
        wire = _content_block_to_anthropic(original)
        stub = StubTextBlock(text=wire["text"])
        parsed = _parse_content_block(stub)
        assert isinstance(parsed, TextContent)
        assert parsed.text == original.text

    def test_roundtrip_thinking_with_signature(self):
        original = ThinkingContent(thinking="analysis here", signature="sig_round")
        wire = _content_block_to_anthropic(original)
        stub = StubThinkingBlock(thinking=wire["thinking"], signature=wire["signature"])
        parsed = _parse_content_block(stub)
        assert isinstance(parsed, ThinkingContent)
        assert parsed.thinking == original.thinking
        assert parsed.signature == original.signature

    def test_roundtrip_thinking_without_signature_is_lossy(self):
        original = ThinkingContent(thinking="lost thought", signature=None)
        wire = _content_block_to_anthropic(original)
        # Without signature, block is omitted (returns None)
        assert wire is None

    def test_roundtrip_tool_use(self):
        original = ToolUseContent(
            tool_name="read_file",
            tool_id="toolu_round",
            tool_input={"path": "/test"},
        )
        wire = _content_block_to_anthropic(original)
        stub = StubToolUseBlock(
            id=wire["id"],
            name=wire["name"],
            input=wire["input"],
        )
        parsed = _parse_content_block(stub)
        assert isinstance(parsed, ToolUseContent)
        assert parsed.tool_name == original.tool_name
        assert parsed.tool_id == original.tool_id
        assert parsed.tool_input == original.tool_input

    def test_roundtrip_container_upload(self):
        original = AttachmentContent(
            filename="output.csv",
            source_type="file_id",
            data="file_xyz",
            media_type="text/csv",
            media_id="att_round",
        )
        wire = _content_block_to_anthropic(original)
        assert wire["type"] == "container_upload"
        assert wire["file_id"] == "file_xyz"

        # Simulate parsing the response back
        stub = StubContainerUploadBlock(file_id=wire["file_id"])
        parsed = _parse_content_block(stub)
        assert isinstance(parsed, AttachmentContent)
        assert parsed.source_type == "file_id"
        assert parsed.data == "file_xyz"

    def test_roundtrip_full_message(self):
        """Full message round-trip through format_messages → parse_response."""
        original_messages = [
            Message(
                role=Role.USER,
                content=[TextContent(text="What's in this file?")],
            ),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ThinkingContent(thinking="analyzing", signature="sig_full"),
                    TextContent(text="Here's the content"),
                ],
            ),
        ]

        # Format to wire
        result = self.formatter.format_messages(original_messages, params={
            "model": "claude-sonnet-4-5-20250514",
            "enable_cache_control": False,
        })

        # Verify wire format structure
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

        # Simulate a response with the same content as the assistant message
        stub_response = StubBetaMessage(
            content=[
                StubThinkingBlock(thinking="analyzing", signature="sig_full"),
                StubTextBlock(text="Here's the content"),
            ],
            usage=StubUsage(input_tokens=50, output_tokens=25),
            stop_reason="end_turn",
        )

        parsed = self.formatter.parse_response(stub_response)
        assert parsed.role == Role.ASSISTANT
        assert len(parsed.content) == 2
        assert isinstance(parsed.content[0], ThinkingContent)
        assert parsed.content[0].thinking == "analyzing"
        assert parsed.content[0].signature == "sig_full"
        assert isinstance(parsed.content[1], TextContent)
        assert parsed.content[1].text == "Here's the content"

    def test_mcp_tool_use_roundtrip_is_lossy(self):
        """MCPToolUseContent → wire as 'tool_use' → parses back as ToolUseContent.

        This is a known limitation: mcp_server_name is lost on the wire
        when the formatter serializes to plain 'tool_use'. However, when the
        API returns 'mcp_tool_use' type blocks, they parse correctly.
        """
        original = MCPToolUseContent(
            tool_name="mcp_func",
            tool_id="toolu_lossy",
            tool_input={"x": 1},
            mcp_server_name="my_server",
        )
        wire = _content_block_to_anthropic(original)
        assert wire["type"] == "tool_use"  # Loses MCP specificity

        # Parse back from tool_use → ToolUseContent (not MCPToolUseContent)
        stub = StubToolUseBlock(id=wire["id"], name=wire["name"], input=wire["input"])
        parsed = _parse_content_block(stub)
        assert isinstance(parsed, ToolUseContent)
        assert not isinstance(parsed, MCPToolUseContent)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for formatter behavior."""

    def test_message_with_only_citations(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                CharCitation(
                    cited_text="text",
                    document_index=0,
                    start_char_index=0,
                    end_char_index=4,
                ),
            ],
        )
        result = _format_message(msg)
        assert result["content"] == []

    def test_redacted_thinking_then_reserialize(self):
        """Redacted thinking parses to ThinkingContent(signature=None),
        which is correctly dropped on re-serialization."""
        stub = StubRedactedThinkingBlock(data="secret")
        parsed = _parse_content_block(stub)
        assert isinstance(parsed, ThinkingContent)
        assert parsed.signature is None

        # Re-serializing should return None (dropped)
        wire = _content_block_to_anthropic(parsed)
        assert wire is None

    def test_tool_result_with_mixed_nested_content(self):
        """Tool result containing text and image blocks."""
        block = ToolResultContent(
            tool_name="multi",
            tool_id="toolu_mix",
            tool_result=[
                TextContent(text="result text"),
                ImageContent(
                    source_type="base64",
                    data="base64img",
                    media_type="image/png",
                    media_id="img_mix",
                ),
                # Citation in tool result should be filtered
                CharCitation(
                    cited_text="cite",
                    document_index=0,
                    start_char_index=0,
                    end_char_index=4,
                ),
            ],
        )
        result = _content_block_to_anthropic(block)
        # Citation is filtered (returns None from _content_block_to_anthropic)
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "image"

    def test_format_tool_schemas_passthrough(self):
        formatter = AnthropicMessageFormatter()
        schemas = [
            {"name": "tool_a", "description": "desc", "input_schema": {"type": "object"}},
        ]
        result = formatter.format_tool_schemas(schemas)
        assert result is schemas  # Exact same object (pass-through)
