"""Tests for SubAgentTool — single dispatcher for spawning subagents."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anthropic_agent.common_tools.sub_agent_tool import SubAgentTool


# ---------------------------------------------------------------------------
# Helpers — lightweight stubs that mimic AnthropicAgent's interface
# ---------------------------------------------------------------------------

@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FakeFinalMessage:
    role: str = "assistant"
    content: list = None
    model: str = "claude-sonnet-4-5"
    stop_reason: str = "end_turn"
    usage: FakeUsage = None

    def __post_init__(self):
        if self.content is None:
            self.content = []
        if self.usage is None:
            self.usage = FakeUsage()


@dataclass
class FakeAgentResult:
    """Mimics AgentResult with the fields SubAgentTool._format_result reads."""
    final_message: Any = None
    final_answer: str = ""
    conversation_history: list = None
    stop_reason: str = "end_turn"
    model: str = "claude-sonnet-4-5"
    usage: Any = None
    container_id: Optional[str] = None
    total_steps: int = 3
    agent_logs: Optional[list] = None
    generated_files: Optional[list] = None
    cost: Optional[dict] = None
    cumulative_usage: Optional[dict] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.usage is None:
            self.usage = FakeUsage()
        if self.final_message is None:
            self.final_message = FakeFinalMessage()


class FakeAgent:
    """Minimal stub that exposes the attributes SubAgentTool reads."""

    def __init__(
        self,
        *,
        description: str = "",
        system_prompt: str = "You are helpful.",
        model: str = "claude-sonnet-4-5",
        max_steps: int = 10,
        thinking_tokens: int = 0,
        max_tokens: int = 2048,
        max_retries: int = 5,
        base_delay: float = 5.0,
        max_parallel_tool_calls: int = 5,
        formatter: str = "json",
        compactor: Any = None,
        memory_store: Any = None,
        config_adapter: Any = None,
        conversation_adapter: Any = None,
        run_adapter: Any = None,
        file_backend: Any = None,
    ):
        self.description = description
        self.system_prompt = system_prompt
        self.model = model
        self.max_steps = max_steps
        self.thinking_tokens = thinking_tokens
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_parallel_tool_calls = max_parallel_tool_calls
        self.formatter = formatter
        self.compactor = compactor
        self.memory_store = memory_store
        self.config_adapter = config_adapter
        self.conversation_adapter = conversation_adapter
        self.run_adapter = run_adapter
        self.file_backend = file_backend
        self._tool_functions = []
        self._sub_agent_tool = None
        self._parent_agent_uuid: str | None = None

    async def run(self, prompt, queue=None, formatter=None):
        """Mock run — returns a FakeAgentResult."""
        return FakeAgentResult(final_answer="Done!")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_agents():
    """Return a dict of two fake agents with descriptions."""
    return {
        "researcher": FakeAgent(description="Researches topics"),
        "coder": FakeAgent(description="Writes code", model="claude-haiku-3"),
    }


@pytest.fixture()
def sub_tool(two_agents):
    """Return a SubAgentTool instance with two fake agents."""
    return SubAgentTool(agents=two_agents)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tool_schema_includes_agent_definitions(sub_tool):
    """The rendered docstring should list registered agents."""
    func = sub_tool.get_tool()
    schema = getattr(func, "__tool_schema__", None)
    assert schema is not None
    desc = schema.get("description", "")
    assert "researcher" in desc
    assert "Researches topics" in desc
    assert "coder" in desc
    assert "Writes code" in desc


def test_unknown_agent_name_returns_error(sub_tool):
    """Calling with an invalid agent_name should return a descriptive error."""
    func = sub_tool.get_tool()

    async def run():
        # The decorated function should accept tool_input-style kwargs
        result = await func(agent_name="unknown", task="Do something")
        assert "Error" in result
        assert "unknown" in result
        assert "researcher" in result  # lists available agents
        assert "coder" in result

    asyncio.run(run())


def test_spawn_subagent_calls_child_run(sub_tool):
    """The child agent's run() should be called with the correct task and queue."""
    queue = asyncio.Queue()
    sub_tool.set_run_context(queue, "json")
    sub_tool.set_agent_uuid("parent-uuid-123")

    # Patch _create_child_agent to return a mock child
    mock_child = AsyncMock()
    mock_child.run = AsyncMock(return_value=FakeAgentResult(final_answer="Research done"))

    func = sub_tool.get_tool()

    async def run():
        with patch.object(sub_tool, "_create_child_agent", return_value=mock_child):
            result = await func(
                agent_name="researcher",
                task="Explain quantum computing",
            )
            mock_child.run.assert_called_once_with(
                prompt="Explain quantum computing",
                queue=queue,
                formatter="json",
            )
            assert "Research done" in result
            assert "researcher" in result

    asyncio.run(run())


def test_queue_shared_with_child(sub_tool):
    """The child's run() should receive the same queue injected via set_run_context."""
    queue = asyncio.Queue()
    sub_tool.set_run_context(queue, "json")

    captured_queue = None

    async def mock_run(prompt, queue=None, formatter=None):
        nonlocal captured_queue
        captured_queue = queue
        return FakeAgentResult(final_answer="ok")

    mock_child = MagicMock()
    mock_child.run = mock_run

    func = sub_tool.get_tool()

    async def run():
        with patch.object(sub_tool, "_create_child_agent", return_value=mock_child):
            await func(agent_name="researcher", task="test")

    asyncio.run(run())
    assert captured_queue is queue


def test_resume_uuid_passed_to_child(sub_tool):
    """When resume_agent_uuid is provided, _create_child_agent receives it."""
    sub_tool.set_agent_uuid("parent-uuid")

    captured_resume_uuid = None

    original_create = sub_tool._create_child_agent

    def spy_create(template, resume_uuid=None):
        nonlocal captured_resume_uuid
        captured_resume_uuid = resume_uuid
        # Return a mock instead of creating a real agent
        child = MagicMock()
        child.run = AsyncMock(return_value=FakeAgentResult(final_answer="resumed"))
        return child

    func = sub_tool.get_tool()

    async def run():
        with patch.object(sub_tool, "_create_child_agent", side_effect=spy_create):
            await func(
                agent_name="researcher",
                task="Continue analysis",
                resume_agent_uuid="child-uuid-abc",
            )

    asyncio.run(run())
    assert captured_resume_uuid == "child-uuid-abc"


def test_fresh_call_gets_none_resume_uuid(sub_tool):
    """When resume_agent_uuid is omitted, _create_child_agent gets None."""
    sub_tool.set_agent_uuid("parent-uuid")

    captured_resume_uuid = "SENTINEL"

    def spy_create(template, resume_uuid=None):
        nonlocal captured_resume_uuid
        captured_resume_uuid = resume_uuid
        child = MagicMock()
        child.run = AsyncMock(return_value=FakeAgentResult(final_answer="fresh"))
        return child

    func = sub_tool.get_tool()

    async def run():
        with patch.object(sub_tool, "_create_child_agent", side_effect=spy_create):
            await func(agent_name="coder", task="Write code")

    asyncio.run(run())
    assert captured_resume_uuid is None


def test_child_error_returns_error_string(sub_tool):
    """If the child's run() raises, the tool returns an error string (no crash)."""
    mock_child = MagicMock()
    mock_child.run = AsyncMock(side_effect=RuntimeError("boom"))

    func = sub_tool.get_tool()

    async def run():
        with patch.object(sub_tool, "_create_child_agent", return_value=mock_child):
            result = await func(agent_name="researcher", task="fail")
            assert "error" in result.lower()
            assert "RuntimeError" in result
            assert "boom" in result

    asyncio.run(run())


def test_set_run_context_and_clear(sub_tool):
    """set_run_context stores queue/formatter; passing None clears them."""
    queue = asyncio.Queue()
    sub_tool.set_run_context(queue, "json")
    assert sub_tool._current_queue is queue
    assert sub_tool._current_formatter == "json"

    sub_tool.set_run_context(None, None)
    assert sub_tool._current_queue is None
    assert sub_tool._current_formatter is None


def test_format_result_with_final_answer():
    """_format_result should include the final answer, stop reason, and steps."""
    result = FakeAgentResult(
        final_answer="42 is the answer",
        stop_reason="end_turn",
        total_steps=5,
        model="claude-sonnet-4-5",
    )
    formatted = SubAgentTool._format_result("researcher", "child-uuid-123", result)
    assert "researcher" in formatted
    assert "child-uuid-123" in formatted
    assert "42 is the answer" in formatted
    assert "end_turn" in formatted
    assert "steps=5" in formatted


def test_format_result_without_final_answer():
    """_format_result should handle missing final_answer gracefully."""
    result = FakeAgentResult(
        final_answer="",
        stop_reason="max_tokens",
        total_steps=10,
    )
    formatted = SubAgentTool._format_result("coder", "child-uuid-456", result)
    assert "coder" in formatted
    assert "child-uuid-456" in formatted
    assert "No final answer extracted" in formatted
    assert "max_tokens" in formatted
    assert "steps=10" in formatted


def test_validation_rejects_missing_description():
    """SubAgentTool should raise ValueError if an agent lacks a description."""
    agent = FakeAgent(description="")
    with pytest.raises(ValueError, match="non-empty `description`"):
        SubAgentTool(agents={"bad_agent": agent})


def test_validation_rejects_none_description():
    """SubAgentTool should raise ValueError if an agent's description is None."""
    agent = FakeAgent()
    agent.description = None
    with pytest.raises(ValueError, match="non-empty `description`"):
        SubAgentTool(agents={"bad_agent": agent})


def test_set_agent_uuid(sub_tool):
    """set_agent_uuid should store the parent UUID for hierarchy tracking."""
    sub_tool.set_agent_uuid("parent-123")
    assert sub_tool._parent_agent_uuid == "parent-123"
