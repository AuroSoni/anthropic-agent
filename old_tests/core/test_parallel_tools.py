"""Tests for parallel backend tool execution via _execute_tools_parallel."""

from __future__ import annotations

import asyncio
import json as _json
import time
from dataclasses import dataclass
from typing import Any

import pytest

from anthropic_agent.tools.base import ToolRegistry
from anthropic_agent.tools.decorators import tool
from anthropic_agent.core.agent import MAX_PARALLEL_TOOL_CALLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeToolCall:
    """Mimics the tool_use content block returned by the Anthropic API."""
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


class StubAgent:
    """Minimal stub that exposes only what _execute_tools_parallel needs."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        max_parallel: int = MAX_PARALLEL_TOOL_CALLS,
        formatter: str = "json",
    ):
        self.tool_registry = tool_registry
        self.max_parallel_tool_calls = max_parallel
        self.formatter = formatter
        self.agent_uuid = "test-agent"
        self.file_backend = None
        self._log_entries: list[dict] = []

    async def execute_tool_call(self, tool_name: str, tool_input: dict):
        return await self.tool_registry.execute(
            tool_name, tool_input,
            file_backend=self.file_backend,
            agent_uuid=self.agent_uuid,
        )

    def _log_action(self, action: str, details: dict, step_number: int = 0):
        self._log_entries.append({"action": action, **details, "step": step_number})

    # Bind the real methods from AnthropicAgent so tests exercise actual code.
    from anthropic_agent.core.agent import AnthropicAgent as _Real
    _execute_tools_parallel = _Real._execute_tools_parallel
    _execute_tools_sequential = _Real._execute_tools_sequential


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def slow_tools():
    """Three tools that each sleep for a short duration."""
    @tool
    def slow_a(x: int) -> str:
        """Slow tool A."""
        time.sleep(0.2)
        return f"a:{x}"

    @tool
    def slow_b(x: int) -> str:
        """Slow tool B."""
        time.sleep(0.2)
        return f"b:{x}"

    @tool
    def slow_c(x: int) -> str:
        """Slow tool C."""
        time.sleep(0.2)
        return f"c:{x}"

    return slow_a, slow_b, slow_c


@pytest.fixture()
def registry(slow_tools):
    reg = ToolRegistry()
    reg.register_tools(list(slow_tools))
    return reg


@pytest.fixture()
def agent(registry):
    return StubAgent(registry, max_parallel=5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_tool_uses_sequential_path(agent):
    """A single tool call should go through the fast (sequential) path."""
    async def run():
        calls = [FakeToolCall(id="t1", name="slow_a", input={"x": 1})]
        results = await agent._execute_tools_parallel(calls, queue=None, formatter=None, step=1)
        assert len(results) == 1
        assert results[0]["content"] == "a:1"
        assert results[0]["tool_use_id"] == "t1"
    asyncio.run(run())


def test_multiple_tools_run_in_parallel(agent):
    """Three tools sleeping 0.2s each should finish in < 0.5s with parallelism."""
    async def run():
        calls = [
            FakeToolCall(id="t1", name="slow_a", input={"x": 1}),
            FakeToolCall(id="t2", name="slow_b", input={"x": 2}),
            FakeToolCall(id="t3", name="slow_c", input={"x": 3}),
        ]
        start = time.monotonic()
        results = await agent._execute_tools_parallel(calls, queue=None, formatter=None, step=1)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        # With parallelism, wall time should be ~0.2s, not ~0.6s.
        assert elapsed < 0.5, f"Expected < 0.5s but took {elapsed:.2f}s (tools not parallel?)"
    asyncio.run(run())


def test_semaphore_bounds_concurrency():
    """Concurrency should never exceed max_parallel_tool_calls."""
    concurrent_count = 0
    max_observed = 0

    @tool
    def counting_tool(x: int) -> str:
        """Tool that tracks concurrency."""
        nonlocal concurrent_count, max_observed
        concurrent_count += 1
        if concurrent_count > max_observed:
            max_observed = concurrent_count
        time.sleep(0.15)
        concurrent_count -= 1
        return str(x)

    reg = ToolRegistry()
    reg.register_tools([counting_tool])
    agent = StubAgent(reg, max_parallel=2)

    async def run():
        calls = [FakeToolCall(id=f"t{i}", name="counting_tool", input={"x": i}) for i in range(5)]
        await agent._execute_tools_parallel(calls, queue=None, formatter=None, step=1)
    asyncio.run(run())

    assert max_observed <= 2, f"Max concurrency was {max_observed}, expected <= 2"


def test_results_preserve_original_order():
    """Results must be in the same order as the input tool calls."""
    @tool
    def fast_tool(x: int) -> str:
        """Finishes instantly."""
        return f"fast:{x}"

    @tool
    def slowish_tool(x: int) -> str:
        """Takes a bit longer."""
        time.sleep(0.15)
        return f"slowish:{x}"

    reg = ToolRegistry()
    reg.register_tools([fast_tool, slowish_tool])
    agent = StubAgent(reg, max_parallel=5)

    async def run():
        # slowish first, fast second — results must still match input order.
        calls = [
            FakeToolCall(id="t1", name="slowish_tool", input={"x": 1}),
            FakeToolCall(id="t2", name="fast_tool", input={"x": 2}),
        ]
        results = await agent._execute_tools_parallel(calls, queue=None, formatter=None, step=1)
        assert results[0]["tool_use_id"] == "t1"
        assert results[0]["content"] == "slowish:1"
        assert results[1]["tool_use_id"] == "t2"
        assert results[1]["content"] == "fast:2"
    asyncio.run(run())


def test_error_in_one_tool_does_not_block_others():
    """If one tool raises, the others should still complete."""
    @tool
    def good_tool(x: int) -> str:
        """Works fine."""
        return f"ok:{x}"

    @tool
    def bad_tool(x: int) -> str:
        """Always fails."""
        raise RuntimeError("boom")

    reg = ToolRegistry()
    reg.register_tools([good_tool, bad_tool])
    agent = StubAgent(reg, max_parallel=5)

    async def run():
        calls = [
            FakeToolCall(id="t1", name="good_tool", input={"x": 1}),
            FakeToolCall(id="t2", name="bad_tool", input={"x": 2}),
            FakeToolCall(id="t3", name="good_tool", input={"x": 3}),
        ]
        results = await agent._execute_tools_parallel(calls, queue=None, formatter=None, step=1)

        assert len(results) == 3
        assert results[0]["content"] == "ok:1"
        assert "is_error" not in results[0]
        # ToolRegistry.execute() catches the exception internally and returns
        # an error string, so the parallel executor sees a normal result.
        assert "Error" in results[1]["content"]
        assert results[2]["content"] == "ok:3"
    asyncio.run(run())


def test_queue_none_skips_emission(agent):
    """When queue is None, no emission errors should occur."""
    async def run():
        calls = [
            FakeToolCall(id="t1", name="slow_a", input={"x": 1}),
            FakeToolCall(id="t2", name="slow_b", input={"x": 2}),
        ]
        results = await agent._execute_tools_parallel(calls, queue=None, formatter=None, step=1)
        assert len(results) == 2
        assert results[0]["content"] == "a:1"
        assert results[1]["content"] == "b:2"
    asyncio.run(run())


def test_sse_emission_is_atomic():
    """Each tool's queue.put sequence must not interleave with another tool's."""
    @tool
    def tool_a(x: int) -> str:
        """Tool A."""
        time.sleep(0.05)
        return "result_a"

    @tool
    def tool_b(x: int) -> str:
        """Tool B."""
        return "result_b"

    reg = ToolRegistry()
    reg.register_tools([tool_a, tool_b])
    agent = StubAgent(reg, max_parallel=5)

    async def run():
        queue = asyncio.Queue()
        calls = [
            FakeToolCall(id="t1", name="tool_a", input={"x": 1}),
            FakeToolCall(id="t2", name="tool_b", input={"x": 2}),
        ]
        await agent._execute_tools_parallel(calls, queue=queue, formatter="json", step=1)

        # Drain all items from the queue.
        items: list[str] = []
        while not queue.empty():
            items.append(await queue.get())

        # For JSON format, each tool emits one or more chunks. All chunks for a
        # given tool_use_id must appear as a contiguous block (not interleaved).
        seen_ids: list[str] = []
        for item in items:
            try:
                parsed = _json.loads(item)
                tid = parsed.get("id", "")
            except (ValueError, TypeError):
                continue
            if tid and (not seen_ids or seen_ids[-1] != tid):
                seen_ids.append(tid)

        # Each tool id should appear exactly once in the contiguous-block list.
        assert len(seen_ids) == len(set(seen_ids)), (
            f"Tool emissions interleaved: {seen_ids}"
        )
    asyncio.run(run())
