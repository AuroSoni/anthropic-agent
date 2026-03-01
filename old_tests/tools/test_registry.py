import asyncio
import time
import threading

import pytest

from anthropic_agent.tools.base import ToolRegistry
from anthropic_agent.tools.decorators import tool


@pytest.fixture()
def sample_tools():
    @tool
    def add(a: float, b: float) -> str:
        """Add two numbers."""
        return str(a + b)

    @tool
    def multiply(a: float, b: float) -> str:
        """Multiply two numbers."""
        return str(a * b)

    return add, multiply


@pytest.fixture()
def registry(sample_tools):
    reg = ToolRegistry()
    reg.register_tools(list(sample_tools))
    return reg


def test_register_tools_populates_registry(registry, sample_tools) -> None:
    add, multiply = sample_tools
    assert set(registry.tools) == {add.__tool_schema__["name"], multiply.__tool_schema__["name"]}
    assert registry.schemas  # schemas captured during registration


def test_register_tools_requires_decorated_function() -> None:
    registry = ToolRegistry()

    def undecorated(a: int) -> int:
        return a

    with pytest.raises(ValueError):
        registry.register_tools([undecorated])


def test_execute_returns_function_result(registry) -> None:
    async def run():
        content, image_refs = await registry.execute("add", {"a": 2, "b": 3})
        assert content == "5"
        assert image_refs == []
    asyncio.run(run())


def test_execute_unknown_tool_returns_error(registry) -> None:
    async def run():
        content, image_refs = await registry.execute("unknown", {})
        assert content.startswith("Error:")
        assert image_refs == []
    asyncio.run(run())


def test_get_schemas_anthropic_format(registry, sample_tools) -> None:
    schemas = registry.get_schemas()
    assert isinstance(schemas, list)
    assert schemas[0]["input_schema"]["type"] == "object"
    names = {schema["name"] for schema in schemas}
    assert names == {tool.__tool_schema__["name"] for tool in sample_tools}


def test_get_schemas_openai_format(registry) -> None:
    schemas = registry.get_schemas(schema_type="openai")
    assert all(entry["type"] == "function" for entry in schemas)
    assert "function" in schemas[0]
    assert {"name", "description", "parameters"} <= schemas[0]["function"].keys()


def test_get_schemas_invalid_type_errors(registry) -> None:
    with pytest.raises(ValueError):
        registry.get_schemas(schema_type="unsupported")


def test_register_method_overwrites_existing_schema() -> None:
    registry = ToolRegistry()

    @tool
    def add(a: int, b: int) -> str:
        """Add numbers."""
        return str(a + b)

    registry.register("add", add, add.__tool_schema__)

    @tool
    def add_new(a: int, b: int) -> str:
        """Updated add function."""
        return str(a + b + 1)

    registry.register("add", add_new, add_new.__tool_schema__)

    async def run():
        content, image_refs = await registry.execute("add", {"a": 1, "b": 1})
        assert content == "3"
        assert image_refs == []
    asyncio.run(run())


def test_async_tool_execution() -> None:
    """Async tool functions should be awaited directly."""
    @tool
    async def greet(name: str) -> str:
        """Greet someone."""
        await asyncio.sleep(0)  # confirm we are actually awaited
        return f"Hello, {name}!"

    registry = ToolRegistry()
    registry.register_tools([greet])

    async def run():
        content, image_refs = await registry.execute("greet", {"name": "World"})
        assert content == "Hello, World!"
        assert image_refs == []
    asyncio.run(run())


def test_sync_tool_runs_off_event_loop() -> None:
    """Sync tools should be dispatched to a thread via asyncio.to_thread."""
    @tool
    def blocking_tool(x: int) -> str:
        """A tool that blocks."""
        # Record the thread; if it ran in the main thread the event loop
        # would have been blocked.
        blocking_tool._thread = threading.current_thread()
        time.sleep(0.05)
        return str(x * 2)

    registry = ToolRegistry()
    registry.register_tools([blocking_tool])

    async def run():
        main_thread = threading.current_thread()
        content, _ = await registry.execute("blocking_tool", {"x": 5})
        assert content == "10"
        # The tool should have executed in a *different* thread.
        assert blocking_tool._thread is not main_thread
    asyncio.run(run())

