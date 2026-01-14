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
    content, image_refs = registry.execute("add", {"a": 2, "b": 3})
    assert content == "5"
    assert image_refs == []


def test_execute_unknown_tool_returns_error(registry) -> None:
    content, image_refs = registry.execute("unknown", {})
    assert content.startswith("Error:")
    assert image_refs == []


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
    content, image_refs = registry.execute("add", {"a": 1, "b": 1})
    assert content == "3"
    assert image_refs == []

