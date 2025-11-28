import pytest

from anthropic_agent.tools.decorators import tool
from anthropic_agent.tools.type_hint_utils import TypeHintParsingException


def test_tool_decorator_attaches_schema() -> None:
    @tool
    def add(a: float, b: float) -> str:
        """Add two numbers.

        Args:
            a: First operand.
            b: Second operand.

        Returns:
            Sum as string.
        """
        return str(a + b)

    assert hasattr(add, "__tool_schema__")
    schema = add.__tool_schema__
    assert schema["name"] == "add"
    assert schema["description"] == "Add two numbers."
    properties = schema["input_schema"]["properties"]
    assert properties["a"]["type"] == "number"
    assert properties["b"]["type"] == "number"


def test_tool_decorator_missing_type_hint_raises() -> None:
    with pytest.raises(TypeHintParsingException):

        @tool
        def invalid(a, b: int) -> str:  # type: ignore[no-untyped-def]
            """Invalid tool because parameter `a` lacks a type hint."""
            return str(a or b)


def test_tool_decorator_optional_parameters_nullable() -> None:
    @tool
    def greet(name: str, language: str = "en") -> str:
        """Greet a user in an optional language.

        Args:
            name: Name of the user.
            language: ISO language code.
        """
        return f"Hello {name} ({language})"

    schema = greet.__tool_schema__["input_schema"]
    assert "language" not in schema.get("required", [])
    assert schema["properties"]["language"]["nullable"] is True


def test_tool_decorator_handles_nested_types() -> None:
    @tool
    def plan_trip(destinations: list[str], metadata: dict[str, int]) -> str:
        """Plan a trip across multiple destinations.

        Args:
            destinations: Cities to visit in order.
            metadata: Additional trip metadata.
        """
        return f"Trip covering {len(destinations)} stops"

    properties = plan_trip.__tool_schema__["input_schema"]["properties"]
    assert properties["destinations"]["type"] == "array"
    assert properties["metadata"]["type"] == "object"


def test_tool_decorator_docstring_fallback_description() -> None:
    @tool
    def no_google_style(a: int) -> str:
        """Simple fallback description."""
        return str(a)

    assert no_google_style.__tool_schema__["description"] == "Simple fallback description."

