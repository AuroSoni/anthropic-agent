"""Tests for ToolSandboxConfig functionality.

Tests cover:
- Initialization and configuration
- Toolset creation with all tool types
- Schema augmentation with constraints
- Integration with common_tools extension presets
"""
import tempfile
from pathlib import Path
from typing import Callable, Generator, List

import pytest

from anthropic_agent.tools.sandbox_config import ToolSandboxConfig
from anthropic_agent.common_tools import EXTENSION_PRESETS, get_extensions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def default_config(temp_workspace: Path) -> ToolSandboxConfig:
    """Create a ToolSandboxConfig with default settings."""
    return ToolSandboxConfig(base_path=temp_workspace)


@pytest.fixture
def custom_config(temp_workspace: Path) -> ToolSandboxConfig:
    """Create a ToolSandboxConfig with custom settings."""
    return ToolSandboxConfig(
        base_path=temp_workspace,
        allowed_extensions={".py", ".js", ".md"},
        read_file_max_lines=200,
        grep_max_matches=50,
        list_dir_max_depth=10,
    )


# ---------------------------------------------------------------------------
# Tests for extension presets
# ---------------------------------------------------------------------------
class TestExtensionPresets:
    def test_docs_preset(self) -> None:
        """Docs preset should include documentation file extensions."""
        exts = get_extensions("docs")
        assert ".md" in exts
        assert ".mmd" in exts
        assert ".rst" in exts
        assert ".txt" in exts
    
    def test_code_preset(self) -> None:
        """Code preset should include code file extensions."""
        exts = get_extensions("code")
        assert ".py" in exts
        assert ".js" in exts
        assert ".ts" in exts
        assert ".json" in exts
    
    def test_all_text_preset(self) -> None:
        """All_text preset should include both docs and code extensions."""
        exts = get_extensions("all_text")
        assert ".md" in exts
        assert ".py" in exts
        assert ".html" in exts
    
    def test_unknown_preset_defaults_to_docs(self) -> None:
        """Unknown preset should default to docs."""
        exts = get_extensions("unknown_preset")
        assert exts == EXTENSION_PRESETS["docs"]
    
    def test_custom_set_passed_through(self) -> None:
        """Custom set should be returned as-is."""
        custom = {".custom", ".ext"}
        result = get_extensions(custom)
        assert result == custom
    
    def test_presets_return_copy(self) -> None:
        """Presets should return a copy, not the original."""
        exts1 = get_extensions("docs")
        exts1.add(".new")
        exts2 = get_extensions("docs")
        assert ".new" not in exts2


# ---------------------------------------------------------------------------
# Tests for ToolSandboxConfig initialization
# ---------------------------------------------------------------------------
class TestToolSandboxConfigInit:
    def test_init_with_path_object(self, temp_workspace: Path) -> None:
        """Initialize with Path object."""
        config = ToolSandboxConfig(base_path=temp_workspace)
        assert config.base_path == temp_workspace
    
    def test_init_with_string_path(self, temp_workspace: Path) -> None:
        """Initialize with string path, should be converted to Path."""
        config = ToolSandboxConfig(base_path=str(temp_workspace))
        assert config.base_path == Path(temp_workspace)
    
    def test_default_values(self, default_config: ToolSandboxConfig) -> None:
        """Verify default configuration values."""
        assert default_config.allowed_extensions == {".md", ".mmd"}
        assert default_config.read_file_max_lines == 100
        assert default_config.grep_max_matches == 20
        assert default_config.list_dir_max_depth == 5
    
    def test_custom_values(self, custom_config: ToolSandboxConfig) -> None:
        """Verify custom configuration values are set."""
        assert custom_config.allowed_extensions == {".py", ".js", ".md"}
        assert custom_config.read_file_max_lines == 200
        assert custom_config.grep_max_matches == 50
        assert custom_config.list_dir_max_depth == 10


# ---------------------------------------------------------------------------
# Tests for toolset creation
# ---------------------------------------------------------------------------
class TestCreateToolset:
    def test_creates_five_tools(self, default_config: ToolSandboxConfig) -> None:
        """create_toolset should return exactly 5 tools."""
        tools = default_config.create_toolset()
        assert len(tools) == 5
    
    def test_all_tools_are_callable(self, default_config: ToolSandboxConfig) -> None:
        """All tools should be callable functions."""
        tools = default_config.create_toolset()
        for tool in tools:
            assert callable(tool)
    
    def test_all_tools_have_schema(self, default_config: ToolSandboxConfig) -> None:
        """All tools should have __tool_schema__ attribute."""
        tools = default_config.create_toolset()
        for tool in tools:
            assert hasattr(tool, "__tool_schema__")
            schema = tool.__tool_schema__
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema
    
    def test_tool_names(self, default_config: ToolSandboxConfig) -> None:
        """Verify tool names in the toolset."""
        tools = default_config.create_toolset()
        names = {tool.__tool_schema__["name"] for tool in tools}
        expected_names = {
            "read_file",
            "apply_patch",
            "glob_file_search",
            "grep_search",
            "list_dir",
        }
        assert names == expected_names


# ---------------------------------------------------------------------------
# Tests for schema augmentation
# ---------------------------------------------------------------------------
class TestSchemaAugmentation:
    def test_constraints_added_to_description(
        self, default_config: ToolSandboxConfig
    ) -> None:
        """Tool descriptions should include constraints section."""
        tools = default_config.create_toolset()
        for tool in tools:
            desc = tool.__tool_schema__["description"]
            assert "Constraints:" in desc
    
    def test_base_path_in_constraints(
        self, temp_workspace: Path
    ) -> None:
        """Constraints should include base_path."""
        config = ToolSandboxConfig(base_path=temp_workspace)
        tools = config.create_toolset()
        for tool in tools:
            desc = tool.__tool_schema__["description"]
            assert str(temp_workspace) in desc
    
    def test_allowed_extensions_in_constraints(
        self, default_config: ToolSandboxConfig
    ) -> None:
        """Constraints should include allowed_extensions."""
        tools = default_config.create_toolset()
        for tool in tools:
            desc = tool.__tool_schema__["description"]
            assert "allowed_extensions" in desc
            # Check for actual extensions
            assert ".md" in desc or ".mmd" in desc
    
    def test_custom_limits_in_constraints(
        self, custom_config: ToolSandboxConfig
    ) -> None:
        """Custom limits should appear in constraints."""
        tools = custom_config.create_toolset()
        
        # Find read_file tool and check its max_lines constraint
        read_tool = next(
            t for t in tools if t.__tool_schema__["name"] == "read_file"
        )
        desc = read_tool.__tool_schema__["description"]
        assert "200" in desc  # Custom read_file_max_lines
        
        # Find grep_search tool and check its max_matches constraint
        grep_tool = next(
            t for t in tools if t.__tool_schema__["name"] == "grep_search"
        )
        desc = grep_tool.__tool_schema__["description"]
        assert "50" in desc  # Custom grep_max_matches
    
    def test_original_description_preserved(
        self, default_config: ToolSandboxConfig
    ) -> None:
        """Original tool description should be preserved before constraints."""
        tools = default_config.create_toolset()
        read_tool = next(
            t for t in tools if t.__tool_schema__["name"] == "read_file"
        )
        desc = read_tool.__tool_schema__["description"]
        # Original description content should be present
        assert "UTF-8" in desc or "text file" in desc


# ---------------------------------------------------------------------------
# Tests for tool functionality with custom config
# ---------------------------------------------------------------------------
class TestToolFunctionality:
    def test_read_file_respects_custom_extensions(
        self, temp_workspace: Path
    ) -> None:
        """read_file should use custom allowed_extensions."""
        config = ToolSandboxConfig(
            base_path=temp_workspace,
            allowed_extensions={".txt"},
        )
        
        # Create a .txt file
        txt_file = temp_workspace / "test.txt"
        txt_file.write_text("Hello from txt")
        
        # Create a .md file (should not be found with .txt only)
        md_file = temp_workspace / "test.md"
        md_file.write_text("Hello from md")
        
        tools = config.create_toolset()
        read_tool = next(
            t for t in tools if t.__tool_schema__["name"] == "read_file"
        )
        
        # .txt should work
        result = read_tool("test.txt")
        assert "Hello from txt" in result
        
        # .md should not be found (not in allowed_extensions)
        result = read_tool("test.md")
        assert "does not exist" in result
    
    def test_glob_search_respects_custom_extensions(
        self, temp_workspace: Path
    ) -> None:
        """glob_file_search should use custom allowed_extensions."""
        config = ToolSandboxConfig(
            base_path=temp_workspace,
            allowed_extensions={".py"},
        )
        
        # Create files
        (temp_workspace / "script.py").write_text("print('hello')")
        (temp_workspace / "doc.md").write_text("# Doc")
        
        tools = config.create_toolset()
        glob_tool = next(
            t for t in tools if t.__tool_schema__["name"] == "glob_file_search"
        )
        
        result = glob_tool("*")
        assert "script.py" in result
        assert "doc.md" not in result
    
    def test_list_dir_respects_custom_depth(
        self, temp_workspace: Path
    ) -> None:
        """list_dir should use custom max_depth."""
        config = ToolSandboxConfig(
            base_path=temp_workspace,
            allowed_extensions={".md"},
            list_dir_max_depth=2,  # Lower than default 5
        )
        
        # Create nested structure
        (temp_workspace / "d1" / "d2" / "d3").mkdir(parents=True)
        (temp_workspace / "d1" / "d2" / "d3" / "deep.md").write_text("deep")
        
        tools = config.create_toolset()
        list_tool = next(
            t for t in tools if t.__tool_schema__["name"] == "list_dir"
        )
        
        result = list_tool(".")
        # At depth 2, d2 should show depth limit summary
        assert "depth limit reached" in result


# ---------------------------------------------------------------------------
# Tests for integration with Agent
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_toolset_schemas_valid_for_anthropic_api(
        self, default_config: ToolSandboxConfig
    ) -> None:
        """Tool schemas should be valid Anthropic tool format."""
        tools = default_config.create_toolset()
        
        for tool in tools:
            schema = tool.__tool_schema__
            
            # Required fields
            assert isinstance(schema["name"], str)
            assert len(schema["name"]) > 0
            assert isinstance(schema["description"], str)
            assert isinstance(schema["input_schema"], dict)
            
            # input_schema structure
            input_schema = schema["input_schema"]
            assert input_schema.get("type") == "object"
            assert "properties" in input_schema
    
    def test_multiple_toolset_instances_independent(
        self, temp_workspace: Path
    ) -> None:
        """Multiple ToolSandboxConfig instances should be independent."""
        config1 = ToolSandboxConfig(
            base_path=temp_workspace,
            read_file_max_lines=100,
        )
        config2 = ToolSandboxConfig(
            base_path=temp_workspace,
            read_file_max_lines=200,
        )
        
        tools1 = config1.create_toolset()
        tools2 = config2.create_toolset()
        
        read1 = next(t for t in tools1 if t.__tool_schema__["name"] == "read_file")
        read2 = next(t for t in tools2 if t.__tool_schema__["name"] == "read_file")
        
        # Schemas should be different
        assert "100" in read1.__tool_schema__["description"]
        assert "200" in read2.__tool_schema__["description"]
