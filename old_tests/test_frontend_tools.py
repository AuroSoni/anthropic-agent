"""Tests for frontend tool relay functionality."""
import asyncio
import pytest
from anthropic_agent.tools import tool
from anthropic_agent.core import AnthropicAgent
from anthropic_agent.storage import MemoryAgentConfigAdapter


# Define a frontend tool for testing
@tool(executor="frontend")
def user_confirm(message: str) -> str:
    """Ask the user for yes/no confirmation.
    
    Args:
        message: The confirmation message to display to the user
    
    Returns:
        "yes" or "no" based on user response
    """
    pass  # Never executed server-side


# Define a backend tool for testing
@tool
def add_numbers(a: float, b: float) -> str:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum as a string
    """
    return str(a + b)


class TestToolDecorator:
    """Test the @tool decorator with executor parameter."""
    
    def test_frontend_tool_has_executor_attribute(self):
        """Frontend tools should have __tool_executor__ = 'frontend'."""
        assert hasattr(user_confirm, '__tool_executor__')
        assert user_confirm.__tool_executor__ == 'frontend'
    
    def test_backend_tool_has_executor_attribute(self):
        """Backend tools should have __tool_executor__ = 'backend'."""
        assert hasattr(add_numbers, '__tool_executor__')
        assert add_numbers.__tool_executor__ == 'backend'
    
    def test_frontend_tool_has_schema(self):
        """Frontend tools should have __tool_schema__ generated."""
        assert hasattr(user_confirm, '__tool_schema__')
        schema = user_confirm.__tool_schema__
        assert schema['name'] == 'user_confirm'
        assert 'input_schema' in schema
    
    def test_backend_tool_has_schema(self):
        """Backend tools should have __tool_schema__ generated."""
        assert hasattr(add_numbers, '__tool_schema__')
        schema = add_numbers.__tool_schema__
        assert schema['name'] == 'add_numbers'


class TestAgentFrontendToolsInit:
    """Test AnthropicAgent initialization with frontend tools."""
    
    def test_agent_accepts_frontend_tools_parameter(self):
        """Agent should accept frontend_tools parameter."""
        agent = AnthropicAgent(
            tools=[add_numbers],
            frontend_tools=[user_confirm],
        )
        
        # Check frontend tool schemas are stored
        assert len(agent.frontend_tool_schemas) == 1
        assert agent.frontend_tool_schemas[0]['name'] == 'user_confirm'
        
        # Check frontend tool names set
        assert 'user_confirm' in agent.frontend_tool_names
        
        # Check backend tools still work
        assert len(agent.tool_schemas) == 1
        assert agent.tool_schemas[0]['name'] == 'add_numbers'
    
    def test_agent_initializes_frontend_relay_state(self):
        """Agent should initialize frontend tool relay state."""
        agent = AnthropicAgent(
            frontend_tools=[user_confirm],
        )
        
        assert agent._pending_frontend_tools == []
        assert agent._pending_backend_results == []
        assert agent._awaiting_frontend_tools == False
        assert agent._current_step == 0
    
    def test_agent_without_frontend_tools(self):
        """Agent without frontend tools should still work."""
        agent = AnthropicAgent(
            tools=[add_numbers],
        )
        
        assert len(agent.frontend_tool_schemas) == 0
        assert len(agent.frontend_tool_names) == 0


class TestContinueWithToolResultsValidation:
    """Test continue_with_tool_results validation."""
    
    def test_raises_when_not_awaiting(self):
        """Should raise ValueError when not awaiting frontend tools."""
        agent = AnthropicAgent(
            frontend_tools=[user_confirm],
        )
        
        async def run_test():
            with pytest.raises(ValueError, match="not awaiting frontend tools"):
                await agent.continue_with_tool_results([
                    {"tool_use_id": "test_123", "content": "yes"}
                ])
        
        asyncio.run(run_test())
    
    def test_raises_when_no_pending_tools(self):
        """Should raise ValueError when pending tools list is empty."""
        agent = AnthropicAgent(
            frontend_tools=[user_confirm],
        )
        # Manually set awaiting flag without pending tools
        agent._awaiting_frontend_tools = True
        agent._pending_frontend_tools = []
        
        async def run_test():
            with pytest.raises(ValueError, match="No pending frontend tools"):
                await agent.continue_with_tool_results([
                    {"tool_use_id": "test_123", "content": "yes"}
                ])
        
        asyncio.run(run_test())
    
    def test_raises_on_tool_id_mismatch(self):
        """Should raise ValueError when tool_use_ids don't match."""
        agent = AnthropicAgent(
            frontend_tools=[user_confirm],
        )
        # Set up pending state
        agent._awaiting_frontend_tools = True
        agent._pending_frontend_tools = [
            {"tool_use_id": "expected_id", "name": "user_confirm", "input": {"message": "test"}}
        ]
        agent._pending_backend_results = []
        
        async def run_test():
            with pytest.raises(ValueError, match="Tool result mismatch"):
                await agent.continue_with_tool_results([
                    {"tool_use_id": "wrong_id", "content": "yes"}
                ])
        
        asyncio.run(run_test())


class TestPrepareRequestParamsIncludesFrontendTools:
    """Test that _prepare_request_params includes frontend tool schemas."""
    
    def test_frontend_tools_included_in_request(self):
        """Frontend tool schemas should be included in API request tools."""
        agent = AnthropicAgent(
            tools=[add_numbers],
            frontend_tools=[user_confirm],
        )
        
        params = agent._prepare_request_params()
        
        # Should have both tools
        assert 'tools' in params
        tool_names = [t['name'] for t in params['tools']]
        assert 'add_numbers' in tool_names
        assert 'user_confirm' in tool_names


class TestStatePersistence:
    """Test that frontend tool relay state is properly saved and loaded."""
    
    def test_save_agent_config_includes_relay_state(self):
        """_save_agent_config should include frontend tool relay state."""
        adapter = MemoryAgentConfigAdapter()
        
        # Create agent with memory adapter for testing
        agent = AnthropicAgent(
            frontend_tools=[user_confirm],
            tools=[add_numbers],
            config_adapter=adapter,
        )
        
        # Set some relay state
        agent._pending_frontend_tools = [
            {"tool_use_id": "test_001", "name": "user_confirm", "input": {"message": "Proceed?"}}
        ]
        agent._pending_backend_results = [
            {"type": "tool_result", "tool_use_id": "backend_001", "content": "42"}
        ]
        agent._awaiting_frontend_tools = True
        agent._current_step = 3
        
        # Save and verify the config contains relay state
        async def run_test():
            await agent._save_agent_config()
            
            # Load config back via the adapter
            loaded_config = await adapter.load(agent.agent_uuid)
            
            assert loaded_config is not None
            assert loaded_config.pending_frontend_tools == agent._pending_frontend_tools
            assert loaded_config.pending_backend_results == agent._pending_backend_results
            assert loaded_config.awaiting_frontend_tools == True
            assert loaded_config.current_step == 3
        
        asyncio.run(run_test())
    
    def test_agent_rehydration_restores_relay_state(self):
        """Agent created with agent_uuid should restore relay state from DB."""
        # Shared adapter so agent2 can read what agent1 wrote
        adapter = MemoryAgentConfigAdapter()
        
        # Create first agent and set state
        agent1 = AnthropicAgent(
            frontend_tools=[user_confirm],
            tools=[add_numbers],
            config_adapter=adapter,
        )
        
        # Set relay state
        expected_frontend_tools = [
            {"tool_use_id": "test_002", "name": "user_confirm", "input": {"message": "OK?"}}
        ]
        expected_backend_results = [
            {"type": "tool_result", "tool_use_id": "backend_002", "content": "100"}
        ]
        agent1._pending_frontend_tools = expected_frontend_tools
        agent1._pending_backend_results = expected_backend_results
        agent1._awaiting_frontend_tools = True
        agent1._current_step = 5
        
        # Save state (need async)
        asyncio.run(agent1._save_agent_config())
        saved_uuid = agent1.agent_uuid
        
        # Create second agent with same UUID and same adapter
        agent2 = AnthropicAgent(
            frontend_tools=[user_confirm],
            tools=[add_numbers],
            config_adapter=adapter,
            agent_uuid=saved_uuid,
        )
        
        # State is loaded asynchronously via initialize()
        asyncio.run(agent2.initialize())
        
        # Verify state was restored
        assert agent2._pending_frontend_tools == expected_frontend_tools
        assert agent2._pending_backend_results == expected_backend_results
        assert agent2._awaiting_frontend_tools == True
        assert agent2._current_step == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

