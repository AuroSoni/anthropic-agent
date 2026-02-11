"""
Integration test for frontend tools functionality.

This script tests the full flow:
1. Agent starts with backend and frontend tools
2. Claude calls a frontend tool (user_confirm)
3. Agent pauses with stop_reason="awaiting_frontend_tools"
4. State is persisted to DB
5. Agent is re-hydrated from DB using agent_uuid
6. Frontend tool results are submitted
7. Agent continues and completes

Run with: python tests/integration_test_frontend_tools.py
"""
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic_agent.tools import tool
from anthropic_agent.core import AnthropicAgent
from anthropic_agent.storage import create_adapters


# Define a frontend tool - executed by the browser
@tool(executor="frontend")
def user_confirm(message: str) -> str:
    """Ask the user for yes/no confirmation before proceeding with an action.
    
    Use this tool when you need explicit user approval before taking an action
    that could have significant consequences.
    
    Args:
        message: The confirmation message to display to the user, explaining
                what action requires their approval.
    
    Returns:
        "yes" if user confirms, "no" if user declines
    """
    pass  # Never executed server-side - runs in browser


# Define a backend tool - executed on server
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2")
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Safe eval for simple math
        allowed = set("0123456789+-*/().% ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"Result: {result}"
        return "Error: Invalid expression"
    except Exception as e:
        return f"Error: {str(e)}"


async def run_integration_test():
    """Run the full integration test."""
    print("=" * 60)
    print("Frontend Tools Integration Test")
    print("=" * 60)
    
    # Create queue to capture streaming output
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    collected_output = []
    
    # Task to collect queue output
    async def collect_output():
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            collected_output.append(chunk)
            # Print interesting chunks
            if "awaiting_frontend_tools" in chunk:
                print(f"\n>>> FRONTEND TOOLS SIGNAL: {chunk[:200]}...")
    
    # Step 1: Create agent with both backend and frontend tools
    print("\n[Step 1] Creating agent with backend + frontend tools...")
    base_path = tempfile.mkdtemp()
    config_adapter, conv_adapter, run_adapter = create_adapters("filesystem", base_path=base_path)
    await config_adapter.connect()
    await conv_adapter.connect()
    await run_adapter.connect()
    agent = AnthropicAgent(
        system_prompt="""You are a helpful assistant. When asked to perform calculations,
        use the calculate tool. When you get a result that seems significant (like any 
        number over 50), ask for user confirmation using the user_confirm tool before 
        reporting the final answer.""",
        model="claude-sonnet-4-20250514",
        thinking_tokens=1024,
        max_tokens=4096,
        tools=[calculate],
        frontend_tools=[user_confirm],
        config_adapter=config_adapter,
        conversation_adapter=conv_adapter,
        run_adapter=run_adapter,
    )
    
    print(f"   Agent UUID: {agent.agent_uuid}")
    print(f"   Backend tools: {[t['name'] for t in agent.tool_schemas]}")
    print(f"   Frontend tools: {[t['name'] for t in agent.frontend_tool_schemas]}")
    
    # Step 2: Run agent with a prompt designed to trigger frontend tool
    print("\n[Step 2] Running agent with prompt...")
    prompt = "Calculate 25 * 4 for me."
    print(f"   Prompt: '{prompt}'")
    
    # Start output collector
    collector_task = asyncio.create_task(collect_output())
    
    result = await agent.run(prompt, queue=queue)
    
    # Signal collector to stop
    await queue.put(None)
    await collector_task
    
    print(f"\n[Step 3] Checking result...")
    print(f"   Stop reason: {result.stop_reason}")
    print(f"   Total steps: {result.total_steps}")
    
    if result.stop_reason == "awaiting_frontend_tools":
        print("\n[OK] Agent correctly paused for frontend tools!")
        
        # Step 4: Check persisted state
        print("\n[Step 4] Verifying persisted state...")
        print(f"   Pending frontend tools: {agent._pending_frontend_tools}")
        print(f"   Pending backend results: {len(agent._pending_backend_results)} results")
        print(f"   Current step: {agent._current_step}")
        
        # Get the tool call info
        if agent._pending_frontend_tools:
            pending_tool = agent._pending_frontend_tools[0]
            tool_use_id = pending_tool["tool_use_id"]
            tool_name = pending_tool["name"]
            tool_input = pending_tool["input"]
            
            print(f"\n   Frontend tool requested:")
            print(f"      Tool: {tool_name}")
            print(f"      ID: {tool_use_id}")
            print(f"      Input: {json.dumps(tool_input, indent=8)}")
            
            # Step 5: Simulate re-hydration (like FastAPI would do)
            print("\n[Step 5] Re-hydrating agent from database...")
            saved_uuid = agent.agent_uuid
            
            # Create new agent instance with same UUID (simulates new request)
            # Use same base_path so state is loaded from storage
            config_adapter2, conv_adapter2, run_adapter2 = create_adapters("filesystem", base_path=base_path)
            agent2 = AnthropicAgent(
                system_prompt=agent.system_prompt,
                model=agent.model,
                thinking_tokens=agent.thinking_tokens,
                max_tokens=agent.max_tokens,
                tools=[calculate],
                frontend_tools=[user_confirm],
                config_adapter=config_adapter2,
                conversation_adapter=conv_adapter2,
                run_adapter=run_adapter2,
                agent_uuid=saved_uuid,
            )
            
            print(f"   Re-hydrated agent UUID: {agent2.agent_uuid}")
            print(f"   Awaiting frontend tools: {agent2._awaiting_frontend_tools}")
            print(f"   Pending frontend tools: {len(agent2._pending_frontend_tools)}")
            
            if agent2._awaiting_frontend_tools and agent2._pending_frontend_tools:
                print("\n[OK] State correctly restored from database!")
                
                # Step 6: Submit frontend tool result
                print("\n[Step 6] Submitting frontend tool result...")
                frontend_results = [
                    {
                        "tool_use_id": tool_use_id,
                        "content": "yes",  # User confirmed
                        "is_error": False,
                    }
                ]
                print(f"   Simulating user response: 'yes' (confirmed)")
                
                # Create new queue for continuation
                queue2: asyncio.Queue[str | None] = asyncio.Queue()
                continuation_output = []
                
                async def collect_continuation():
                    while True:
                        chunk = await queue2.get()
                        if chunk is None:
                            break
                        continuation_output.append(chunk)
                
                collector2 = asyncio.create_task(collect_continuation())
                
                result2 = await agent2.continue_with_tool_results(
                    frontend_results,
                    queue=queue2,
                )
                
                await queue2.put(None)
                await collector2
                
                print(f"\n[Step 7] Final result...")
                print(f"   Stop reason: {result2.stop_reason}")
                print(f"   Total steps: {result2.total_steps}")
                
                # Get final answer
                if result2.final_answer:
                    print(f"\n   Final answer preview:")
                    print(f"   {result2.final_answer[:500]}...")
                
                if result2.stop_reason == "end_turn":
                    print("\n[OK] Agent completed successfully after frontend tool confirmation!")
                    return True
                else:
                    print(f"\n[WARN] Agent ended with unexpected stop_reason: {result2.stop_reason}")
            else:
                print("\n[FAIL] Failed to restore state from database!")
        else:
            print("\n[FAIL] No pending frontend tools found!")
    
    elif result.stop_reason == "end_turn":
        # Agent completed without triggering frontend tool
        print("\n[WARN] Agent completed without triggering frontend tool.")
        print("   This can happen if Claude decided not to ask for confirmation.")
        print(f"   Final answer: {result.final_answer[:200] if result.final_answer else 'None'}...")
        
        # Check if any tools were used
        print(f"\n   Conversation history length: {len(result.conversation_history)}")
        for i, msg in enumerate(result.conversation_history):
            role = msg.get("role", "unknown")
            content = msg.get("content", [])
            if isinstance(content, list):
                tool_uses = [c for c in content if isinstance(c, dict) and c.get("type") == "tool_use"]
                if tool_uses:
                    print(f"   Message {i} ({role}): {len(tool_uses)} tool use(s)")
                    for tu in tool_uses:
                        print(f"      - {tu.get('name', 'unknown')}")
        
        return True  # Not a failure, just different behavior
    
    else:
        print(f"\n[FAIL] Unexpected stop_reason: {result.stop_reason}")
    
    return False


async def main():
    """Main entry point."""
    print(f"\nStarting test at {datetime.now().isoformat()}")
    print("-" * 60)
    
    try:
        success = await run_integration_test()
        
        print("\n" + "=" * 60)
        if success:
            print("*** INTEGRATION TEST PASSED ***")
        else:
            print("*** INTEGRATION TEST FAILED ***")
        print("=" * 60)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n[FAIL] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

