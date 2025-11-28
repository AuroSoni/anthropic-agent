"""Example usage of AnthropicAgent with math tools."""
import asyncio
from anthropic_agent import AnthropicAgent
from anthropic_agent.src.tools.sample_tools import SAMPLE_TOOL_SCHEMAS, execute_tool


async def main():
    """Example usage of AnthropicAgent with math tools."""
    print("=" * 80)
    print("Anthropic Agent - Math Tools Example")
    print("=" * 80)
    
    # Create agent with math tools
    agent = AnthropicAgent(
        system_prompt="You are a helpful assistant that can perform mathematical calculations. Use the available tools to solve math problems.",
        model="claude-sonnet-4-5",
        max_steps=50,
        thinking_tokens=1024,  # Budget for thinking tokens
        max_tokens=16000,  # Must be greater than thinking_tokens
        tools=SAMPLE_TOOL_SCHEMAS,
        beta_headers=[]
    )
    
    # Create a queue for streaming output
    queue = asyncio.Queue()
    
    # Test prompt that requires multiple tool calls
    test_prompt = "Calculate (15 + 27) * 3 - 8. Show your work step by step."
    
    print(f"\nUser Query: {test_prompt}\n")
    print("Agent Response:")
    print("-" * 80)
    
    # Run agent and consume stream concurrently
    async def consume_stream():
        """Consume and print streaming output."""
        while True:
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                print(chunk, end="", flush=True)
            except asyncio.TimeoutError:
                # Check if agent task is done
                if agent_task.done():
                    # Drain any remaining items
                    while not queue.empty():
                        chunk = queue.get_nowait()
                        print(chunk, end="", flush=True)
                    break
            except Exception as e:
                print(f"\nError consuming stream: {e}")
                break
    
    # Start agent task
    agent_task = asyncio.create_task(
        agent.run(
            prompt=test_prompt,
            queue=queue,
            tool_executor=execute_tool
        )
    )
    
    # Start stream consumer
    stream_task = asyncio.create_task(consume_stream())
    
    # Wait for completion
    try:
        result = await agent_task
        await stream_task
        
        print("\n" + "-" * 80)
        print(f"\n✓ Agent completed successfully")
        print(f"  Total messages in conversation: {len(result.conversation_history)}")
        print(f"  Total agent steps: {result.total_steps}")
        print(f"  Final stop reason: {result.stop_reason}")
        print(f"  Model: {result.model}")
        print(f"  Token usage: {result.usage}")
        
        # Print message summary
        print("\nConversation Summary:")
        for i, msg in enumerate(result.conversation_history, 1):
            role = msg.get('role', 'unknown')
            content = msg.get('content', [])
            
            if role == 'user':
                # Check if it's tool results or user message
                if isinstance(content, list) and any(
                    isinstance(c, dict) and c.get('type') == 'tool_result' 
                    for c in content
                ):
                    tool_count = sum(1 for c in content if isinstance(c, dict) and c.get('type') == 'tool_result')
                    print(f"  {i}. [Tool Results] {tool_count} tool result(s)")
                else:
                    # Regular user message
                    if isinstance(content, list):
                        text_content = next((c.get('text', '') for c in content if isinstance(c, dict) and c.get('type') == 'text'), '')
                    else:
                        text_content = str(content)
                    preview = text_content[:60] + "..." if len(text_content) > 60 else text_content
                    print(f"  {i}. [User] {preview}")
            elif role == 'assistant':
                # Count tool uses
                tool_uses = sum(1 for c in content if hasattr(c, 'type') and c.type == 'tool_use')
                if tool_uses > 0:
                    print(f"  {i}. [Assistant] Used {tool_uses} tool(s)")
                else:
                    print(f"  {i}. [Assistant] Final response")
        
    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        stream_task.cancel()
        raise
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

