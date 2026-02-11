"""Example usage of AnthropicAgent with math tools."""
import asyncio
from anthropic_agent.core import AnthropicAgent
from anthropic_agent.storage import create_adapters
from anthropic_agent.tools import SAMPLE_TOOL_FUNCTIONS


async def main():
    """Example usage of AnthropicAgent with math tools."""
    print("=" * 80)
    print("Anthropic Agent - Math Tools Example")
    print("=" * 80)
    
    # Create filesystem adapters for persistence
    config_adapter, conv_adapter, run_adapter = create_adapters(
        "filesystem", base_path="./data"
    )
    
    # Create agent with math tools
    agent = AnthropicAgent(
        system_prompt="You are a helpful assistant that can perform mathematical calculations. Use the available tools to solve math problems.",
        model="claude-sonnet-4-5",
        tools=SAMPLE_TOOL_FUNCTIONS,
        config_adapter=config_adapter,
        conversation_adapter=conv_adapter,
        run_adapter=run_adapter,
    )
    
    print(f"Agent UUID: {agent.agent_uuid}")
    
    # Test prompt that requires multiple tool calls
    test_prompt = "Calculate (15 + 27) * 3 - 8. Show your work step by step."
    
    print(f"\nUser Query: {test_prompt}\n")
    print("Agent Response:")
    print("-" * 80)
    
    # Create a queue for streaming output
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def print_queue() -> None:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            print(chunk, end="", flush=True)
            queue.task_done()

    printer = asyncio.create_task(print_queue())
    
    result = await agent.run(test_prompt, queue)
    
    await queue.put(None)  # signal printer to stop
    await printer
    
    print("\n" + "-" * 80)
    print(f"\n✓ Agent completed successfully")
    print(f"  Total messages in conversation: {len(result.conversation_history)}")
    
    # Print final answer
    if result.final_message and result.final_message.content:
        print("\nFinal Answer:")
        print(result.final_message.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
