## Creating Agents with `anthropic_agent`

### Quick Start

```python
import asyncio
from anthropic_agent.core import AnthropicAgent
from anthropic_agent.database import FilesystemBackend
from anthropic_agent.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.
    
    Args:
        city: Name of the city
    """
    return f"<weather city='{city}'>Sunny, 72°F</weather>"

async def main():
    agent = AnthropicAgent(
        system_prompt="You are a helpful assistant.",
        model="claude-sonnet-4-5",
        tools=[get_weather],
        db_backend=FilesystemBackend(base_path="./data"),
    )
    
    result = await agent.run("What's the weather in Tokyo?")
    print(result.final_answer)

asyncio.run(main())
```

---

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | `str` | `"You are a helpful assistant..."` | System prompt guiding agent behavior |
| `model` | `str` | `"claude-sonnet-4-5"` | Anthropic model name |
| `max_steps` | `int` | `50` | Maximum conversation turns before stopping |
| `max_tokens` | `int` | `2048` | Maximum tokens in each response |
| `thinking_tokens` | `int` | `0` | Budget for extended thinking (0 = disabled) |
| `tools` | `list[Callable]` | `None` | Backend tools decorated with `@tool` |
| `frontend_tools` | `list[Callable]` | `None` | Browser-executed tools (schema-only on server) |
| `server_tools` | `list[dict]` | `None` | Anthropic-executed tools (MCP, code_execution) |
| `db_backend` | `str \| DatabaseBackend` | `"filesystem"` | Database for persistence |
| `file_backend` | `str \| FileStorageBackend` | `None` | File storage for generated files |
| `agent_uuid` | `str` | Auto-generated | Session UUID for resuming agents |
| `compactor` | `str \| Compactor` | `None` | Context compaction strategy |
| `memory_store` | `str \| MemoryStore` | `None` | Semantic context injection |
| `formatter` | `"xml" \| "raw"` | `"xml"` | Stream output format |
| `enable_cache_control` | `bool` | `True` | Enable Anthropic cache_control |
| `beta_headers` | `list[str]` | `None` | Beta feature headers |
| `**api_kwargs` | `Any` | — | Pass-through to Anthropic API (temperature, top_p, etc.) |

---

### Running the Agent

#### Basic Execution

```python
result = await agent.run("Your prompt here")
```

#### With Streaming Output

```python
import asyncio

queue: asyncio.Queue[str | None] = asyncio.Queue()

async def print_stream():
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        print(chunk, end="", flush=True)

# Run printer and agent concurrently
printer = asyncio.create_task(print_stream())
result = await agent.run("Calculate 15 * 27", queue)
await queue.put(None)  # Signal end
await printer
```

#### AgentResult Object

```python
@dataclass
class AgentResult:
    final_message: BetaMessage      # Last assistant message
    conversation_history: list[dict] # Full conversation (uncompacted)
    stop_reason: str                 # "end_turn", "tool_use", "max_tokens", etc.
    model: str                       # Model used
    usage: BetaUsage                 # Token usage statistics
    container_id: str | None         # Container ID (if applicable)
    total_steps: int                 # Number of agent steps taken
    agent_logs: list[dict] | None    # Compaction and action logs
    generated_files: list[dict] | None  # Files created during run
    final_answer: str                # Extracted final answer text
```

---

### Persistence & Session Resumption

Agents automatically persist state after each run. To resume a session:

```python
# First run - creates new session
agent = AnthropicAgent(
    system_prompt="You are a helpful assistant.",
    tools=[my_tool],
    db_backend=FilesystemBackend(base_path="./data"),
)
print(f"Session: {agent.agent_uuid}")  # Save this UUID
result = await agent.run("Hello!")

# Later - resume the same session
agent = AnthropicAgent(
    agent_uuid="<saved-uuid>",  # Resume this session
    tools=[my_tool],            # Must provide same tools
    db_backend=FilesystemBackend(base_path="./data"),
)
result = await agent.run("What did I say earlier?")  # Has context
```

#### Database Backends

| Backend | Usage | Best For |
|---------|-------|----------|
| `"filesystem"` | `FilesystemBackend(base_path="./data")` | Development, single-server |
| `"sql"` | `SQLBackend(connection_string="...")` | Production, multi-server |

```python
from anthropic_agent.database import FilesystemBackend, SQLBackend

# Filesystem (default)
db = FilesystemBackend(base_path="./data")

# PostgreSQL
db = SQLBackend(connection_string="postgresql://user:pass@host/db")

agent = AnthropicAgent(db_backend=db, ...)
```

#### Background Task Management

Agent state is persisted asynchronously. Before shutdown:

```python
await agent.drain_background_tasks()  # Ensure all saves complete
```

---

### Tool Integration

The agent supports three types of tools:

| Type | Execution | Use Case |
|------|-----------|----------|
| **Backend tools** | Server-side via `ToolRegistry` | Database queries, API calls, file operations |
| **Frontend tools** | Browser-side (client executes) | DOM manipulation, user interactions |
| **Server tools** | Anthropic-side (MCP servers) | Code execution, web search |

#### Backend Tools

```python
from anthropic_agent.tools import tool

@tool
def query_database(sql: str) -> str:
    """Execute a SQL query.
    
    Args:
        sql: The SQL query to execute
    """
    results = db.execute(sql)
    return f"<results count='{len(results)}'>{results}</results>"

agent = AnthropicAgent(tools=[query_database], ...)
```

#### Frontend Tools

Frontend tools pause the agent and emit an event for browser execution:

```python
from anthropic_agent.tools import tool

@tool(executor="frontend")
def get_user_location() -> str:
    """Get user's current location from browser.
    
    Returns:
        JSON with latitude and longitude
    """
    pass  # Implementation in browser

agent = AnthropicAgent(frontend_tools=[get_user_location], ...)
```

When Claude calls a frontend tool:
1. Agent pauses with `stop_reason="awaiting_frontend_tools"`
2. Client receives `<awaiting_frontend_tools>` event
3. Browser executes tool and POSTs results to `/agent/tool_results`
4. Agent resumes via `continue_with_tool_results()`

#### Server Tools (MCP)

Pass Anthropic-executed tool schemas directly:

```python
server_tools = [
    {
        "name": "code_execution",
        "description": "Execute Python code in a sandbox",
        "input_schema": {...}
    }
]

agent = AnthropicAgent(server_tools=server_tools, beta_headers=["code-execution-2024"], ...)
```

---

### Advanced Features

#### Extended Thinking

Enable Claude's extended thinking for complex reasoning:

```python
agent = AnthropicAgent(
    thinking_tokens=8000,  # Budget for thinking
    max_tokens=4096,       # Response budget
    ...
)
```

#### Context Compaction

Manage long conversations by removing old tool results:

```python
from anthropic_agent.core import get_compactor

# String shorthand (no threshold - always compacts)
agent = AnthropicAgent(compactor="tool_result_removal", ...)

# With threshold (compact only when estimated tokens exceed)
compactor = get_compactor("tool_result_removal", threshold=50000)
agent = AnthropicAgent(compactor=compactor, ...)
```

| Compactor | Behavior |
|-----------|----------|
| `"none"` | No compaction (default) |
| `"tool_result_removal"` | Removes tool results from older messages |

#### Memory Store

Inject relevant context from external memory:

```python
from anthropic_agent.memory import get_memory_store

memory = get_memory_store("placeholder")  # Or custom implementation
agent = AnthropicAgent(memory_store=memory, ...)
```

#### Final Answer Validation

Validate the agent's final response before accepting:

```python
def validate_answer(answer: str) -> tuple[bool, str]:
    """Return (success, error_message)."""
    if "FINAL:" not in answer:
        return False, "Response must include 'FINAL:' prefix"
    return True, ""

agent = AnthropicAgent(final_answer_check=validate_answer, ...)
```

If validation fails, the error is injected and the agent continues.

#### API Pass-through Arguments

Pass any Anthropic API parameter:

```python
agent = AnthropicAgent(
    temperature=0.7,
    top_p=0.9,
    stop_sequences=["END"],
    metadata={"user_id": "123"},
    ...
)
```

---

### File Backend

Store files generated by the agent (e.g., via code execution):

```python
from anthropic_agent.file_backends import LocalFilesystemBackend, S3Backend

# Local storage
file_backend = LocalFilesystemBackend(base_path="./agent-files")

# S3 storage
file_backend = S3Backend(
    bucket="my-bucket",
    prefix="agent-files/",
    region="us-east-1",
)

agent = AnthropicAgent(
    file_backend=file_backend,
    beta_headers=["files-api-2025"],  # Auto-added when file_backend set
    ...
)
```

Generated files are tracked in `result.generated_files`:

```python
for file in result.generated_files or []:
    print(f"File: {file['filename']}, Path: {file['storage_location']}")
```

---

### Common Patterns

#### Multi-turn Conversation

```python
agent = AnthropicAgent(...)

# Turn 1
result1 = await agent.run("My name is Alice")

# Turn 2 - agent remembers context
result2 = await agent.run("What's my name?")
```

#### Multimodal Input (Images)

```python
import base64

with open("image.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode()

prompt = [
    {"type": "text", "text": "What's in this image?"},
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": image_data,
        }
    }
]

result = await agent.run(prompt)
```

#### Custom Formatter

```python
# XML format (default) - structured tags for parsing
result = await agent.run(prompt, queue, formatter="xml")

# Raw format - plain text streaming
result = await agent.run(prompt, queue, formatter="raw")
```

#### Accessing Logs

```python
result = await agent.run(prompt)

for log in result.agent_logs or []:
    print(f"[{log['type']}] {log.get('message', '')}")
```

---

### Checklist Before Deploying

- [ ] `system_prompt` clearly defines agent behavior and constraints
- [ ] `max_steps` set appropriately (default 50 may be too high/low)
- [ ] `max_tokens` sufficient for expected response length
- [ ] All required tools registered and tested
- [ ] `db_backend` configured for production persistence
- [ ] `file_backend` configured if agent generates files
- [ ] `compactor` configured for long conversations
- [ ] `drain_background_tasks()` called before shutdown
- [ ] Error handling for `AgentResult.stop_reason` edge cases
- [ ] Streaming queue properly closed with `None` sentinel

---
