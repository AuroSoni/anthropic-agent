# Logging Guide

This guide covers the structured logging framework available in `anthropic_agent`.

## Quick Start

The library works out of the box with sensible defaults. No configuration required:

```python
from anthropic_agent import AnthropicAgent

agent = AnthropicAgent(...)
await agent.run("Hello!")  # Logs go to stderr automatically
```

To customize logging behavior:

```python
from anthropic_agent import configure_logging, LogConfig, LogLevel

configure_logging(LogConfig(level=LogLevel.DEBUG))
```

## Configuration

### LogConfig Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `level` | `LogLevel` | `INFO` | Default log level for all loggers |
| `format` | `LogFormat` | `PLAIN` | Output format (`PLAIN` or `JSON`) |
| `log_file` | `Path \| None` | `None` | Optional file path for log output |
| `max_bytes` | `int` | `10MB` | Max log file size before rotation |
| `backup_count` | `int` | `5` | Number of rotated files to keep |
| `module_levels` | `dict[str, LogLevel]` | `{}` | Per-module log level overrides |
| `filters` | `list[Callable]` | `[]` | Custom filter functions |

### Log Levels

```python
from anthropic_agent import LogLevel

LogLevel.DEBUG     # Detailed diagnostic information
LogLevel.INFO      # General operational messages
LogLevel.WARNING   # Unexpected but handled situations
LogLevel.ERROR     # Errors that need attention
LogLevel.CRITICAL  # System-critical failures
```

### Output Formats

**PLAIN** (default) - Human-readable format for development:

```
2026-01-28 10:15:30 [info     ] API call started               agent_uuid=abc-123 model=claude-sonnet-4-5
2026-01-28 10:15:31 [debug    ] Tool execution completed       tool=read_file duration_ms=45
```

**JSON** - Structured format for production/log aggregation:

```json
{"timestamp": "2026-01-28T10:15:30.123Z", "level": "info", "event": "API call started", "agent_uuid": "abc-123", "model": "claude-sonnet-4-5"}
{"timestamp": "2026-01-28T10:15:31.456Z", "level": "debug", "event": "Tool execution completed", "tool": "read_file", "duration_ms": 45}
```

## Usage Examples

### Basic Configuration

```python
from anthropic_agent import configure_logging, LogConfig, LogLevel, LogFormat

# Development: verbose console output
configure_logging(LogConfig(
    level=LogLevel.DEBUG,
    format=LogFormat.PLAIN,
))

# Production: JSON output to file
from pathlib import Path

configure_logging(LogConfig(
    level=LogLevel.INFO,
    format=LogFormat.JSON,
    log_file=Path("./logs/agent.log"),
    max_bytes=50 * 1024 * 1024,  # 50MB
    backup_count=10,
))
```

### Per-Module Log Levels

Silence noisy modules or enable debug for specific ones:

```python
configure_logging(LogConfig(
    level=LogLevel.INFO,  # Default level
    module_levels={
        # Silence retry noise
        "anthropic_agent.core.retry": LogLevel.WARNING,
        # Debug database issues
        "anthropic_agent.database": LogLevel.DEBUG,
        # Only errors from executors
        "anthropic_agent.python_executors": LogLevel.ERROR,
    }
))
```

### File Logging with Rotation

```python
from pathlib import Path

configure_logging(LogConfig(
    level=LogLevel.INFO,
    format=LogFormat.JSON,
    log_file=Path("./logs/agent.log"),
    max_bytes=10 * 1024 * 1024,  # 10MB per file
    backup_count=5,              # Keep 5 backup files
))
# Creates: agent.log, agent.log.1, agent.log.2, etc.
```

## Context Binding (Request Tracing)

Bind context values that appear in all subsequent log entries. This is essential for tracing logs across async agent runs.

### Basic Usage

```python
from anthropic_agent import bind_context, clear_context

# Bind context for a request
bind_context(request_id="req-abc123", user_id="user-456")

# All logs now include request_id and user_id
await agent.run("Process this data")

# Clean up after request
clear_context()
```

### Web Framework Integration

**FastAPI example:**

```python
from fastapi import FastAPI, Request
from anthropic_agent import bind_context, clear_context, AnthropicAgent

app = FastAPI()
agent = AnthropicAgent(...)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Bind request context
    bind_context(
        request_id=request.headers.get("X-Request-ID", str(uuid.uuid4())),
        path=request.url.path,
        method=request.method,
    )
    try:
        response = await call_next(request)
        return response
    finally:
        clear_context()

@app.post("/chat")
async def chat(message: str):
    # All logs from agent.run() will include request context
    result = await agent.run(message)
    return {"response": result.final_answer}
```

### Context in Async Tasks

Context automatically propagates across `await` boundaries:

```python
async def process_batch(items: list[str]):
    bind_context(batch_id="batch-123", item_count=len(items))
    
    for i, item in enumerate(items):
        bind_context(item_index=i)  # Add/update context
        await agent.run(item)       # Logs include batch_id, item_count, item_index
    
    clear_context()
```

### Removing Specific Context

```python
from anthropic_agent.logging import unbind_context

bind_context(a=1, b=2, c=3)
unbind_context("b", "c")  # Only 'a' remains
```

## Custom Filters

Filter or transform log entries before output:

```python
def drop_debug_from_retry(event_dict: dict) -> dict | None:
    """Drop debug logs from the retry module."""
    if (event_dict.get("logger", "").endswith("retry") 
        and event_dict.get("level") == "debug"):
        return None  # Drop this log
    return event_dict

def redact_secrets(event_dict: dict) -> dict | None:
    """Redact sensitive information."""
    if "api_key" in event_dict:
        event_dict["api_key"] = "***REDACTED***"
    return event_dict

configure_logging(LogConfig(
    filters=[drop_debug_from_retry, redact_secrets]
))
```

## Integration with Existing Logging

The framework bridges with Python's standard `logging` module. Existing code using `logging.getLogger()` will work seamlessly:

```python
import logging

# Your existing setup
existing_handler = logging.handlers.SysLogHandler(address='/dev/log')
logging.getLogger().addHandler(existing_handler)

# anthropic_agent logs flow through stdlib -> your handler
from anthropic_agent import configure_logging
configure_logging()  # Integrates with existing handlers
```

### Using get_logger for Rich Features

For new code, use `get_logger()` to access structured logging features:

```python
from anthropic_agent import get_logger

logger = get_logger(__name__)

# Structured logging with key-value pairs
logger.info("Processing started", item_count=42, user="alice")
logger.debug("Cache hit", key="user:123", ttl=300)
logger.error("Operation failed", error_code="E001", retry_count=3)
```

## Silencing Library Logs

To completely silence library logs:

```python
configure_logging(LogConfig(level=LogLevel.CRITICAL))
```

Or silence specific modules:

```python
configure_logging(LogConfig(
    level=LogLevel.INFO,
    module_levels={
        "anthropic_agent": LogLevel.ERROR,  # Only errors from library
    }
))
```

## Best Practices

1. **Call `configure_logging()` early** - Configure before creating any `AnthropicAgent` instances.

2. **Use JSON format in production** - Structured logs are easier to search and aggregate.

3. **Bind context at request boundaries** - Always `clear_context()` when a request completes.

4. **Use meaningful context keys** - `request_id`, `user_id`, `agent_uuid`, `run_id` are common choices.

5. **Don't log sensitive data** - Use filters to redact API keys, passwords, and PII.

6. **Set appropriate log levels** - Use `DEBUG` only in development; `INFO` or `WARNING` in production.
