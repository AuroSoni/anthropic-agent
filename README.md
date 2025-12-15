# anthropic-agent

A small Python library for building **Claude-powered agents** with:
- **streaming** responses
- **tool calling** (client tools)
- **optional server tools** (passed through to Anthropic)
- **persistence** (filesystem or Postgres) and **file storage** (local or S3)

The primary package is `anthropic_agent` (install name: `anthropic-agent`).

## AnthropicAgent features

- **Resumable sessions**: pass `agent_uuid` to reload `messages`, `container_id`, token counters, and file registry from the DB backend.
- **Streaming output**: stream formatted chunks to an `asyncio.Queue` (`formatter="xml"` or `formatter="raw"`).
- **Tool execution loop**: executes *client tools* locally when the model returns `stop_reason == "tool_use"`, then feeds tool results back to the model.
- **Server tools passthrough**: include Anthropic server tools configs (e.g. code execution / web search) in requests.
- **Retry + backoff**: resilient streaming calls via `anthropic_stream_with_backoff(...)`.
- **Context compaction** (optional): plug in a `Compactor` to shrink message history before calls.
- **Memory hooks** (optional): plug in a `MemoryStore` to inject retrieved context and integrate with compaction lifecycle.
- **Persistence backends**: filesystem (default) or Postgres for agent config, conversation history, and per-run logs.
- **Files API + storage backends** (optional): detect generated `file_id`s, download via Anthropic Files API, and store via local filesystem or S3.

For a step-by-step diagram of the run loop, see `anthropic_agent/agent-flow.md`.

## Requirements

- Python **3.12+**
- `uv` for dependency management
- `ANTHROPIC_API_KEY` for runtime calls to the Anthropic API

## Install

```bash
uv sync
```

## Quickstart (library)

```bash
uv run python - <<'PY'
import asyncio

from anthropic_agent import AnthropicAgent


async def main() -> None:
    agent = AnthropicAgent(model="claude-sonnet-4-5")
    result = await agent.run("Hello!")
    print(result.final_answer)


asyncio.run(main())
PY
```

Notes:
- For local tool calling, pass `tools=[...]` (see `anthropic_agent/tools/sample_tools.py`).
- Runs are persisted under `./data/` by default (agent config, conversation history, run logs).

## Demos

### FastAPI server (SSE streaming + Files API upload)

The demo app lives in `demos/fastapi_server/` and exposes:
- `POST /agent/run`: streams output as Server-Sent Events (SSE)
- `POST /agent/upload`: uploads files (multipart or URL) to the Anthropic Files API

**Environment variables**:
- `ANTHROPIC_API_KEY` (required)
- `DATABASE_URL` (required by the demo: Postgres for runs)
- `S3_BUCKET` (required by the demo: store downloaded Files API artifacts)

Run it:

```bash
uv run --directory demos/fastapi_server uvicorn main:app --reload
```

Then try:

```bash
curl -N -X POST "http://127.0.0.1:8000/agent/run" \
  -H "content-type: application/json" \
  -d '{"user_prompt":"Calculate (15 + 27) * 3 - 8"}'
```

See `demos/fastapi_server/README.md` for request/response details.

### Vite UI (optional)

There’s also a small React app in `demos/vite_app/` that can consume the SSE stream.
It expects the FastAPI demo to allow CORS for `http://localhost:5173`.

## Project layout

- `anthropic_agent/`: library code (agent, tools, streaming, persistence backends)
- `demos/fastapi_server/`: runnable FastAPI demo
- `demos/vite_app/`: optional UI demo
- `tests/`: pytest tests

