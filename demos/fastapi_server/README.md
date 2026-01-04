# FastAPI Server Demo

This demo provides a complete backend API for the Anthropic Agent, exposing endpoints for agent execution, conversation management, file uploads, and frontend tool coordination.

All endpoints are defined in `demos/fastapi_server/agent_router.py` under the `/agent` prefix.

## Prerequisites

### Environment Variables

- **`ANTHROPIC_API_KEY`**: Required for Anthropic SDK client
- **`S3_BUCKET`**: S3 bucket for file storage (used by `S3Backend`)
- **`DATABASE_URL`**: PostgreSQL connection string (used by `SQLBackend`)
- **`OPENAI_API_KEY`**: Optional, used for conversation title generation via LiteLLM

### Dependencies

- **`python-multipart`**: Required for `multipart/form-data` parsing (`/agent/upload`)

## API Endpoints

### `POST /agent/run` - Run Agent (SSE Streaming)

Runs the agent and streams output via Server-Sent Events.

#### Request

```json
{
  "agent_uuid": "optional-uuid-to-resume",
  "agent_type": "agent_frontend_tools",
  "user_prompt": "Your message here"
}
```

- **`agent_uuid`** (optional): Resume an existing agent session
- **`agent_type`** (optional): Agent configuration to use. Options:
  - `agent_no_tools` - Basic agent without tools
  - `agent_client_tools` - Agent with client-side tools
  - `agent_all_raw` - Agent with all tools (raw format)
  - `agent_all_xml` - Agent with all tools (XML format)
  - `agent_frontend_tools` - Agent with browser-executed tools (default)
- **`user_prompt`**: String, list of content blocks, or message dict

#### Example

```bash
curl -N -X POST "http://127.0.0.1:8000/agent/run" \
  -H "Content-Type: application/json" \
  -d '{"user_prompt":"Calculate (15 + 27) * 3 - 8"}'
```

#### Response (SSE)

```text
data: <chunk>\n\n
data: <chunk>\n\n
data: [DONE]\n\n
```

---

### `POST /agent/tool_results` - Submit Frontend Tool Results

Resume agent execution after browser-executed tools (e.g., `user_confirm`) complete.

#### Request

```json
{
  "agent_uuid": "uuid-of-paused-agent",
  "tool_results": [
    {
      "tool_use_id": "toolu_xxx",
      "content": "yes",
      "is_error": false
    }
  ]
}
```

#### Response (SSE)

Same as `/agent/run` - streams remaining agent output.

---

### `GET /agent/sessions` - List Agent Sessions

Returns all agent sessions with metadata for sidebar display.

#### Query Parameters

- **`limit`** (default: 50, max: 100): Maximum sessions to return
- **`offset`** (default: 0): Pagination offset
- **`agent_type`** (default: `agent_frontend_tools`): Determines database backend

#### Example

```bash
curl "http://127.0.0.1:8000/agent/sessions?limit=20"
```

#### Response

```json
{
  "sessions": [
    {
      "agent_uuid": "abc-123-def",
      "title": "Math Calculation",
      "created_at": "2025-01-04T10:00:00Z",
      "updated_at": "2025-01-04T10:05:00Z",
      "total_runs": 3
    }
  ],
  "total": 42
}
```

---

### `GET /agent/{agent_uuid}/conversations` - Get Conversation History

Returns paginated conversation history for an agent (cursor-based pagination).

#### Query Parameters

- **`before`** (optional): Load conversations with `sequence_number < before`
- **`limit`** (default: 20, max: 100): Maximum conversations to return
- **`agent_type`** (default: `agent_frontend_tools`): Determines database backend

#### Example

```bash
curl "http://127.0.0.1:8000/agent/abc-123/conversations?limit=10"
```

#### Response

```json
{
  "conversations": [
    {
      "conversation_id": "conv-001",
      "run_id": "run-001",
      "sequence_number": 3,
      "user_message": "What is 2+2?",
      "final_response": "The answer is 4.",
      "started_at": "2025-01-04T10:00:00Z",
      "completed_at": "2025-01-04T10:00:05Z",
      "stop_reason": "end_turn",
      "total_steps": 1,
      "usage": {"input_tokens": 50, "output_tokens": 20},
      "generated_files": [],
      "messages": [...]
    }
  ],
  "has_more": true,
  "title": "Math Questions"
}
```

---

### `POST /agent/{agent_uuid}/title` - Generate Conversation Title

Generates a title for the conversation using LLM (GPT-4o-mini via LiteLLM).

#### Request

```json
{
  "user_message": "The first message from the user"
}
```

#### Query Parameters

- **`agent_type`** (default: `agent_frontend_tools`): Determines database backend

#### Example

```bash
curl -X POST "http://127.0.0.1:8000/agent/abc-123/title" \
  -H "Content-Type: application/json" \
  -d '{"user_message":"Help me debug my Python code"}'
```

#### Response

```json
{
  "title": "Python Debugging Help"
}
```

---

### `POST /agent/upload` - Upload Files

Uploads files to Anthropic Files API.

#### Request (`multipart/form-data`)

- **`files`**: File field (repeatable)
- **`urls`**: URL to download and upload (repeatable)

#### Example

```bash
curl -X POST "http://127.0.0.1:8000/agent/upload" \
  -F "files=@./document.pdf" \
  -F "urls=https://example.com/image.png"
```

#### Response

```json
{
  "files": [
    {
      "id": "file_xxx",
      "filename": "document.pdf",
      "mime_type": "application/pdf",
      "size_bytes": 12345,
      "created_at": "2025-01-04T10:00:00Z",
      "downloadable": false
    }
  ]
}
```

## Agent Types

| Type | Description | Tools | Format |
|------|-------------|-------|--------|
| `agent_no_tools` | Basic assistant | None | Raw |
| `agent_client_tools` | With client tools | Sample tools | Raw |
| `agent_all_raw` | Full featured | All + server tools | Raw |
| `agent_all_xml` | Full featured | All + server tools | XML |
| `agent_frontend_tools` | Browser tools | Sample + frontend | XML |

## Running the Server

```bash
cd demos/fastapi_server
uvicorn main:app --reload --port 8000
```
