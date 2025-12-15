### FastAPI server demo

This demo exposes two APIs from `demos/fastapi_server/agent_router.py` under the `/agent` prefix:

- **`POST /agent/run`**: streams agent output via Server-Sent Events (SSE)
- **`POST /agent/upload`**: uploads one or more files (local multipart uploads and/or URLs) to the Anthropic **Files API** and returns file metadata

The upload endpoint uses Anthropic’s Files API as described in the docs: [`https://platform.claude.com/docs/en/build-with-claude/files`](https://platform.claude.com/docs/en/build-with-claude/files).

### Prerequisites

- **Environment variables**
  - **`ANTHROPIC_API_KEY`**: required (used by the Anthropic SDK client)
  - **`S3_BUCKET`**: used by the agent’s `S3Backend` (for agent runs)
  - **`DATABASE_URL`**: used by the agent’s `SQLBackend` (for agent runs)
- **Dependencies**
  - **`python-multipart`** is required for `multipart/form-data` parsing (needed by `/agent/upload`)

### API: `POST /agent/run` (SSE streaming)

Runs the agent and streams raw output chunks as **SSE events**.

#### Request (JSON)

Body schema:

- **`agent_uuid`** (optional, string): resume an existing agent run
- **`user_prompt`** (required): one of:
  - string
  - list of dicts
  - dict

Example:

```bash
curl -N -X POST "http://127.0.0.1:8000/agent/run" \
  -H "content-type: application/json" \
  -d '{"user_prompt":"Calculate (15 + 27) * 3 - 8"}'
```

#### Response (SSE)

Response headers include:

- **`content-type: text/event-stream`**
- **`cache-control: no-cache`**
- **`connection: keep-alive`**

Each chunk is sent as an SSE `data:` line, terminated by a blank line:

```text
data: <chunk>\n\n
```

Important formatting details:

- **Newlines inside a chunk** are escaped to literal `\n` (the server does `chunk.replace(chr(10), "\\n")`).
- When the stream completes, the server emits a final marker:

```text
data: [DONE]\n\n
```

Error behavior:

- On exception, the stream yields a single SSE event like:

```text
data: Error: <message>\n\n
```

### API: `POST /agent/upload` (multipart form upload)

Uploads one or more files to Anthropic Files API and returns metadata + `file_id`s.

#### Request (`multipart/form-data`)

Form fields:

- **`files`**: optional, repeatable file field
- **`urls`**: optional, repeatable text field containing a URL to download and upload

You can send any combination of both. At least one must be provided.

Example: upload local files

```bash
curl -X POST "http://127.0.0.1:8000/agent/upload" \
  -F "files=@./path/to/document.pdf" \
  -F "files=@./path/to/image.png"
```

Example: upload from URL(s)

```bash
curl -X POST "http://127.0.0.1:8000/agent/upload" \
  -F "urls=https://example.com/some-file.jpg"
```

Example: mixed (files + URL)

```bash
curl -X POST "http://127.0.0.1:8000/agent/upload" \
  -F "files=@./path/to/file.txt" \
  -F "urls=https://example.com/another-file.png"
```

#### Response (JSON)

Success response schema:

- **`files`**: array of file metadata objects (one per successfully uploaded file)

Each metadata object has:

- **`id`**: string (Anthropic `file_id`, e.g. `file_...`)
- **`filename`**: string
- **`mime_type`**: string
- **`size_bytes`**: integer
- **`created_at`**: string (ISO-8601 timestamp; the server returns `result.created_at.isoformat()`)
- **`downloadable`**: boolean

Example response:

```json
{
  "files": [
    {
      "id": "file_011CW7uSsiQEEF4L8K3jiYCA",
      "filename": "test_upload.txt",
      "mime_type": "text/plain",
      "size_bytes": 147,
      "created_at": "2025-12-15T05:16:36.249000+00:00",
      "downloadable": false
    }
  ]
}
```

Error behavior:

- **400** if neither `files` nor `urls` are provided:
  - `{"detail":"At least one file or URL must be provided"}`
- **400** if a URL cannot be downloaded (HTTP error)
- **500** if uploading to Anthropic fails

