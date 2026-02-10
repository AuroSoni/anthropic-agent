# New Streaming Paradigm — JSON Envelope

This document defines the new streaming format for the anthropic_agent library. It replaces the legacy XML tag-based streaming schema (documented in `current_xml_schema.md`) with a JSON envelope protocol designed for reliable SSE transport and trivial parsing.

## Motivation

The legacy XML format uses stateful open/close tags (`<content-block-text>...</content-block-text>`) with raw text deltas flowing between them. This breaks down when:

1. **Large payloads** — Tags like `meta_init` and `server_tool_result` dump entire escaped JSON blobs into a single SSE `data:` line. Oversized messages can trigger SSE transport fragmentation, splitting a logical message across multiple `data:` lines.
2. **No agent scoping** — The flat tag sequence has no mechanism to route chunks to the correct UI subtree when subagents are involved.

## Design Principles

1. **Every SSE message is self-contained** — Each `data:` line is a complete JSON object with routing context. No stateful open/close pairs.
2. **Proactive chunking** — The server controls message size. Large payloads are split into multiple messages, preventing the SSE transport layer from fragmenting them.
3. **Trivial parsing** — The frontend calls `JSON.parse()` on each `data:` line. No regex, no XML parser, no entity decoding.
4. **Arrival-order reconstruction** — SSE is an ordered, single-connection protocol. The consumer reconstructs content by appending deltas in the order they arrive. The `final` flag marks block boundaries.

## Message Format

Every SSE `data:` line is a JSON object with a standardized set of fields.

### Base Fields (present on every message)

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Content type identifier (see [Message Types](#message-types)) |
| `agent` | `string` (UUID) | Agent scope — identifies which agent produced this chunk |
| `final` | `bool` | `true` if this is the last chunk for the current content block |
| `delta` | `string` | The content payload for this chunk (may be empty on final markers) |

### Extra Fields (per message type)

Some message types carry additional fields beyond the base set. These are documented per type below.

---

## Message Types

There are **13 message types** organized into four categories:

| Category | Types | Emission |
|----------|-------|----------|
| **Metadata** | `meta_init`, `meta_final` | Buffered, from agent control flow |
| **Streamed content** | `thinking`, `text`, `citation` | Incremental deltas from Anthropic API |
| **Tool calls** | `tool_call`, `server_tool_call` | Buffered from Anthropic API |
| **Tool results** | `tool_result`, `tool_result_image`, `server_tool_result` | Buffered, from agent tool execution / API |
| **Control flow** | `awaiting_frontend_tools`, `meta_files`, `error` | Buffered, from agent control flow / API |

---

### `meta_init`

Emitted once at stream start. Contains agent configuration and session metadata.

**Source:** `agent.py` — emitted directly via `_chunk_and_emit()` before the first API call.

**Emission:** Buffered. The `delta` carries a JSON string (the serialized metadata object). If the metadata is large, it is chunked across multiple messages.

**Delta JSON keys:**

| Key | Type | Description |
|-----|------|-------------|
| `format` | `string` | Active formatter (`"xml"`, `"json"`, `"raw"`) |
| `user_query` | `string` | The user's input prompt |
| `agent_uuid` | `string` | UUID of the agent |
| `model` | `string` | Claude model name (e.g., `"claude-sonnet-4-5"`) |
| `message_history` | `array` (optional) | Conversation history (only when `stream_meta_history_and_tool_results` is enabled) |

```json
{"type":"meta_init","agent":"abc-123","final":false,"delta":"{\"format\":\"json\",\"user_query\":\"He"}
{"type":"meta_init","agent":"abc-123","final":true,"delta":"llo\",\"model\":\"claude-sonnet-4-5\"}"}
```

---

### `thinking`

Extended thinking content. Streamed incrementally as deltas arrive from the Anthropic API.

**Source:** `json_formatter` in `formatters.py` — emits each `thinking_delta` event immediately.

**Emission:** Incremental. Every API delta is forwarded immediately with `final: false`. When `content_block_stop` is received, a final marker with `final: true` and empty `delta` is emitted.

**Extra fields:** None.

```json
{"type":"thinking","agent":"abc-123","final":false,"delta":"The user is asking about"}
{"type":"thinking","agent":"abc-123","final":false,"delta":" the color of grass."}
{"type":"thinking","agent":"abc-123","final":true,"delta":""}
```

---

### `text`

Assistant text content. Streamed incrementally.

**Source:** `json_formatter` in `formatters.py` — emits each `text_delta` event immediately.

**Emission:** Incremental. Same pattern as `thinking`. May be followed by `citation` messages after the final marker.

**Extra fields:** None.

```json
{"type":"text","agent":"abc-123","final":false,"delta":"Based on the document, "}
{"type":"text","agent":"abc-123","final":false,"delta":"the grass is green."}
{"type":"text","agent":"abc-123","final":true,"delta":""}
```

---

### `citation`

Citation metadata attached to the preceding `text` block. Emitted immediately after the text block's final marker.

**Source:** `json_formatter` in `formatters.py` — extracted from `content_block_stop` event's `citations` array.

**Emission:** Buffered. Each citation is emitted as a single message. Multiple citations for the same text block arrive in sequence; the last one carries `final: true`.

**Extra fields:**

| Field | Type | Description |
|-------|------|-------------|
| `citation_type` | `string` | One of `char_location`, `page_location`, `web_search_result_location` |
| `document_index` | `int` (optional) | Index of the source document |
| `document_title` | `string` (optional) | Title of the source document |
| `start_char_index` | `int` (optional) | Start position for `char_location` citations |
| `end_char_index` | `int` (optional) | End position for `char_location` citations |
| `start_page_number` | `int` (optional) | Start page for `page_location` citations |
| `end_page_number` | `int` (optional) | End page for `page_location` citations |
| `url` | `string` (optional) | Source URL for `web_search_result_location` citations |
| `title` | `string` (optional) | Source title for `web_search_result_location` citations |

`delta` carries the cited text.

```json
{"type":"citation","agent":"abc-123","final":true,"delta":"The grass is green.","citation_type":"char_location","document_index":0,"document_title":"My Document","start_char_index":0,"end_char_index":20}
```

Multiple citations:

```json
{"type":"citation","agent":"abc-123","final":false,"delta":"First cited passage.","citation_type":"char_location","document_index":0,"start_char_index":0,"end_char_index":20}
{"type":"citation","agent":"abc-123","final":true,"delta":"Second cited passage.","citation_type":"web_search_result_location","url":"https://example.com","title":"Example"}
```

---

### `tool_call`

A client-side (backend) tool invocation. The full tool call is buffered server-side and emitted as a single message (or chunked if the arguments are large).

**Source:** `json_formatter` in `formatters.py` — buffered during `input_json_delta` events, emitted at `content_block_stop`.

**Emission:** Buffered. Passed through `_chunk_and_emit()` with `final_on_last=True`.

**Extra fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Tool use ID (matches `tool_result.id` in the response) |
| `name` | `string` | Tool function name |

`delta` carries the JSON-serialized arguments object.

```json
{"type":"tool_call","agent":"abc-123","final":true,"id":"toolu_01","name":"grep_search","delta":"{\"pattern\":\"TODO\",\"path\":\"src/\"}"}
```

If arguments are large, they are chunked:

```json
{"type":"tool_call","agent":"abc-123","final":false,"id":"toolu_01","name":"grep_search","delta":"{\"pattern\":\"TODO\",\"pa"}
{"type":"tool_call","agent":"abc-123","final":true,"id":"toolu_01","name":"grep_search","delta":"th\":\"src/\"}"}
```

---

### `server_tool_call`

A server-side tool invocation (e.g., `web_search`, `code_execution`). Same structure as `tool_call` but for Anthropic's built-in server tools.

**Source:** `json_formatter` in `formatters.py` — buffered and emitted at `content_block_stop`.

**Emission:** Buffered. Same chunking behavior as `tool_call`.

**Extra fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Server tool use ID |
| `name` | `string` | Server tool name (e.g., `web_search`, `code_execution`) |

```json
{"type":"server_tool_call","agent":"abc-123","final":true,"id":"srvtoolu_01","name":"web_search","delta":"{\"query\":\"latest ai news\"}"}
```

---

### `tool_result`

Result of a backend tool execution (your `@tool` functions) or a frontend tool execution (browser-side tools).

**Source:** `_emit_tool_result()` in `agent.py` — emitted after each tool is executed in the agent loop, or when frontend tool results are received via `continue_with_tool_results()`.

**Emission:** Buffered. Passed through `_chunk_and_emit()` with `final_on_last=True`. Large results are automatically chunked.

**Extra fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Tool use ID this result responds to |
| `name` | `string` | Tool function name |

`delta` carries the result content (plain text, JSON string, etc.).

#### Text-only result

```json
{"type":"tool_result","agent":"abc-123","final":true,"id":"toolu_01","name":"grep_search","delta":"Found 4 matches in src/"}
```

#### Chunked large result

```json
{"type":"tool_result","agent":"abc-123","final":false,"id":"toolu_02","name":"read_file","delta":"Line 1: import os\nLine 2: import sy"}
{"type":"tool_result","agent":"abc-123","final":true,"id":"toolu_02","name":"read_file","delta":"s\nLine 3: from pathlib import Path"}
```

#### Multimodal result (text + images)

When a tool result contains images alongside text, the result is split into a multi-message sequence:

1. A `tool_result` message with the text content and `final: false`
2. One or more `tool_result_image` messages (one per image)
3. A final `tool_result` message with empty `delta` and `final: true`

```json
{"type":"tool_result","agent":"abc-123","final":false,"id":"toolu_03","name":"screenshot","delta":"Screenshot captured successfully"}
{"type":"tool_result_image","agent":"abc-123","final":false,"id":"toolu_03","name":"screenshot","delta":"","src":"data:image/png;base64,iVBOR...","media_type":"image/png"}
{"type":"tool_result","agent":"abc-123","final":true,"id":"toolu_03","name":"screenshot","delta":""}
```

See [`tool_result_image`](#tool_result_image) for image message details.

---

### `tool_result_image`

Image data within a multimodal tool result. Always appears between a non-final `tool_result` and the final `tool_result` marker.

**Source:** `_emit_tool_result()` in `agent.py` — emitted when the tool returns `image_refs`.

**Emission:** Buffered. One message per image. Always carries `final: false` (the enclosing `tool_result` sequence handles finality).

**Extra fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Tool use ID (same as parent `tool_result`) |
| `name` | `string` | Tool function name (same as parent `tool_result`) |
| `src` | `string` | Image data URI (e.g., `data:image/png;base64,...`) or URL |
| `media_type` | `string` | MIME type (e.g., `image/png`, `image/jpeg`) |

`delta` is always empty (`""`).

```json
{"type":"tool_result_image","agent":"abc-123","final":false,"id":"toolu_03","name":"screenshot","delta":"","src":"data:image/png;base64,iVBOR...","media_type":"image/png"}
```

---

### `server_tool_result`

Result of a server-side tool execution (e.g., web search results, code execution output). These are Anthropic's built-in tools, not user-defined tools.

**Source:** `json_formatter` in `formatters.py` — buffered during API streaming, emitted at `content_block_stop`.

**Emission:** Buffered. Passed through `_chunk_and_emit()` with `final_on_last=True`. Only emitted when `stream_tool_results` is `True`.

**Extra fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Server tool use ID (matches the `server_tool_call.id`) |
| `name` | `string` | Result type name from the API (e.g., `web_search_tool_result`, `bash_code_execution_tool_result`) |

```json
{"type":"server_tool_result","agent":"abc-123","final":false,"id":"srvtoolu_01","name":"web_search_tool_result","delta":"[{\"title\":\"Resul"}
{"type":"server_tool_result","agent":"abc-123","final":true,"id":"srvtoolu_01","name":"web_search_tool_result","delta":"t 1\"}]"}
```

---

### `awaiting_frontend_tools`

Signals that the agent is paused, waiting for the frontend to execute client tools and return results via `POST /agent/tool_results`.

**Source:** `agent.py` — emitted in `run()` and `_resume_run()` when Claude requests frontend tool execution.

**Emission:** Buffered. Passed through `_chunk_and_emit()` with `final_on_last=True`.

**Extra fields:** None.

`delta` carries a JSON array of pending tool invocations. Each element has:

| Key | Type | Description |
|-----|------|-------------|
| `tool_use_id` | `string` | Tool use ID to include in the result |
| `name` | `string` | Frontend tool function name |
| `input` | `object` | Arguments passed to the tool |

```json
{"type":"awaiting_frontend_tools","agent":"abc-123","final":true,"delta":"[{\"tool_use_id\":\"toolu_01\",\"name\":\"user_confirm\",\"input\":{\"question\":\"Continue?\"}}]"}
```

---

### `meta_files`

File metadata generated during the run (e.g., files created in a code execution container).

**Source:** `agent.py` — emitted via `_stream_file_metadata()`.

**Emission:** Buffered. Passed through `_chunk_and_emit()` with `final_on_last=True`.

**Extra fields:** None.

`delta` carries a JSON object with a `files` array:

| Key | Type | Description |
|-----|------|-------------|
| `files` | `array` | Array of file metadata objects |
| `files[].file_id` | `string` | Unique file identifier |
| `files[].filename` | `string` | Original filename |
| `files[].storage_location` | `string` | Download URL or storage path |

```json
{"type":"meta_files","agent":"abc-123","final":true,"delta":"{\"files\":[{\"file_id\":\"file_01\",\"filename\":\"report.pdf\",\"storage_location\":\"https://...\"}]}"}
```

If the file list is large, it is chunked:

```json
{"type":"meta_files","agent":"abc-123","final":false,"delta":"{\"files\":[{\"file_id\":\"file_01\",\"filename\":\"report.pdf\",\"storage_lo"}
{"type":"meta_files","agent":"abc-123","final":true,"delta":"cation\":\"https://...\"}]}"}
```

---

### `error`

Error event from the Anthropic API.

**Source:** `json_formatter` in `formatters.py` — emitted when an `error` event is received from the API stream.

**Emission:** Buffered. Passed through `_chunk_and_emit()` with `final_on_last=True`.

**Extra fields:** None.

`delta` carries a JSON-serialized error payload.

```json
{"type":"error","agent":"abc-123","final":true,"delta":"{\"type\":\"api_error\",\"message\":\"rate_limit\"}"}
```

---

### `meta_final`

Emitted once at stream end. Contains run summary and final conversation state.

**Source:** `agent.py` — emitted via `_emit_meta_final()`. Only emitted when `stream_meta_history_and_tool_results` is enabled. Not emitted for partial returns (e.g., `awaiting_frontend_tools`).

**Emission:** Buffered. Passed through `_chunk_and_emit()` with `final_on_last=True`.

**Extra fields:** None.

**Delta JSON keys:**

| Key | Type | Description |
|-----|------|-------------|
| `conversation_history` | `array` | Full conversation history |
| `stop_reason` | `string` | Why the run ended (`"end_turn"`, `"max_tokens"`, etc.) |
| `total_steps` | `int` | Number of API round-trips |
| `generated_files` | `array \| null` | File metadata from the run |
| `cost` | `object \| null` | Cost breakdown |
| `cumulative_usage` | `object` | Token usage totals (`input_tokens`, `output_tokens`) |

```json
{"type":"meta_final","agent":"abc-123","final":true,"delta":"{\"stop_reason\":\"end_turn\",\"total_steps\":3,\"cost\":null,\"cumulative_usage\":{\"input_tokens\":1000,\"output_tokens\":300}}"}
```

---

## Emission Sources

Message types are emitted from two distinct layers:

### 1. Formatter layer (`formatters.py` → `json_formatter`)

Processes the Anthropic API streaming response. Handles all content that comes from Claude's output.

| Type | Trigger |
|------|---------|
| `text` | `text_delta` events (incremental) |
| `thinking` | `thinking_delta` events (incremental) |
| `citation` | `content_block_stop` with `citations` array |
| `tool_call` | `content_block_stop` for `tool_use` blocks (buffered) |
| `server_tool_call` | `content_block_stop` for `server_tool_use` blocks (buffered) |
| `server_tool_result` | `content_block_stop` for `*_tool_result` blocks (buffered) |
| `error` | `error` API event |

### 2. Agent layer (`agent.py`)

Handles metadata, tool execution, and control flow — events that originate from the agent framework, not from Claude's API stream.

| Type | Trigger |
|------|---------|
| `meta_init` | Start of `run()` — before first API call |
| `meta_final` | End of `run()` / `_resume_run()` — after final API response |
| `tool_result` | After each backend tool executes, or when frontend results arrive |
| `tool_result_image` | Within multimodal `tool_result` sequence (images) |
| `awaiting_frontend_tools` | When Claude requests frontend tool execution |
| `meta_files` | When files are generated during the run |

---

## Final Flag Semantics

### Streamed blocks (`thinking`, `text`)

Every delta received from the API is emitted immediately as `final: false`. When `content_block_stop` is received, a message with `final: true` and `delta: ""` is emitted. This provides zero-latency streaming — no buffering delay on any intermediate chunk.

```
API delta "Hello"        → {"final":false, "delta":"Hello"}
API delta " world"       → {"final":false, "delta":" world"}
API content_block_stop   → {"final":true,  "delta":""}
```

### Buffered blocks (all other types)

The entire payload is known before emission. The last chunk (or the only chunk if not chunked) carries `final: true` **with** the content in `delta`.

```
Buffered tool call       → {"final":true, "delta":"{\"pattern\":\"TODO\"}"}
```

```
Chunked large result     → {"final":false, "delta":"[{\"title\":\"Resul"}
                            {"final":true,  "delta":"t 1\"}]"}
```

### Multimodal tool results

The `tool_result` + `tool_result_image` sequence uses a compound final pattern:

```
Text content             → {"type":"tool_result",       "final":false, "delta":"Screenshot taken"}
Image 1                  → {"type":"tool_result_image",  "final":false, "delta":"", "src":"...", "media_type":"image/png"}
Image 2                  → {"type":"tool_result_image",  "final":false, "delta":"", "src":"...", "media_type":"image/jpeg"}
Final marker             → {"type":"tool_result",       "final":true,  "delta":""}
```

The `tool_result_image` messages always have `final: false`. The enclosing `tool_result` sequence's final marker closes the entire multimodal block.

### Citations

Citations arrive immediately after a `text` block's final marker. Multiple citations carry `final: false` on all but the last:

```
Text final               → {"type":"text",     "final":true,  "delta":""}
Citation 1               → {"type":"citation",  "final":false, "delta":"cited text 1", ...}
Citation 2               → {"type":"citation",  "final":true,  "delta":"cited text 2", ...}
```

---

## Frontend Handling

The frontend tracks the current open block per `(agent, type)`. On each message:

1. Look up or create the block state for the current `(agent, type)`.
2. Append `delta` to the block's content buffer.
3. If `final` is `true`, mark the block as complete and close the tracker — the next message of the same type opens a new block.
4. For `citation` messages, attach citation metadata to the most recently completed `text` block.
5. For `tool_result_image` messages, attach the image to the currently open `tool_result` block.

---

## Proactive Chunking

The server enforces a maximum SSE message size to prevent transport-layer fragmentation.

### Algorithm

```
MAX_SSE_CHUNK_BYTES = 2048

To emit a payload for a given (type, agent, extra_fields):

1. Build the JSON envelope with delta="" to measure overhead bytes.
2. max_delta_bytes = MAX_SSE_CHUNK_BYTES - overhead
3. Encode payload as UTF-8.
4. Split into chunks of at most max_delta_bytes, respecting UTF-8
   character boundaries.
5. Emit each chunk as a message.
   The last chunk carries final=true (for buffered blocks) or
   final=false (for streamed blocks where final comes separately).
```

### UTF-8 Safety

When splitting by byte length, the split point must not fall in the middle of a multi-byte UTF-8 sequence. If the split point lands on a continuation byte (`0b10xxxxxx`), back up to the start of that character.

### Overhead Calculation

The overhead is everything in the JSON message except the `delta` value content. For example:

```json
{"type":"text","agent":"abc-123","final":false,"delta":""}
```

This is ~52 bytes of overhead. With `MAX_SSE_CHUNK_BYTES = 2048`, the effective max delta size is ~1996 bytes.

Extra fields increase overhead. A `tool_result` with `id` and `name` has ~80 bytes of overhead, leaving ~1968 bytes for the delta.

---

## Multi-Agent Routing

When subagents are introduced, each agent has its own `agent` UUID. Messages from different agents can be freely interleaved on the same SSE connection:

```json
{"type":"text","agent":"parent-uuid","final":false,"delta":"Let me search for that."}
{"type":"thinking","agent":"child-uuid","final":false,"delta":"I need to find the file..."}
{"type":"text","agent":"parent-uuid","final":false,"delta":" One moment."}
{"type":"text","agent":"child-uuid","final":false,"delta":"Found the file at src/main.py"}
```

The frontend uses `agent` to route each message to the correct agent's UI subtree, and `type` + `final` to track block boundaries within that subtree.

---

## Stream Lifecycle

A complete single-agent stream follows this sequence:

```
 1. meta_init                       — session metadata
 2. thinking                        — extended thinking (if enabled)
 3. text                            — assistant response text
    citation                        — citations (if any, after text final)
 4. server_tool_call                — server tool invocation (if any)
 5. server_tool_result              — server tool result (if any)
 6. text                            — continued response after server tool use
 7. tool_call                       — client tool invocation (if any)
 8. tool_result                     — backend tool result (per tool)
      tool_result_image             — image data (if multimodal result)
 9. [loop back to step 2 for next API turn]
10. meta_files                      — generated file metadata (if any)
11. meta_final                      — run summary
12. [DONE]                          — SSE stream termination signal
```

**With frontend tools**, the sequence diverges at step 7:

```
 7. tool_call                       — client tool invocation (backend + frontend)
 8. tool_result                     — backend tool results (executed server-side)
 9. awaiting_frontend_tools         — pause signal with pending frontend tools
    ... (frontend executes tools, POSTs results to /agent/tool_results) ...
10. tool_result                     — frontend tool results (streamed on resume)
11. [loop back to step 2 for next API turn]
```

The exact sequence depends on the model's output — tool calls, citations, and thinking blocks may or may not appear in any given run.

---

## Complete Type Reference

| `type` | Extra Fields | Emission | Source |
|--------|-------------|----------|--------|
| `meta_init` | — | Buffered | `agent.py` |
| `thinking` | — | Incremental | `json_formatter` |
| `text` | — | Incremental | `json_formatter` |
| `citation` | `citation_type`, `document_index`, `document_title`, `start_char_index`, `end_char_index`, `start_page_number`, `end_page_number`, `url`, `title` | Buffered | `json_formatter` |
| `tool_call` | `id`, `name` | Buffered | `json_formatter` |
| `server_tool_call` | `id`, `name` | Buffered | `json_formatter` |
| `tool_result` | `id`, `name` | Buffered | `agent.py` |
| `tool_result_image` | `id`, `name`, `src`, `media_type` | Buffered | `agent.py` |
| `server_tool_result` | `id`, `name` | Buffered | `json_formatter` |
| `awaiting_frontend_tools` | — | Buffered | `agent.py` |
| `meta_files` | — | Buffered | `agent.py` |
| `error` | — | Buffered | `json_formatter` |
| `meta_final` | — | Buffered | `agent.py` |

---

## Migration from Legacy XML Format

| Legacy XML Tag | New JSON `type` |
|---|---|
| `<meta_init data="...">` | `meta_init` |
| `<content-block-thinking>` | `thinking` |
| `<content-block-text>` | `text` |
| `<citations><citation ...>` | `citation` |
| `<content-block-tool_call ...>` | `tool_call` |
| `<content-block-server_tool_call ...>` | `server_tool_call` |
| `<content-block-tool_result ...>` | `tool_result` + `tool_result_image` |
| `<content-block-server_tool_result ...>` | `server_tool_result` |
| `<awaiting_frontend_tools data="...">` | `awaiting_frontend_tools` |
| `<content-block-meta_files>` | `meta_files` |
| `<content-block-error>` | `error` |
| `<meta_final data="...">` | `meta_final` |
