## Creating Frontend Tools for `anthropic_agent` (Browser-Executed)

Frontend tools are **schema-only** on the backend: the LLM can call them, but the backend **does not execute** them.
Instead, `AnthropicAgent` pauses, streams an `<awaiting_frontend_tools ...>` tag, and the browser executes the tool and POSTs results back to resume the run.

---

### Quick Start (End-to-End)

#### 1) Backend: declare a schema-only frontend tool

```python
from anthropic_agent.tools import tool

@tool(executor="frontend")
def user_confirm(message: str) -> str:
    """Ask the user for a yes/no confirmation in the browser.
    
    Args:
        message: The prompt to show the user.
    
    Returns:
        "yes" or "no"
    """
    pass  # Never executed server-side
```

Register it when constructing the agent:

```python
from anthropic_agent.core import AnthropicAgent

agent = AnthropicAgent(
    tools=[...],                  # backend tools (executed on server)
    frontend_tools=[user_confirm],# frontend tools (executed in browser)
    formatter="raw",              
)
```

#### 2) Frontend: implement handler + UI

```ts
// demos/vite_app/src/lib/frontend-tools.ts
export const frontendToolHandlers = {
  user_confirm: (tool, userResponse) => ({
    tool_use_id: tool.tool_use_id,
    content: userResponse,
  }),
};
```

The UI receives `pendingFrontendTools` and calls `POST /agent/tool_results` with:

```json
{
  "agent_uuid": "<uuid>",
  "tool_results": [
    { "tool_use_id": "<tool_use_id>", "content": "yes", "is_error": false }
  ]
}
```

---

### Core Contract (Backend ↔ Frontend)

#### What the backend sends to the browser

When the LLM calls a frontend tool, the agent emits a single tag containing a JSON array:

```xml
<awaiting_frontend_tools data="[{&quot;tool_use_id&quot;:&quot;...&quot;,&quot;name&quot;:&quot;user_confirm&quot;,&quot;input&quot;:{&quot;message&quot;:&quot;...&quot;}}]"></awaiting_frontend_tools>
```

Shape (as used by the Vite demo):

```ts
export interface PendingFrontendTool {
  tool_use_id: string;
  name: string;
  input: Record<string, unknown>;
}
```

#### What the frontend must send back to resume

POST body to `POST /agent/tool_results`:

- `agent_uuid`: the current session UUID (from `<meta_init>` or state metadata)
- `tool_results`: one entry per pending tool (the backend validates IDs match exactly)

Shape:

```ts
export interface FrontendToolResult {
  tool_use_id: string;
  content: string;
  is_error?: boolean;
}
```

Important: the backend requires that the set of `tool_use_id`s in `tool_results` equals the set emitted in `awaiting_frontend_tools`.

---

### Backend Guidelines (Declaring Frontend Tools)

#### 1) Always use `@tool(executor="frontend")`

- This repo distinguishes frontend tools by name using the schemas passed as `frontend_tools=...`.
- The function body is **never executed** server-side; treat it as a typed schema declaration.

#### 2) Inputs must be JSON-serializable and stable

The tool `input` is sent over SSE as JSON embedded in an XML attribute. Prefer:

- Primitives: `str`, `int`, `float`, `bool`
- Collections: `list[...]`, `dict[str, ...]`
- Optionals: `param: str | None = None`
- Simple `Literal[...]` enums

Avoid:

- Passing arbitrary objects/classes
- Large blobs (files, base64, huge arrays) — these bloat the context and SSE payload

#### 3) Keep the “return contract” simple

In this repo’s FastAPI + Vite demo, frontend tool results are returned as:

- `content: string`

If you need structured output, encode it as JSON **string** consistently:

```python
@tool(executor="frontend")
def pick_color() -> str:
    """Prompt user to pick a color.
    
    Returns:
        JSON string: {"hex": "#RRGGBB"}
    """
    pass
```

#### 4) Choose good tool names

- Use `snake_case`
- Keep names stable; renaming tools breaks old conversations and frontend handler routing
- Prefer a “verb + object” name: `user_confirm`, `pick_file`, `select_option`

#### 5) Document tool behavior precisely

The schema is generated from type hints + docstring. For predictable model usage:

- First line: what the tool is for and when to call it
- `Args:`: all parameters, with constraints and examples when helpful
- `Returns:`: exact expected values (and any JSON string schema)

#### 6) Understand how the agent pauses

Behavior in `anthropic_agent/core/agent.py`:

- Backend tools are executed first (tool results are already produced)
- If any frontend tools are present, the agent:
  - Stores backend results + pending frontend tool calls in persisted state
  - Streams `<awaiting_frontend_tools ...>`
  - Returns early with `stop_reason="awaiting_frontend_tools"`

That means frontend tools are ideal for:

- Asking user confirmation
- Collecting user input that requires UI
- Using browser-only capabilities (clipboard, geolocation, file picker)

---

### Frontend Guidelines (Integrating Frontend Tools)

#### 1) Parse `awaiting_frontend_tools` from the stream

In the Vite demo, parsing happens by searching parsed XML nodes for the `awaiting_frontend_tools` tag and `JSON.parse(...)` on its `data` attribute.

Best practices:

- Treat parsing failures as non-fatal; show an error card and allow retry
- Consider the agent can emit *multiple* frontend tool pauses in a single conversation (nested tool calls)

#### 2) Implement a tool handler registry

Centralize mapping from tool name → serializer for `FrontendToolResult`:

- Keep it in one file (the demo uses `src/lib/frontend-tools.ts`)
- Provide a safe fallback for unknown tools (set `is_error: true`)

Guideline:

- The handler should be a pure function (no side effects). Do UI side effects in components/hooks.

#### 3) Build UI components per tool “interaction type”

Common interaction types:

- **Confirm**: yes/no buttons (`user_confirm`)
- **Text input**: input + submit
- **Select**: dropdown list
- **File picker**: `<input type="file" />` and return metadata (or upload separately)

Best practices:

- Disable chat input while `status === "awaiting_tools"` to avoid state confusion
- Show which tool is awaiting and what the `input` was (sanitized)
- Prevent double-submit (button disable + local submitted state)

#### 4) Always submit results for all pending tool IDs

The backend validates that the returned `tool_use_id`s match the pending set exactly.

If you render multiple tool prompts at once, you must still send a result for each one.
If a tool is “skipped” in UI, return an explicit result such as:

- `content: "skipped"`
- `is_error: true` (if you want the model to treat it as a failure)

Pick one convention and keep it consistent.

#### 5) Resume via `/agent/tool_results` and keep streaming

After submitting results:

- Switch UI state back to streaming
- Start reading SSE from `/agent/tool_results`
- Append new nodes to the existing transcript (the demo merges old + new nodes)

---

### Error Handling & Safety

#### Backend

- If frontend tool results are missing/mismatched, the agent raises a `ValueError`.
- Always assume tool results are untrusted user input. Validate/sanitize if you later use them server-side.

#### Frontend

- Unknown tool name: return `is_error: true` with a descriptive message
- Network failure when posting results: keep the tool prompt visible and allow retry
- If the stream ends unexpectedly, show a “resume” action if you have `agent_uuid`

---

### Performance & Token Hygiene

- Keep tool inputs small; avoid embedding long UI text or large lists
- Keep tool results short; return only what the model needs
- If returning structured data, return JSON strings without indentation
- Prefer `formatter="xml"` for reliable parsing of `<awaiting_frontend_tools>`

---

### Checklist Before Using a New Frontend Tool

#### Backend

- [ ] Tool function is decorated with `@tool(executor="frontend")`
- [ ] All parameters have type hints; docstring includes `Args:` and clear `Returns:`
- [ ] Inputs are JSON-serializable and small
- [ ] Tool name is stable and matches the frontend registry key
- [ ] Tool is passed via `frontend_tools=[...]` when constructing `AnthropicAgent`

#### Frontend

- [ ] Parser recognizes `awaiting_frontend_tools` and sets `pendingFrontendTools`
- [ ] Handler exists in the registry for the tool name
- [ ] UI collects required input and prevents double-submit
- [ ] Submits `tool_results` for all pending `tool_use_id`s
- [ ] Handles unknown tools, network errors, and retry paths

---
