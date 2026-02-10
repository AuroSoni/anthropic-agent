# Current XML Stream Schema

This documents the XML schema currently used by the streaming pipeline in this repo (legacy `content-block-*` format).

Primary sources in code:
- `anthropic_agent/streaming/formatters.py`
- `anthropic_agent/core/agent.py`
- `demos/vite_app/src/lib/parsers/xml-parser.ts`
- `demos/vite_app/src/lib/parsers/xml-stream-parser.ts`

## Transport and encoding

- Data is streamed as SSE `data:` chunks.
- XML attribute values are HTML-escaped.
- Large payloads are usually wrapped in CDATA.
- `meta_init` is emitted at stream start (when queue streaming is used).
- `meta_final` is emitted only when `stream_meta_history_and_tool_results=true` and the run fully completes.

## Block types and attributes

### 1) `meta_init`

Tag:
```xml
<meta_init data="{...escaped json...}"></meta_init>
```

Attributes:
- `data` (required): escaped JSON string.

`data` JSON keys:
- `format` (required): `"xml"` or `"raw"`.
- `user_query` (required): prompt text or JSON-serialized prompt.
- `agent_uuid` (required)
- `model` (required)
- `message_history` (optional): included only when `stream_meta_history_and_tool_results=true`.

Example:
```xml
<meta_init data="{&quot;format&quot;:&quot;xml&quot;,&quot;user_query&quot;:&quot;Hello&quot;,&quot;agent_uuid&quot;:&quot;abc-123&quot;,&quot;model&quot;:&quot;claude-sonnet-4-5&quot;}"></meta_init>
```

---

### 2) `content-block-thinking`

Tag:
```xml
<content-block-thinking>...</content-block-thinking>
```

Attributes:
- None

Body:
- Plain streamed text (thinking content).

Example:
```xml
<content-block-thinking>I should check the file first.</content-block-thinking>
```

---

### 3) `content-block-text`

Tag:
```xml
<content-block-text>...</content-block-text>
```

Attributes:
- None

Body:
- Plain streamed assistant text.

Example:
```xml
<content-block-text>I found the issue and applied a fix.</content-block-text>
```

---

### 4) `content-block-tool_call`

Tag:
```xml
<content-block-tool_call id="..." name="..." arguments="{...escaped json...}"></content-block-tool_call>
```

Attributes:
- `id` (required): tool use id.
- `name` (required): client tool name.
- `arguments` (required): escaped JSON object string.

Body:
- Empty (arguments are in attributes).

Example:
```xml
<content-block-tool_call id="toolu_01" name="grep_search" arguments="{&quot;pattern&quot;:&quot;TODO&quot;,&quot;path&quot;:&quot;src/&quot;}"></content-block-tool_call>
```

---

### 5) `content-block-server_tool_call`

Tag:
```xml
<content-block-server_tool_call id="..." name="..." arguments="{...escaped json...}"></content-block-server_tool_call>
```

Attributes:
- `id` (required): server tool use id.
- `name` (required): server tool name.
- `arguments` (required): escaped JSON object string.

Body:
- Empty.

Example:
```xml
<content-block-server_tool_call id="srvtoolu_01" name="web_search" arguments="{&quot;query&quot;:&quot;latest ai news&quot;}"></content-block-server_tool_call>
```

---

### 6) `content-block-tool_result`

Tag:
```xml
<content-block-tool_result id="..." name="..."><![CDATA[...]]></content-block-tool_result>
```

Attributes:
- `id` (required): tool use id this result belongs to.
- `name` (required): tool name (or tool result label).

Body variants:
- CDATA text/json string.
- Multimodal variant with nested tags:
  - `<text><![CDATA[...]]></text>`
  - `<image src="..." media_type="..." />` (one or more)

Examples:
```xml
<content-block-tool_result id="toolu_01" name="grep_search"><![CDATA[Found 4 matches]]></content-block-tool_result>
```

```xml
<content-block-tool_result id="toolu_img_01" name="image_tool">
  <text><![CDATA[Generated image successfully]]></text>
  <image src="data:image/png;base64,AAA..." media_type="image/png" />
</content-block-tool_result>
```

Nested `image` attributes:
- `src` (required)
- `media_type` (required)

---

### 7) `content-block-server_tool_result`

Tag:
```xml
<content-block-server_tool_result id="..." name="..."><![CDATA[...]]></content-block-server_tool_result>
```

Attributes:
- `id` (required): server tool use id.
- `name` (required): result type/name (for example `web_search_tool_result`).

Body:
- CDATA payload (string or JSON-serialized content).

Example:
```xml
<content-block-server_tool_result id="srvtoolu_01" name="web_search_tool_result"><![CDATA[{"results":[{"title":"Example"}]}]]></content-block-server_tool_result>
```

---

### 8) `citations` and `citation`

Tags:
```xml
<citations> ... </citations>
<citation ...>...</citation>
```

Attributes:
- `citations`: none.
- `citation`: always has `type`; other attributes depend on citation type.

Supported citation attribute sets:
- `type="char_location"`:
  - `document_index`, `document_title`, `start_char_index`, `end_char_index`
- `type="page_location"`:
  - `document_index`, `document_title`, `start_page_number`, `end_page_number`
- `type="web_search_result_location"`:
  - `url`, `title`
- fallback:
  - `document_index` (optional)

Body:
- cited text (often in CDATA in formatter output).

Example:
```xml
<citations>
  <citation type="char_location" document_index="0" document_title="My Doc" start_char_index="10" end_char_index="35"><![CDATA[Quoted source text]]></citation>
</citations>
```

---

### 9) `awaiting_frontend_tools`

Tag:
```xml
<awaiting_frontend_tools data="{...escaped json array...}"></awaiting_frontend_tools>
```

Attributes:
- `data` (required): escaped JSON array of pending frontend tools.

Each tool object:
- `tool_use_id`
- `name`
- `input`

Example:
```xml
<awaiting_frontend_tools data="[{&quot;tool_use_id&quot;:&quot;toolu_01&quot;,&quot;name&quot;:&quot;user_confirm&quot;,&quot;input&quot;:{&quot;question&quot;:&quot;Continue?&quot;}}]"></awaiting_frontend_tools>
```

---

### 10) `content-block-meta_files`

Tag:
```xml
<content-block-meta_files><![CDATA[{...json...}]]></content-block-meta_files>
```

Attributes:
- None

Body:
- JSON object with key `files` (array of file metadata records).

Example:
```xml
<content-block-meta_files><![CDATA[{"files":[{"file_id":"file_01","filename":"report.pdf","storage_location":"https://..."}]}]]></content-block-meta_files>
```

---

### 11) `content-block-error`

Tag:
```xml
<content-block-error><![CDATA[{...error json...}]]></content-block-error>
```

Attributes:
- None

Body:
- CDATA containing serialized error payload.

Example:
```xml
<content-block-error><![CDATA[{"type":"api_error","message":"rate_limit"}]]></content-block-error>
```

---

### 12) `meta_final`

Tag:
```xml
<meta_final data="{...escaped json...}"></meta_final>
```

Attributes:
- `data` (required): escaped JSON summary object.

`data` JSON keys:
- `conversation_history`
- `stop_reason`
- `total_steps`
- `generated_files`
- `cost`
- `cumulative_usage`

Example:
```xml
<meta_final data="{&quot;stop_reason&quot;:&quot;end_turn&quot;,&quot;total_steps&quot;:3,&quot;generated_files&quot;:null,&quot;cost&quot;:null,&quot;cumulative_usage&quot;:{&quot;input_tokens&quot;:1000,&quot;output_tokens&quot;:300}}"></meta_final>
```

## Parser-accepted compatibility tags

The frontend parser also accepts these XML tags for compatibility:

- Dynamic server result tags ending in `_tool_result` (for example `content-block-web_search_tool_result` or `bash_code_execution_tool_result`).
- Embedded tags inside text blocks: `chart`, `table`.

These are parser-compatible, but canonical backend emission now uses the block types listed above.
