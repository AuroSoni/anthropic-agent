```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ AnthropicAgent.__init__(...)                                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ - db_backend: filesystem/sql (always present)                                │
│ - optional: tool_registry, compactor, memory_store, file_backend             │
│ - optional: server_tools (Anthropic-executed), beta_headers, api_kwargs      │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ agent_uuid given?│
              └──────────────────┘
                │            │
            yes │            │ no
                ▼            ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│ load_agent_config from DB│   │ new session state        │
│ - messages               │   │ - messages = []          │
│ - container_id           │   │ - container_id optional  │
│ - file_registry          │   │ - file_registry = {}     │
│ - token counters         │   │ - token counters = 0     │
└──────────────────────────┘   └──────────────────────────┘
                \            /
                 \          /
                  ▼        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ run(prompt, queue?, formatter?)                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Build user_message                                                           │
│ - prompt is str  -> {role:user, content:[{type:text, text:...}]}             │
│ - prompt is list -> {role:user, content:[...blocks...]}                      │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Append user_message                                                          │
│ - self.messages += [user_message]                                            │
│ - self.conversation_history += [user_message]                                │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ If queue: emit <meta_init ...> (format, user_query, message_history, model...)│
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │ memory_store configured?      │
        └───────────────────────────────┘
                 │               │
             yes │               │ no
                 ▼               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ memory_store.retrieve(...)                                                   │
│ - injects context into self.messages ONLY (not conversation_history)         │
└──────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Token baseline: _count_tokens_api(...)                                       │
│ (best-effort; updates _last_known_input_tokens)                              │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Loop: step = 1..max_steps                                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │ compactor configured?         │
        └───────────────────────────────┘
                 │               │
             yes │               │ no
                 ▼               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ _apply_compaction(step)                                                      │
│ (memory before/after hooks + logs)                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ _prepare_request_params()                                                    │
│ - model, max_tokens, messages                                                │
│ - system_prompt, thinking budget                                             │
│ - tools = client tool_schemas + server_tools                                 │
│ - betas, container_id, api_kwargs                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ anthropic_stream_with_backoff(...)                                           │
│ - retries transient API errors                                               │
│ - if queue: render_stream(stream, queue, formatter=xml/raw)                  │
│ - returns accumulated_message (BetaMessage)                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Append assistant message; update tokens + container_id                       │
│ - self.messages += [assistant_message]                                       │
│ - self.conversation_history += [assistant_message]                           │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ stop_reason ==   │
              │ "tool_use" ?     │
              └──────────────────┘
                │            │
            yes │            │ no
                ▼            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Local tool execution (client tools only)                                     │
│ - ToolRegistry.execute(...) -> append tool_result message -> continue loop   │
└──────────────────────────────────────────────────────────────────────────────┘
                │
                └─────────────────────────────── back to Loop ─────────────────┘

                              (stop_reason != "tool_use")
                                       │
                                       ▼
                         ┌───────────────────────────────┐
                         │ final_answer_check configured? │
                         └───────────────────────────────┘
                                  │               │
                              yes │               │ no
                                  ▼               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Validate assistant_message                                                   │
│ - if invalid: inject error as user message; continue loop                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌───────────────────────────────┐
        │ memory_store configured?      │
        └───────────────────────────────┘
                 │               │
             yes │               │ no
                 ▼               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ memory_store.update(messages, conversation_history, tools, model)            │
│ (logs memory_update)                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Build AgentResult                                                            │
│ - final_message, final_answer, conversation_history, usage, container_id     │
│ - agent_logs                                                                 │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ _finalize_file_processing(queue)                                             │
│ (discover file_ids; optional download+store; optional stream meta_files)     │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Save run data asynchronously (background tasks)                              │
│ - agent_config (resumable state)                                             │
│ - conversation_history entry (UI)                                            │
│ - agent_runs logs (_run_logs_buffer)                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ return AgentResult                                                           │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ If max_steps reached: _generate_final_summary()                              │
│ (tools disabled) -> then same tail: memory/update -> files -> persist        │
└──────────────────────────────────────────────────────────────────────────────┘

Notes:
- self.messages = live API context (may include memory injection + compaction)
- conversation_history = per-run full history (no memory injection; not compacted)
- server_tools are executed by Anthropic; client tools are executed via ToolRegistry
```