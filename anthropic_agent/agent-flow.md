```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ AnthropicAgent.__init__(...)                                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ - config_adapter, conversation_adapter, run_adapter (storage adapters, default: memory) │
│ - optional: tool_registry (backend tools), frontend_tools (browser-executed)│
│ - optional: compactor, memory_store, file_backend                            │
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
│ - frontend tool relay:   │   │ - frontend tool relay:   │
│   - pending_frontend_tools│  │   - all fields = empty   │
│   - pending_backend_results│ │                          │
│   - awaiting_frontend_tools│ │                          │
│   - current_step         │   │                          │
│   - conversation_history │   │                          │
│     (only if awaiting)   │   │                          │
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
│ Initialize per-run state                                                     │
│ - _run_id, _run_start_time, _run_logs_buffer                                 │
│ - conversation_history = [] (fresh per run)                                  │
│ - agent_logs = []                                                            │
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
│ Token baseline: _estimate_tokens(...)                                        │
│ (heuristic estimation; updates _last_known_input_tokens)                     │
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
│ - tools = client tool_schemas + frontend_tool_schemas + server_tools         │
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
           yes │            │ no ──────────────────────────────────────────────┐
               ▼                                                               │
┌──────────────────────────────────────────────────────────────────────────────┐
│ Separate tool calls into backend_tools vs frontend_tools                     │
│ (based on tool name membership in frontend_tool_names set)                   │
└──────────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Execute ALL backend tools first                                              │
│ - ToolRegistry.execute(...) for each backend tool                            │
│ - Collect tool_results list                                                  │
│ - Stream each <content-block-tool_result> to queue                           │
└──────────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ frontend_tools   │
             │ present?         │
             └──────────────────┘
               │            │
           yes │            │ no
               ▼            │
┌──────────────────────────────────────────────────────────────────────────────┐
│ PAUSE FOR FRONTEND TOOLS                                                     │
│ - Store backend results in _pending_backend_results                          │
│ - Store frontend tool details in _pending_frontend_tools                     │
│ - Set _awaiting_frontend_tools = True, _current_step = step                  │
│ - Emit <awaiting_frontend_tools data="[...]"> to queue                       │
│ - _save_agent_config() (persist state for re-hydration)                      │
│ - Return AgentResult with stop_reason="awaiting_frontend_tools"              │
└──────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ *** AGENT INSTANCE MAY BE DESTROYED HERE ***                                 │
│ (stateless server - frontend executes tools in browser)                      │
└──────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                    FRONTEND TOOL CONTINUATION FLOW
═══════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────────┐
│ POST /agent/tool_results {agent_uuid, tool_results: [...]}                   │
└──────────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ AnthropicAgent.__init__(agent_uuid=..., ...)                                 │
│ - Re-hydrate agent from DB (messages, relay state, conversation_history)     │
│ - _awaiting_frontend_tools = True (loaded from DB)                           │
│ - _loaded_conversation_history restored for AgentResult                      │
└──────────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ continue_with_tool_results(frontend_results, queue?, formatter?)             │
└──────────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Validation                                                                   │
│ - Check _awaiting_frontend_tools == True                                     │
│ - Validate tool_use_ids match _pending_frontend_tools                        │
└──────────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Restore per-run state                                                        │
│ - conversation_history = _loaded_conversation_history.copy()                 │
│ - agent_logs, _run_logs_buffer, _run_id, _run_start_time                     │
└──────────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Combine results                                                              │
│ - all_results = _pending_backend_results + frontend_results                  │
│ - Stream frontend <content-block-tool_result> to queue                       │
│ - Append tool_result_message to messages + conversation_history              │
└──────────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Clear relay state                                                            │
│ - _pending_frontend_tools = []                                               │
│ - _pending_backend_results = []                                              │
│ - _awaiting_frontend_tools = False                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ _resume_run(queue, formatter) -> re-enters main loop at _current_step        │
└──────────────────────────────────────────────────────────────────────────────┘
                      │
                      └──────────────── continues normal loop flow ───────────┘


═══════════════════════════════════════════════════════════════════════════════
                         NORMAL COMPLETION FLOW
═══════════════════════════════════════════════════════════════════════════════

               (no frontend tools, OR all results combined)
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Append tool_result_message                                                   │
│ - self.messages += [tool_result_message]                                     │
│ - self.conversation_history += [tool_result_message]                         │
│ - Update token estimate                                                      │
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
│ - agent_logs, stop_reason ("end_turn", "max_tokens", etc.)                   │
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
│ - conversation_history TABLE entry (UI, cross-turn thread history)           │
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


═══════════════════════════════════════════════════════════════════════════════
                                  NOTES
═══════════════════════════════════════════════════════════════════════════════

State Distinction:
- self.messages = live API context (may include memory injection + compaction)
- conversation_history (per-run) = full history for AgentResult (no injection/compaction)
- conversation_history TABLE = permanent records of completed runs (cross-turn thread)

Tool Types:
- backend tools: executed via ToolRegistry on server
- frontend tools: schema-only on server; executed in browser; results via POST
- server tools: executed by Anthropic (code_execution, web_search, etc.)

Frontend Tool Relay State (persisted in agent_config):
- pending_frontend_tools: [{tool_use_id, name, input}, ...]
- pending_backend_results: tool results waiting to combine
- awaiting_frontend_tools: boolean flag
- current_step: step number to resume from
- conversation_history: per-run history for AgentResult (preserved during pause)

SSE Stream Tags:
- <meta_init>: format, user_query, agent_uuid, model
- <content-block-tool_result>: tool execution results
- <awaiting_frontend_tools>: signals pause for browser execution
- <content-block-meta_files>: file metadata
```
