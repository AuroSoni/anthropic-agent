## Schema Design: Three-Table Philosophy

---

## Table 1: agent_config (Session State for Resumption)

**Purpose**: Everything needed to resume an agent exactly where it left off

### SQL Schema (PostgreSQL)
```sql
CREATE TABLE agent_config (
    agent_uuid UUID PRIMARY KEY,
    
    -- Core configuration
    system_prompt TEXT NOT NULL,
    model VARCHAR(100) NOT NULL,
    max_steps INTEGER DEFAULT 50,
    thinking_tokens INTEGER DEFAULT 0,
    max_tokens INTEGER DEFAULT 2048,
    
    -- State for resumption
    container_id VARCHAR(255),  -- Anthropic container for code execution
    messages JSONB,  -- Current compacted messages state
    
    -- Tools configuration
    tool_schemas JSONB,  -- Client-side tool schemas [{name, description, input_schema}]
    tool_names TEXT[],  -- Quick reference list of tool names
    server_tools JSONB,  -- Anthropic server tools (code_execution, web_search, etc.)
    
    -- Beta features
    beta_headers TEXT[],  -- e.g., ["code-execution-2025-08-25"]
    
    -- API configuration
    api_kwargs JSONB,  -- Pass-through API params (temperature, top_p, stop_sequences, etc.)
    
    -- Component configuration
    formatter VARCHAR(50),  -- "xml", "json", etc.
    stream_meta_history_and_tool_results BOOLEAN DEFAULT FALSE,  -- Include metadata in stream
    compactor_type VARCHAR(50),  -- "tool_result_removal", "none" (type name only)
    memory_store_type VARCHAR(50),  -- "placeholder", "none" (type name only)
    
    -- File registry (consolidated across all runs)
    file_registry JSONB,  -- {file_id: {filename, storage_path, ...}}

    -- Custom extension point
    extras JSONB DEFAULT '{}'::jsonb,
    
    -- Retry configuration
    max_retries INTEGER DEFAULT 5,
    base_delay FLOAT DEFAULT 1.0,
    
    -- Token tracking for last run
    last_known_input_tokens INTEGER DEFAULT 0,
    last_known_output_tokens INTEGER DEFAULT 0,
    
    -- Frontend tool relay state (for pause/resume with browser-executed tools)
    pending_frontend_tools JSONB DEFAULT '[]',  -- [{tool_use_id, name, input}, ...]
    pending_backend_results JSONB DEFAULT '[]',  -- Backend tool results waiting to combine
    awaiting_frontend_tools BOOLEAN DEFAULT FALSE,  -- Is agent paused for frontend tools?
    current_step INTEGER DEFAULT 0,  -- Step to resume from
    -- NOTE: This conversation_history is the per-run history used to populate AgentResult.
    -- When frontend tools cause a pause, this preserves the current run's history so it
    -- can be returned in the AgentResult after continuation. This is distinct from the
    -- conversation_history TABLE (Table 2) which stores completed runs across user turns.
    conversation_history JSONB DEFAULT '[]',  -- Preserved for frontend tool resume only
    
    -- UI metadata
    title VARCHAR(255),  -- Auto-generated conversation title
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_run_at TIMESTAMP,
    total_runs INTEGER DEFAULT 0
);

-- Indexes
CREATE INDEX idx_agent_config_updated ON agent_config(updated_at DESC);
CREATE INDEX idx_agent_config_last_run ON agent_config(last_run_at DESC);
```

### Document Schema (Filesystem/JSON)
```python
# Stored as: data/agent_config/{agent_uuid}.json
{
  "agent_uuid": "abc-123-def-456",
  
  # Core configuration
  "system_prompt": "You are a helpful assistant...",
  "model": "claude-sonnet-4-5",
  "max_steps": 50,
  "thinking_tokens": 0,
  "max_tokens": 2048,
  
  # State for resumption
  "container_id": "container_xyz789",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    # Current compacted state
  ],
  
  # Tools configuration
  "tool_schemas": [
    {
      "name": "web_search",
      "description": "...",
      "input_schema": {...}
    }
  ],
  "tool_names": ["web_search", "calculator"],
  "server_tools": [
    {"type": "code_execution_20250825"}  # Anthropic server tools
  ],
  
  # Beta features
  "beta_headers": ["code-execution-2025-08-25"],
  
  # API configuration
  "api_kwargs": {
    "temperature": 0.7,
    "top_p": 0.9
  },
  
  # Component configuration
  "formatter": "xml",
  "stream_meta_history_and_tool_results": false,
  "compactor_type": "ToolResultRemovalCompactor",  # Type name only
  "memory_store_type": "PlaceholderMemoryStore",  # Type name only
  
  # File registry
  "file_registry": {
    "file_abc123": {
      "filename": "chart.png",
      "storage_path": "s3://bucket/agent-data/abc-123/file_abc123_chart.png",
      "size": 45231,
      "mime_type": "image/png",
      "created_in_run": "run-001",
      "last_updated_run": "run-003",
      "last_updated_at": "2025-11-25T14:30:00Z"
    }
  },
  
  # Retry configuration
  "max_retries": 5,
  "base_delay": 1.0,
  
  # Token tracking for last run
  "last_known_input_tokens": 12345,
  "last_known_output_tokens": 678,
  
  # Frontend tool relay state (for pause/resume with browser-executed tools)
  "pending_frontend_tools": [
    # Populated when agent pauses for frontend tools
    # {"tool_use_id": "toolu_123", "name": "user_confirm", "input": {"message": "..."}}
  ],
  "pending_backend_results": [
    # Backend tool results waiting to combine with frontend results
    # {"type": "tool_result", "tool_use_id": "toolu_456", "content": "..."}
  ],
  "awaiting_frontend_tools": false,  # True when agent is paused for frontend tools
  "current_step": 0,  # Step number to resume from
  # NOTE: This conversation_history is the per-run history used to populate AgentResult.
  # When frontend tools cause a pause, this preserves the current run's history so it
  # can be returned in the AgentResult after continuation. This is distinct from the
  # conversation_history TABLE (Table 2) which stores completed runs across user turns.
  "conversation_history": [],  # Preserved only when awaiting_frontend_tools=true
  
  # UI metadata
  "title": "Python Function for List Sorting",  # Auto-generated conversation title (nullable)

  # Custom extension point
  "extras": {
    "user_id": "user_123"
  },
  
  # Metadata
  "created_at": "2025-11-25T10:00:00Z",
  "updated_at": "2025-11-25T15:30:00Z",
  "last_run_at": "2025-11-25T15:30:00Z",
  "total_runs": 5
}
```

---

## Table 2: conversation_history (UI Display & Pagination)

**Purpose**: User-facing conversation records, paginated for UI

### SQL Schema (PostgreSQL)
```sql
CREATE TABLE conversation_history (
    conversation_id UUID PRIMARY KEY,
    agent_uuid UUID NOT NULL REFERENCES agent_config(agent_uuid) ON DELETE CASCADE,
    run_id UUID NOT NULL,  -- Links to agent_runs table
    
    -- Run timing
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    
    -- User interaction
    user_message TEXT NOT NULL,
    final_response TEXT,  -- Extracted from final assistant message
    
    -- Full conversation for this run
    messages JSONB NOT NULL,  -- Complete uncompacted conversation_history
    
    -- Run outcome
    stop_reason VARCHAR(50),  -- "end_turn", "max_steps", "tool_use", etc.
    total_steps INTEGER,
    
    -- Token usage (nested structure)
    usage JSONB,  -- {input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}
    
    -- Files generated in this run
    generated_files JSONB,  -- [{file_id, filename, storage_path, step, ...}]

    -- Cost breakdown for this run
    cost JSONB DEFAULT '{}'::jsonb,  -- {input_cost, output_cost, cache_write_cost, cache_read_cost, total_cost, total_input_tokens, total_output_tokens, total_cache_creation_tokens, total_cache_read_tokens, model_id, long_context_applied, currency}

    -- Custom extension point
    extras JSONB DEFAULT '{}'::jsonb,
    
    -- Sequence for pagination
    sequence_number INTEGER,  -- Auto-incrementing within agent_uuid
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX idx_conv_agent_uuid ON conversation_history(agent_uuid, sequence_number DESC);
CREATE INDEX idx_conv_started_at ON conversation_history(agent_uuid, started_at DESC);
CREATE INDEX idx_conv_run_id ON conversation_history(run_id);

-- Auto-increment sequence_number per agent
CREATE OR REPLACE FUNCTION set_conversation_sequence()
RETURNS TRIGGER AS $$
BEGIN
    NEW.sequence_number := COALESCE(
        (SELECT MAX(sequence_number) + 1 
         FROM conversation_history 
         WHERE agent_uuid = NEW.agent_uuid),
        1
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_conversation_sequence
BEFORE INSERT ON conversation_history
FOR EACH ROW
EXECUTE FUNCTION set_conversation_sequence();
```

### Document Schema (Filesystem/JSON)
```python
# Stored as: data/conversation_history/{agent_uuid}/{sequence_number}.json
# OR: data/conversation_history/{agent_uuid}/{run_id}.json

{
  "conversation_id": "conv-001",
  "agent_uuid": "abc-123-def-456",
  "run_id": "run-001",
  
  # Run timing
  "started_at": "2025-11-25T10:00:00Z",
  "completed_at": "2025-11-25T10:02:30Z",
  
  # User interaction
  "user_message": "Create a chart showing sales data",
  "final_response": "I've created a chart showing your sales data...",
  
  # Full conversation
  "messages": [
    {
      "role": "user",
      "content": "Create a chart showing sales data"
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "I'll create that chart for you..."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_123",
          "name": "bash_code_execution",
          "input": {...}
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_123",
          "content": [...]
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "I've created a chart..."}
      ]
    }
  ],
  
  # Run outcome
  "stop_reason": "end_turn",
  "total_steps": 3,
  
  # Token usage
  "usage": {
    "input_tokens": 1500,
    "output_tokens": 800,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  },
  
  # Files generated
  "generated_files": [
    {
      "file_id": "file_abc123",
      "filename": "sales_chart.png",
      "storage_path": "/data/files/abc-123/file_abc123_sales_chart.png",
      "size": 45231,
      "mime_type": "image/png",
      "step": 2,
      "timestamp": "2025-11-25T10:01:45Z"
    }
  ],

  # Cost breakdown for this run (from CostBreakdown)
  "cost": {
    "input_cost": 0.0057,
    "output_cost": 0.0195,
    "cache_write_cost": 0.00075,
    "cache_read_cost": 0.00012,
    "total_cost": 0.02607,
    "total_input_tokens": 2500,
    "total_output_tokens": 1300,
    "total_cache_creation_tokens": 200,
    "total_cache_read_tokens": 400,
    "model_id": "claude-sonnet-4-5",
    "long_context_applied": false,
    "currency": "USD"
  },

  # Custom extension point
  "extras": {
    "credits_used": 42
  },

  # Metadata
  "sequence_number": 1,
  "created_at": "2025-11-25T10:02:30Z"
}
```

**Filesystem structure**:
```
data/
  conversation_history/
    abc-123-def-456/
      001.json  # sequence_number padded
      002.json
      003.json
      index.json  # {last_sequence: 3, total_conversations: 3}
```

---

## Table 3: agent_runs (Detailed Execution Logs)

**Purpose**: Granular debugging and evaluation data

### SQL Schema (PostgreSQL)
```sql
CREATE TABLE agent_runs (
    log_id BIGSERIAL PRIMARY KEY,
    agent_uuid UUID NOT NULL REFERENCES agent_config(agent_uuid) ON DELETE CASCADE,
    run_id UUID NOT NULL,
    
    -- Chronological ordering
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    step_number INTEGER,  -- Which step in the run (0 = pre-loop, 1+ = loop iterations)
    
    -- Action type
    action_type VARCHAR(100) NOT NULL,
    -- Currently implemented: "run_started", "api_response_received", "tool_execution", 
    --                        "compaction", "run_completed", "final_summary_generation"
    
    -- Action details (flexible JSON)
    action_data JSONB NOT NULL,
    
    -- Context state tracking
    messages_snapshot JSONB,  -- Snapshot of messages at this point (optional, for key actions)
    messages_count INTEGER,  -- Quick reference
    estimated_tokens INTEGER,  -- Token count at this action
    
    -- Performance metrics
    duration_ms INTEGER  -- How long this action took (optional)
);

-- Indexes for efficient querying
CREATE INDEX idx_runs_agent_run ON agent_runs(agent_uuid, run_id, timestamp);
CREATE INDEX idx_runs_action_type ON agent_runs(action_type);
CREATE INDEX idx_runs_timestamp ON agent_runs(timestamp DESC);
```

### Document Schema (Filesystem/JSON)
```python
# Stored as: data/agent_runs/{agent_uuid}/{run_id}.jsonl
# JSONL format - one line per action (append-only)

# Line 1: Run started
{
  "log_id": 1,
  "agent_uuid": "abc-123-def-456",
  "run_id": "run-001",
  "timestamp": "2025-11-25T10:00:00.000Z",
  "step_number": 0,
  "action_type": "run_started",
  "action_data": {
    "user_message": "Create a chart showing sales data",
    "queue_present": true,
    "formatter": "xml"
  }
}

# Line 2: Memory retrieval
{
  "log_id": 2,
  "agent_uuid": "abc-123-def-456",
  "run_id": "run-001",
  "timestamp": "2025-11-25T10:00:00.125Z",
  "step_number": 0,
  "action_type": "memory_retrieval",
  "action_data": {
    "memories_found": 3,
    "memories_injected": 2,
    "memory_store_type": "placeholder"
  },
  "messages_count": 3,
  "estimated_tokens": 450,
  "duration_ms": 125
}

# Line 3: Compaction check
{
  "log_id": 3,
  "agent_uuid": "abc-123-def-456",
  "run_id": "run-001",
  "timestamp": "2025-11-25T10:00:00.250Z",
  "step_number": 1,
  "action_type": "compaction",
  "action_data": {
    "compaction_applied": false,
    "reason": "below_threshold",
    "current_tokens": 450,
    "threshold": 100000
  },
  "messages_count": 3,
  "estimated_tokens": 450,
  "duration_ms": 5
}

# Line 4: API call started
{
  "log_id": 4,
  "agent_uuid": "abc-123-def-456",
  "run_id": "run-001",
  "timestamp": "2025-11-25T10:00:00.300Z",
  "step_number": 1,
  "action_type": "api_call_started",
  "action_data": {
    "model": "claude-sonnet-4-5",
    "max_tokens": 2048,
    "thinking_tokens": 0,
    "tools_count": 5,
    "messages_sent": [
      {"role": "user", "content": "Create a chart showing sales data"},
      # ... full messages
    ]
  },
  "messages_snapshot": [...],  # Optional: full messages at this point
  "messages_count": 3,
  "estimated_tokens": 450
}

# Line 5: API response received
{
  "log_id": 5,
  "agent_uuid": "abc-123-def-456",
  "run_id": "run-001",
  "timestamp": "2025-11-25T10:00:02.145Z",
  "step_number": 1,
  "action_type": "api_response_received",
  "action_data": {
    "stop_reason": "tool_use",
    "usage": {
      "input_tokens": 450,
      "output_tokens": 120
    },
    "content": [
      {"type": "text", "text": "I'll create that chart..."},
      {"type": "tool_use", "id": "toolu_123", "name": "bash_code_execution", "input": {...}}
    ]
  },
  "duration_ms": 1845
}

# Line 6: Tool execution
{
  "log_id": 6,
  "agent_uuid": "abc-123-def-456",
  "run_id": "run-001",
  "timestamp": "2025-11-25T10:00:02.200Z",
  "step_number": 1,
  "action_type": "tool_execution",
  "action_data": {
    "tool_name": "bash_code_execution",
    "tool_use_id": "toolu_123",
    "input": {...},
    "result": {
      "stdout": "...",
      "stderr": "",
      "return_code": 0,
      "files": [{"file_id": "file_abc123", "filename": "chart.png"}]
    },
    "execution_time_ms": 850
  },
  "duration_ms": 850
}

# Line 7: Context transformation (messages updated)
{
  "log_id": 7,
  "agent_uuid": "abc-123-def-456",
  "run_id": "run-001",
  "timestamp": "2025-11-25T10:00:02.205Z",
  "step_number": 1,
  "action_type": "context_transformation",
  "action_data": {
    "transformation_type": "tool_result_added",
    "messages_before_count": 4,
    "messages_after_count": 5,
    "tokens_before": 570,
    "tokens_after": 620
  },
  "messages_count": 5,
  "estimated_tokens": 620
}

# ... more actions ...

# Final line: Run completed
{
  "log_id": 25,
  "agent_uuid": "abc-123-def-456",
  "run_id": "run-001",
  "timestamp": "2025-11-25T10:02:30.500Z",
  "step_number": 3,
  "action_type": "run_completed",
  "action_data": {
    "stop_reason": "end_turn",
    "total_steps": 3,
    "final_response": "I've created a chart showing your sales data...",
    "total_input_tokens": 1500,
    "total_output_tokens": 800,
    "files_generated": 1
  },
  "duration_ms": 150500  # Total run time
}
```

**Filesystem structure**:
```
data/
  agent_runs/
    abc-123-def-456/
      run-001.jsonl
      run-002.jsonl
      run-003.jsonl
```

---

## Action Types Taxonomy

For `agent_runs.action_type`, here's the list of actions. Actions marked with ✓ are currently implemented.

### Run Lifecycle
- ✓ `run_started` - Agent.run() called
- ✓ `run_completed` - Successful completion
- `run_failed` - Error occurred (not implemented)
- `max_steps_reached` - Hit max steps limit (not implemented)

### Memory Operations
- `memory_retrieval` - Retrieved context from memory store (not implemented)
- `memory_update` - Updated memory store post-run (not implemented)
- `memory_before_compact` - Memory hook before compaction (not implemented)
- `memory_after_compact` - Memory hook after compaction (not implemented)

### Compaction
- ✓ `compaction` - Compaction applied (logged when compaction_applied=true)
- `context_transformation` - Any message list modification (not implemented)

### API Communication
- `api_call_started` - Request sent to Anthropic (not implemented)
- ✓ `api_response_received` - Response received with stop_reason and token usage
- `api_retry` - Retry after error (not implemented)
- `api_error` - API call failed (not implemented)

### Tool Execution
- ✓ `tool_execution` - Tool called and executed (with tool_name, tool_use_id, success)
- `tool_error` - Tool execution failed (not implemented, errors logged within tool_execution)

### Validation
- `final_answer_validation` - Final answer check applied (not implemented)
- `final_answer_validation_failed` - Validation failed, retrying (not implemented)

### File Operations
- `file_generated` - File created by code execution (not implemented)
- `file_downloaded` - File downloaded from Anthropic (not implemented)
- `file_stored` - File stored in backend (not implemented)
- `file_error` - File operation failed (not implemented)

### Special Cases
- ✓ `final_summary_generation` - Max steps summary generated
- `streaming_chunk` - (optional) Stream chunk sent (not implemented)
- `error` - General error (not implemented)

---

## Query Patterns & Examples

### Pattern 1: Resume Agent (Load config)
```sql
SELECT * FROM agent_config WHERE agent_uuid = 'abc-123-def-456';
```
```python
# Filesystem
config = load_json(f"data/agent_config/{agent_uuid}.json")
```

### Pattern 2: Get Paginated Conversation History (UI)
```sql
SELECT conversation_id, user_message, final_response, started_at, 
       generated_files, total_steps
FROM conversation_history
WHERE agent_uuid = 'abc-123-def-456'
ORDER BY sequence_number DESC
LIMIT 20 OFFSET 0;
```
```python
# Filesystem
files = sorted(glob(f"data/conversation_history/{agent_uuid}/*.json"), reverse=True)
conversations = [load_json(f) for f in files[offset:offset+limit]]
```

### Pattern 3: Debug Specific Run
```sql
SELECT timestamp, step_number, action_type, action_data, duration_ms
FROM agent_runs
WHERE run_id = 'run-001'
ORDER BY timestamp ASC;
```
```python
# Filesystem
with open(f"data/agent_runs/{agent_uuid}/{run_id}.jsonl") as f:
    logs = [json.loads(line) for line in f]
```

### Pattern 4: Analytics - Average tokens per run
```sql
SELECT agent_uuid, AVG(input_tokens + output_tokens) as avg_tokens
FROM conversation_history
WHERE started_at > NOW() - INTERVAL '7 days'
GROUP BY agent_uuid;
```

### Pattern 5: Find errors
```sql
SELECT agent_uuid, run_id, timestamp, action_data
FROM agent_runs
WHERE action_type = 'error'
ORDER BY timestamp DESC
LIMIT 50;
```

---

## Storage Size Estimates

For a typical agent with 100 runs:

**agent_config**: ~10-50 KB (1 row)
**conversation_history**: ~5-20 MB (100 rows × 50-200 KB each)
**agent_runs**: ~50-200 MB (100 runs × 500-2000 KB each, depends on detail level)

**Total per agent**: ~55-220 MB for 100 runs

---

## Implementation Notes

### For SQL Backend:
- Use JSONB for flexibility
- Index on common query patterns
- Consider partitioning `agent_runs` by timestamp for large-scale
- Use transactions for atomic updates

### For Filesystem Backend:
- JSONL for agent_runs (append-only, efficient)
- JSON for config and conversation_history (structured)
- Directory structure mirrors UUID hierarchy
- Consider compression for old runs
