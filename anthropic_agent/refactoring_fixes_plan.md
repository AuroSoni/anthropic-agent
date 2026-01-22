# Fix Provider-Agnostic Design Flaws

## Overview

Replace `container_id` with a generic `provider_state: dict[str, Any] | None` pattern, and clean up other hardcoded provider-specific fields in the base agent.

---

## 1. Replace `container_id` with `provider_state`

### 1.1 Update `LLMResponse` dataclass ([base_agent.py](anthropic_agent/core/base_agent.py) line 132-154)

```python
@dataclass
class LLMResponse:
    raw: Any
    assistant_message: dict[str, Any]
    stop_reason: str
    model: str | None
    usage: dict[str, Any] | None = None
    provider_state: dict[str, Any] | None = None  # Replaces container_id
```

### 1.2 Update `AgentResult` dataclass ([types.py](anthropic_agent/core/types.py) line 69)

```python
provider_state: dict[str, Any] | None = None  # Replaces container_id
```

Update docstring to explain the pattern (Anthropic stores `{"container_id": "..."}`, OpenAI may store `{"thread_id": "..."}`).

### 1.3 Add `_provider_state` instance variable and hooks in `BaseAgent`

**In `__init__`** (around line 304):

```python
self._provider_state: dict[str, Any] = {}
```

**Add hook methods** (in the "Provider hooks" section around line 315):

```python
def _merge_provider_state(self, response: LLMResponse) -> None:
    """Merge provider-specific state from response into _provider_state.
    
    Called after each LLM response. Subclasses can override to customize
    how provider state is accumulated (e.g., only keep latest container_id).
    """
    if response.provider_state:
        self._provider_state.update(response.provider_state)

def _get_provider_state(self) -> dict[str, Any] | None:
    """Return provider state to include in AgentResult.
    
    Subclasses can override to filter or transform the state.
    Returns None if no provider state exists.
    """
    return self._provider_state if self._provider_state else None
```

**Usage pattern**: The base agent calls `_merge_provider_state(response)` after each LLM call, then uses `_get_provider_state()` when constructing `AgentResult`. Subclasses can override either hook to customize behavior.

### 1.4 Fix broken references in `_run_loop` and `_generate_final_summary`

Replace all occurrences of:

```python
if response.container_id:
    self.container_id = response.container_id
```

With:

```python
self._merge_provider_state(response)
```

Replace all `AgentResult` constructions:

```python
container_id=self.container_id,  # OLD
provider_state=self._get_provider_state(),  # NEW (uses hook)
```

**Locations**: Lines 614-615, 661, 723, 802-803, 831

### 1.5 Update Anthropic agent subclass

In `_stream_llm_response`, return:

```python
provider_state={"container_id": container_id} if container_id else None
```

---

## 2. Fix Hardcoded Cache Tokens in Persistence

### 2.1 Add hook for provider-specific conversation fields ([base_agent.py](anthropic_agent/core/base_agent.py) line 1236)

```python
def _get_provider_specific_conversation_fields(self, usage: dict[str, Any]) -> dict[str, Any]:
    """Return provider-specific fields to include in conversation persistence."""
    return {}
```

### 2.2 Update `_save_conversation_entry` (lines 1268-1269)

Remove hardcoded cache fields:

```python
# REMOVE these lines:
"cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
"cache_read_input_tokens": usage.get("cache_read_input_tokens"),
```

Replace with:

```python
**self._get_provider_specific_conversation_fields(usage),
```

### 2.3 Override in Anthropic agent

```python
def _get_provider_specific_conversation_fields(self, usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
    }
```

---

## 3. Clean Up Block Type Checking

### 3.1 Add class attribute for tool-related block types ([base_agent.py](anthropic_agent/core/base_agent.py))

```python
# In BaseAgent class definition (around line 172)
TOOL_BLOCK_TYPES: set[str] = {"tool_use", "tool_result"}
TEXT_BLOCK_TYPES: set[str] = {"text"}
```

### 3.2 Update `_extract_final_answer` (lines 1000-1015)

Replace hardcoded sets with class attributes:

```python
if block.get("type") in self.TOOL_BLOCK_TYPES:
    start_index = i + 1
...
if block.get("type") in self.TEXT_BLOCK_TYPES:
```

### 3.3 Override in provider subclasses

**Anthropic agent**:

```python
TOOL_BLOCK_TYPES = {"tool_use", "tool_result", "server_tool_use", "server_tool_result", 
                   "web_search_tool_use", "web_search_tool_result"}
```

**OpenAI agent**:

```python
TOOL_BLOCK_TYPES = {"tool_use", "tool_result", "function_call", "function_result"}
TEXT_BLOCK_TYPES = {"text", "output_text"}
```

---

## 4. Update State Persistence/Restoration

### 4.1 Save `_provider_state` in `_save_agent_config` (line 1191)

Add to config dict:

```python
"provider_state": self._provider_state,
```

### 4.2 Restore in `_restore_state_from_config` (line 1402)

```python
self._provider_state = config.get("provider_state", {})
```

---

## Files to Modify

| File | Changes |

|------|---------|

| `anthropic_agent/core/types.py` | Replace `container_id` → `provider_state` |

| `anthropic_agent/core/base_agent.py` | Add hooks, fix references, add class attributes |

| `anthropic_agent/core/agent.py` | Update LLMResponse construction, add hook overrides |

| `anthropic_agent/core/openai_agent.py` | Add hook overrides for block types |