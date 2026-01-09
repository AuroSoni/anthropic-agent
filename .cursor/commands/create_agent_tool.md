## Creating Backend Tools for `anthropic_agent`

### Quick Start

```python
from anthropic_agent.tools import tool, ToolRegistry

@tool
def my_tool(param1: str, param2: int, optional_param: float = 0.0) -> str:
    """Brief description of what the tool does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        optional_param: Description with default value
    
    Returns:
        String result (tools must return str)
    """
    # Your implementation
    return f"Result: {param1}, {param2}, {optional_param}"
```

### Requirements

| Requirement | Details |
|------------|---------|
| **Type hints** | Required for ALL parameters |
| **Docstring** | Google-style with `Args:` section |
| **Return type** | Must return `str` or `ToolResult` |

### Supported Types

- Primitives: `int`, `float`, `str`, `bool`
- Collections: `list`, `dict`, `list[str]`, `dict[str, int]`
- Optional: `param: str | None = None`
- Literal enums: `Literal["option1", "option2"]`

### Registering Tools with the Agent

```python
from anthropic_agent.tools import ToolRegistry

# Create registry and register tools
registry = ToolRegistry()
registry.register_tools([my_tool, another_tool])

# Get schemas for the agent
schemas = registry.get_schemas()  # Anthropic format (default)

# Execute tools by name
result = registry.execute("my_tool", {"param1": "hello", "param2": 42})
```

### Common Patterns

**Enum choices in docstring:**
```python
@tool
def select_mode(mode: str) -> str:
    """Select operating mode.
    
    Args:
        mode: The mode to use (choices: ["fast", "slow", "auto"])
    """
```

**Complex types:**
```python
@tool  
def process_items(items: list[str], config: dict[str, int]) -> str:
    """Process a list of items with configuration.
    
    Args:
        items: List of item names to process
        config: Configuration mapping keys to values
    """
```

### Best Practices for Tool Output

#### 1. Keep Return Strings Concise
Tool output is injected into the LLM context window. Large outputs waste tokens and can degrade reasoning.

```python
# ❌ Bad: Returns entire file contents
@tool
def read_log(path: str) -> str:
    with open(path) as f:
        return f.read()  # Could be megabytes!

# ✅ Good: Returns summarized/truncated output
@tool
def read_log(path: str, max_lines: int = 100) -> str:
    with open(path) as f:
        lines = f.readlines()[-max_lines:]
    return f"<log lines='{len(lines)}' truncated='{'yes' if len(lines) == max_lines else 'no'}'>\n{''.join(lines)}</log>"
```

#### 2. Use Structured Formats (XML or JSON)
Structured output improves LLM parsing accuracy. **XML is preferred** for nested/hierarchical data.

```python
# ✅ XML format (preferred for complex data)
@tool
def search_files(query: str) -> str:
    results = perform_search(query)
    xml_parts = [f"<results query='{query}' count='{len(results)}'>"]
    for r in results[:10]:  # Limit results
        xml_parts.append(f"  <file path='{r.path}' score='{r.score}'>{r.snippet}</file>")
    xml_parts.append("</results>")
    return "\n".join(xml_parts)

# ✅ JSON format (good for flat data)
@tool
def get_config(key: str) -> str:
    import json
    config = {"key": key, "value": fetch_value(key), "source": "database"}
    return json.dumps(config, indent=2)
```

#### 3. Position Key Information Strategically
LLMs have **primacy bias** (beginning) and **recency bias** (end). Place critical info at these positions.

```python
# ✅ Good: Summary first, details last
@tool
def analyze_code(file_path: str) -> str:
    issues = run_analysis(file_path)
    return f"""<analysis status='{"error" if issues else "ok"}' file='{file_path}'>
<summary>Found {len(issues)} issues</summary>
<details>
{format_issues(issues)}
</details>
<recommendation>{"Fix critical issues before deployment" if issues else "Code looks good"}</recommendation>
</analysis>"""
```

#### 4. Additional Guidelines

| Guideline | Rationale |
|-----------|-----------|
| **Include metadata** | Add counts, status, timestamps to help LLM assess completeness |
| **Use consistent naming** | Same tag/key names across tools reduce cognitive load |
| **Indicate truncation** | Always tell the LLM if output was cut off |
| **Avoid raw error dumps** | Parse errors into actionable summaries |
| **Escape special chars** | Sanitize `<`, `>`, `&` in XML; escape quotes in JSON |

```python
# ✅ Error handling with structured output
@tool
def run_command(cmd: str) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return f"<result status='{'success' if result.returncode == 0 else 'error'}' code='{result.returncode}'>\n<stdout>{result.stdout[:2000]}</stdout>\n<stderr>{result.stderr[:500]}</stderr>\n</result>"
    except subprocess.TimeoutExpired:
        return "<result status='timeout'><error>Command exceeded 30s timeout</error></result>"
```

---

### Multimodal Tool Results (Images)

Tools can return images alongside text using the `ToolResult` wrapper class. Images are automatically:
- Encoded as base64 and sent to the Anthropic API
- Stored via the configured file backend (local or S3)
- Streamed to clients as references (URLs or paths)

#### Basic Image Return

```python
from anthropic_agent.tools import tool, ToolResult

@tool
def capture_screenshot(url: str) -> ToolResult:
    """Capture a screenshot of a webpage.
    
    Args:
        url: The URL to capture
        
    Returns:
        Screenshot image with description
    """
    screenshot_bytes = take_screenshot(url)  # Your implementation
    return ToolResult.with_image(
        text=f"Screenshot of {url}",
        image_data=screenshot_bytes,
        media_type="image/png",
    )
```

#### Multiple Images

```python
from anthropic_agent.tools import tool, ToolResult, ImageBlock

@tool
def compare_images(image1_path: str, image2_path: str) -> ToolResult:
    """Compare two images side by side.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        
    Returns:
        Both images with comparison notes
    """
    img1 = Path(image1_path).read_bytes()
    img2 = Path(image2_path).read_bytes()
    
    return ToolResult(content=[
        "Comparison of the two images:",
        ImageBlock(data=img1, media_type="image/png"),
        ImageBlock(data=img2, media_type="image/png"),
        "Note: Images are displayed in order provided.",
    ])
```

#### Text-Only with ToolResult

For consistency, you can use `ToolResult.text()` for text-only results:

```python
@tool
def simple_tool(param: str) -> ToolResult:
    """A simple tool that returns text.
    
    Args:
        param: Input parameter
    """
    return ToolResult.text(f"Processed: {param}")
```

#### Streaming Behavior

When images are returned, the streaming output format changes:

| Backend | Image `src` in Stream | Client Action |
|---------|----------------------|---------------|
| Local | `/agent/{uuid}/images/{id}` | Fetch via API |
| S3 | Direct presigned URL | Fetch from S3 |

```xml
<!-- Streamed output with image -->
<content-block-tool_result id="toolu_123" name="capture_screenshot">
  <text><![CDATA[Screenshot of https://example.com]]></text>
  <image src="/agent/abc-123/images/img_def456" media_type="image/png" />
</content-block-tool_result>
```

---

### Checklist Before Committing

- [ ] All parameters have type hints
- [ ] Docstring has description + `Args:` section with all params described
- [ ] Function returns `str` or `ToolResult`
- [ ] Output is concise (truncate/summarize large data)
- [ ] Output uses structured format (XML/JSON) for text
- [ ] Key info at beginning or end of output
- [ ] Truncation/limits clearly indicated in output
- [ ] Tool tested via `registry.execute()`