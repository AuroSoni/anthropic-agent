# Claude Cowork — Tools & Subagents Reference

---

## 1. Tools

### File & Code Operations

#### `Read`
Read a file from the local filesystem (text, images, PDFs, notebooks).
```
Parameters:
  file_path : string    (required) — Absolute path to the file
  offset    : number    (optional) — Line number to start reading from
  limit     : number    (optional) — Number of lines to read
  pages     : string    (optional) — Page range for PDFs, e.g. "1-5" (max 20 pages)

Returns: string — File contents in `cat -n` format (line-numbered), or rendered image/PDF content.
```

**How it works:**
Read is the primary file-inspection tool. It opens a file by absolute path and returns its
contents with line numbers prepended (in `cat -n` format: `  <line_number>\t<content>`).
It is **multimodal** — it handles text files, images (PNG, JPG, etc.), PDFs, and Jupyter
notebooks, adapting its output format to the file type.

Key behaviors:

- **Text files** — Returns line-numbered content. By default reads up to 2000 lines from
  the start of the file. Lines longer than 2000 characters are truncated.
- **Images** — The image is rendered visually in the agent's context (Claude is a
  multimodal LLM). No textual transcription is returned — the agent *sees* the image.
- **PDFs** — Rendered as images (page-by-page). For large PDFs (>10 pages), the `pages`
  parameter is **required** (e.g. `"1-5"`). Maximum 20 pages per call. Reading a large
  PDF without specifying pages will fail.
- **Jupyter notebooks (.ipynb)** — Returns all cells with their outputs, combining code,
  markdown, and visualization content.
- **Empty files** — If a file exists but is empty, a system warning is returned in place
  of content.
- **Non-existent files** — Returns an error. It is safe to attempt; no side effects occur.

Behavioral rules:

- Read only works on **files**, not directories. To list directory contents, use `ls` via
  the Bash tool instead.
- The `offset` and `limit` parameters enable windowed reading of very large files. For
  most files, omit them to read the entire file.
- Read must be called on a file **at least once** in the conversation before the Edit tool
  can be used on that same file. This prevents blind edits.
- Multiple Read calls can be issued **in parallel** in a single message for speculative
  exploration of several files at once.

Typical patterns:

- **Full read**: `Read(file_path="/path/to/file.py")` — read entire file.
- **Windowed read**: `Read(file_path="/path/to/big.log", offset=500, limit=100)` — read
  lines 500-599.
- **PDF page range**: `Read(file_path="/path/to/doc.pdf", pages="3-7")` — read pages 3
  through 7.

---

#### `Write`
Create or overwrite a file on the local filesystem.
```
Parameters:
  file_path : string    (required) — Absolute path to write to
  content   : string    (required) — Full file content

Returns: confirmation of write success or error.
```

**How it works:**
Write is a **full-replacement** file creation tool. It takes an absolute path and a
complete content string, then writes (or overwrites) the file at that location. There is
no append mode — the entire content parameter becomes the file's new content.

Key behaviors:

- **Overwrite semantics** — If the file already exists, its previous contents are
  completely replaced. There is no merge or diff — the content parameter is the sole
  source of truth for the resulting file.
- **Prerequisite for existing files** — If the file already exists, Read must have been
  called on it earlier in the conversation. Write will fail if you haven't read the file
  first. This prevents accidental data loss from blind overwrites.
- **New file creation** — If the file does not exist, Write creates it (along with any
  necessary parent directories implied by the path). No prior Read is needed for new files.
- **Preferred tool for new files** — Use Write for creating entirely new files. For
  modifying existing files, prefer Edit (targeted string replacement) over Write (full
  replacement), since Edit is less error-prone for partial changes.

Behavioral rules:

- Always prefer **Edit** over Write when making targeted changes to existing files. Write
  should only be used for existing files when the changes are so extensive that a full
  rewrite is cleaner.
- File paths must be **absolute**, not relative.
- Do not use Write to produce documentation files (*.md, README) proactively — only
  create them when explicitly requested by the user.

Typical patterns:

- **New file**: `Write(file_path="/sessions/.../output.py", content="def main():...")` —
  creates a new Python script.
- **Full rewrite**: Read the file first, then `Write(file_path="...", content="<entirely
  new content>")` — replaces everything.

---

#### `Edit`
Perform exact string replacement in an existing file.
```
Parameters:
  file_path   : string   (required) — Absolute path to the file
  old_string  : string   (required) — Exact text to find
  new_string  : string   (required) — Replacement text (must differ from old_string)
  replace_all : boolean  (optional, default: false) — Replace all occurrences

Returns: confirmation of edit success; error if old_string is not found or not unique.
```

**How it works:**
Edit performs **surgical, exact-match string replacement** within an existing file. It
finds the `old_string` in the file and replaces it with `new_string`. This is the
preferred tool for modifying existing files because it minimizes the blast radius of
changes — only the targeted text is affected.

Key behaviors:

- **Exact match** — The `old_string` must appear in the file **exactly** as specified,
  including whitespace, indentation, and line breaks. Copy the target text precisely from
  the Read output (excluding the line-number prefix).
- **Uniqueness requirement** — By default (`replace_all: false`), the `old_string` must
  be **unique** within the file. If it appears more than once, the edit fails with an
  error. To fix this, either provide a longer string with more surrounding context to make
  it unique, or set `replace_all: true` to replace every occurrence.
- **replace_all mode** — When `replace_all: true`, every occurrence of `old_string` in
  the file is replaced. This is useful for renaming variables, updating imports, or
  changing repeated patterns across a file.
- **Prerequisite** — Read must have been called on the file at least once earlier in the
  conversation. Edit will fail on files that haven't been read yet.
- **Indentation preservation** — When copying text from Read output, preserve the exact
  indentation as it appears *after* the line-number prefix. The line-number format is:
  `<spaces><line_number><tab><content>`. Everything after the tab is the actual file
  content to match. Never include any part of the line-number prefix in `old_string` or
  `new_string`.

Behavioral rules:

- Prefer Edit over Write for targeted changes — it's safer and more precise.
- If an edit fails due to non-uniqueness, expand the `old_string` to include more
  surrounding lines until it becomes unique.
- `old_string` and `new_string` must differ — no-op edits are rejected.

Typical patterns:

- **Single replacement**: `Edit(file_path="...", old_string="foo = 42",
  new_string="foo = 99")` — change one specific value.
- **Rename across file**: `Edit(file_path="...", old_string="oldName",
  new_string="newName", replace_all=true)` — rename a variable everywhere.
- **Multi-line replacement**: Both `old_string` and `new_string` can span multiple lines
  with embedded newlines.

---

#### `Glob`
Find files by glob pattern, sorted by modification time.
```
Parameters:
  pattern : string   (required) — Glob pattern, e.g. "**/*.ts", "src/**/*.jsx"
  path    : string   (optional) — Directory to search in (default: cwd)

Returns: string[] — List of matching file paths, sorted by modification time.
```

**How it works:**
Glob is a **fast file-discovery tool** that matches file paths against glob patterns. It
searches the filesystem starting from a given directory (or the current working directory)
and returns all matching file paths, sorted by modification time (most recently modified
first).

Key behaviors:

- **Pattern syntax** — Standard glob patterns are supported:
  - `*` matches any sequence of characters within a single path segment.
  - `**` matches zero or more path segments (recursive descent).
  - `?` matches a single character.
  - `{a,b}` matches either `a` or `b`.
  - Examples: `"**/*.py"` (all Python files recursively), `"src/**/*.{ts,tsx}"` (all
    TypeScript files under src), `"*.json"` (JSON files in the search root only).
- **Sort order** — Results are sorted by modification time, most recent first. This makes
  it easy to find recently changed files.
- **Scales to any codebase size** — Glob is optimized for large repositories and will not
  choke on huge directory trees.
- **File paths only** — Returns paths, not file contents. Use Read to inspect matched files.

Behavioral rules:

- Use Glob (not `find` or `ls` via Bash) for file-discovery tasks. It is the dedicated
  tool for this purpose.
- Multiple Glob calls can be issued in parallel in a single message for speculative
  searches (e.g. searching for both `"**/*.py"` and `"**/*.js"` simultaneously).
- When looking for a specific class or function definition, Glob on the filename pattern
  is often faster than a Grep content search.

Typical patterns:

- **Find all Python files**: `Glob(pattern="**/*.py")`
- **Find test files in a subtree**: `Glob(pattern="tests/**/*_test.py")`
- **Find config files**: `Glob(pattern="**/*.{json,yaml,yml,toml}")`
- **Scoped search**: `Glob(pattern="*.rs", path="/project/src")` — Rust files in a
  specific directory only.

---

#### `Grep`
Search file contents using ripgrep regex.
```
Parameters:
  pattern     : string   (required) — Regex pattern to search for
  path        : string   (optional) — File or directory to search (default: cwd)
  output_mode : enum     (optional) — "files_with_matches" (default) | "content" | "count"
  glob        : string   (optional) — File glob filter, e.g. "*.js"
  type        : string   (optional) — File type filter, e.g. "py", "ts", "rust"
  -A          : number   (optional) — Lines of context after match
  -B          : number   (optional) — Lines of context before match
  -C / context: number   (optional) — Lines of context before and after match
  -i          : boolean  (optional) — Case-insensitive search
  -n          : boolean  (optional, default: true) — Show line numbers
  head_limit  : number   (optional) — Limit output to first N entries
  offset      : number   (optional) — Skip first N entries before applying head_limit
  multiline   : boolean  (optional, default: false) — Enable multiline matching

Returns: Depending on output_mode:
  - files_with_matches → list of file paths
  - content → matching lines with context
  - count → match counts per file
```

**How it works:**
Grep is a **content-search tool** built on top of ripgrep (`rg`). It searches file
contents for a regex pattern and returns results in one of three output modes. It is the
primary tool for "find where X is used/defined/referenced" tasks.

Key behaviors:

- **Regex syntax** — Uses ripgrep's regex engine (Rust regex). Supports full regex
  including character classes, quantifiers, alternation, lookahead, etc. Literal braces
  need escaping (e.g. `interface\\{\\}` to find `interface{}` in Go code).
- **Output modes**:
  - `files_with_matches` (default) — Returns only the file paths that contain at least
    one match. Fastest mode for "which files contain X?"
  - `content` — Returns the matching lines themselves, with optional surrounding context
    lines (`-A`, `-B`, `-C`). Line numbers are shown by default (`-n: true`). Use this
    mode to see *what* matched and the code around it.
  - `count` — Returns the number of matches per file. Useful for gauging how heavily a
    pattern is used across the codebase.
- **File filtering** — Two mechanisms to narrow the search scope:
  - `glob` — Glob pattern on file paths (e.g. `"*.tsx"`, `"src/**/*.py"`).
  - `type` — Ripgrep's built-in file type groups (e.g. `"js"` covers `.js`, `.mjs`,
    `.cjs`; `"py"` covers `.py`, `.pyi`). More efficient than glob for standard types.
- **Pagination** — `head_limit` and `offset` let you paginate through large result sets.
  `head_limit` caps the output, `offset` skips the first N entries.
- **Multiline** — By default, patterns match within single lines. Set `multiline: true`
  to match patterns that span across line boundaries (e.g. `struct \\{[\\s\\S]*?field`).

Behavioral rules:

- Always use Grep (not `grep` or `rg` via Bash) for content searches. Grep has been
  optimized for correct permissions and access.
- For open-ended searches that may require multiple rounds of iteration, consider
  delegating to an Explore subagent via the Task tool instead.
- Multiple Grep calls can be issued in parallel when searching for different patterns.

Typical patterns:

- **Find files containing a function**: `Grep(pattern="def calculate_total",
  type="py")` → `files_with_matches`
- **See usage with context**: `Grep(pattern="calculateTotal", output_mode="content",
  -C=3, glob="*.ts")` — show 3 lines of context around each match in TypeScript files.
- **Count occurrences**: `Grep(pattern="TODO", output_mode="count")` — how many TODOs
  per file.
- **Case-insensitive search**: `Grep(pattern="error", -i=true, output_mode="content")`

---

#### `Bash`
Execute a shell command in the Linux VM.
```
Parameters:
  command                  : string   (required) — The bash command to run
  description              : string   (optional) — Human-readable description of the command
  timeout                  : number   (optional, max: 600000ms) — Timeout in milliseconds (default: 120000)
  dangerouslyDisableSandbox: boolean  (optional) — Override sandbox mode

Returns: string — stdout/stderr output of the command (truncated at 30000 chars).
```

**How it works:**
Bash executes arbitrary shell commands in a sandboxed Linux VM (Ubuntu 22). It is the
general-purpose "escape hatch" for anything that doesn't have a dedicated tool — running
builds, installing packages, executing scripts, managing git, running tests, starting
servers, and performing system operations.

Key behaviors:

- **Stateless shell, persistent directory** — Each Bash call runs in a fresh shell
  process (no environment variable or alias persistence between calls), but the
  **working directory** persists across calls. This means `cd` in one call affects the
  next, but `export VAR=value` does not carry over.
- **Output truncation** — stdout/stderr is capped at 30,000 characters. For commands
  that produce more output, pipe through `head`, `tail`, or redirect to a file.
- **Timeout** — Commands time out after 120 seconds by default. Use the `timeout`
  parameter (max: 600,000ms = 10 minutes) for long-running operations like builds or
  large installs.
- **Combined stdout/stderr** — Both streams are captured and returned together.
- **Shell initialization** — The shell is initialized from the user's profile (bash or
  zsh), so standard tools and PATH entries are available.
- **Package management**:
  - `npm` works normally (global packages install to a session-specific directory).
  - `pip` requires the `--break-system-packages` flag (e.g. `pip install pandas
    --break-system-packages`).

Behavioral rules:

- **Do not** use Bash for file reading (`cat`, `head`, `tail`), file searching (`find`,
  `grep`), or file writing (`echo >`, `sed`). Use the dedicated Read, Glob, Grep, Edit,
  and Write tools instead — they are optimized and permissions-aware.
- Use Bash for terminal operations: git, npm, docker, build tools, compilers, test
  runners, system utilities.
- Quote file paths containing spaces with double quotes.
- For independent commands, issue multiple Bash calls in parallel in a single message.
  For dependent commands, chain with `&&` in a single call.
- Prefer absolute paths over `cd` to maintain a stable working directory.
- The `description` parameter is for human readability — keep it concise for simple
  commands ("Install dependencies"), more detailed for complex piped commands.

Git-specific rules:

- Never force-push, hard-reset, or run destructive git commands without explicit user
  request.
- Always create new commits rather than amending (unless the user asks to amend).
- Stage specific files by name rather than `git add -A` or `git add .`.
- Pass commit messages via HEREDOC for clean formatting.

---

#### `NotebookEdit`
Replace, insert, or delete a cell in a Jupyter `.ipynb` notebook.
```
Parameters:
  notebook_path : string  (required) — Absolute path to the .ipynb file
  new_source    : string  (required) — New cell source content
  cell_id       : string  (optional) — ID of the cell to edit or insert after
  cell_type     : enum    (optional) — "code" | "markdown"
  edit_mode     : enum    (optional) — "replace" (default) | "insert" | "delete"

Returns: confirmation of notebook edit success or error.
```

**How it works:**
NotebookEdit is a **cell-level editor** for Jupyter `.ipynb` notebooks. Rather than
treating the notebook as a text file (which would require manipulating raw JSON), this
tool provides a structured interface for replacing, inserting, or deleting individual
cells by their ID.

Key behaviors:

- **Cell identification** — Cells are identified by `cell_id`, which is a unique
  identifier assigned to each cell in the notebook's JSON structure. Use the Read tool
  on the `.ipynb` file first to discover cell IDs and their contents.
- **Edit modes**:
  - `replace` (default) — Replaces the content of the cell identified by `cell_id` with
    `new_source`. The cell type remains the same unless `cell_type` is also specified.
  - `insert` — Inserts a **new cell** after the cell identified by `cell_id` (or at the
    beginning of the notebook if `cell_id` is omitted). Requires `cell_type` to be set.
  - `delete` — Removes the cell identified by `cell_id`. The `new_source` parameter is
    still required syntactically but is not used for the content.
- **Cell types** — `"code"` for executable code cells, `"markdown"` for documentation
  cells. When inserting, `cell_type` is required. When replacing, it defaults to the
  cell's existing type.
- **Zero-indexed** — The `cell_number` (if used instead of `cell_id`) is 0-indexed.

Behavioral rules:

- Always Read the notebook first to understand its structure and discover cell IDs.
- For substantial notebook modifications (restructuring, adding many cells), it may be
  more efficient to use Write to replace the entire notebook file, but NotebookEdit is
  safer for targeted changes since it preserves cell metadata and outputs.
- Notebooks are JSON under the hood — NotebookEdit handles the JSON manipulation so
  you don't have to.

Typical patterns:

- **Fix a code cell**: `NotebookEdit(notebook_path="...", cell_id="abc123",
  new_source="import pandas as pd\ndf = pd.read_csv('data.csv')")` — replace cell
  content.
- **Add a markdown header**: `NotebookEdit(notebook_path="...", cell_id="abc123",
  edit_mode="insert", cell_type="markdown", new_source="# Analysis Results")` — insert
  a new markdown cell after cell abc123.
- **Remove a cell**: `NotebookEdit(notebook_path="...", cell_id="abc123",
  edit_mode="delete", new_source="")` — delete the cell.

---

### Web & Search

#### `WebSearch`
Search the web and return result snippets with URLs.
```
Parameters:
  query           : string    (required) — Search query (min 2 chars)
  allowed_domains : string[]  (optional) — Only include results from these domains
  blocked_domains : string[]  (optional) — Exclude results from these domains

Returns: Search result blocks with titles, snippets, and markdown hyperlinks.
```

---

#### `WebFetch`
Fetch a URL, convert HTML to markdown, and analyze with a prompt.
```
Parameters:
  url    : string (URI)  (required) — Fully-formed URL to fetch
  prompt : string        (required) — What to extract/analyze from the page

Returns: string — AI-processed summary/extraction of the page content (15-min cache).
```

---

### Browser Automation (Claude in Chrome MCP)

#### `mcp__Claude_in_Chrome__read_page`
Get an accessibility tree representation of the current page.
```
Parameters:
  tabId     : number  (required) — Tab ID
  filter    : enum    (optional) — "all" (default) | "interactive"
  depth     : number  (optional, default: 15) — Max tree depth
  ref_id    : string  (optional) — Focus on a specific element by reference ID
  max_chars : number  (optional, default: 50000) — Max output characters

Returns: Accessibility tree with element reference IDs (ref_1, ref_2, etc.).
```

---

#### `mcp__Claude_in_Chrome__find`
Find page elements using natural language.
```
Parameters:
  query : string  (required) — Natural language description, e.g. "login button"
  tabId : number  (required) — Tab ID

Returns: Up to 20 matching elements with reference IDs. Notifies if >20 matches.
```

---

#### `mcp__Claude_in_Chrome__javascript_tool`
Execute JavaScript in the page context.
```
Parameters:
  action : string  (required) — Must be "javascript_exec"
  text   : string  (required) — JavaScript code to evaluate (last expression is returned)
  tabId  : number  (required) — Tab ID

Returns: Result of the last evaluated expression, or error details.
```

---

#### `mcp__Claude_in_Chrome__form_input`
Set a value on a form element.
```
Parameters:
  ref   : string  (required) — Element reference ID from read_page/find
  tabId : number  (required) — Tab ID
  value : any     (required) — Value to set (boolean for checkboxes, string/number otherwise)

Returns: confirmation of form input success or error.
```

---

#### `mcp__Claude_in_Chrome__computer`
Mouse, keyboard, and screenshot interactions.
```
Parameters:
  action           : enum    (required) — "left_click" | "right_click" | "double_click" |
                                           "triple_click" | "type" | "screenshot" | "wait" |
                                           "scroll" | "key" | "left_click_drag" | "zoom" |
                                           "scroll_to" | "hover"
  tabId            : number  (required) — Tab ID
  coordinate       : [x, y]  (required for click/scroll actions) — Pixel coordinates
  text             : string  (required for type/key) — Text to type or keys to press
  scroll_direction : enum    (required for scroll) — "up" | "down" | "left" | "right"
  scroll_amount    : number  (optional, 1-10, default: 3) — Scroll wheel ticks
  duration         : number  (required for wait, max: 30) — Seconds to wait
  modifiers        : string  (optional) — "ctrl", "shift", "alt", "cmd", combinable with "+"
  start_coordinate : [x, y]  (required for left_click_drag) — Drag start position
  region           : [x0, y0, x1, y1]  (required for zoom) — Rectangular region to capture
  ref              : string  (required for scroll_to, optional for clicks) — Element reference ID
  repeat           : number  (optional, 1-100, default: 1) — Repeat count for key action

Returns: Screenshot image (for screenshot/zoom), or confirmation of action.
```

---

#### `mcp__Claude_in_Chrome__navigate`
Navigate to a URL or go forward/back in history.
```
Parameters:
  url   : string  (required) — URL, "forward", or "back"
  tabId : number  (required) — Tab ID

Returns: confirmation of navigation.
```

---

#### `mcp__Claude_in_Chrome__resize_window`
Resize the browser window.
```
Parameters:
  width  : number  (required) — Target width in pixels
  height : number  (required) — Target height in pixels
  tabId  : number  (required) — Tab ID

Returns: confirmation of resize.
```

---

#### `mcp__Claude_in_Chrome__gif_creator`
Record browser interactions and export as animated GIF.
```
Parameters:
  action   : enum    (required) — "start_recording" | "stop_recording" | "export" | "clear"
  tabId    : number  (required) — Tab ID
  download : boolean (optional) — Set true for export to download the GIF
  filename : string  (optional) — Custom filename for exported GIF
  options  : object  (optional) — {
      showClickIndicators : boolean (default: true),
      showDragPaths       : boolean (default: true),
      showActionLabels    : boolean (default: true),
      showProgressBar     : boolean (default: true),
      showWatermark       : boolean (default: true),
      quality             : number  (1-30, default: 10)
    }

Returns: confirmation of recording action or exported GIF.
```

---

#### `mcp__Claude_in_Chrome__upload_image`
Upload a screenshot or image to a file input or drag-and-drop target.
```
Parameters:
  imageId    : string  (required) — ID of a previously captured screenshot or uploaded image
  tabId      : number  (required) — Tab ID
  ref        : string  (optional) — Element reference ID (for file inputs)
  coordinate : [x, y]  (optional) — Viewport coordinates (for drag-and-drop targets)
  filename   : string  (optional, default: "image.png") — Filename for the upload

Returns: confirmation of upload success or error.
```

---

#### `mcp__Claude_in_Chrome__get_page_text`
Extract raw text content from the page (article-first).
```
Parameters:
  tabId : number  (required) — Tab ID

Returns: string — Plain text content of the page.
```

---

#### `mcp__Claude_in_Chrome__tabs_context_mcp`
Get info about the current MCP tab group.
```
Parameters:
  createIfEmpty : boolean  (optional) — Create a new tab group if none exists

Returns: List of tab IDs in the current MCP group.
```

---

#### `mcp__Claude_in_Chrome__tabs_create_mcp`
Create a new empty tab in the MCP tab group.
```
Parameters: (none)

Returns: New tab ID.
```

---

#### `mcp__Claude_in_Chrome__read_console_messages`
Read browser console messages from a tab.
```
Parameters:
  tabId      : number   (required) — Tab ID
  pattern    : string   (optional) — Regex filter for messages
  limit      : number   (optional, default: 100) — Max messages to return
  onlyErrors : boolean  (optional, default: false) — Only return errors/exceptions
  clear      : boolean  (optional, default: false) — Clear messages after reading

Returns: Array of console messages (log, warn, error, etc.) from the current domain.
```

---

#### `mcp__Claude_in_Chrome__read_network_requests`
Read HTTP network requests from a tab.
```
Parameters:
  tabId      : number  (required) — Tab ID
  urlPattern : string  (optional) — URL substring filter, e.g. "/api/"
  limit      : number  (optional, default: 100) — Max requests to return
  clear      : boolean (optional, default: false) — Clear requests after reading

Returns: Array of network request records (URL, method, status, type, etc.).
```

---

#### `mcp__Claude_in_Chrome__shortcuts_list`
List all available shortcuts and workflows.
```
Parameters:
  tabId : number  (required) — Tab ID

Returns: Array of shortcuts with command names, descriptions, and workflow flag.
```

---

#### `mcp__Claude_in_Chrome__shortcuts_execute`
Execute a shortcut or workflow in a new sidepanel.
```
Parameters:
  tabId      : number  (required) — Tab ID
  command    : string  (optional) — Command name, e.g. "debug"
  shortcutId : string  (optional) — Shortcut ID

Returns: confirmation that execution has started (non-blocking).
```

---

#### `mcp__Claude_in_Chrome__switch_browser`
Switch to a different Chrome browser instance.
```
Parameters: (none)

Returns: Broadcasts connection request; user clicks "Connect" in desired browser.
```

---

#### `mcp__Claude_in_Chrome__update_plan`
Present a plan to the user for domain approval.
```
Parameters:
  domains  : string[]  (required) — Domains to be visited, e.g. ["github.com"]
  approach : string[]  (required) — High-level steps (3-7 items)

Returns: User approval or rejection of the plan.
```

---

### Connectors & Filesystem

#### `mcp__mcp-registry__search_mcp_registry`
Search for available app connectors.
```
Parameters:
  keywords : string[]  (required) — Search terms, e.g. ["asana", "tasks"]

Returns: Array of connector results with names, descriptions, and connected status.
```

---

#### `mcp__mcp-registry__suggest_connectors`
Display connector suggestions with Connect buttons.
```
Parameters:
  directoryUuids : string[]  (required) — UUIDs from search results

Returns: UI with Connect buttons for the user.
```

---

#### `mcp__cowork__request_cowork_directory`
Request access to a directory on the user's computer.
```
Parameters: (none)

Returns: Directory picker dialog; mounts selected directory on approval.
```

---

#### `mcp__cowork__allow_cowork_file_delete`
Request permission to delete a file.
```
Parameters:
  file_path : string  (required) — VM path of the file to delete

Returns: Approval or denial of delete permission.
```

---

### Workflow & Planning

#### `TodoWrite`
Create and manage a structured task list.
```
Parameters:
  todos : array  (required) — Array of:
    {
      content    : string  (required) — Task description (imperative form)
      status     : enum    (required) — "pending" | "in_progress" | "completed"
      activeForm : string  (required) — Present-continuous form, e.g. "Running tests"
    }

Returns: confirmation of todo list update.
```

**How it works:**
TodoWrite renders a visible progress widget in the Cowork UI, giving the user real-time
visibility into what the agent is doing. It is a *stateless replacement* tool — every call
sends the complete, updated list and overwrites the previous one. There is no append or
patch operation; the caller must always include the full array of todos in the desired state.

Behavioral rules:

- Exactly **one** todo should be `in_progress` at any time. No more, no fewer (while work
  is active).
- Mark a todo `completed` **immediately** after finishing it — do not batch completions.
- A todo should only be marked `completed` when the work genuinely succeeded. If tests
  fail, the build breaks, or the task is only partially done, leave it `in_progress` and
  create a new todo describing the blocker.
- Todos that become irrelevant should be removed from the array entirely rather than left
  in a stale state.
- Each todo requires **two description forms**: `content` (imperative, e.g. "Run tests")
  and `activeForm` (present-continuous, e.g. "Running tests"). The UI switches between
  them depending on the task's status.

Typical invocation cadence in a session:

1. **Initial creation** — Build the full list with the first item `in_progress` and the
   rest `pending`.
2. **Progress updates** — After completing a step, mark it `completed`, flip the next to
   `in_progress`, and send the whole array.
3. **Final update** — All items `completed`.

A final "verification" step (e.g. reviewing output, running tests, fact-checking) should
almost always be included as the last todo for non-trivial tasks.

---

#### `AskUserQuestion`
Ask the user multiple-choice questions via a structured UI widget.
```
Parameters:
  questions : array (1-4 items, required) — Array of:
    {
      question    : string   (required) — The question text
      header      : string   (required) — Short label (max 12 chars)
      options     : array    (required, 2-4 items) — [{label: string, description: string}]
      multiSelect : boolean  (required) — Allow multiple selections
    }
  metadata : object (optional) — {source: string} for analytics tracking
  answers  : object (optional) — Pre-collected answers (used internally by permission component)

Returns: object — User's selected answers. An implicit "Other" free-text option is always
         appended to every question automatically — you never add it yourself.
```

**How it works:**
AskUserQuestion renders a rich multiple-choice card in the Cowork chat UI. The user sees
each question as a set of clickable chips or radio/checkbox options. This is **not** the
same as simply typing a question in prose — the structured widget is easier to answer and
parses unambiguously.

Behavioral rules:

- Use this tool **before starting real work** whenever requirements are underspecified.
  Even seemingly simple requests ("make a presentation") have implicit choices (audience,
  tone, length, format) that are worth clarifying upfront.
- You may ask **1 to 4 questions** in a single call. Each question offers **2 to 4
  explicit options** plus the automatic "Other" free-text fallback.
- If you have a recommended option, place it **first** in the options list and append
  `"(Recommended)"` to its label.
- `multiSelect: true` allows the user to pick more than one option — use it when the
  choices are not mutually exclusive (e.g. "Which features do you want?").
- **Do not** use AskUserQuestion to ask "Is my plan okay?" or "Should I proceed?" — those
  are the jobs of `ExitPlanMode`. AskUserQuestion is for gathering *requirements and
  preferences*, not for approval gates.
- The `header` field is a very short tag (≤12 chars) shown on the chip, e.g. "Format",
  "Audience", "Depth".

Typical usage pattern:

1. User sends a request.
2. Agent identifies ambiguities or design choices.
3. Agent calls AskUserQuestion with 1-4 clarifying questions.
4. User responds via the widget.
5. Agent proceeds with the work, using the answers to guide decisions.

---

#### `EnterPlanMode`
Transition into plan mode for implementation design.
```
Parameters: (none)

Returns: Enters plan mode (requires user consent). The agent's tool access is restricted
         to read-only exploration tools (Glob, Grep, Read, WebFetch, WebSearch) plus
         AskUserQuestion. Write-side tools (Edit, Write, Bash, NotebookEdit) are
         unavailable until the plan is approved and plan mode is exited.
```

**How it works:**
EnterPlanMode is a **state transition** tool. Calling it switches the agent from normal
execution mode into a restricted "planning phase." The purpose is to force thorough
exploration and design *before* any code is written, preventing wasted effort on the wrong
approach.

Once in plan mode the agent should:

1. **Explore the codebase** — Read files, search for patterns, understand the existing
   architecture using read-only tools.
2. **Identify trade-offs** — Consider multiple approaches and their implications.
3. **Write a plan** — Produce a step-by-step implementation plan in a designated plan file.
4. **Clarify if needed** — Use `AskUserQuestion` to resolve remaining ambiguities.
5. **Exit** — Call `ExitPlanMode` to present the plan for user approval.

When to use:

- New feature implementation with architectural choices.
- Multi-file refactors or changes that affect existing behavior.
- Tasks with multiple valid approaches (caching strategies, auth methods, etc.).
- Unclear or underspecified requirements that need investigation first.

When **not** to use:

- Single-line fixes, typos, or trivial edits.
- Tasks where the user gave very specific, detailed instructions.
- Pure research or information-gathering (use the Explore subagent instead).

Requires user consent — the user must agree to enter plan mode before the transition
takes effect.

---

#### `ExitPlanMode`
Signal that the plan is ready for user approval and leave plan mode.
```
Parameters:
  allowedPrompts     : array   (optional) — [{tool: "Bash", prompt: string}]
                       Prompt-based permissions needed to implement the plan.
                       These are semantic descriptions of actions rather than
                       specific commands (e.g. "run tests", "install dependencies").
  pushToRemote       : boolean (optional) — Push plan to a remote Claude.ai session
  remoteSessionId    : string  (optional) — Remote session ID if pushed
  remoteSessionTitle : string  (optional) — Remote session title if pushed
  remoteSessionUrl   : string  (optional) — Remote session URL if pushed

Returns: The plan file contents are presented to the user. The user can approve,
         reject, or request modifications. On approval, the agent exits plan mode
         and regains access to all tools (Edit, Write, Bash, etc.) for implementation.
```

**How it works:**
ExitPlanMode is the counterpart to `EnterPlanMode`. It does **not** take the plan as a
parameter — instead, it reads the plan from the plan file that the agent wrote during
plan mode. The tool simply signals "I'm done planning, please review."

Behavioral rules:

- Only call this when the plan is **complete and unambiguous**. If you still have open
  questions, use `AskUserQuestion` first to resolve them, *then* call ExitPlanMode.
- **Do not** use AskUserQuestion to ask "Is this plan okay?" or "Should I proceed?" —
  that is exactly what ExitPlanMode does. It inherently requests user approval.
- Only use this tool for tasks that involve **writing code or making changes**. For
  pure research tasks (searching files, reading code, gathering information), do not
  use this tool — just return the findings directly.
- The `allowedPrompts` parameter lets you pre-declare what Bash actions the
  implementation will need (e.g. "run tests", "install dependencies"), so the user
  can approve them upfront.

Lifecycle:

```
  User request
       │
       ▼
  EnterPlanMode ──► Explore codebase (read-only)
       │                    │
       │              AskUserQuestion (if needed)
       │                    │
       │              Write plan file
       │                    │
       ▼                    ▼
  ExitPlanMode ──► User reviews plan
       │                    │
       │         ┌──────────┴──────────┐
       │         ▼                     ▼
       │     Approved              Rejected
       │         │                     │
       ▼         ▼                     ▼
  Implementation          Revise plan / re-enter plan mode
```

---

#### `Skill`
Invoke a specialized skill to load domain-specific best practices.
```
Parameters:
  skill : string  (required) — Skill name, e.g. "pdf", "docx", "pptx",
                                or fully qualified "ms-office-suite:pdf"
  args  : string  (optional) — Arguments for the skill

Returns: The skill's instruction set is expanded inline into the conversation context.
         The agent then follows these instructions for the remainder of the task.
```

**How it works:**
Skills are pre-authored instruction bundles (stored as SKILL.md files) that encode
battle-tested best practices for specific output types. When invoked, the skill's
full prompt is injected into the conversation, giving the agent detailed guidance on
libraries, patterns, formatting conventions, and common pitfalls.

Skills are **not code libraries** — they are *instructional documents* that teach the
agent how to produce high-quality outputs. Think of them as expert playbooks.

Behavioral rules:

- Skills should be loaded **before starting any work**. Reading the SKILL.md file first
  is critical — the patterns it contains dramatically improve output quality.
- Multiple skills can be loaded in a single task if the work spans domains (e.g. reading
  a PDF and writing a DOCX requires both the `pdf` and `docx` skills).
- Skills are matched to tasks by trigger keywords. For example, any mention of "Word",
  ".docx", "report", or "memo" should trigger the `docx` skill.
- User-uploaded skills (in the `.skills/skills/` directory) take high priority when
  relevant, since the user added them for a reason.

Invocation pattern:

1. User requests an output (e.g. "Create a presentation about Q4 results").
2. Agent identifies the relevant skill(s) (`pptx` in this case).
3. Agent calls `Skill` with `skill: "pptx"`.
4. The skill prompt loads, providing detailed instructions.
5. Agent follows those instructions to produce the output.

---

#### `Task`
Launch an autonomous subagent (subprocess) to handle a complex subtask.
```
Parameters:
  description   : string  (required) — Short summary (3-5 words), e.g. "Explore auth system"
  prompt        : string  (required) — Detailed task description with all necessary context
  subagent_type : string  (required) — Agent type: "Bash" | "general-purpose" | "Explore" |
                                       "Plan" | "claude-code-guide" | "statusline-setup"
  model         : enum    (optional) — "sonnet" | "opus" | "haiku" (inherits from parent if omitted)
  max_turns     : integer (optional) — Max agentic turns (API round-trips) before stopping
  resume        : string  (optional) — Agent ID from a previous invocation to resume with
                                       full prior context preserved

Returns: string — The subagent's final response message + an agent ID that can be used
         to resume the agent later. The subagent's output is returned to the parent
         agent only — it is NOT directly visible to the user. The parent must relay
         relevant information back to the user.
```

**How it works:**
Task spawns an independent subprocess (subagent) that runs autonomously with its own
tool access and context window. The parent agent delegates a piece of work, waits for the
result, and then integrates it into the main conversation.

Each subagent type has a different tool profile (see §2 below) and is suited for different
kinds of work. The subagent starts with a fresh context unless `resume` is provided, in
which case it continues with the full transcript from the prior run.

Key behaviors:

- **Parallelization** — Multiple Task calls can be made in a *single message* to run
  subagents concurrently. This is the primary mechanism for parallel work. Use it whenever
  you have 2+ independent work items that each require multiple steps.
- **Context isolation** — Subagents have their own context window, separate from the
  parent. This is useful for offloading high-token-cost subtasks (large file reads,
  exhaustive searches) without polluting the main conversation's context.
- **Invisible to user** — The subagent's output goes to the parent agent, not the user.
  The parent must summarize or relay findings to the user explicitly.
- **Resumable** — Every completed subagent returns an agent ID. Passing that ID to
  `resume` in a later Task call continues from where it left off with full context.

When to use Task:

- **Parallelizable work**: Researching 3 competitors simultaneously, analyzing 5 files
  in parallel, making independent design variants.
- **Context-hiding**: Having a subagent explore a large codebase, parse lengthy emails,
  analyze big document sets, or verify earlier work — without consuming the parent's
  context window.
- **Verification**: Spawning a subagent to independently verify the parent's work
  (fact-check, run tests, review diffs).

When **not** to use Task:

- Reading a single specific file (use Read directly).
- Searching for a known class/function (use Glob or Grep directly).
- Simple tasks that don't benefit from isolation or parallelism.

Prompt guidance:

- Non-resumed subagents start with zero context — provide **all** necessary information
  in the `prompt` parameter (file paths, requirements, constraints).
- Resumed subagents retain their full prior transcript — you can use short, contextual
  prompts like "now also check the error handling."
- Be explicit about whether the subagent should **write code** or just **do research** —
  it cannot infer the user's intent since it has no visibility into the parent conversation.

---

#### `TaskOutput`
Retrieve output from a running or completed background task.
```
Parameters:
  task_id : string   (required) — The task ID (from a previous Task invocation or /tasks)
  block   : boolean  (optional, default: true) — Whether to wait for the task to complete
  timeout : number   (optional, default: 30000, max: 600000) — Max wait time in milliseconds

Returns: The task's output text along with its current status:
  - "running"   — Task is still in progress (only returned when block=false)
  - "completed" — Task finished; output contains the final result
  - "failed"    — Task encountered an error; output contains error details
```

**How it works:**
TaskOutput is a polling/waiting tool for monitoring tasks launched via the `Task` tool or
background shell processes. By default it **blocks** — the call waits until the task
finishes (up to `timeout` ms) and then returns the full output.

Set `block: false` for a **non-blocking** check: the call returns immediately with whatever
output is available so far, plus the current status. This is useful for monitoring
long-running tasks without stalling the conversation.

Typical patterns:

- **Fire and wait** (default): Launch a Task, then call TaskOutput with `block: true` to
  get the result when done. This is the most common pattern.
- **Poll periodically**: Launch a Task, then call TaskOutput with `block: false` and a
  short timeout to check progress. Repeat until status is "completed" or "failed".
- **Parallel harvest**: Launch multiple Tasks concurrently, then call TaskOutput on each
  to collect results.

Note: Task IDs for all active/recent tasks can be discovered via the `/tasks` command.

---

#### `TaskStop`
Terminate a running background task.
```
Parameters:
  task_id  : string  (optional) — The ID of the background task to stop
  shell_id : string  (deprecated) — Legacy parameter; use task_id instead

Returns: Success or failure status indicating whether the task was stopped.
```

**How it works:**
TaskStop forcefully terminates a running task identified by its task ID. This is used when
a background process or subagent is taking too long, is no longer needed, or is stuck in
an unproductive loop.

When to use:

- A long-running shell command (e.g. a build or server) needs to be killed.
- A subagent has been running past its expected duration with no useful progress.
- The user's requirements changed and the in-flight task is no longer relevant.
- You need to free up resources before launching a new task.

The stopped task cannot be resumed — once terminated, its context is lost. If you may need
to continue the work later, consider using TaskOutput with `block: false` to check status
before deciding to stop.

---

## 2. Subagents (via the Task tool)

### `Bash`
**Command execution specialist.**

| Property | Value |
|---|---|
| Tools available | `Bash` only |
| Best for | Git operations, shell commands, build/test pipelines, system tasks |
| When to use | Running terminal commands, install dependencies, execute scripts |
| Model preference | `haiku` for simple commands; `sonnet`/`opus` for complex multi-step shell work |

---

### `general-purpose`
**Research and multi-step task agent.**

| Property | Value |
|---|---|
| Tools available | All tools (full toolkit) |
| Best for | Complex research, multi-file code searches, multi-step workflows |
| When to use | Open-ended investigations, tasks requiring many sequential tool calls, context-heavy subtasks you want to offload |
| Model preference | `sonnet` for most tasks; `opus` for the hardest problems |

---

### `Explore`
**Fast codebase exploration agent.**

| Property | Value |
|---|---|
| Tools available | All tools *except* Task, ExitPlanMode, Edit, Write, NotebookEdit |
| Best for | Finding files, searching code, understanding codebase structure |
| When to use | "Find all API endpoints", "How does auth work?", "Where is X defined?" |
| Thoroughness levels | Specify in prompt: `"quick"` (basic), `"medium"` (moderate), `"very thorough"` (comprehensive) |
| Key constraint | Read-only — cannot modify files |

---

### `Plan`
**Software architect agent.**

| Property | Value |
|---|---|
| Tools available | All tools *except* Task, ExitPlanMode, Edit, Write, NotebookEdit |
| Best for | Designing implementation strategies, identifying critical files, evaluating trade-offs |
| When to use | Planning a feature before coding, evaluating architectural options |
| Key constraint | Read-only — produces plans, not code |

---

### `claude-code-guide`
**Claude Code / Agent SDK / API knowledge agent.**

| Property | Value |
|---|---|
| Tools available | Glob, Grep, Read, WebFetch, WebSearch |
| Best for | Answering questions about Claude Code CLI, Claude Agent SDK, and the Anthropic API |
| When to use | "How do I use hooks?", "What MCP servers are available?", "How does tool_use work in the API?" |
| Resumable | Yes — check for a running/recently completed instance before spawning a new one |

---

### `statusline-setup`
**Status line configuration agent.**

| Property | Value |
|---|---|
| Tools available | Read, Edit |
| Best for | Configuring the Claude Code status line display |
| When to use | User wants to customize their status line settings |

---

## 3. Skills (via the Skill tool)

| Skill | Trigger keywords | Description |
|---|---|---|
| `docx` | Word, .docx, report, letter, memo | Word document creation, editing, tracked changes, formatting |
| `xlsx` | Excel, spreadsheet, .xlsx, budget, chart | Excel creation, formulas, formatting, charting, data analysis |
| `pptx` | PowerPoint, .pptx, slides, deck, presentation | Slide deck creation, editing, layouts, speaker notes |
| `pdf` | PDF, .pdf, form, merge, split, extract | PDF creation, text extraction, merge/split, form filling, OCR |
| `skill-creator` | create skill, improve skill, run evals | Build new skills, optimize existing ones, benchmark performance |
| `create-shortcut` | shortcut, schedule, automation | Create on-demand or scheduled shortcuts |
| `keybindings-help` | keybindings, rebind, keyboard shortcuts | Customize keyboard shortcuts and chord bindings |
| `cowork-plugin-customizer` | customize plugin, configure MCP | Personalize plugins for specific org workflows |

---

*Generated: February 10, 2026*
