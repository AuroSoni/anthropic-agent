# Anthropic Agent Chat UI

A React-based chat interface for the Anthropic Agent, featuring real-time streaming, conversation history, and rich content rendering.

## Features

- **Real-time Streaming**: Live streaming of agent responses via SSE
- **Rich Content Rendering**: Displays thinking blocks, tool calls, tool results, citations, and more
- **Conversation History**: Persistent chat history with infinite scroll
- **Conversation Sidebar**: Browse and switch between past conversations
- **Frontend Tools**: Support for browser-executed tools (e.g., `user_confirm`)
- **Multi-format Support**: Handles both XML and raw JSON stream formats
- **Server Tools**: Full support for Anthropic server tools (code execution, web search, etc.)

## Getting Started

### Prerequisites

- Node.js 18+
- FastAPI backend running on `http://localhost:8000`

### Installation

```bash
cd demos/vite_app
npm install
```

### Development

```bash
npm run dev
```

Opens the app at `http://localhost:5173`

### Build

```bash
npm run build
```

## Architecture

```
src/
├── components/
│   ├── agent/           # Agent-specific renderers
│   │   ├── agent-viewer.tsx      # Main chat interface
│   │   ├── node-renderer.tsx     # Rich content renderer
│   │   ├── thinking-block.tsx    # Thinking block display
│   │   ├── tool-call-block.tsx   # Tool call display
│   │   └── tool-result-block.tsx # Tool result display
│   ├── chat/            # Chat UI components
│   │   ├── chat-thread.tsx       # Message list with scroll
│   │   ├── user-message.tsx      # User message bubble
│   │   └── assistant-message.tsx # Assistant message bubble
│   ├── sidebar/         # Conversation sidebar
│   │   └── conversation-sidebar.tsx
│   └── ui/              # Base UI components (shadcn/ui)
├── hooks/
│   ├── use-conversation.ts  # Conversation state management
│   └── use-url-state.ts     # URL-based agent UUID
├── lib/
│   ├── api.ts               # Backend API client
│   ├── agent-stream.ts      # SSE streaming logic
│   ├── message-converter.ts # History to nodes converter
│   └── parsers/             # Stream format parsers
│       ├── xml-parser.ts
│       ├── xml-stream-parser.ts
│       └── anthropic-parser.ts
```

## Key Components

### AgentViewer

Main chat interface containing:
- Header with conversation title
- Chat thread with message bubbles
- Input form with agent type selector

### ConversationSidebar

Collapsible left sidebar showing:
- List of past conversations
- Title and relative timestamp
- Click to load conversation
- "New Chat" button

### useConversation Hook

Manages conversation state:
- Message history loading (cursor-based pagination)
- Real-time streaming updates
- Frontend tool coordination
- Title generation

## URL State

The current conversation is persisted in the URL:

```
http://localhost:5173/?agent=abc-123-def
```

This enables:
- Shareable conversation links
- Browser back/forward navigation
- Page refresh without losing context

## Supported Content Types

The UI renders these Anthropic content block types:

| Type | Display |
|------|---------|
| `text` | Plain text |
| `thinking` | Collapsible thinking block |
| `tool_use` | Tool call with arguments |
| `tool_result` | Tool execution result |
| `server_tool_use` | Server tool call (code_execution, etc.) |
| `*_tool_result` | Server tool results |
| `citations` | Source citations |

## API Integration

The frontend connects to these backend endpoints:

- `POST /agent/run` - Start/continue agent execution
- `POST /agent/tool_results` - Submit frontend tool results
- `GET /agent/sessions` - List all conversations
- `GET /agent/{uuid}/conversations` - Get conversation history
- `POST /agent/{uuid}/title` - Generate conversation title

## Configuration

### Agent Types

Select from the dropdown in the input area:

- **Frontend Tools** (default) - Includes browser-executed tools
- **All Tools (Raw)** - Full tools with raw format
- **All Tools (XML)** - Full tools with XML format
- **Client Tools** - Sample client-side tools only
- **No Tools** - Basic assistant

### Styling

Uses Tailwind CSS with:
- Dark mode support (`.dark` class)
- CSS variables for theming
- shadcn/ui component library

## Development

### Type Checking

```bash
npm run typecheck
# or
npx tsc --noEmit
```

### Linting

```bash
npm run lint
```

### Testing

```bash
npm run test
```
