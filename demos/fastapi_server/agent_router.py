"""Agent endpoint for streaming responses via SSE."""
import asyncio
import json
import logging
import mimetypes
import os
from typing import Any, AsyncGenerator, Literal, Optional
from urllib.parse import urlparse

import anthropic
import httpx
import litellm
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, model_validator

from anthropic_agent.core import AnthropicAgent
from anthropic_agent.database import SQLBackend
from anthropic_agent.file_backends import S3Backend
from anthropic_agent.tools import SAMPLE_TOOL_FUNCTIONS, tool

logger = logging.getLogger(__name__)


########################################################
# FRONTEND TOOLS (executed by browser, schema only on server)
########################################################

@tool(executor="frontend")
def user_confirm(message: str) -> str:
    """Ask the user for yes/no confirmation before proceeding with an action.
    
    Use this tool when you need explicit user approval before taking an action
    that could have significant consequences, such as:
    - Deleting or modifying data
    - Making purchases or transactions
    - Sending communications
    - Any irreversible operation
    
    Args:
        message: The confirmation message to display to the user, explaining
                what action requires their approval and any relevant details.
    
    Returns:
        "yes" if the user confirms, "no" if the user declines
    """
    pass  # Never executed server-side - runs in browser


# List of all frontend tools for easy registration
FRONTEND_TOOL_FUNCTIONS = [user_confirm]

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentConfig(BaseModel):
    """Configuration for an agent instance."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    system_prompt: str
    model: str
    thinking_tokens: int
    max_tokens: int
    tools: list = []
    frontend_tools: list = []  # Frontend-executed tools (schema only on server)
    server_tools: list = []
    context_management: dict | None = None
    beta_headers: list[str] = []
    file_backend: Any = None
    db_backend: Any = None
    formatter: str = "raw"


# Agent type literal for request validation
AgentType = Literal["agent_no_tools", "agent_client_tools", "agent_all_raw", "agent_all_xml", "agent_frontend_tools", "agent_frontend_tools_raw"]


class AgentRunRequest(BaseModel):
    """Request to run an agent with a user prompt."""
    agent_uuid: Optional[str] = None
    agent_type: Optional[AgentType] = None
    user_prompt: str | list[dict] | dict

    @model_validator(mode='after')
    def validate_agent_source(self) -> "AgentRunRequest":
        # Allow both agent_uuid and agent_type (agent_type provides config)
        # Default to agent_frontend_tools if neither provided
        if not self.agent_uuid and not self.agent_type:
            self.agent_type = "agent_frontend_tools"
        return self


class FileMetadata(BaseModel):
    """Metadata for an uploaded file from Anthropic Files API."""
    id: str
    filename: str
    mime_type: str
    size_bytes: int
    created_at: str
    downloadable: bool


class UploadResponse(BaseModel):
    """Response containing metadata for all uploaded files."""
    files: list[FileMetadata]


class ToolResult(BaseModel):
    """A single frontend tool result."""
    tool_use_id: str
    content: str
    is_error: bool = False


class ToolResultsRequest(BaseModel):
    """Request to submit frontend tool results and resume agent execution."""
    agent_uuid: str
    tool_results: list[ToolResult]


class ConversationItem(BaseModel):
    """A single conversation turn in the history."""
    conversation_id: str
    run_id: str
    sequence_number: int
    user_message: str
    final_response: str | None
    started_at: str | None
    completed_at: str | None
    stop_reason: str | None
    total_steps: int | None
    usage: dict | None
    generated_files: list[dict] | None
    messages: list[dict]


class ConversationListResponse(BaseModel):
    """Response for paginated conversation history."""
    conversations: list[ConversationItem]
    has_more: bool
    title: str | None = None  # Session title if available


class GenerateTitleRequest(BaseModel):
    """Request to generate a conversation title."""
    user_message: str


class GenerateTitleResponse(BaseModel):
    """Response with generated title."""
    title: str


class AgentSessionItem(BaseModel):
    """A single agent session in the list."""
    agent_uuid: str
    title: str | None
    created_at: str | None
    updated_at: str | None
    total_runs: int


class AgentSessionListResponse(BaseModel):
    """Response for paginated agent sessions list."""
    sessions: list[AgentSessionItem]
    total: int


########################################################
# AGENT CONFIGURATIONS
########################################################
agent_no_tools = AgentConfig(
    system_prompt="You are a helpful assistant that should help the user with their questions.",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    file_backend=S3Backend(bucket=os.getenv("S3_BUCKET")),
    db_backend=SQLBackend(connection_string=os.getenv("DATABASE_URL")),
    formatter="raw",
)

agent_client_tools = AgentConfig(
    system_prompt="You are a helpful assistant that should help the user with their questions.",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=SAMPLE_TOOL_FUNCTIONS,
    file_backend=S3Backend(bucket=os.getenv("S3_BUCKET")),
    db_backend=SQLBackend(connection_string=os.getenv("DATABASE_URL")),
    formatter="raw",
)

agent_all_raw = AgentConfig(
    system_prompt="You are a helpful assistant that should help the user with their questions.",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=SAMPLE_TOOL_FUNCTIONS,
    server_tools=[{
        "type": "code_execution_20250825",
        "name": "code_execution"
    },
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 50
    },
    {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 50
    }],
    context_management={
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                # Trigger clearing when threshold is exceeded
                "trigger": {
                    "type": "input_tokens",
                    "value": 30000
                },
                # Number of tool uses to keep after clearing
                "keep": {
                    "type": "tool_uses",
                    "value": 3
                },
                # Optional: Clear at least this many tokens
                "clear_at_least": {
                    "type": "input_tokens",
                    "value": 5000
                },
                # Exclude these tools from being cleared
                "exclude_tools": ["web_search"]
            }
        ]
    },
    beta_headers=["code-execution-2025-08-25", "web-fetch-2025-09-10", "files-api-2025-04-14", "context-management-2025-06-27"],
    file_backend=S3Backend(bucket=os.getenv("S3_BUCKET")),
    db_backend=SQLBackend(connection_string=os.getenv("DATABASE_URL")),
    formatter="raw",
)

agent_all_xml = AgentConfig(
    system_prompt="You are a helpful assistant that should help the user with their questions.",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=SAMPLE_TOOL_FUNCTIONS,
    server_tools=[{
        "type": "code_execution_20250825",
        "name": "code_execution"
    },
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 50
    },
    {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 50
    }],
    context_management={
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                # Trigger clearing when threshold is exceeded
                "trigger": {
                    "type": "input_tokens",
                    "value": 30000
                },
                # Number of tool uses to keep after clearing
                "keep": {
                    "type": "tool_uses",
                    "value": 3
                },
                # Optional: Clear at least this many tokens
                "clear_at_least": {
                    "type": "input_tokens",
                    "value": 5000
                },
                # Exclude these tools from being cleared
                "exclude_tools": ["web_search"]
            }
        ]
    },
    beta_headers=["code-execution-2025-08-25", "web-fetch-2025-09-10", "files-api-2025-04-14", "context-management-2025-06-27"],
    file_backend=S3Backend(bucket=os.getenv("S3_BUCKET")),
    db_backend=SQLBackend(connection_string=os.getenv("DATABASE_URL")),
    formatter="xml",
)

# Agent with frontend tools (for browser-executed tools like user_confirm)
agent_frontend_tools = AgentConfig(
    system_prompt="""You are a helpful assistant that should help the user with their questions.
When performing calculations that result in significant values (over 50), or when taking 
any action that could have consequences, ask for user confirmation using the user_confirm tool.""",
    model="claude-sonnet-4-20250514",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=SAMPLE_TOOL_FUNCTIONS,
    frontend_tools=FRONTEND_TOOL_FUNCTIONS,  # Include frontend tools
    file_backend=S3Backend(bucket=os.getenv("S3_BUCKET")),
    db_backend=SQLBackend(connection_string=os.getenv("DATABASE_URL")),
    formatter="xml",  # XML formatter for proper tag streaming
)

# Agent with frontend tools - RAW format (for testing parser compatibility)
agent_frontend_tools_raw = AgentConfig(
    system_prompt="""You are a helpful assistant that should help the user with their questions.
When performing calculations that result in significant values (over 50), or when taking 
any action that could have consequences, ask for user confirmation using the user_confirm tool.""",
    model="claude-sonnet-4-20250514",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=SAMPLE_TOOL_FUNCTIONS,
    frontend_tools=FRONTEND_TOOL_FUNCTIONS,
    file_backend=S3Backend(bucket=os.getenv("S3_BUCKET")),
    db_backend=SQLBackend(connection_string=os.getenv("DATABASE_URL")),
    formatter="raw",  # Raw JSON format for testing
)

# Agent config registry
AGENT_CONFIGS: dict[AgentType, AgentConfig] = {
    "agent_no_tools": agent_no_tools,
    "agent_client_tools": agent_client_tools,
    "agent_all_raw": agent_all_raw,
    "agent_all_xml": agent_all_xml,
    "agent_frontend_tools": agent_frontend_tools,
    "agent_frontend_tools_raw": agent_frontend_tools_raw,
}

async def stream_agent_response(
    user_prompt: str | list[dict] | dict,
    agent_uuid: Optional[str] = None,
    agent_type: Optional[AgentType] = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE-formatted stream of agent responses.
    
    Args:
        user_prompt: The user's question or task (str, list, or dict)
        agent_uuid: Optional UUID to resume existing agent
        agent_type: Optional agent type to use for configuration
        
    Yields:
        SSE-formatted strings containing raw agent output chunks
    """
    try:
        # Get config from registry (default to agent_all_raw)
        config = AGENT_CONFIGS[agent_type or "agent_all_raw"]
        
        # Create the agent with config
        agent = AnthropicAgent(
            **config.model_dump(exclude_none=True),
            agent_uuid=agent_uuid,
        )
        
        # Create queue for streaming
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        
        # Run the agent in background (this will populate the queue)
        async def run_agent_and_signal():
            result = await agent.run(user_prompt, queue)
            await queue.put(None)  # Signal completion
            return result
        
        agent_task = asyncio.create_task(run_agent_and_signal())
        
        # Yield raw chunks as they arrive from the queue
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield f"data: {chunk}\n\n"
                queue.task_done()
        except asyncio.CancelledError:
            agent_task.cancel()
            raise
        
        # Wait for agent to complete
        await agent_task
        
        # Wait for background persistence tasks to complete
        # This prevents "Event loop is closed" errors from asyncpg
        try:
            drain_result = await agent.drain_background_tasks(timeout=10.0)
            if drain_result.get("timed_out", 0) > 0:
                logger.warning(f"Some persistence tasks timed out: {drain_result}")
        except Exception as e:
            logger.warning(f"Error draining background tasks: {e}")
        
        # Send final SSE marker to close stream
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in agent stream: {e}", exc_info=True)
        yield f"data: Error: {str(e)}\n\n"


async def stream_tool_results_response(
    request: ToolResultsRequest,
) -> AsyncGenerator[str, None]:
    """Generate SSE-formatted stream after frontend tools complete.
    
    Re-hydrates the agent from the database using agent_uuid, submits
    the frontend tool results, and continues streaming the response.
    
    Args:
        request: Tool results request containing agent_uuid and results
        
    Yields:
        SSE-formatted strings containing agent output chunks
    """
    try:
        # Re-hydrate agent from DB (state is loaded automatically via agent_uuid)
        # Use agent_frontend_tools config since we know this agent has frontend tools
        config = AGENT_CONFIGS["agent_frontend_tools"]
        
        agent = AnthropicAgent(
            **config.model_dump(exclude_none=True),
            agent_uuid=request.agent_uuid,
        )
        
        # Create queue for streaming
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        
        # Continue the agent with frontend tool results
        async def run_continuation():
            result = await agent.continue_with_tool_results(
                [r.model_dump() for r in request.tool_results],
                queue=queue,
            )
            await queue.put(None)  # Signal completion
            return result
        
        agent_task = asyncio.create_task(run_continuation())
        
        # Yield chunks as they arrive
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield f"data: {chunk}\n\n"
                queue.task_done()
        except asyncio.CancelledError:
            agent_task.cancel()
            raise
        
        # Wait for agent to complete
        await agent_task
        
        # Wait for background persistence tasks to complete
        # This prevents "Event loop is closed" errors from asyncpg
        try:
            drain_result = await agent.drain_background_tasks(timeout=10.0)
            if drain_result.get("timed_out", 0) > 0:
                logger.warning(f"Some persistence tasks timed out: {drain_result}")
        except Exception as e:
            logger.warning(f"Error draining background tasks: {e}")
        
        # Send final SSE marker
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in tool results stream: {e}", exc_info=True)
        yield f"data: Error: {str(e)}\n\n"


@router.post("/run")
async def run_agent(request: AgentRunRequest) -> StreamingResponse:
    """Run an agent with the given prompt and stream responses via SSE.
    
    This endpoint creates a new agent (or resumes an existing one) and streams
    its responses in real-time using Server-Sent Events (SSE).
    
    Only one of agent_type or agent_uuid can be provided. If neither is provided,
    defaults to agent_all_raw.
    
    Args:
        request: Agent run request containing user_prompt and optional agent_uuid/agent_type
        
    Returns:
        StreamingResponse with text/event-stream content type
        
    Example:
        ```
        POST /agent/run
        {
            "user_prompt": "Calculate (15 + 27) * 3 - 8",
            "agent_type": "agent_all_raw"
        }
        ```
    """
    return StreamingResponse(
        stream_agent_response(
            user_prompt=request.user_prompt,
            agent_uuid=request.agent_uuid,
            agent_type=request.agent_type,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )


@router.post("/tool_results")
async def submit_tool_results(request: ToolResultsRequest) -> StreamingResponse:
    """Resume agent execution after frontend tools have been executed.
    
    This endpoint is called by the browser after it executes frontend tools
    (e.g., user_confirm). The agent state is re-hydrated from the database
    using the agent_uuid, and execution continues with the tool results.
    
    Args:
        request: Tool results request containing agent_uuid and tool results
        
    Returns:
        StreamingResponse with text/event-stream content type
        
    Example:
        ```
        POST /agent/tool_results
        {
            "agent_uuid": "abc123...",
            "tool_results": [
                {"tool_use_id": "tool_001", "content": "yes"},
                {"tool_use_id": "tool_002", "content": "confirmed"}
            ]
        }
        ```
    """
    return StreamingResponse(
        stream_tool_results_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )


@router.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(default=[]),
    urls: list[str] = Form(default=[]),
) -> UploadResponse:
    """Upload files to Anthropic Files API.
    
    Accepts files via form data (direct uploads or URLs). For URLs, the server
    downloads the file first and then uploads to Anthropic.
    
    Args:
        files: List of files to upload directly
        urls: List of URLs to download and upload
        
    Returns:
        UploadResponse containing metadata for all uploaded files
        
    Example:
        ```
        POST /agent/upload
        Content-Type: multipart/form-data
        
        files: <file1>, <file2>
        urls: https://example.com/image.png
        ```
    """
    if not files and not urls:
        raise HTTPException(
            status_code=400,
            detail="At least one file or URL must be provided",
        )
    
    client = anthropic.Anthropic()
    uploaded_files: list[FileMetadata] = []
    
    # Upload direct files
    for upload_file in files:
        if upload_file.filename:
            try:
                content = await upload_file.read()
                mime_type = upload_file.content_type or "application/octet-stream"
                
                result = client.beta.files.upload(
                    file=(upload_file.filename, content, mime_type),
                )
                
                uploaded_files.append(FileMetadata(
                    id=result.id,
                    filename=result.filename,
                    mime_type=result.mime_type,
                    size_bytes=result.size_bytes,
                    created_at=result.created_at.isoformat(),
                    downloadable=result.downloadable,
                ))
            except Exception as e:
                logger.error(f"Failed to upload file {upload_file.filename}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload file {upload_file.filename}: {str(e)}",
                )
    
    # Download and upload from URLs
    async with httpx.AsyncClient() as http_client:
        for url in urls:
            try:
                # Download file from URL
                response = await http_client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                # Extract filename from URL or Content-Disposition header
                content_disposition = response.headers.get("content-disposition")
                if content_disposition and "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[-1].strip('"\'')
                else:
                    parsed_url = urlparse(url)
                    filename = os.path.basename(parsed_url.path) or "downloaded_file"
                
                # Determine MIME type
                content_type = response.headers.get("content-type", "").split(";")[0]
                if not content_type:
                    content_type, _ = mimetypes.guess_type(filename)
                    content_type = content_type or "application/octet-stream"
                
                content = response.content
                
                result = client.beta.files.upload(
                    file=(filename, content, content_type),
                )
                
                uploaded_files.append(FileMetadata(
                    id=result.id,
                    filename=result.filename,
                    mime_type=result.mime_type,
                    size_bytes=result.size_bytes,
                    created_at=result.created_at.isoformat(),
                    downloadable=result.downloadable,
                ))
            except httpx.HTTPError as e:
                logger.error(f"Failed to download from URL {url}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download from URL {url}: {str(e)}",
                )
            except Exception as e:
                logger.error(f"Failed to upload file from URL {url}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload file from URL {url}: {str(e)}",
                )
    
    return UploadResponse(files=uploaded_files)


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(default=50, le=100, description="Maximum number of sessions to return"),
    offset: int = Query(default=0, ge=0, description="Number of sessions to skip"),
    agent_type: AgentType = Query(default="agent_frontend_tools", description="Agent type to determine database backend"),
) -> AgentSessionListResponse:
    """List all agent sessions with metadata.
    
    Returns sessions sorted by updated_at descending (newest first).
    Supports offset-based pagination.
    
    Args:
        limit: Maximum number of sessions to return (max 100)
        offset: Number of sessions to skip
        agent_type: Agent type to determine which database backend to query
        
    Returns:
        AgentSessionListResponse with list of sessions and total count
    """
    # Get the database backend from the agent config
    config = AGENT_CONFIGS.get(agent_type)
    if not config or not config.db_backend:
        raise HTTPException(
            status_code=400,
            detail=f"Agent type '{agent_type}' does not have a database backend configured"
        )
    
    db_backend = config.db_backend
    
    sessions, total = await db_backend.list_agent_sessions(limit=limit, offset=offset)
    
    return AgentSessionListResponse(
        sessions=[
            AgentSessionItem(
                agent_uuid=s["agent_uuid"],
                title=s.get("title"),
                created_at=s.get("created_at"),
                updated_at=s.get("updated_at"),
                total_runs=s.get("total_runs", 0),
            )
            for s in sessions
        ],
        total=total,
    )


@router.get("/{agent_uuid}/conversations")
async def get_conversations(
    agent_uuid: str,
    before: int | None = Query(default=None, description="Load conversations with sequence_number < before"),
    limit: int = Query(default=20, le=100, description="Maximum conversations to return"),
    agent_type: AgentType = Query(default="agent_frontend_tools", description="Agent type to determine database backend"),
) -> ConversationListResponse:
    """Get paginated conversation history for an agent (newest first).
    
    Uses cursor-based pagination for efficient infinite scroll. On initial load,
    omit the `before` parameter to get the newest conversations. For subsequent
    pages, pass the smallest `sequence_number` from the previous response as `before`.
    
    Args:
        agent_uuid: The agent's UUID
        before: Load conversations with sequence_number < this value (None = latest)
        limit: Maximum number of conversations to return (max 100)
        agent_type: Agent type to determine which database backend to query
        
    Returns:
        ConversationListResponse with conversations and has_more flag
        
    Example:
        ```
        # Initial load (newest conversations)
        GET /agent/{uuid}/conversations?limit=20
        
        # Load older conversations (scroll up)
        GET /agent/{uuid}/conversations?before=31&limit=20
        ```
    """
    # Get the database backend from the agent config
    config = AGENT_CONFIGS.get(agent_type)
    if not config or not config.db_backend:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent_type or no database backend configured: {agent_type}",
        )
    
    db_backend = config.db_backend
    
    # Load conversations using cursor-based pagination
    conversations, has_more = await db_backend.load_conversation_history_cursor(
        agent_uuid=agent_uuid,
        before=before,
        limit=limit,
    )
    
    # Convert to response models
    items = [
        ConversationItem(
            conversation_id=conv.get("conversation_id", ""),
            run_id=conv.get("run_id", ""),
            sequence_number=conv.get("sequence_number", 0),
            user_message=conv.get("user_message", ""),
            final_response=conv.get("final_response"),
            started_at=conv.get("started_at"),
            completed_at=conv.get("completed_at"),
            stop_reason=conv.get("stop_reason"),
            total_steps=conv.get("total_steps"),
            usage=conv.get("usage"),
            generated_files=conv.get("generated_files"),
            messages=conv.get("messages", []),
        )
        for conv in conversations
    ]
    
    # Get session title from agent config
    agent_config = await db_backend.load_agent_config(agent_uuid)
    session_title = agent_config.get("title") if agent_config else None
    
    return ConversationListResponse(
        conversations=items,
        has_more=has_more,
        title=session_title,
    )


@router.post("/{agent_uuid}/title")
async def generate_title(
    agent_uuid: str,
    request: GenerateTitleRequest,
    agent_type: AgentType = Query(default="agent_frontend_tools", description="Agent type to determine database backend"),
) -> GenerateTitleResponse:
    """Generate a conversation title using LiteLLM.
    
    Uses OpenAI GPT-5-mini for fast, cost-effective title generation.
    The title is persisted to the agent_config in the database.
    
    Args:
        agent_uuid: The agent's UUID
        request: Contains the user_message to generate title from
        agent_type: Agent type to determine which database backend to use
        
    Returns:
        GenerateTitleResponse with the generated title
    """
    # Get the database backend from the agent config
    config = AGENT_CONFIGS.get(agent_type)
    if not config or not config.db_backend:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent_type or no database backend configured: {agent_type}",
        )
    
    db_backend = config.db_backend
    
    # Generate title using LiteLLM
    try:
        response = await litellm.acompletion(
            model="openai/gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a short, descriptive title (max 50 characters) for a conversation "
                        "based on the user's first message. The title should capture the main topic "
                        "or intent. Respond with only JSON in this format: {\"title\": \"...\"}"
                    )
                },
                {"role": "user", "content": request.user_message}
            ],
            response_format={"type": "json_object"},
            max_tokens=100,
        )
        
        result = json.loads(response.choices[0].message.content)
        title = result.get("title", "New Conversation")[:50]  # Enforce max length
        
    except Exception as e:
        logger.error(f"Error generating title: {e}")
        # Fallback: use truncated user message
        title = request.user_message[:47] + "..." if len(request.user_message) > 50 else request.user_message
    
    # Save title to database
    try:
        await db_backend.update_agent_title(agent_uuid, title)
    except Exception as e:
        logger.error(f"Error saving title to database: {e}")
        # Continue anyway - we have the title even if save failed
    
    return GenerateTitleResponse(title=title)
