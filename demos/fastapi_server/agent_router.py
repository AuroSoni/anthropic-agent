"""Agent endpoint for streaming responses via SSE."""
import asyncio
import json
import logging
import mimetypes
import os
import traceback
from typing import Any, AsyncGenerator, Literal, Optional
from urllib.parse import urlparse

import anthropic
import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, model_validator

from anthropic_agent.core import AnthropicAgent
from anthropic_agent.common_tools import (
    ReadFileTool, GlobFileSearchTool, GrepSearchTool,
    ListDirTool, ApplyPatchTool, CodeExecutionTool,
)
from anthropic_agent.cowork_style_tools import (
    BashTool, create_read_tool, create_write_tool,
    create_edit_tool, create_glob_tool, create_grep_tool,
)
from anthropic_agent.file_backends import S3Backend
from anthropic_agent.tools import SAMPLE_TOOL_FUNCTIONS, tool, ToolResult
from storage import config_adapter, conversation_adapter, run_adapter

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

@tool
def read_image_raw(image_path: str, description: str = "") -> str:
    """Read an image file and return it with a description.
    
    Args:
        image_path: Path to the image file to read
        description: Optional description to include with the image
        
    Returns:
        JSON content blocks with image and text
    """
    import json
    import base64
    from pathlib import Path
    
    path = Path(image_path)
    
    if not path.exists():
        return f"Error: Image not found at {image_path}"
    
    # Determine media type from extension
    ext = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg", 
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/png")
    
    # Read and encode image
    image_bytes = path.read_bytes()
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    
    # Build content blocks as dictionary structure
    content_blocks = [
        {
            "type": "text",
            "text": description if description else f"Image from {path.name}"
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data
            }
        }
    ]
    
    # Return as JSON string (current tools return str)
    return ToolResult.with_image(
        image_data=image_bytes,
        media_type=media_type,
        text=description if description else f"Image from {path.name}"
    )

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
    config_adapter: Any = None
    conversation_adapter: Any = None
    run_adapter: Any = None
    subagents: dict | None = None  # dict[str, AnthropicAgent] for subagent delegation
    formatter: str = "json"
    stream_meta_history_and_tool_results: bool = False  # Include tool results in stream output


# Agent type literal for request validation
AgentType = Literal[
    "agent_no_tools",
    "agent_client_tools",
    "agent_frontend_tools",
    "agent_all_json",
    "agent_subagents",
    "agent_cowork",
]


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


class FrontendToolResult(BaseModel):
    """A single frontend tool result."""
    tool_use_id: str
    content: str
    is_error: bool = False


class ToolResultsRequest(BaseModel):
    """Request to submit frontend tool results and resume agent execution."""
    agent_uuid: str
    tool_results: list[FrontendToolResult]
    agent_type: Optional[AgentType] = None  # Original agent type for correct config (defaults to agent_frontend_tools)


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

# Shared config
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "./workspace")
_s3 = S3Backend(bucket=os.getenv("S3_BUCKET"))

# Common tools (sandboxed to WORKSPACE_PATH, .md/.mmd files)
COMMON_TOOL_FUNCTIONS = [
    ReadFileTool(base_path=WORKSPACE_PATH).get_tool(),
    GlobFileSearchTool(base_path=WORKSPACE_PATH).get_tool(),
    GrepSearchTool(base_path=WORKSPACE_PATH).get_tool(),
    ListDirTool(base_path=WORKSPACE_PATH).get_tool(),
    ApplyPatchTool(base_path=WORKSPACE_PATH).get_tool(),
    CodeExecutionTool(output_base_path=os.path.join(WORKSPACE_PATH, "code_outputs")).get_tool(),
]

# Cowork-style tools (unrestricted, absolute paths)
COWORK_TOOL_FUNCTIONS = [
    create_read_tool(),
    create_write_tool(),
    create_edit_tool(),
    create_glob_tool(),
    create_grep_tool(),
    BashTool(base_path=WORKSPACE_PATH).get_tool(),
]

# --- 1. Plain chat (no tools) ---
agent_no_tools = AgentConfig(
    system_prompt="You are a helpful assistant that should help the user with their questions.",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    file_backend=_s3,
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    formatter="json",
    stream_meta_history_and_tool_results=True,
)

# --- 2. Calculator tools only ---
agent_client_tools = AgentConfig(
    system_prompt="You are a helpful assistant that should help the user with their questions.",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=SAMPLE_TOOL_FUNCTIONS,
    file_backend=_s3,
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    formatter="json",
    stream_meta_history_and_tool_results=True,
)

# --- 3. Frontend tools (browser-executed, pause/resume) ---
agent_frontend_tools = AgentConfig(
    system_prompt="""You are a helpful assistant that should help the user with their questions.
When performing calculations that result in significant values (over 50), or when taking
any action that could have consequences, ask for user confirmation using the user_confirm tool.""",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=SAMPLE_TOOL_FUNCTIONS + [read_image_raw],
    frontend_tools=FRONTEND_TOOL_FUNCTIONS,
    file_backend=_s3,
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    formatter="json",
    stream_meta_history_and_tool_results=True,
)

# --- 4. All common tools + server tools ---
agent_all_json = AgentConfig(
    system_prompt="You are a helpful assistant that should help the user with their questions.",
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=COMMON_TOOL_FUNCTIONS + [read_image_raw],
    server_tools=[
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 50},
        {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 50},
    ],
    context_management={
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 30000},
                "keep": {"type": "tool_uses", "value": 3},
                "clear_at_least": {"type": "input_tokens", "value": 5000},
                "exclude_tools": ["web_search"],
            }
        ]
    },
    beta_headers=[
        "web-fetch-2025-09-10",
        "files-api-2025-04-14",
        "context-management-2025-06-27",
    ],
    file_backend=_s3,
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    formatter="json",
    stream_meta_history_and_tool_results=True,
)

# --- 5. Subagent orchestration ---
_calculator_agent = AnthropicAgent(
    system_prompt="You are a calculator specialist. Use the provided math tools to perform calculations accurately.",
    description="Performs arithmetic calculations using add, subtract, multiply, and divide tools",
    model="claude-sonnet-4-5",
    tools=SAMPLE_TOOL_FUNCTIONS,
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
)

_researcher_agent = AnthropicAgent(
    system_prompt="You are a research specialist. Search the web for information and provide comprehensive answers.",
    description="Searches the web for information and provides summarised results",
    model="claude-sonnet-4-5",
    server_tools=[
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 50},
        {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 50},
    ],
    beta_headers=["web-fetch-2025-09-10"],
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
)

agent_subagents = AgentConfig(
    system_prompt=(
        "You are a coordinator agent. You MUST delegate tasks to your specialist subagents:\n"
        "- Use the 'calculator' subagent for any arithmetic or math questions.\n"
        "- Use the 'researcher' subagent for any questions requiring web search or factual lookup.\n"
        "After receiving results from subagents, synthesise and present them to the user."
    ),
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    subagents={
        "calculator": _calculator_agent,
        "researcher": _researcher_agent,
    },
    file_backend=_s3,
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    formatter="json",
    stream_meta_history_and_tool_results=True,
)

# --- 6. Cowork-style coding agent (unrestricted file I/O + bash) ---
agent_cowork = AgentConfig(
    system_prompt=(
        "You are a coding assistant with full access to the workspace. "
        "Use your tools to explore, read, write, edit, and search files, "
        "and execute bash commands to help the user with coding tasks."
    ),
    model="claude-sonnet-4-5",
    thinking_tokens=1024,
    max_tokens=64000,
    tools=COWORK_TOOL_FUNCTIONS,
    file_backend=_s3,
    config_adapter=config_adapter,
    conversation_adapter=conversation_adapter,
    run_adapter=run_adapter,
    formatter="json",
    stream_meta_history_and_tool_results=True,
)

# Agent config registry
AGENT_CONFIGS: dict[AgentType, AgentConfig] = {
    "agent_no_tools": agent_no_tools,
    "agent_client_tools": agent_client_tools,
    "agent_frontend_tools": agent_frontend_tools,
    "agent_all_json": agent_all_json,
    "agent_subagents": agent_subagents,
    "agent_cowork": agent_cowork,
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
        # Get config from registry (default to agent_all_json)
        config = AGENT_CONFIGS[agent_type or "agent_all_json"]
        
        # Create the agent with config
        agent = AnthropicAgent(
            **config.model_dump(exclude_none=True),
            agent_uuid=agent_uuid,
        )
        
        # Create queue for streaming
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        
        # Run the agent in background (this will populate the queue)
        async def run_agent_and_signal():
            try:
                result = await agent.run(user_prompt, queue)
                return result
            finally:
                await queue.put(None)  # Always signal completion
        
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
        
        # Wait for agent to complete (re-raises if agent.run() failed)
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
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        logger.error(f"Error in agent stream: {e}\n{''.join(tb)}")
        error_detail = {
            "error": type(e).__name__,
            "message": str(e),
            "traceback": "".join(tb),
        }
        yield f"data: {json.dumps(error_detail)}\n\n"


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
        # Use the specified agent_type config, defaulting to agent_frontend_tools
        agent_type = request.agent_type or "agent_frontend_tools"
        config = AGENT_CONFIGS.get(agent_type, AGENT_CONFIGS["agent_frontend_tools"])
        
        agent = AnthropicAgent(
            **config.model_dump(exclude_none=True),
            agent_uuid=request.agent_uuid,
        )
        
        # Create queue for streaming
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        
        # Continue the agent with frontend tool results
        async def run_continuation():
            try:
                result = await agent.continue_with_tool_results(
                    [r.model_dump() for r in request.tool_results],
                    queue=queue,
                )
                return result
            finally:
                await queue.put(None)  # Always signal completion
        
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
        
        # Wait for agent to complete (re-raises if continuation failed)
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
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        logger.error(f"Error in tool results stream: {e}\n{''.join(tb)}")
        error_detail = {
            "error": type(e).__name__,
            "message": str(e),
            "traceback": "".join(tb),
        }
        yield f"data: {json.dumps(error_detail)}\n\n"


@router.post("/run")
async def run_agent(request: AgentRunRequest) -> StreamingResponse:
    """Run an agent with the given prompt and stream responses via SSE.
    
    This endpoint creates a new agent (or resumes an existing one) and streams
    its responses in real-time using Server-Sent Events (SSE).
    
    Only one of agent_type or agent_uuid can be provided. If neither is provided,
    defaults to agent_all_json.
    
    Args:
        request: Agent run request containing user_prompt and optional agent_uuid/agent_type
        
    Returns:
        StreamingResponse with text/event-stream content type
        
    Example:
        ```
        POST /agent/run
        {
            "user_prompt": "Calculate (15 + 27) * 3 - 8",
            "agent_type": "agent_all_json"
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


@router.get("/{agent_uuid}/images/{image_id}")
async def get_tool_image(
    agent_uuid: str,
    image_id: str,
) -> StreamingResponse:
    """Fetch an image generated by a tool result.
    
    This endpoint serves images stored by the local file backend. When using
    S3 backend, images are served directly via presigned URLs and this endpoint
    is not needed.
    
    Args:
        agent_uuid: The agent session UUID
        image_id: The image identifier (e.g., "img_abc123")
        
    Returns:
        StreamingResponse with the image content
        
    Raises:
        HTTPException 404: If image not found
        HTTPException 500: If file backend not configured for local storage
    """
    from pathlib import Path
    from anthropic_agent.file_backends import LocalFilesystemBackend
    
    # Get file backend configuration
    # For now, use default local path - in production this should match agent config
    base_path = Path("./agent-files")
    agent_dir = base_path / agent_uuid
    
    if not agent_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Agent directory not found: {agent_uuid}",
        )
    
    # Find image file matching the image_id pattern
    # Files are stored as: {image_id}_{filename}.{ext}
    matching_files = list(agent_dir.glob(f"{image_id}_*"))
    
    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {image_id}",
        )
    
    image_path = matching_files[0]
    
    # Determine media type from extension
    ext = image_path.suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "application/octet-stream")
    
    # Read and return image
    try:
        content = image_path.read_bytes()
        return StreamingResponse(
            iter([content]),
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="{image_path.name}"',
                "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
            },
        )
    except Exception as e:
        logger.error(f"Failed to read image {image_path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read image: {str(e)}",
        )


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(default=50, le=100, description="Maximum number of sessions to return"),
    offset: int = Query(default=0, ge=0, description="Number of sessions to skip"),
) -> AgentSessionListResponse:
    """List all agent sessions with metadata.
    
    Returns sessions sorted by updated_at descending (newest first).
    Supports offset-based pagination.
    
    Args:
        limit: Maximum number of sessions to return (max 100)
        offset: Number of sessions to skip
        
    Returns:
        AgentSessionListResponse with list of sessions and total count
    """
    # Use shared adapter directly
    sessions, total = await config_adapter.list_sessions(limit=limit, offset=offset)
    
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
) -> ConversationListResponse:
    """Get paginated conversation history for an agent (newest first).
    
    Uses cursor-based pagination for efficient infinite scroll. On initial load,
    omit the `before` parameter to get the newest conversations. For subsequent
    pages, pass the smallest `sequence_number` from the previous response as `before`.
    
    Args:
        agent_uuid: The agent's UUID
        before: Load conversations with sequence_number < this value (None = latest)
        limit: Maximum number of conversations to return (max 100)
        
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
    # Use conversation adapter - returns list of Conversation dataclasses
    conversations, has_more = await conversation_adapter.load_cursor(
        agent_uuid=agent_uuid,
        before=before,
        limit=limit,
    )
    
    # Convert Conversation dataclasses to response models
    items = [
        ConversationItem(
            conversation_id=conv.conversation_id,
            run_id=conv.run_id,
            sequence_number=conv.sequence_number or 0,
            user_message=conv.user_message,
            final_response=conv.final_response,
            started_at=conv.started_at,
            completed_at=conv.completed_at,
            stop_reason=conv.stop_reason,
            total_steps=conv.total_steps,
            usage=conv.usage,
            generated_files=conv.generated_files,
            messages=conv.messages,
        )
        for conv in conversations
    ]
    
    # Get session title from config adapter
    agent_config = await config_adapter.load(agent_uuid)
    session_title = agent_config.title if agent_config else None
    
    return ConversationListResponse(
        conversations=items,
        has_more=has_more,
        title=session_title,
    )
