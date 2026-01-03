"""Agent endpoint for streaming responses via SSE."""
import asyncio
import logging
import mimetypes
import os
from typing import Any, AsyncGenerator, Literal, Optional
from urllib.parse import urlparse

import anthropic
import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, model_validator

from anthropic_agent.core import AnthropicAgent
from anthropic_agent.database import SQLBackend, FilesystemBackend
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
        if self.agent_uuid and self.agent_type:
            raise ValueError("Only one of agent_uuid or agent_type can be provided")
        # Default to agent_all_raw if neither provided
        if not self.agent_uuid and not self.agent_type:
            self.agent_type = "agent_all_raw"
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
# NOTE: Uses FilesystemBackend to avoid asyncpg event loop conflicts when re-hydrating
# from the /agent/tool_results endpoint (asyncpg connections are bound to specific event loops)
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
    db_backend=FilesystemBackend(base_path="./agent_data"),  # Use filesystem to avoid asyncpg issues
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
    db_backend=FilesystemBackend(base_path="./agent_data"),
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
