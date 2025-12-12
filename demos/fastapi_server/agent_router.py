"""Agent endpoint for streaming responses via SSE."""
import asyncio
import logging
import os
from typing import Optional, AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from anthropic_agent.core import AnthropicAgent
from anthropic_agent.database import SQLBackend
from anthropic_agent.file_backends import S3Backend
from anthropic_agent.tools import SAMPLE_TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRunRequest(BaseModel):
    """Request to run an agent with a user prompt."""
    agent_uuid: Optional[str] = None
    user_prompt: str | list[dict] | dict


async def stream_agent_response(
    user_prompt: str | list[dict] | dict,
    agent_uuid: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE-formatted stream of agent responses.
    
    Args:
        user_prompt: The user's question or task (str, list, or dict)
        agent_uuid: Optional UUID to resume existing agent
        
    Yields:
        SSE-formatted strings containing raw agent output chunks
    """
    try:
        # Create the agent with hardcoded config
        agent = AnthropicAgent(
            system_prompt="You are a helpful assistant that should help the user with their questions.",
            model="claude-sonnet-4-5",
            thinking_tokens=1024,
            max_tokens=64000,
            tools=SAMPLE_TOOL_FUNCTIONS,
            server_tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            beta_headers=["files-api-2025-04-14", "code-execution-2025-08-25"],
            file_backend=S3Backend(bucket=os.getenv("S3_BUCKET")),
            db_backend=SQLBackend(connection_string=os.getenv("DATABASE_URL")),
            agent_uuid=agent_uuid,
            formatter="raw",
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


@router.post("/run")
async def run_agent(request: AgentRunRequest) -> StreamingResponse:
    """Run an agent with the given prompt and stream responses via SSE.
    
    This endpoint creates a new agent (or resumes an existing one) and streams
    its responses in real-time using Server-Sent Events (SSE).
    
    Args:
        request: Agent run request containing user_prompt and optional agent_uuid
        
    Returns:
        StreamingResponse with text/event-stream content type
        
    Example:
        ```
        POST /agent/run
        {
            "user_prompt": "Calculate (15 + 27) * 3 - 8",
            "agent_uuid": "optional-uuid-to-resume"
        }
        ```
    """
    return StreamingResponse(
        stream_agent_response(
            user_prompt=request.user_prompt,
            agent_uuid=request.agent_uuid,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )
