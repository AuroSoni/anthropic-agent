"""Agent endpoint for streaming responses via SSE."""
import asyncio
import json
import logging
from typing import Any, Optional, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from anthropic_agent.core import AnthropicAgent
from anthropic_agent.database import FilesystemBackend
from anthropic_agent.tools import SAMPLE_TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentConfigRequest(BaseModel):
    """Configuration for initializing an AnthropicAgent.
    
    Accepts all agent initialization parameters except db_backend, file_backend, and tools
    which are managed internally by the endpoint.
    """
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    max_steps: Optional[int] = None
    thinking_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    stream_meta_history_and_tool_results: Optional[bool] = None
    server_tools: Optional[list[dict[str, Any]]] = None
    beta_headers: Optional[list[str]] = None
    container_id: Optional[str] = None
    messages: Optional[list[dict]] = None
    max_retries: Optional[int] = None
    base_delay: Optional[float] = None
    formatter: Optional[str] = None
    compactor: Optional[str] = None
    memory_store: Optional[str] = None
    api_kwargs: Optional[dict[str, Any]] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    """Request to run an agent with a user prompt.
    
    Attributes:
        user_prompt: The user's question or task for the agent
        agent_config: Optional configuration for agent initialization
        agent_uuid: Optional UUID to resume an existing agent session
    """
    user_prompt: str = Field(..., description="User's question or task")
    agent_config: Optional[AgentConfigRequest] = Field(
        default=None, 
        description="Agent configuration (optional, uses defaults if not provided)"
    )
    agent_uuid: Optional[str] = Field(
        default=None,
        description="UUID of existing agent to resume (optional)"
    )


async def stream_agent_response(
    user_prompt: str,
    agent_config: Optional[AgentConfigRequest] = None,
    agent_uuid: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE-formatted stream of agent responses.
    
    Args:
        user_prompt: The user's question or task
        agent_config: Optional agent configuration
        agent_uuid: Optional UUID to resume existing agent
        
    Yields:
        SSE-formatted strings containing agent output chunks and metadata
    """
    try:
        # Prepare agent initialization parameters
        agent_params: dict[str, Any] = {
            "tools": SAMPLE_TOOL_FUNCTIONS,
            "db_backend": FilesystemBackend(base_path="./data"),
        }
        
        # Add agent_uuid if provided (for resuming sessions)
        if agent_uuid:
            agent_params["agent_uuid"] = agent_uuid
        
        # Add configuration parameters if provided
        if agent_config:
            config_dict = agent_config.model_dump(exclude_none=True, exclude={"api_kwargs"}, warnings=False)
            agent_params.update(config_dict)
            
            # Handle api_kwargs separately to merge them properly
            if agent_config.api_kwargs:
                agent_params.update(agent_config.api_kwargs)
        
        # Create the agent
        agent = AnthropicAgent(**agent_params)
        
        # Send initial metadata
        metadata = {
            "event": "metadata",
            "agent_uuid": agent.agent_uuid,
            "model": agent.model,
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        
        # Create queue for streaming
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        
        # Run the agent in background (this will populate the queue)
        async def run_agent_and_signal():
            result = await agent.run(user_prompt, queue)
            await queue.put(None)  # Signal completion
            return result
        
        agent_task = asyncio.create_task(run_agent_and_signal())
        
        # Yield chunks as they arrive from the queue
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                # Format as SSE event
                yield f"data: {json.dumps({'event': 'chunk', 'content': chunk})}\n\n"
                queue.task_done()
        except asyncio.CancelledError:
            agent_task.cancel()
            raise
        
        # Wait for agent to complete and get result
        result = await agent_task
        
        # Send completion metadata
        completion = {
            "event": "complete",
            "agent_uuid": agent.agent_uuid,
            "total_steps": result.total_steps,
            "stop_reason": result.stop_reason,
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
            },
            "container_id": result.container_id,
        }
        yield f"data: {json.dumps(completion)}\n\n"
        
        # Send final SSE comment to close stream
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in agent stream: {e}", exc_info=True)
        error_event = {
            "event": "error",
            "error": str(e),
            "type": type(e).__name__,
        }
        yield f"data: {json.dumps(error_event)}\n\n"


@router.post("/run")
async def run_agent(request: AgentRunRequest) -> StreamingResponse:
    """Run an agent with the given prompt and stream responses via SSE.
    
    This endpoint creates a new agent (or resumes an existing one) and streams
    its responses in real-time using Server-Sent Events (SSE).
    
    Args:
        request: Agent run request containing prompt, config, and optional UUID
        
    Returns:
        StreamingResponse with text/event-stream content type
        
    Example:
        ```
        POST /agent/run
        {
            "user_prompt": "Calculate (15 + 27) * 3 - 8",
            "agent_config": {
                "system_prompt": "You are a helpful math assistant",
                "model": "claude-sonnet-4-5",
                "max_steps": 50
            }
        }
        ```
        
    SSE Event Types:
        - metadata: Initial agent information (agent_uuid, model)
        - chunk: Text chunks from agent response
        - complete: Final metadata (total_steps, usage, etc.)
        - error: Error information if something goes wrong
    """
    return StreamingResponse(
        stream_agent_response(
            user_prompt=request.user_prompt,
            agent_config=request.agent_config,
            agent_uuid=request.agent_uuid,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )

