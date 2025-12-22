"""Anthropic-specific type definitions and converters.

This module provides type aliases and conversion functions for working
with Anthropic API types and converting them to provider-agnostic types.
"""

from typing import Any, TypeAlias
from dataclasses import dataclass

from anthropic.types.beta import BetaMessage, BetaUsage

from ...core.types import GenericUsage, GenericAgentResult


# Type aliases for Anthropic types
AnthropicUsage: TypeAlias = BetaUsage
AnthropicResponse: TypeAlias = BetaMessage


def convert_to_generic_usage(usage: BetaUsage) -> GenericUsage:
    """Convert Anthropic BetaUsage to GenericUsage.
    
    Args:
        usage: Anthropic BetaUsage object
        
    Returns:
        GenericUsage with values extracted from Anthropic usage
    """
    return GenericUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_tokens=getattr(usage, 'cache_creation_input_tokens', 0) or 0,
        cache_read_tokens=getattr(usage, 'cache_read_input_tokens', 0) or 0,
    )


def convert_to_generic_result(
    final_message: BetaMessage,
    final_answer: str,
    conversation_history: list[dict],
    stop_reason: str,
    model: str,
    total_steps: int = 1,
    agent_logs: list[dict] | None = None,
    generated_files: list[dict] | None = None,
    container_id: str | None = None,
) -> GenericAgentResult:
    """Convert Anthropic response data to GenericAgentResult.
    
    Args:
        final_message: Anthropic BetaMessage object
        final_answer: Extracted text answer
        conversation_history: Full message history
        stop_reason: Stop reason from the response
        model: Model identifier used
        total_steps: Number of agent steps taken
        agent_logs: Optional log entries
        generated_files: Optional file metadata
        container_id: Optional Anthropic container ID
        
    Returns:
        GenericAgentResult with all data converted to generic format
    """
    # Convert BetaMessage to dict
    if hasattr(final_message, 'model_dump'):
        final_message_dict = final_message.model_dump(
            mode="json",
            exclude_unset=True,
            warnings=False
        )
    else:
        final_message_dict = {
            "role": getattr(final_message, 'role', 'assistant'),
            "content": getattr(final_message, 'content', []),
        }
    
    # Build provider metadata
    provider_metadata: dict[str, Any] = {
        "provider": "anthropic",
    }
    if container_id:
        provider_metadata["container_id"] = container_id
    
    return GenericAgentResult(
        final_message=final_message_dict,
        final_answer=final_answer,
        conversation_history=conversation_history,
        stop_reason=stop_reason,
        model=model,
        usage=convert_to_generic_usage(final_message.usage),
        total_steps=total_steps,
        agent_logs=agent_logs,
        generated_files=generated_files,
        provider_metadata=provider_metadata,
    )


def extract_text_from_message(message: BetaMessage) -> str:
    """Extract text content from an Anthropic message.
    
    Extracts text from all text blocks that appear after the last tool use block.
    If no tool use block is present, all text blocks are included.
    
    Args:
        message: Anthropic BetaMessage object
        
    Returns:
        Concatenated text from relevant text blocks
    """
    if not message or not message.content:
        return ""

    start_index = 0
    # Find the index of the last tool_use block
    for i, block in enumerate(message.content):
        if getattr(block, 'type', '') in ['server_tool_use', 'web_search_tool_result', 
                                           'tool_use', 'tool_result']:
            start_index = i + 1
    
    full_text = []
    for block in message.content[start_index:]:
        if hasattr(block, 'text') and getattr(block, 'type', '') == 'text':
            full_text.append(block.text)
    
    return "".join(full_text)


def extract_file_ids_from_message(message: BetaMessage | dict[str, Any]) -> list[str]:
    """Extract file IDs from an Anthropic message.
    
    Scans message content for bash_code_execution_tool_result blocks
    and extracts file IDs from them.
    
    Args:
        message: Anthropic message (BetaMessage or dict)
        
    Returns:
        List of file IDs found in the message
    """
    file_ids: list[str] = []
    
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
        
    if not content or not isinstance(content, list):
        return file_ids

    for item in content:
        if isinstance(item, dict):
            item_type = item.get("type")
            item_content = item.get("content")
        else:
            item_type = getattr(item, "type", "")
            item_content = getattr(item, "content", None)

        if item_type in ('bash_code_execution_tool_result', 'tool_result'):
            if not item_content:
                continue

            if isinstance(item_content, dict):
                inner_type = item_content.get("type")
                files = item_content.get("content", [])
            elif hasattr(item_content, "type"):
                inner_type = getattr(item_content, "type", "")
                files = getattr(item_content, "content", [])
            else:
                continue

            if inner_type == 'bash_code_execution_result' and isinstance(files, list):
                for file in files:
                    if isinstance(file, dict):
                        file_id = file.get("file_id")
                    else:
                        file_id = getattr(file, "file_id", None)
                        
                    if file_id:
                        file_ids.append(file_id)
    
    return file_ids

