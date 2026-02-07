"""
Anthropic Agent Library

A library for building AI agents using Anthropic's Claude API with
streaming support, retry logic, and tool execution capabilities.

Main exports:
    - AnthropicAgent: Main agent class
    - AgentResult: Result dataclass from agent.run()
    - anthropic_stream_with_backoff: Streaming with retry logic
    - render_stream: Stream renderer with custom formatting
    - FormatterType: Type alias for formatter options ("xml" | "raw")

Logging:
    - configure_logging: Configure the logging framework
    - LogConfig, LogLevel, LogFormat: Configuration classes
    - bind_context, clear_context: Context management for request tracing
    
Example:
    >>> from anthropic_agent import AnthropicAgent
    >>> agent = AnthropicAgent(
    ...     system_prompt="You are a helpful assistant",
    ...     model="claude-sonnet-4-5"
    ... )
    >>> result = await agent.run("Hello!", formatter="xml")
"""
# Optional: Load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip loading .env file

# Core exports
from .core import AgentResult, AnthropicAgent, anthropic_stream_with_backoff
from .streaming import render_stream, stream_to_aqueue, FormatterType

# Logging exports
from .logging import (
    configure_logging,
    LogConfig,
    LogLevel,
    LogFormat,
    bind_context,
    clear_context,
    get_logger,
)

# Pricing exports
from .pricing import (
    CostBreakdown,
    calculate_run_cost,
    load_pricing,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    'AgentResult',
    'AnthropicAgent',
    'anthropic_stream_with_backoff',
    'render_stream',
    'stream_to_aqueue',
    'FormatterType',
    # Logging
    'configure_logging',
    'LogConfig',
    'LogLevel',
    'LogFormat',
    'bind_context',
    'clear_context',
    'get_logger',
    # Pricing
    'CostBreakdown',
    'calculate_run_cost',
    'load_pricing',
]

