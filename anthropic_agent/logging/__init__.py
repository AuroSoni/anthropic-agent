"""Structured logging framework for anthropic_agent.

This module provides a centralized logging configuration with support for:
- Structured JSON output for production environments
- Human-readable console output for development
- Context propagation across async boundaries
- Per-module log level control
- File rotation and backup

Quick Start:
    >>> from anthropic_agent.logging import configure_logging, LogConfig, LogLevel
    >>> configure_logging(LogConfig(level=LogLevel.DEBUG))

For request tracing:
    >>> from anthropic_agent.logging import bind_context, clear_context
    >>> bind_context(request_id="req-123")
    >>> # ... all logs will include request_id
    >>> clear_context()
"""
import structlog

from .config import (
    LogConfig,
    LogFormat,
    LogLevel,
    configure_logging,
    ensure_configured,
    is_configured,
)
from .context import bind_context, clear_context, get_context, unbind_context


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.
    
    This returns a structlog BoundLogger that integrates with the
    configured logging setup. If logging hasn't been configured yet,
    it will be configured with default settings.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
              If None, returns the root logger.
    
    Returns:
        A bound logger instance.
    
    Example:
        >>> from anthropic_agent.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", item_count=42)
    """
    ensure_configured()
    return structlog.get_logger(name)


__all__ = [
    # Configuration
    "LogConfig",
    "LogFormat",
    "LogLevel",
    "configure_logging",
    "is_configured",
    # Logger
    "get_logger",
    # Context management
    "bind_context",
    "unbind_context",
    "clear_context",
    "get_context",
]
