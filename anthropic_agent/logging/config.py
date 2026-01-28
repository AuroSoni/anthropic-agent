"""Logging configuration for anthropic_agent."""
import logging
import logging.handlers
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import structlog

from .context import get_context
from .processors import inject_context, add_logger_name


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_int(self) -> int:
        """Convert to logging module integer level."""
        return getattr(logging, self.value)


class LogFormat(str, Enum):
    """Output format for logs."""
    PLAIN = "plain"  # Human-readable for development
    JSON = "json"    # Structured for production


@dataclass
class LogConfig:
    """Configuration for the logging framework.
    
    Attributes:
        level: Default log level for all loggers.
        format: Output format (PLAIN for console, JSON for production).
        log_file: Optional path to write logs to a file.
        max_bytes: Maximum size of log file before rotation (default 10MB).
        backup_count: Number of backup files to keep (default 5).
        module_levels: Per-module log level overrides.
        filters: List of filter functions to apply to log entries.
    """
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.PLAIN
    log_file: Optional[Path] = None
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    module_levels: dict[str, LogLevel] = field(default_factory=dict)
    filters: list[Callable[[dict], Optional[dict]]] = field(default_factory=list)


# Track if logging has been configured
_configured: bool = False


def _get_processors(config: LogConfig) -> list:
    """Build the processor chain based on config."""
    processors: list = [
        structlog.contextvars.merge_contextvars,
        inject_context,
        add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add user-defined filters
    for filter_func in config.filters:
        def make_filter(f: Callable):
            def processor(logger, method_name, event_dict):
                result = f(event_dict)
                if result is None:
                    raise structlog.DropEvent
                return result
            return processor
        processors.append(make_filter(filter_func))
    
    return processors


def _get_renderer(config: LogConfig):
    """Get the appropriate renderer based on format."""
    if config.format == LogFormat.JSON:
        return structlog.processors.JSONRenderer()
    else:
        return structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
            exception_formatter=structlog.dev.plain_traceback,
        )


def _setup_stdlib_logging(config: LogConfig) -> None:
    """Configure stdlib logging to work with structlog."""
    root_logger = logging.getLogger()
    root_logger.setLevel(config.level.to_int())
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(config.level.to_int())
    root_logger.addHandler(console_handler)
    
    # File handler with rotation (if configured)
    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
        )
        file_handler.setLevel(config.level.to_int())
        root_logger.addHandler(file_handler)
    
    # Apply per-module log levels
    for module_name, level in config.module_levels.items():
        logging.getLogger(module_name).setLevel(level.to_int())


def configure_logging(config: Optional[LogConfig] = None) -> None:
    """Configure the logging framework.
    
    This function sets up structlog with a stdlib logging bridge, enabling
    structured logging while maintaining compatibility with existing code
    that uses the standard logging module.
    
    Args:
        config: Logging configuration. If None, uses default settings.
    
    Example:
        >>> from anthropic_agent.logging import configure_logging, LogConfig, LogLevel
        >>> configure_logging(LogConfig(
        ...     level=LogLevel.DEBUG,
        ...     format=LogFormat.JSON,
        ...     log_file=Path("./logs/agent.log"),
        ... ))
    """
    global _configured
    
    if config is None:
        config = LogConfig()
    
    # Set up stdlib logging
    _setup_stdlib_logging(config)
    
    # Build processor chain
    processors = _get_processors(config)
    
    # Configure structlog
    structlog.configure(
        processors=processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure the formatter for stdlib handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _get_renderer(config),
        ],
    )
    
    # Apply formatter to all handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    
    _configured = True


def is_configured() -> bool:
    """Check if logging has been configured."""
    return _configured


def ensure_configured() -> None:
    """Ensure logging is configured with defaults if not already configured."""
    if not _configured:
        configure_logging()
