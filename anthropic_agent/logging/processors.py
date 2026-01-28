"""Custom structlog processors for anthropic_agent.

Processors are functions that transform log event dictionaries as they
pass through the logging pipeline. They can add, remove, or modify fields.
"""
from typing import Any

from .context import get_context


def inject_context(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Inject bound context variables into the log event.
    
    This processor adds any context bound via bind_context() to the
    log event dictionary.
    
    Args:
        logger: The logger instance.
        method_name: The name of the log method called (e.g., "info").
        event_dict: The log event dictionary.
    
    Returns:
        The modified event dictionary with context values injected.
    """
    context = get_context()
    # Context values are added but don't override explicit values
    for key, value in context.items():
        if key not in event_dict:
            event_dict[key] = value
    return event_dict


def add_logger_name(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add the logger name to the log event.
    
    This processor adds a 'logger' field containing the name of the
    logger that produced the event.
    
    Args:
        logger: The logger instance.
        method_name: The name of the log method called.
        event_dict: The log event dictionary.
    
    Returns:
        The modified event dictionary with logger name added.
    """
    # Get logger name from the record if available, otherwise from logger
    record = event_dict.get("_record")
    if record is not None:
        event_dict["logger"] = record.name
    elif hasattr(logger, "name"):
        event_dict["logger"] = logger.name
    return event_dict


def filter_by_module_level(
    module_levels: dict[str, int],
) -> Any:
    """Create a processor that filters by per-module log levels.
    
    Args:
        module_levels: Mapping of module names to their minimum log levels.
    
    Returns:
        A processor function that filters events based on module levels.
    """
    import structlog
    
    def processor(
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        record = event_dict.get("_record")
        if record is None:
            return event_dict
        
        logger_name = record.name
        
        # Check if any configured module matches this logger
        for module, min_level in module_levels.items():
            if logger_name.startswith(module):
                if record.levelno < min_level:
                    raise structlog.DropEvent
                break
        
        return event_dict
    
    return processor
