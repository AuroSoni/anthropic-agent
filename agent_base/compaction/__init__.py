"""Compaction module for agent_base.

Context window management strategies that operate on canonical ``Message``
objects.  Completely provider-agnostic.

Usage::

    from agent_base.compaction import NoOpCompactor, SummarizingCompactor

    # Disable compaction
    compactor = NoOpCompactor()

    # LLM-based summarization (provider supplies a CompactionLLM)
    compactor = SummarizingCompactor(
        llm=my_compaction_llm,
        threshold=160_000,
    )

    # Factory
    compactor = get_compactor("none")
"""
from typing import Any

from .base import Compactor, CompactorType, CompactionLLM
from .strategies import NoOpCompactor, SummarizingCompactor

# Registry mapping string names to compactor classes.
COMPACTORS: dict[str, type[Compactor]] = {
    "summarizing": SummarizingCompactor,
    "none": NoOpCompactor,
}


def get_compactor(name: CompactorType, **kwargs: Any) -> Compactor:
    """Get a compactor instance by name.

    Args:
        name: Compactor name (``"summarizing"`` or ``"none"``).
        **kwargs: Arguments passed to the compactor constructor.
            For ``SummarizingCompactor``, an ``llm`` (``CompactionLLM``) kwarg is required.

    Returns:
        An instance of the requested compactor.

    Raises:
        ValueError: If compactor name is not recognized or required kwargs
            are missing.
    """
    if name not in COMPACTORS:
        available = ", ".join(COMPACTORS.keys())
        raise ValueError(f"Unknown compactor '{name}'. Available: {available}")

    if name == "summarizing" and "llm" not in kwargs:
        raise ValueError(
            "SummarizingCompactor requires an 'llm' (CompactionLLM) kwarg."
        )

    return COMPACTORS[name](**kwargs)


__all__ = [
    # ABC / Protocol
    "Compactor",
    "CompactorType",
    "CompactionLLM",
    # Implementations
    "NoOpCompactor",
    "SummarizingCompactor",
    # Factory
    "COMPACTORS",
    "get_compactor",
]
