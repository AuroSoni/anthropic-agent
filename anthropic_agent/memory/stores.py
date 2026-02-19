"""Memory store implementations for persistent cross-session knowledge.

Memory stores are the agent's **cross-session knowledge managers**. They
operate strictly at run boundaries — ``retrieve()`` at the start of a run to
inject relevant prior knowledge, and ``update()`` at the end to extract and
persist new learnings for future runs.

Memory stores never participate in context compaction. Compaction (shrinking
the live message list to fit the model's token budget) is handled entirely by
the ``Compactor`` in ``anthropic_agent.core.compaction``. The two systems are
independent: a compactor manages *within-session* context size, while a memory
store manages *across-session* knowledge.

This module defines the ``MemoryStore`` protocol and provides placeholder
implementations. Real implementations (vector-based, semantic, etc.) will be
added in future iterations.
"""

from typing import Protocol, Literal, Any
from datetime import datetime

from ..logging import get_logger

logger = get_logger(__name__)

# Type alias for memory store names
MemoryStoreType = Literal["placeholder", "none"]


class MemoryStore(Protocol):
    """Protocol for memory store implementations.
    
    Memory stores manage persistent cross-session knowledge that can be
    injected into agent conversations. They operate at run boundaries only:
    retrieve at start, update at end. Independent of context compaction.
    """
    
    def retrieve(
        self,
        tools: list[dict],
        user_message: dict,
        messages: list[dict],
        model: str
    ) -> list[dict]:
        """Retrieve relevant memories and inject into messages.
        
        Called once per agent.run() after the user message is added.
        The memory store can query its storage and inject relevant context
        as additional messages.
        
        Args:
            tools: List of tool schemas available to the agent
            user_message: The current user message dict
            messages: Current message list (including user_message)
            model: Model name being used
            
        Returns:
            Updated messages list with memory context injected.
            Memory-injected messages should NOT be added to conversation_history.
        """
        ...
    
    def update(
        self,
        messages: list[dict],
        conversation_history: list[dict],
        tools: list[dict],
        model: str
    ) -> dict[str, Any]:
        """Update memory store with conversation results.
        
        Called after agent.run() completes successfully. The memory store
        can extract facts, entities, or summaries to store for future retrieval.
        
        Args:
            messages: Compacted message list (what was sent to API)
            conversation_history: Full uncompacted conversation history
            tools: List of tool schemas
            model: Model name used
            
        Returns:
            Metadata dict with format: {"created": [...], "updated": [...], ...}
            This will be logged to agent_logs.
        """
        ...


class NoOpMemoryStore:
    """No-operation memory store that does nothing.
    
    Useful for disabling memory functionality or as a baseline.
    Returns all inputs unchanged.
    """
    
    def __init__(self, **kwargs):
        """Initialize no-op memory store.
        
        Args:
            **kwargs: Ignored (accepted for interface consistency)
        """
        pass
    
    def retrieve(
        self,
        tools: list[dict],
        user_message: dict,
        messages: list[dict],
        model: str
    ) -> list[dict]:
        """Return messages unchanged."""
        return messages
    
    def update(
        self,
        messages: list[dict],
        conversation_history: list[dict],
        tools: list[dict],
        model: str
    ) -> dict[str, Any]:
        """Return empty metadata."""
        return {
            "store_type": "none",
            "memories_created": 0,
            "memories_updated": 0
        }
    


class PlaceholderMemoryStore:
    """Placeholder memory store for future implementation.
    
    This is a stub implementation that logs method calls and returns
    data unchanged. It serves as:
    1. A template for implementing real memory stores
    2. A way to test memory integration without actual storage
    3. Documentation of expected behavior
    
    TODO: Implement real memory storage and retrieval:
    - Vector embeddings for semantic search
    - Storage backend (Pinecone, Postgres, etc.)
    - Entity extraction and fact storage
    - Similarity-based retrieval
    """
    
    def __init__(self, top_k: int = 5, **kwargs):
        """Initialize placeholder memory store.
        
        Args:
            top_k: Number of memories to retrieve (not used yet)
            **kwargs: Additional config for future implementation
        """
        self.top_k = top_k
        self.config = kwargs
        logger.info("PlaceholderMemoryStore initialized", top_k=top_k)
    
    def retrieve(
        self,
        tools: list[dict],
        user_message: dict,
        messages: list[dict],
        model: str
    ) -> list[dict]:
        """Retrieve relevant memories and inject into messages.
        
        TODO: Implement real retrieval:
        1. Extract query from user_message
        2. Generate embedding for query
        3. Search vector store for similar memories
        4. Format top_k results as context message
        5. Inject into messages list
        
        Current behavior: Returns messages unchanged.
        """
        # TODO: Implement memory retrieval and injection
        # Example structure for injected memory:
        # memory_message = {
        #     "role": "user",
        #     "content": [{
        #         "type": "text",
        #         "text": f"[Relevant context from memory:\n{formatted_memories}\n]"
        #     }]
        # }
        # messages.append(memory_message)
        
        return messages
    
    def update(
        self,
        messages: list[dict],
        conversation_history: list[dict],
        tools: list[dict],
        model: str
    ) -> dict[str, Any]:
        """Update memory store with conversation results.
        
        TODO: Implement memory extraction and storage:
        1. Extract entities and facts from conversation
        2. Generate embeddings for extracted information
        3. Store in vector database with metadata
        4. Update existing memories if applicable
        5. Return created/updated memory IDs
        
        Current behavior: Returns empty metadata.
        """
        # TODO: Implement memory extraction and storage
        # Example:
        # - Extract facts from assistant responses
        # - Extract user preferences from conversation
        # - Store with embeddings and metadata
        
        return {
            "store_type": "placeholder",
            "memories_created": 0,
            "memories_updated": 0,
            "timestamp": datetime.now().isoformat(),
            "note": "Placeholder - no actual storage performed"
        }
    


# Memory store registry mapping string names to store classes
MEMORY_STORES: dict[str, type[MemoryStore]] = {
    "placeholder": PlaceholderMemoryStore,
    "none": NoOpMemoryStore,
}


def get_memory_store(name: MemoryStoreType, **kwargs) -> MemoryStore:
    """Get a memory store instance by name.
    
    Args:
        name: Memory store name ("placeholder" or "none")
        **kwargs: Additional arguments to pass to the memory store constructor
            (e.g., top_k=5 for PlaceholderMemoryStore)
        
    Returns:
        An instance of the requested memory store
        
    Raises:
        ValueError: If memory store name is not recognized
    """
    if name not in MEMORY_STORES:
        raise ValueError(
            f"Unknown memory store '{name}'. Available stores: {list(MEMORY_STORES.keys())}"
        )
    
    store_class = MEMORY_STORES[name]
    return store_class(**kwargs)

