import json
from dataclasses import dataclass
from typing import Optional
from anthropic.types.beta import BetaMessage, BetaUsage

@dataclass
class AgentResult:
    """Result object returned from agent.run() containing the full agent execution context.
    
    Attributes:
        final_message: The last message from the assistant (BetaMessage object)
        conversation_history: Full conversation history including all messages (uncompacted)
        stop_reason: Why the model stopped generating ("end_turn", "tool_use", "max_tokens", etc.)
        model: The model used for this execution
        usage: Token usage statistics from the final message (last step only)
        container_id: Container ID for multi-turn conversations (if applicable)
        total_steps: Number of agent steps taken (including tool use loops)
        agent_logs: List of log entries tracking compactions and other agent actions
        generated_files: List of metadata dicts for files generated during the run
        cost: Cost breakdown dict (from CostBreakdown.to_dict()), None if model pricing unknown
        cumulative_usage: Summed token counts across all steps in the run
    """
    final_message: BetaMessage
    conversation_history: list[dict]
    stop_reason: str
    model: str
    usage: BetaUsage
    container_id: Optional[str] = None
    total_steps: int = 1
    agent_logs: Optional[list[dict]] = None
    generated_files: Optional[list[dict]] = None
    final_answer: str = ""
    cost: Optional[dict] = None
    cumulative_usage: Optional[dict] = None

    def __str__(self) -> str:
        """Return a JSON-formatted representation with all fields."""
        payload = {
            "final_message": self.final_message,
            "final_answer": self.final_answer,
            "conversation_history": self.conversation_history,
            "stop_reason": self.stop_reason,
            "model": self.model,
            "usage": self.usage,
            "container_id": self.container_id,
            "total_steps": self.total_steps,
            "agent_logs": self.agent_logs,
            "generated_files": self.generated_files,
            "cost": self.cost,
            "cumulative_usage": self.cumulative_usage,
        }
        return json.dumps(payload, indent=2, default=str)