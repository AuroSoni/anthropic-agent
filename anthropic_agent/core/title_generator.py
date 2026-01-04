"""Title generation for agent conversations using LiteLLM."""

import json
import logging

import litellm

logger = logging.getLogger(__name__)

TITLE_SYSTEM_PROMPT = (
    "Generate a short, descriptive title (max 50 characters) for a conversation "
    "based on the user's first message. The title should capture the main topic "
    "or intent. Respond with only JSON in this format: {\"title\": \"...\"}"
)


async def generate_title(user_message: str, model: str = "openai/gpt-4o-mini") -> str:
    """Generate a conversation title using LLM.

    Args:
        user_message: The first user message to generate title from
        model: LiteLLM model identifier (default: gpt-4o-mini)

    Returns:
        Generated title (max 50 chars), or truncated user_message as fallback
    """
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": TITLE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            max_tokens=100,
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("title", "New Conversation")[:50]
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        # Fallback: truncate user message
        return user_message[:47] + "..." if len(user_message) > 50 else user_message

