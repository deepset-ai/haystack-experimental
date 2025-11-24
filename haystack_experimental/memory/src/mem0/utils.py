from typing import Any, Optional

from haystack import default_from_dict


class Mem0Scope:
    """
    This object is used to scope the memory store.
    """

    def __init__(self, user_id: Optional[str] = None, run_id: Optional[str] = None, session_id: Optional[str] = None):
        self.user_id = user_id
        self.run_id = run_id
        self.session_id = session_id

        self._check_one_id_is_set()

    def _check_one_id_is_set(self):
        # we need to check that at least one of the ids is set
        if not self.user_id and not self.run_id and not self.session_id:
            raise ValueError("At least one of user_id, run_id, or session_id must be set")

    def _to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "Mem0Scope":
        return default_from_dict(cls, data)

    def _get_scope(self) -> dict[str, Any]:
        return {key: value for key, value in self.to_dict().items() if value is not None}


SEMANTIC_MEMORY_EXTRACTION_PROMPT = """
You are extracting semantic memories from a conversation chain. Semantic memories are general
preferences or personality traits or facts about the user that occurred during the conversation.


WHAT TO EXTRACT:
- General preferences or personality traits
- Facts about the user


WHAT NOT TO EXTRACT:
- Vague statements without concrete facts
- Specific events or actions that occurred during the conversation
- Contextual details from the conversation flow
- Problem-solution pairs that emerged during the conversation
- Questions without answers
- Speculative or hypothetical statements


OUTPUT FORMAT:
You MUST return a JSON object with ONLY a "facts" key containing an array of strings.
Each fact should be a string that contains fact about the user.

{
    "facts": [
        "User likes to listen to Russian pop music"
    ]
}

IMPORTANT:
- Return JSON with ONLY the "facts" key
- Each fact must be a STRING, not an object
- If no semantic memories can be extracted, return {"facts": []}

EXAMPLES:

Example 1:
Conversation:
USER: I like to listen to Russian pop music
ASSISTANT: That's interesting! What other types of music do you like?
USER: I also like to listen to jazz music
ASSISTANT: Jazz is great! Do you have a favorite jazz artist?
USER: Yes, I like to listen to John Coltrane

Extracted semantic memory:
{
    "facts": [
        "User likes to listen to Russian pop music"
        "User likes to listen to jazz music"
        "User likes to listen to John Coltrane"
    ]
}

Example 2:
Conversation:
USER: I live in Florence Italy and I love mountains.
ASSISTANT: That sounds exciting! Which countries are you planning to visit?
USER: I'm thinking of France, Italy, and Spain. I've already booked flights to Paris for next month.
ASSISTANT: Great itinerary! Paris is beautiful. Have you been to Europe before?
USER: Yes, I visited London last year for a conference. It was my first time in Europe.

Extracted semantic memory:
{
    "facts": [
        "User lives in Florence Italy and loves mountains"
    ]
}

Example 3 (Empty Output):
Conversation:
USER: Hello, how are you?
ASSISTANT: I'm doing well, thank you! How can I help you today?
USER: Just checking in, nothing specific.

Extracted semantic memory:
{
    "facts": []
}


"""

semantic_memory_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4.1-nano-2025-04-14",
            "temperature": 0.2,
            "max_tokens": 2000,
        },
    },
    "custom_fact_extraction_prompt": SEMANTIC_MEMORY_EXTRACTION_PROMPT,
}
