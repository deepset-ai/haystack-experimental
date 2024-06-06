from enum import Enum


class LLMProvider(Enum):
    """
    Enum for the supported LLM providers.
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
