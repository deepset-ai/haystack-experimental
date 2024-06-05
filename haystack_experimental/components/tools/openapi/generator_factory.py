import importlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"


PROVIDER_DETAILS = {
    LLMProvider.OPENAI: {
        "class_path": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
        "patterns": [re.compile(r"^gpt.*")],
    },
    LLMProvider.ANTHROPIC: {
        "class_path": "haystack_integrations.components.generators.anthropic.AnthropicChatGenerator",
        "patterns": [re.compile(r"^claude.*")],
    },
    LLMProvider.COHERE: {
        "class_path": "haystack_integrations.components.generators.cohere.CohereChatGenerator",
        "patterns": [re.compile(r"^command-r.*")],
    },
}


def load_class(full_class_path: str):
    """
    Load a class from a string representation of its path e.g. "module.submodule.class_name"
    """
    module_path, _, class_name = full_class_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@dataclass
class LLMIdentifier:
    provider: LLMProvider
    model_name: str

    def __post_init__(self):
        if not isinstance(self.provider, LLMProvider):
            raise ValueError(f"Invalid provider: {self.provider}")

        if not isinstance(self.model_name, str):
            raise ValueError(f"Model name must be a string: {self.model_name}")

        details = PROVIDER_DETAILS.get(self.provider)
        if not details or not any(
            pattern.match(self.model_name) for pattern in details["patterns"]
        ):
            raise ValueError(
                f"Invalid combination of provider {self.provider} and model name {self.model_name}"
            )


def create_generator(
    model_name: str, provider: Optional[str] = None, **model_kwargs
) -> Tuple[LLMIdentifier, Any]:
    """
    Create ChatGenerator instance based on the model name and provider.
    """
    if provider:
        try:
            provider_enum = LLMProvider[provider.lower()]
        except KeyError:
            raise ValueError(f"Invalid provider: {provider}")
    else:
        provider_enum = None
        for prov, details in PROVIDER_DETAILS.items():
            if any(pattern.match(model_name) for pattern in details["patterns"]):
                provider_enum = prov
                break

        if provider_enum is None:
            raise ValueError(f"Could not infer provider for model name: {model_name}")

    llm_identifier = LLMIdentifier(provider=provider_enum, model_name=model_name)
    class_path = PROVIDER_DETAILS[llm_identifier.provider]["class_path"]
    return llm_identifier, load_class(class_path)(
        model=llm_identifier.model_name, **model_kwargs
    )
