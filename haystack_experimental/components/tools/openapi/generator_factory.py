# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ChatGeneratorDescriptor:
    """
    Dataclass to describe a Chat Generator
    """

    class_path: str
    patterns: List[re.Pattern]
    name: str
    model_name: str


class ChatGeneratorDescriptorManager:
    """
    Class to manage Chat Generator Descriptors
    """

    def __init__(self):
        self._descriptors: Dict[str, ChatGeneratorDescriptor] = {}
        self._register_default_descriptors()

    def _register_default_descriptors(self):
        """
        Register default Chat Generator Descriptors.
        """
        default_descriptors = [
            ChatGeneratorDescriptor(
                class_path="haystack.components.generators.chat.openai.OpenAIChatGenerator",
                patterns=[re.compile(r"^gpt.*")],
                name="openai",
                model_name="gpt-3.5-turbo",
            ),
            ChatGeneratorDescriptor(
                class_path="haystack_integrations.components.generators.anthropic.AnthropicChatGenerator",
                patterns=[re.compile(r"^claude.*")],
                name="anthropic",
                model_name="claude-1",
            ),
            ChatGeneratorDescriptor(
                class_path="haystack_integrations.components.generators.cohere.CohereChatGenerator",
                patterns=[re.compile(r"^command-r.*")],
                name="cohere",
                model_name="command-r",
            ),
        ]

        for descriptor in default_descriptors:
            self.register_descriptor(descriptor)

    def _load_class(self, full_class_path: str):
        """
        Load a class from a string representation of its path e.g. "module.submodule.class_name"
        """
        module_path, _, class_name = full_class_path.rpartition(".")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def register_descriptor(self, descriptor: ChatGeneratorDescriptor):
        """
        Register a new Chat Generator Descriptor.
        """
        if descriptor.name in self._descriptors:
            raise ValueError(f"Descriptor {descriptor.name} already exists.")

        self._descriptors[descriptor.name] = descriptor

    def _infer_descriptor(self, model_name: str) -> Optional[ChatGeneratorDescriptor]:
        """
        Infer the descriptor based on the model name.
        """
        for descriptor in self._descriptors.values():
            if any(pattern.match(model_name) for pattern in descriptor.patterns):
                return descriptor
        return None

    def create_generator(
        self, model_name: str, descriptor_name: Optional[str] = None, **model_kwargs
    ) -> Tuple[ChatGeneratorDescriptor, Any]:
        """
        Create ChatGenerator instance based on the model name and descriptor.
        """
        if descriptor_name:
            descriptor = self._descriptors.get(descriptor_name)
            if not descriptor:
                raise ValueError(f"Invalid descriptor name: {descriptor_name}")
        else:
            descriptor = self._infer_descriptor(model_name)
            if not descriptor:
                raise ValueError(
                    f"Could not infer descriptor for model name: {model_name}"
                )

        return descriptor, self._load_class(descriptor.class_path)(
            model=model_name, **(model_kwargs or {})
        )
