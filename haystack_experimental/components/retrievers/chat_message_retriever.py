# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import DeserializationError, component, default_from_dict, default_to_dict, logging
from haystack.core.serialization import import_class_by_name
from haystack.dataclasses import ChatMessage

from haystack_experimental.chat_message_stores.types import ChatMessageStore

logger = logging.getLogger(__name__)


@component
class ChatMessageRetriever:
    """
    Retrieves chat messages from the underlying ChatMessageStore.

    Usage example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_experimental.components.retrievers import ChatMessageRetriever
    from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore

    messages = [
        ChatMessage.from_assistant("Hello, how can I help you?"),
        ChatMessage.from_user("Hi, I have a question about Python. What is a Protocol?"),
    ]

    message_store = InMemoryChatMessageStore()
    message_store.write_messages(messages)
    retriever = ChatMessageRetriever(message_store)

    result = retriever.run()

    print(result["messages"])
    ```
    """

    def __init__(self, message_store: ChatMessageStore):
        """
        Create the ChatMessageRetriever component.

        :param message_store:
            An instance of a ChatMessageStore.
        """
        self.message_store = message_store

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        message_store = self.message_store.to_dict()
        return default_to_dict(self, message_store=message_store)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessageRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        if "message_store" not in init_params:
            raise DeserializationError("Missing 'message_store' in serialization data")
        if "type" not in init_params["message_store"]:
            raise DeserializationError("Missing 'type' in message store's serialization data")

        message_store_data = init_params["message_store"]
        try:
            message_store_class = import_class_by_name(message_store_data["type"])
        except ImportError as e:
            raise DeserializationError(f"Class '{message_store_data['type']}' not correctly imported") from e

        data["init_parameters"]["message_store"] = default_from_dict(message_store_class, message_store_data)
        return default_from_dict(cls, data)

    @component.output_types(messages=List[ChatMessage])
    def run(self, top_k: Optional[int] = None):
        """
        Run the ChatMessageRetriever

        :param top_k: The number of messages to retrieve. If None all messages will be retrieved.
        :returns:
            - `messages` - The retrieved chat messages.
        :raises ValueError: If top_k is not None and is less than 1
        """
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        return {"messages": self.message_store.retrieve()[-top_k:]}
