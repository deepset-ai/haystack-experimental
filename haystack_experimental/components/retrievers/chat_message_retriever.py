# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from haystack import DeserializationError, component, default_from_dict, default_to_dict, logging
from haystack.core.serialization import import_class_by_name
from haystack.dataclasses import ChatMessage, ChatRole

from haystack_experimental.chat_message_stores.types import ChatMessageStore
from haystack_experimental.chat_message_stores.utils import get_last_k_messages

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

    def __init__(self, chat_message_store: ChatMessageStore, last_k: Optional[int] = 10):
        """
        Create the ChatMessageRetriever component.

        :param chat_message_store:
            An instance of a ChatMessageStore.
        :param last_k:
            The number of last messages to retrieve. Defaults to 10 messages if not specified.
        """
        self.chat_message_store = chat_message_store
        if last_k and last_k <= 0:
            raise ValueError(f"last_k must be greater than 0. Currently, last_k is {last_k}")
        self.last_k = last_k

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, chat_message_store=self.chat_message_store.to_dict(), last_k=self.last_k)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessageRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        if "chat_message_store" not in init_params:
            raise DeserializationError("Missing 'chat_message_store' in serialization data")
        if "type" not in init_params["chat_message_store"]:
            raise DeserializationError("Missing 'type' in message store's serialization data")

        message_store_data = init_params["chat_message_store"]
        try:
            message_store_class = import_class_by_name(message_store_data["type"])
        except ImportError as e:
            raise DeserializationError(f"Class '{message_store_data['type']}' not correctly imported") from e
        if not hasattr(message_store_class, "from_dict"):
            raise DeserializationError(f"{message_store_class} does not have from_dict method implemented.")
        init_params["chat_message_store"] = message_store_class.from_dict(message_store_data)

        return default_from_dict(cls, data)

    @component.output_types(messages=list[ChatMessage])
    def run(
        self, index: str, *, last_k: Optional[int] = None, new_messages: Optional[list[ChatMessage]] = None
    ) -> dict[str, list[ChatMessage]]:
        """
        Run the ChatMessageRetriever

        :param index:
            A unique identifier for the chat session or conversation whose messages should be retrieved.
            Each `index` corresponds to a distinct chat history stored in the underlying ChatMessageStore.
            For example, use a session ID or conversation ID to isolate messages from different chat sessions.
        :param last_k: The number of last messages to retrieve. This parameter takes precedence over the last_k
            parameter passed to the ChatMessageRetriever constructor. If unspecified, the last_k parameter passed
            to the constructor will be used.
        :param new_messages:
            A list of new chat messages to append to the retrieved messages. This is useful for retrieving the current
            chat history and appending new messages (e.g. user messages) so the output can be directly used as input
            to a ChatGenerator or an Agent.

        :returns:
            - `messages` - The retrieved chat messages and optionally the new messages appended if provided.
        :raises ValueError: If last_k is not None and is less than 1
        """
        if index is None:
            return {"messages": new_messages or []}

        if last_k is not None and last_k <= 0:
            raise ValueError("last_k must be greater than 0")

        messages = self.chat_message_store.retrieve_messages(index=index, last_k=last_k or self.last_k)

        if not new_messages:
            return {"messages": messages}

        # We maintain the order: system messages first, then stored messages, then new user messages
        system_messages = [msg for msg in new_messages if msg.is_from(ChatRole.SYSTEM)]
        other_messages = [msg for msg in new_messages if not msg.is_from(ChatRole.SYSTEM)]
        return {"messages": system_messages + messages + other_messages}
