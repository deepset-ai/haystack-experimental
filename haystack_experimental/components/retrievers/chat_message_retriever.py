# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from haystack import DeserializationError, component, default_from_dict, default_to_dict, logging
from haystack.core.serialization import import_class_by_name
from haystack.dataclasses import ChatMessage, ChatRole

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
    message_store.write_messages(chat_history_id="user_456_session_123", messages=messages)
    retriever = ChatMessageRetriever(message_store)

    result = retriever.run(chat_history_id="user_456_session_123")

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
        self,
        chat_history_id: str,
        *,
        last_k: Optional[int] = None,
        current_messages: Optional[list[ChatMessage]] = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Run the ChatMessageRetriever

        :param chat_history_id:
            A unique identifier for the chat session or conversation whose messages should be retrieved.
            Each `chat_history_id` corresponds to a distinct chat history stored in the underlying ChatMessageStore.
            For example, use a session ID or conversation ID to isolate messages from different chat sessions.
        :param last_k: The number of last messages to retrieve. This parameter takes precedence over the last_k
            parameter passed to the ChatMessageRetriever constructor. If unspecified, the last_k parameter passed
            to the constructor will be used.
        :param current_messages:
            A list of incoming chat messages to combine with the retrieved messages. System messages from this list
            are prepended before the retrieved history, while all other messages (e.g., user messages) are appended
            after. This is useful for including new conversational context alongside stored history so the output
            can be directly used as input to a ChatGenerator or an Agent. If not provided, only the stored messages
            will be returned.

        :returns:
            A dictionary with the following key:
            - `messages` - The retrieved chat messages combined with any provided current messages.
        :raises ValueError: If last_k is not None and is less than 0.
        """
        if last_k is not None and last_k < 0:
            raise ValueError("last_k must be 0 or greater")

        resolved_last_k = last_k or self.last_k
        if resolved_last_k == 0:
            return {"messages": current_messages or []}

        retrieved_messages = self.chat_message_store.retrieve_messages(
            chat_history_id=chat_history_id, last_k=last_k or self.last_k
        )

        if not current_messages:
            return {"messages": retrieved_messages}

        # We maintain the order: system messages first, then stored messages, then new user messages
        system_messages = [msg for msg in current_messages if msg.is_from(ChatRole.SYSTEM)]
        other_messages = [msg for msg in current_messages if not msg.is_from(ChatRole.SYSTEM)]
        return {"messages": system_messages + retrieved_messages + other_messages}
