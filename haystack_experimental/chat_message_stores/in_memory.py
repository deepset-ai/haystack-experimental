# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Iterable

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

# Global storage for all InMemoryDocumentStore instances, indexed by the index name.
_STORAGES: dict[str, list[ChatMessage]] = {}


class InMemoryChatMessageStore:
    """
    Stores chat messages in-memory.

    The `index` parameter is used as a unique identifier for each conversation or chat session.
    It acts as a namespace that isolates messages from different sessions. Each `index` value corresponds to a
    separate list of `ChatMessage` objects stored in memory.

    Typical usage involves providing a unique `index` (for example, a session ID or conversation ID)
    whenever you write, read, or delete messages. This ensures that chat messages from different
    conversations do not overlap.

    Usage example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore

    message_store = InMemoryChatMessageStore()

    messages = [
        ChatMessage.from_assistant("Hello, how can I help you?"),
        ChatMessage.from_user("Hi, I have a question about Python. What is a Protocol?"),
    ]
    message_store.write_messages(messages, index="user_456_session_123")
    retrieved_messages = message_store.retrieve(index="user_456_session_123")

    print(retrieved_messages)
    ```
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InMemoryChatMessageStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    def count_messages(self, index: str) -> int:
        """
        Returns the number of chat messages stored in this store.

        :param index:
            The index for which to count messages.

        :returns: The number of messages.
        """
        return len(_STORAGES.get(index, []))

    def write_messages(self, index: str, messages: list[ChatMessage]) -> int:
        """
        Writes chat messages to the ChatMessageStore.

        :param index:
            The index under which to store the messages.
        :param messages:
            A list of ChatMessages to write.

        :returns: The number of messages written.

        :raises ValueError: If messages is not a list of ChatMessages.
        """
        if not isinstance(messages, Iterable) or any(not isinstance(message, ChatMessage) for message in messages):
            raise ValueError("Please provide a list of ChatMessages.")

        for message in messages:
            if index not in _STORAGES:
                _STORAGES[index] = []
            _STORAGES[index].append(message)

        return len(messages)

    def delete_messages(self, index: str) -> None:
        """
        Deletes all stored chat messages.

        :param index:
            The index from which to delete messages.
        """
        _STORAGES.pop(index, None)

    def retrieve_messages(self, index: str) -> list[ChatMessage]:
        """
        Retrieves all stored chat messages.

        :param index:
            The index from which to retrieve messages.

        :returns: A list of chat messages.
        """

        messages = _STORAGES.get(index, [])

        return messages
