# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any, Iterable, Optional

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole

# Global storage for all InMemoryDocumentStore instances, indexed by the chat history id.
_STORAGES: dict[str, list[ChatMessage]] = {}


class InMemoryChatMessageStore:
    """
    Stores chat messages in-memory.

    The `chat_history_id` parameter is used as a unique identifier for each conversation or chat session.
    It acts as a namespace that isolates messages from different sessions. Each `chat_history_id` value corresponds to a
    separate list of `ChatMessage` objects stored in memory.

    Typical usage involves providing a unique `chat_history_id` (for example, a session ID or conversation ID)
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
    message_store.write_messages(chat_history_id="user_456_session_123", messages=messages)
    retrieved_messages = message_store.retrieve_messages(chat_history_id="user_456_session_123")

    print(retrieved_messages)
    ```
    """

    def __init__(self, skip_system_messages: bool = True, last_k: Optional[int] = 10) -> None:
        """
        Create an InMemoryChatMessageStore.

        :param skip_system_messages:
            Whether to skip storing system messages. Defaults to True.
        :param last_k:
            The number of last messages to retrieve. Defaults to 10 messages if not specified.
        """
        self.skip_system_messages = skip_system_messages
        self.last_k = last_k

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, skip_system_messages=self.skip_system_messages, last_k=self.last_k)

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

    def count_messages(self, chat_history_id: str) -> int:
        """
        Returns the number of chat messages stored in this store.

        :param chat_history_id:
            The chat history id for which to count messages.

        :returns: The number of messages.
        """
        return len(_STORAGES.get(chat_history_id, []))

    def write_messages(self, chat_history_id: str, messages: list[ChatMessage]) -> int:
        """
        Writes chat messages to the ChatMessageStore.

        :param chat_history_id:
            The chat history id under which to store the messages.
        :param messages:
            A list of ChatMessages to write.

        :returns: The number of messages written.

        :raises ValueError: If messages is not a list of ChatMessages.
        """
        if not isinstance(messages, Iterable) or any(not isinstance(message, ChatMessage) for message in messages):
            raise ValueError("Please provide a list of ChatMessages.")

        # We assign an ID to messages that don't have one yet. The ID simply corresponds to the chat_history_id in the
        # array.
        counter = self.count_messages(chat_history_id)
        messages_with_id = []
        for msg in messages:
            # Skip system messages if configured to do so
            if self.skip_system_messages and msg.is_from(ChatRole.SYSTEM):
                continue

            chat_message_id = msg.meta.get("chat_message_id")
            if chat_message_id is None:
                # We use replace to avoid mutating the original message
                msg = replace(msg, _meta={"chat_message_id": str(counter), **msg.meta})
                counter += 1

            messages_with_id.append(msg)

        # For now, we always skip messages that are already stored based on their ID.
        existing_messages = _STORAGES.get(chat_history_id, [])
        existing_ids = {
            msg.meta.get("chat_message_id") for msg in existing_messages if msg.meta.get("chat_message_id") is not None
        }
        messages_to_write = [
            message for message in messages_with_id if message.meta["chat_message_id"] not in existing_ids
        ]

        for message in messages_to_write:
            if chat_history_id not in _STORAGES:
                _STORAGES[chat_history_id] = []
            _STORAGES[chat_history_id].append(message)

        return len(messages_to_write)

    def retrieve_messages(self, chat_history_id: str, last_k: Optional[int] = None) -> list[ChatMessage]:
        """
        Retrieves all stored chat messages.

        :param chat_history_id:
            The chat history id from which to retrieve messages.
        :param last_k:
            The number of last messages to retrieve. If unspecified, the last_k parameter passed
            to the constructor will be used.

        :returns: A list of chat messages.
        :raises ValueError:
            If last_k is not None and is less than 0.
        """

        if last_k is not None and last_k < 0:
            raise ValueError("last_k must be 0 or greater")

        resolved_last_k = last_k if last_k is not None else self.last_k
        if resolved_last_k == 0:
            return []

        messages = _STORAGES.get(chat_history_id, [])
        if resolved_last_k is not None:
            messages = self._get_last_k_messages(messages=messages, last_k=resolved_last_k)

        return messages

    @staticmethod
    def _get_last_k_messages(messages: list[ChatMessage], last_k: int) -> list[ChatMessage]:
        """
        Get the last_k rounds of messages from the incoming list of messages.

        This is done in such a way such the returned list of messages is always valid. By valid we mean it will
        be submittable to an LLM without causing issues. For example, we want to avoid slicing the chat history in a
        way that a ToolCall is present without its corresponding ToolOutput.

        This is handled by treating ToolCalls and its corresponding ToolOutput(s) as a single unit when slicing the chat
        history.

        :param messages:
            List of chat messages.
        :param last_k:
            The number of last rounds of messages to retrieve. By rounds of messages we mean pairs of
            User -> Assistant messages. ToolCalls and ToolOutputs are considered part of the Assistant message.
        :returns:
            The sliced list of chat messages.
        """
        rounds = []
        current = []

        for msg in messages:
            # Treat system messages as separate rounds
            if msg.role == "system":
                rounds.append([msg])
                continue

            # User messages always start a new round
            if msg.role == "user":
                current.append(msg)
                continue

            # Assistant messages can either end a round or continue it (in case of tool calls)
            if msg.role == "assistant":
                current.append(msg)
                if msg.text and not msg.tool_calls:
                    rounds.append(current)
                    current = []
                continue

            # Append all other messages (e.g., tool outputs) to the current round
            current.append(msg)

        # Catch any remaining messages in the current round
        if current:
            rounds.append(current)

        selected = rounds[-last_k:]
        return [m for r in selected for m in r]

    def delete_messages(self, chat_history_id: str) -> None:
        """
        Deletes all stored chat messages.

        :param chat_history_id:
            The chat history id from which to delete messages.
        """
        _STORAGES.pop(chat_history_id, None)

    def delete_all_messages(self) -> None:
        """
        Deletes all stored chat messages from all chat history ids.
        """
        _STORAGES.clear()
