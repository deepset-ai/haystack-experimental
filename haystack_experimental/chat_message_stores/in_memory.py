# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any, Iterable, Optional

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole

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

        # We assign an ID to messages that don't have one yet. The ID simply corresponds to the index in the array.
        counter = self.count_messages(index)
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
        existing_ids = {msg.meta["chat_message_id"] for msg in self.retrieve_messages(index)}
        messages_to_write = [
            message for message in messages_with_id if message.meta["chat_message_id"] not in existing_ids
        ]

        for message in messages_to_write:
            if index not in _STORAGES:
                _STORAGES[index] = []
            _STORAGES[index].append(message)

        return len(messages)

    def retrieve_messages(self, index: str, last_k: Optional[int] = None) -> list[ChatMessage]:
        """
        Retrieves all stored chat messages.

        :param index:
            The index from which to retrieve messages.
        :param last_k:
            The number of last messages to retrieve. If unspecified, the last_k parameter passed
            to the constructor will be used.

        :returns: A list of chat messages.
        :raises ValueError:
            If last_k is not None and is less than 0.
        """

        if last_k is not None and last_k < 0:
            raise ValueError("last_k must be 0 or greater")

        resolved_last_k = last_k or self.last_k
        if resolved_last_k == 0:
            return []

        messages = _STORAGES.get(index, [])
        if resolved_last_k is not None:
            messages = self._get_last_k_messages(messages=messages, last_k=resolved_last_k)

        return messages

    @staticmethod
    def _get_last_k_messages(messages: list[ChatMessage], last_k: int) -> list[ChatMessage]:
        """
        Get the last_k messages from the list of messages while ensuring the chat history remains valid.

        By valid we mean that if there are ToolCalls in the chat history, we want to ensure that we do not slice
        in a way that a ToolCall is present without its corresponding ToolOutput.

        We handle this by treating ToolCalls and its corresponding ToolOutput(s) as a single unit when slicing the chat
        history.

        :param messages:
            The list of chat messages.
        :param last_k:
            The number of last messages to retrieve.
        :returns:
            The sliced list of chat messages.
        """
        # If ToolCalls are present we try to keep pairs of ToolCalls + ToolOutputs together to ensure a valid
        # chat history. We only slice at indices where the number of ToolCalls matches the number of ToolOutputs.
        has_tool_calls = any(msg.tool_call is not None for msg in messages)
        if has_tool_calls:
            valid_start_indices = []
            for start_idx in range(len(messages)):
                sliced_messages = messages[start_idx:]
                num_tool_calls_in_messages = sum(len(msg.tool_calls) for msg in sliced_messages)
                num_tool_outputs_in_messages = sum(len(msg.tool_call_results) for msg in sliced_messages)
                if num_tool_calls_in_messages == num_tool_outputs_in_messages:
                    valid_start_indices.append(start_idx)

            # If no valid start index is found we return the entire history
            # This should only occur if the chat history consists of only tool call messages
            if not valid_start_indices:
                valid_start_indices = [0]

            new_start_index = valid_start_indices[-last_k] if len(valid_start_indices) >= last_k else 0
            messages = messages[new_start_index:]
        else:
            messages = messages[-last_k:]
        return messages

    def delete_messages(self, index: str) -> None:
        """
        Deletes all stored chat messages.

        :param index:
            The index from which to delete messages.
        """
        _STORAGES.pop(index, None)

    def delete_all_messages(self) -> None:
        """
        Deletes all stored chat messages from all indices.
        """
        _STORAGES.clear()
