# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import DeserializationError, component, default_from_dict, default_to_dict
from haystack.core.serialization import import_class_by_name
from haystack.dataclasses import ChatMessage

from haystack_experimental.chat_message_stores.types import ChatMessageStore


@component
class ChatMessageWriter:
    """
    Writes chat messages to an underlying ChatMessageStore.

    Usage example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_experimental.components.writers import ChatMessageWriter
    from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore

    messages = [
        ChatMessage.from_assistant("Hello, how can I help you?"),
        ChatMessage.from_user("I have a question about Python."),
    ]
    message_store = InMemoryChatMessageStore()
    writer = ChatMessageWriter(message_store)
    writer.run(messages)
    ```
    """

    def __init__(self, message_store: ChatMessageStore) -> None:
        """
        Create a ChatMessageWriter component.

        :param message_store:
            The ChatMessageStore where the chat messages are to be written.
        """
        self.message_store = message_store

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, message_store=self.message_store.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessageWriter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.

        :raises DeserializationError:
            If the message store is not properly specified in the serialization data or its type cannot be imported.
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

    @component.output_types(messages_written=int)
    def run(self, index: str, messages: list[ChatMessage]) -> dict[str, int]:
        """
        Run the ChatMessageWriter on the given input data.

        :param index:
            A unique identifier for the chat session or conversation whose messages should be retrieved.
            Each `index` corresponds to a distinct chat history stored in the underlying ChatMessageStore.
            For example, use a session ID or conversation ID to isolate messages from different chat sessions.
        :param messages:
            A list of chat messages to write to the store.

        :returns:
            - `messages_written`: Number of messages written to the ChatMessageStore.
        """
        if index is None:
            return {"messages_written": 0}

        messages_written = self.message_store.write_messages(index=index, messages=messages)
        return {"messages_written": messages_written}
