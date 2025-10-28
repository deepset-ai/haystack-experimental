# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

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

    def __init__(self, message_store: ChatMessageStore, last_k: int = 10):
        """
        Create the ChatMessageRetriever component.

        :param message_store:
            An instance of a ChatMessageStore.
        :param last_k:
            The number of last messages to retrieve. Defaults to 10 messages if not specified.
        """
        self.message_store = message_store
        if last_k <= 0:
            raise ValueError(f"last_k must be greater than 0. Currently, the last_k is {last_k}")
        self.last_k = last_k

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, message_store=self.message_store.to_dict(), last_k=self.last_k)

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

    def _get_last_k_messages(self, messages: list[ChatMessage], last_k: int) -> list[ChatMessage]:
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
            if new_messages:
                return {"messages": new_messages}
            return {"messages": []}

        if last_k is not None and last_k <= 0:
            raise ValueError("last_k must be greater than 0")

        resolved_last_k = last_k or self.last_k

        messages = self.message_store.retrieve_messages(index)

        if resolved_last_k is not None:
            messages = self._get_last_k_messages(messages, resolved_last_k)

        if new_messages:
            messages.extend(new_messages)

        return {"messages": messages}
