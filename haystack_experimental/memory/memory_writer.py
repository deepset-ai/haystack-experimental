# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.components.memory.protocol import MemoryStore
from haystack.dataclasses.chat_message import ChatMessage


@component
class MemoryWriter:
    """
    Writes chat messages to a MemoryStore for long-term memory.

    This component processes chat messages and stores them as memories with
    appropriate metadata that can be retrieved later for personalized agent interactions.

    Usage example:
    ```python
    from haystack.components.memory import MemoryWriter
    from haystack.components.memory.mem0_store import Mem0MemoryStore
    from haystack.dataclasses import ChatMessage

    memory_store = Mem0MemoryStore(api_key="your-api-key")
    writer = MemoryWriter(memory_store=memory_store)

    # Create messages with identifiers in metadata
    messages = [
        ChatMessage.from_user(
            "I prefer dark mode",
            meta={
                "user_id": "user_123",
                "org_id": "org_456",
                "session_id": "session_789",
                "memory_type": "semantic"
            }
        ),
        ChatMessage.from_assistant(
            "I'll remember your preference",
            meta={
                "user_id": "user_123",
                "org_id": "org_456",
                "session_id": "session_789",
                "memory_type": "episodic"
            }
        )
    ]

    # Write memories - identifiers are extracted from message metadata
    result = writer.run(
        messages=messages,
        metadata={"conversation_id": "conv_001"}  # Optional additional metadata
    )
    print(result["memories_written"])  # Number of memories written
    ```

    :param memory_store: The memory store to write memories to
    """

    def __init__(
        self,
        memory_store: MemoryStore,
    ):
        self.memory_store = memory_store

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            memory_store=self.memory_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryWriter":
        """Deserialize the component from a dictionary."""

    @component.output_types(memories_written=int)
    def run(
        self,
        messages: list[ChatMessage],
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Write chat messages as memories to the memory store.

        The component extracts user_id, org_id, and session_id from each ChatMessage's
        metadata to properly scope the memories in the store.

        :param messages: List of chat messages to store as memories. Each message should
                        have user_id, org_id, and/or session_id in its metadata for proper scoping.
        :param metadata: Additional metadata to attach to all memories

        :returns: Dictionary with "memories_written" key containing the number of memories written
        """
        # Process messages and add any additional metadata
        processed_messages = []

        for message in messages:
            updated_meta = {**message.meta}

            # Add any additional metadata
            if metadata:
                updated_meta.update(metadata)

            # Create new message with updated metadata (if any additional metadata was provided)
            if metadata:
                processed_message = ChatMessage(
                    role=message.role,
                    content=message.content,
                    meta=updated_meta,
                    name=message.name,
                    tool_calls=message.tool_calls,
                    tool_call_result=message.tool_call_result,
                )
                processed_messages.append(processed_message)
            else:
                # Use original message if no additional metadata
                processed_messages.append(message)

        # Write memories to store
        try:
            added_ids = self.memory_store.add(processed_messages)
            return {"memories_written": len(added_ids)}
        except Exception as e:
            raise RuntimeError(f"Failed to write memories: {e}") from e
