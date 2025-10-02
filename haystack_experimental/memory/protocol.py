# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Protocol, runtime_checkable

from haystack.dataclasses.chat_message import ChatMessage


@runtime_checkable
class MemoryStore(Protocol):
    """
    Protocol for memory storage backends that can store and retrieve agent memories.

    This protocol defines the interface for pluggable memory stores that can be used
    with MemoryRetriever and MemoryWriter components. Memories are stored as ChatMessage
    objects with memory-specific metadata.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this store to a dictionary.

        :returns: Dictionary representation of the store configuration
        """

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryStore":
        """
        Deserializes the store from a dictionary.

        :param data: Dictionary containing store configuration
        :returns: Initialized MemoryStore instance
        """

    def add_memories(self, user_id: str, messages: list[ChatMessage]) -> list[str]:
        """
        Adds ChatMessage memories to the store.

        :param messages: List of ChatMessage objects with memory metadata
        :returns: List of memory IDs for the added messages
        """

    def filter_memories(self, user_id: str, filters: Optional[dict[Any, Any]]) -> list[str]:
        """
        Retrieve memories based on the filter.

        :param messages: List of ChatMessage objects with memory metadata
        :returns: List of memory IDs for the added messages
        """

    def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
    ) -> list[ChatMessage]:
        """
        Searches for memories matching the criteria.

        :param query: Text query to search for
        :param user_id: User identifier for scoping
        :param filters: Additional filters to apply
        :param top_k: Maximum number of results to return
        :returns: List of ChatMessage memories matching the criteria
        """

    def update_memories(self, messages: list[ChatMessage]) -> None:
        """
        Updates existing memory messages in the store.

        :param messages: List of ChatMessage memories to update (must have memory_id in meta)
        """

    def delete_all_memories(self, memory_ids: list[str]) -> int:
        """
        Deletes memory records by their IDs.

        :param memory_ids: List of memory IDs to delete
        :returns: Number of records actually deleted
        """
