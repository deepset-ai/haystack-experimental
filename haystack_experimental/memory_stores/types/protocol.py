# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Protocol

from haystack.dataclasses import ChatMessage

# Ellipsis are needed for the type checker, it's safe to disable module-wide
# pylint: disable=unnecessary-ellipsis


class MemoryStore(Protocol):
    """
    Stores ChatMessage-based memories to be used by agents and components.

    Implementations typically persist user- and agent-specific memories and
    support adding, searching, and deleting memories.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this memory store to a dictionary.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryStore":
        """
        Deserializes the memory store from a dictionary.
        """
        ...

    def add_memories(
        self,
        *,
        messages: list[ChatMessage],
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> list[str]:
        """
        Add ChatMessage memories to the store.

        :param messages: List of ChatMessage objects with memory metadata.
        :param user_id: User ID to scope memories.
        :param run_id: Run ID to scope memories.
        :param agent_id: Agent ID to scope memories.
        :returns: List of memory IDs for the added messages.
        """
        ...

    def search_memories(
        self,
        *,
        query: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        include_memory_metadata: bool = False,
    ) -> list[ChatMessage]:
        """
        Search for memories in the store.

        :param query: Text query to search for. If not provided, all memories may be returned.
        :param filters: Backend-specific filter structure.
        :param top_k: Maximum number of results to return.
        :param user_id: User ID to scope memories.
        :param run_id: Run ID to scope memories.
        :param agent_id: Agent ID to scope memories.
        :param include_memory_metadata: Whether to include backend-specific
            memory metadata in ChatMessage.meta.
        :returns: List of ChatMessage memories matching the criteria.
        """
        ...

    def delete_all_memories(
        self,
        *,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Delete all memories in the given scope.

        At least one of user_id, run_id, or agent_id should be set.
        """
        ...

    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a single memory by its ID.

        :param memory_id: The ID of the memory to delete.
        """
        ...
