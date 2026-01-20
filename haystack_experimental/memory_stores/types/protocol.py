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
        user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add ChatMessage memories to the store.

        :param messages: List of ChatMessage objects with memory metadata.
        :param user_id: User ID to scope memories.
        :param kwargs: Additional keyword arguments to pass to the add method.
        """
        ...

    def search_memories(
        self,
        *,
        query: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Search for memories in the store.

        :param query: Text query to search for. If not provided, all memories may be returned.
        :param filters: Haystack filters to apply on search.
        :param top_k: Maximum number of results to return.
        :param user_id: User ID to scope memories.
        :param kwargs: Additional keyword arguments to pass to the search method.

        :returns: List of ChatMessage memories matching the criteria.
        """
        ...

    def delete_all_memories(
        self,
        *,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete all memories in the given scope.

        In case of multiple optional ids, one of the ids must be set.
        :param user_id: User ID to delete memories.
        :param kwargs: Additional keyword arguments to pass to the delete method.
        """
        ...

    def delete_memory(self, memory_id: str, **kwargs: Any) -> None:
        """
        Delete a single memory by its ID.

        :param memory_id: The ID of the memory to delete.
        :param kwargs: Additional keyword arguments to pass to the delete method.
        """
        ...
