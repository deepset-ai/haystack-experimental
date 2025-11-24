# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Optional

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses.chat_message import ChatMessage
from haystack.lazy_imports import LazyImport

from .utils import Mem0Scope

with LazyImport(message="Run 'pip install mem0ai'") as mem0_import:
    from mem0 import Memory, MemoryClient


class Mem0MemoryStore:
    """
    A memory store implementation using Mem0 as the backend.

    """

    def __init__(
        self,
        scope: Mem0Scope,
        api_key: Optional[str] = None,
        memory_config: Optional[dict[str, Any]] = None,
        search_criteria: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the Mem0 memory store.

        :param scope: The scope for the memory store. This defines the identifiers to retrieve or update memories.
        :param api_key: The Mem0 API key (if not provided, uses MEM0_API_KEY environment variable)
        :param memory_config: Configuration dictionary for Mem0 client
        :param search_criteria: Set the search configuration for the memory store.
            This can include query, filters, and top_k.
        """

        mem0_import.check()
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        if not self.api_key:
            raise ValueError("Mem0 API key must be provided either as parameter or MEM0_API_KEY environment variable.")

        self.scope = scope
        self.memory_config = memory_config

        # If a memory config is provided, use it to initialize the Mem0 client
        if self.memory_config:
            self.client = Memory.from_config(self.memory_config)
        else:
            self.client = MemoryClient(
                api_key=self.api_key,
            )

        # User can set the search criteria using the set_search_criteria method
        self.search_criteria = search_criteria
        if not self.search_criteria:
            self.search_criteria = {
                "query": None,
                "filters": None,
                "top_k": 10,
            }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store configuration to a dictionary."""
        return default_to_dict(
            self,
            scope=self.scope,
            api_key=self.api_key,
            memory_config=self.memory_config,
            search_criteria=self.search_criteria,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryStore":
        """Deserialize the store from a dictionary."""
        return default_from_dict(cls, data)

    def add_memories(self, messages: list[ChatMessage], infer: bool = True) -> list[str]:
        """
        Add ChatMessage memories to Mem0.

        :param messages: List of ChatMessage objects with memory metadata
        :returns: List of memory IDs for the added messages
        """
        added_ids = []

        for message in messages:
            if not message.text:
                continue
            mem0_message = [{"role": "user", "content": message.text}]

            try:
                result = self.client.add(
                    messages=mem0_message, metadata=message.meta, infer=infer, **self.scope.get_scope()
                )
                # Mem0 returns different response formats, handle both
                memory_id = result.get("id") or result.get("memory_id") or str(result)
                added_ids.append(memory_id)
            except Exception as e:
                raise RuntimeError(f"Failed to add memory message: {e}") from e

        return list(added_ids)

    def search_memories(
        self,
        query: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
        search_criteria: Optional[dict[str, Any]] = None,
    ) -> list[ChatMessage]:
        """
        Search for memories in Mem0.

        :param query: Text query to search for. If not provided, all memories will be returned.
        :param filters: Additional filters to apply on search. For more details on mem0 filters, see https://mem0.ai/docs/search/
        :param top_k: Maximum number of results to return
        :param search_criteria: Search criteria to search memories from the store.
            This can include query, filters, and top_k.
        :returns: List of ChatMessage memories matching the criteria
        """
        # Prepare filters for Mem0
        search_criteria = search_criteria or self.search_criteria

        search_query = query or search_criteria.get("query", None)
        search_filters = filters or search_criteria.get("filters", {})
        search_top_k = top_k or search_criteria.get("top_k", 10)

        if search_filters:
            mem0_filters = search_filters
        else:
            mem0_filters = self.scope.get_scope()

        try:
            if not search_query:
                memories = self.client.get_all(filters=mem0_filters, **self.scope.get_scope())
            else:
                memories = self.client.search(
                    query=search_query, limit=search_top_k, filters=mem0_filters, **self.scope.get_scope()
                )
            messages = [
                ChatMessage.from_user(text=memory["memory"], meta=memory["metadata"]) for memory in memories["results"]
            ]

            return messages

        except Exception as e:
            raise RuntimeError(f"Failed to search memories: {e}") from e

    # mem0 doesn't allow passing filter to delete endpoint,
    # we can delete all memories for a user by passing the user_id
    def delete_all_memories(self, user_id: Optional[str] = None):
        """
        Delete memory records from Mem0.

        :param user_id: User identifier for scoping the deletion
        """
        try:
            self.client.delete_all(**self.scope.get_scope())
        except Exception as e:
            raise RuntimeError(f"Failed to delete memories for user {user_id}: {e}") from e

    def delete_memory(self, memory_id: str):
        """
        Delete memory from Mem0.

        :param memory_id: The ID of the memory to delete.
        """
        try:
            self.client.delete(memory_id=memory_id)
        except Exception as e:
            raise RuntimeError(f"Failed to delete memory {memory_id}: {e}") from e
