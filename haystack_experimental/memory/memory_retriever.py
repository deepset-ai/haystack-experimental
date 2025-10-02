# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses.chat_message import ChatMessage

from haystack_experimental.memory.mem0_store import Mem0MemoryStore


@component
class MemoryRetriever:
    """
    Retrieves relevant memories from a MemoryStore before agent interactions.

    This component searches for memories that are relevant to the current query
    and returns them as ChatMessage objects that can be used to provide context
    to agents for more personalized responses.

    Usage example:
    ```python
    from haystack.components.memory import MemoryRetriever
    from haystack.components.memory.mem0_store import Mem0MemoryStore

    memory_store = Mem0MemoryStore(api_key="your-api-key")
    retriever = MemoryRetriever(memory_store=memory_store, top_k=5)

    result = retriever.run(
        query="What's my timezone preference?",
        user_id="user_123"
    )
    print(result["memories"])  # List of ChatMessage objects with memory metadata
    ```

    :param memory_store: The memory store to retrieve memories from
    :param top_k: Maximum number of memories to retrieve
    :param default_filters: Default filters to apply to all searches
    """

    def __init__(
        self,
        memory_store: Mem0MemoryStore,
        top_k: int = 10,
        default_filters: Optional[dict[str, Any]] = None,
    ):
        self.memory_store = memory_store
        self.top_k = top_k
        self.default_filters = default_filters or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            memory_store=self.memory_store.to_dict(),
            top_k=self.top_k,
            default_filters=self.default_filters,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRetriever":
        """Deserialize the component from a dictionary."""

        # TODO: Deserialize memory store based on type
        # This would need proper import logic based on store type

        return default_from_dict(cls, data)

    @component.output_types(memories=list[ChatMessage])
    def run(
        self,
        query: Optional[str] = None,
        user_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Retrieve relevant memories from the memory store.

        :param query: Text query to search for relevant memories
        :param user_id: User identifier for scoping the search
        :param org_id: Organization identifier for scoping the search
        :param session_id: Session identifier for scoping the search
        :param filters: Additional filters to apply to the search
        :param top_k: Maximum number of memories to retrieve (overrides default)

        :returns: Dictionary with "memories" key containing list of ChatMessage objects
        """
        # Merge filters
        combined_filters = {**self.default_filters}
        if filters:
            combined_filters.update(filters)

        # Search for memories directly
        memories = self.memory_store.search(
            query=query,
            user_id=user_id,
            filters=combined_filters if combined_filters else None,
            top_k=top_k or self.top_k,
        )

        return {"memories": memories}
