# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses.chat_message import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

with LazyImport(message="Run 'pip install mem0ai'") as mem0_import:
    from mem0 import Memory, MemoryClient


class Mem0MemoryStore:
    """
    A memory store implementation using Mem0 as the backend.
    """

    def __init__(
        self,
        *,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        api_key: Secret = Secret.from_env_var("MEM0_API_KEY"),
        memory_config: Optional[dict[str, Any]] = None,
        search_criteria: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the Mem0 memory store.

        :param user_id: The user ID to to store and retrieve memories from the memory store.
        :param run_id: The run ID to to store and retrieve memories from the memory store.
        :param agent_id: The agent ID to to store and retrieve memories from the memory store.
        :param api_key: The Mem0 API key (if not provided, uses MEM0_API_KEY environment variable)
        :param memory_config: Configuration dictionary for Mem0 client
        :param search_criteria: Set the search configuration for the memory store.
            This can include query, filters, and top_k.
        """

        mem0_import.check()
        self.api_key = api_key
        self.user_id = user_id
        self.run_id = run_id
        self.agent_id = agent_id

        self._check_one_id_is_set()

        self.memory_config = memory_config

        # If a memory config is provided, use it to initialize the Mem0 client
        if self.memory_config:
            self.client = Memory.from_config(self.memory_config)
        else:
            self.client = MemoryClient(
                api_key=self.api_key.resolve_value(),
            )

        # We allow setting search criteria in init because
        # it's needed for use of memorystore in pipelines and agents
        self.search_criteria = search_criteria

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store configuration to a dictionary."""
        return default_to_dict(
            self,
            user_id=self.user_id,
            run_id=self.run_id,
            agent_id=self.agent_id,
            api_key=self.api_key.to_dict(),
            memory_config=self.memory_config,
            search_criteria=self.search_criteria,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryStore":
        """Deserialize the store from a dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])

        return default_from_dict(cls, data)

    def add_memories(self, messages: list[ChatMessage], infer: bool = True) -> list[str]:
        """
        Add ChatMessage memories to Mem0.

        :param messages: List of ChatMessage objects with memory metadata
        :param infer: Whether to infer facts from the messages. If False, the whole message will
            be added as a memory.
        :returns: List of memory IDs for the added messages
        """
        added_ids = []

        for message in messages:
            if not message.text:
                continue
            mem0_message = [{"content": message.text, "role": message.role}]
            mem0_metadata = message.meta
            # we save the role of the message in the metadata
            mem0_metadata.update({"role": message.role})

            try:
                result = self.client.add(messages=mem0_message, metadata=mem0_metadata, infer=infer, **self._get_ids())
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
        _include_memory_metadata: bool = False,
    ) -> list[ChatMessage]:
        """
        Search for memories in Mem0.

        :param query: Text query to search for. If not provided, all memories will be returned.
        :param filters: Additional filters to apply on search. For more details on mem0 filters, see https://mem0.ai/docs/search/
        :param top_k: Maximum number of results to return
        :param _include_memory_metadata: Whether to include the memory metadata in the ChatMessage
        :returns: List of ChatMessage memories matching the criteria
        """
        # Prepare filters for Mem0

        search_query = query or (self.search_criteria.get("query", None) if self.search_criteria else None)
        search_filters = filters or (self.search_criteria.get("filters", {}) if self.search_criteria else None)
        search_top_k = top_k or (self.search_criteria.get("top_k", 10) if self.search_criteria else 10)

        if search_filters:
            mem0_filters = search_filters
        else:
            ids = self._get_ids()
            if len(ids) == 1:
                mem0_filters = dict(ids)
            else:
                mem0_filters = {"AND": [{key: value} for key, value in ids.items()]}
        print(f"Mem0 filters: {mem0_filters}")
        try:
            if self.memory_config:
                memories = self.client.search(
                    query=search_query,
                    filters=search_filters,
                    user_id=self.user_id,
                    run_id=self.run_id,
                    agent_id=self.agent_id,
                )
            else:
                if not search_query:
                    memories = self.client.get_all(filters=mem0_filters)
                else:
                    memories = self.client.search(
                        query=search_query,
                        limit=search_top_k,
                        filters=mem0_filters,
                    )
            if _include_memory_metadata:
                for memory in memories["results"]:
                    meta = memory.copy()
                    meta.pop("memory")
                    messages = [
                        ChatMessage.from_user(text=memory["memory"], meta=meta) for memory in memories["results"]
                    ]
            else:
                messages = [
                    ChatMessage.from_user(text=memory["memory"], meta=memory["metadata"])
                    for memory in memories["results"]
                ]
            return messages

        except Exception as e:
            raise RuntimeError(f"Failed to search memories: {e}") from e

    # mem0 doesn't allow passing filter to delete endpoint,
    # we can delete all memories for a user by passing the user_id
    def delete_all_memories(
        self, user_id: Optional[str] = "", run_id: Optional[str] = "", agent_id: Optional[str] = ""
    ) -> None:
        """
        Delete memory records from Mem0.

        :param user_id: User identifier for scoping the deletion
        :param run_id: Run identifier for scoping the deletion
        :param agent_id: Agent identifier for scoping the deletion
        """

        user_id = user_id or self.user_id
        run_id = run_id or self.run_id
        agent_id = agent_id or self.agent_id

        try:
            self.client.delete_all(user_id=user_id, run_id=run_id, agent_id=agent_id)
            print(f"All memories deleted successfully for scope {user_id}, {run_id}, {agent_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to delete memories with scope {user_id}, {run_id}, {agent_id}: {e}") from e

    def delete_memory(self, memory_id: str) -> None:
        """
        Delete memory from Mem0.

        :param memory_id: The ID of the memory to delete.
        """
        try:
            self.client.delete(memory_id=memory_id)
            print(f"Memory {memory_id} deleted successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to delete memory {memory_id}: {e}") from e

    def _check_one_id_is_set(self) -> None:
        "Check that at least one of the ids is set."
        if not self.user_id and not self.run_id and not self.agent_id:
            raise ValueError("At least one of user_id, run_id, or agent_id must be set")

    def _get_ids(self) -> dict[str, Any]:
        """
        Return the set ids as a dictionary.
        """
        ids = {
            "user_id": self.user_id,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
        }
        return {key: value for key, value in ids.items() if value is not None}
