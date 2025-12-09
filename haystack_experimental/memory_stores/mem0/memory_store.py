# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses.chat_message import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

MemoryClient = None  # type: ignore[assignment]

with LazyImport(message="Run 'pip install mem0ai'") as mem0_import:
    from mem0 import MemoryClient  # pylint: disable=import-error


class Mem0MemoryStore:
    """
    A memory store implementation using Mem0 as the backend.
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("MEM0_API_KEY"),
    ):
        """
        Initialize the Mem0 memory store.

        :param api_key: The Mem0 API key. You can also set it using `MEM0_API_KEY` environment variable.
        """

        mem0_import.check()
        self.api_key = api_key
        self.client = MemoryClient(
            api_key=self.api_key.resolve_value(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store configuration to a dictionary."""
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryStore":
        """Deserialize the store from a dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])

        return default_from_dict(cls, data)

    def add_memories(
        self,
        *,
        messages: list[ChatMessage],
        infer: bool = True,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        async_mode: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Add ChatMessage memories to Mem0.

        :param messages: List of ChatMessage objects with memory metadata
        :param infer: Whether to infer facts from the messages. If False, the whole message will
            be added as a memory.
        :param user_id: The user ID to to store and retrieve memories from the memory store.
        :param run_id: The run ID to to store and retrieve memories from the memory store.
        :param agent_id: The agent ID to to store and retrieve memories from the memory store.
            If you want Mem0 to store chat messages from the assistant, you need to set the agent_id.
        :param async_mode: Whether to add memories asynchronously.
            If True, the method will return immediately and the memories will be added in the background.
        :param kwargs: Additional keyword arguments to pass to the Mem0 client.add method.
            Note: ChatMessage.meta in the list of messages will be ignored because Mem0 doesn't allow
            passing metadata for each message in the list. You can pass metadata for the whole memory
            by passing the `metadata` keyword argument to the method.
        :returns: List of objects with the memory_id and the memory
        """
        added_ids = []
        ids = self._get_ids(user_id, run_id, agent_id)
        mem0_messages = []
        for message in messages:
            if not message.text:
                continue
            # we save the role of the message in the metadata
            mem0_messages.append({"content": message.text, "role": message.role.value})
        try:
            status = self.client.add(messages=mem0_messages, infer=infer, **ids, async_mode=async_mode, **kwargs)
            for result in status["results"]:
                memory_id = {"memory_id": result.get("id"), "memory": result["memory"]}
                added_ids.append(memory_id)
        except Exception as e:
            raise RuntimeError(f"Failed to add memory message: {e}") from e
        return added_ids

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
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Search for memories in Mem0.

        If filters are not provided, at least one of user_id, run_id, or agent_id must be set.
        If filters are provided, the search will be scoped to the provided filters and the other ids will be ignored.
        :param query: Text query to search for. If not provided, all memories will be returned.
        :param filters: Additional filters to apply on search. For more details on mem0 filters, see https://mem0.ai/docs/search/
        :param top_k: Maximum number of results to return
        :param user_id: The user ID to to store and retrieve memories from the memory store.
        :param run_id: The run ID to to store and retrieve memories from the memory store.
        :param agent_id: The agent ID to to store and retrieve memories from the memory store.
            If you want Mem0 to store chat messages from the assistant, you need to set the agent_id.
        :param include_memory_metadata: Whether to include the memory related metadata from the
            retrieved memory in the ChatMessage.
            If True, the metadata will include the mem0 related metadata i.e. memory_id, score, etc.
            in the `mem0_memory_metadata` key.
            If False, the `ChatMessage.meta` will only contain the user defined metadata.
        :param kwargs: Additional keyword arguments to pass to the Mem0 client.
            If query is passed, the kwargs will be passed to the Mem0 client.search method.
            If query is not passed, the kwargs will be passed to the Mem0 client.get_all method.
        :returns: List of ChatMessage memories matching the criteria
        """
        # Prepare filters for Mem0

        if filters:
            mem0_filters = filters
        else:
            ids = self._get_ids(user_id, run_id, agent_id)
            if len(ids) == 1:
                mem0_filters = dict(ids)
            else:
                mem0_filters = {"AND": [{key: value} for key, value in ids.items()]}
        try:
            if not query:
                memories = self.client.get_all(filters=mem0_filters, **kwargs)
            else:
                memories = self.client.search(
                    query=query,
                    top_k=top_k,
                    filters=mem0_filters,
                    **kwargs,
                )
            if include_memory_metadata:
                # we also include the mem0 related metadata i.e. memory_id, score, etc.
                # metadata
                for memory in memories["results"]:
                    meta = memory["metadata"].copy() if memory["metadata"] else {}
                    meta["retrieved_memory_metadata"] = memory.copy()
                    meta["retrieved_memory_metadata"].pop("memory")
                    messages = [
                        ChatMessage.from_system(text=memory["memory"], meta=meta) for memory in memories["results"]
                    ]
            else:
                # we only include the metadata stored in the memory in ChatMessage
                messages = [
                    ChatMessage.from_system(text=memory["memory"], meta=memory["metadata"])
                    for memory in memories["results"]
                ]
            return messages

        except Exception as e:
            raise RuntimeError(f"Failed to search memories: {e}") from e

    def search_memories_as_single_message(
        self,
        *,
        query: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """
        Search for memories in Mem0 and return a single ChatMessage object.

        If filters are not provided, at least one of user_id, run_id, or agent_id must be set.
        If filters are provided, the search will be scoped to the provided filters and the other ids will be ignored.
        :param query: Text query to search for. If not provided, all memories will be returned.
        :param filters: Additional filters to apply on search. For more details on mem0 filters, see https://mem0.ai/docs/search/
        :param top_k: Maximum number of results to return
        :param user_id: The user ID to to store and retrieve memories from the memory store.
        :param run_id: The run ID to to store and retrieve memories from the memory store.
        :param agent_id: The agent ID to to store and retrieve memories from the memory store.
            If you want Mem0 to store chat messages from the assistant, you need to set the agent_id.
        :param kwargs: Additional keyword arguments to pass to the Mem0 client.
            If query is passed, the kwargs will be passed to the Mem0 client.search method.
            If query is not passed, the kwargs will be passed to the Mem0 client.get_all method.
        :returns: A single ChatMessage object with the memories matching the criteria
        """
        # Prepare filters for Mem0
        if filters:
            mem0_filters = filters
        else:
            ids = self._get_ids(user_id, run_id, agent_id)
            if len(ids) == 1:
                mem0_filters = dict(ids)
            else:
                mem0_filters = {"AND": [{key: value} for key, value in ids.items()]}

        try:
            if not query:
                memories = self.client.get_all(filters=mem0_filters, **kwargs)
            else:
                memories = self.client.search(
                    query=query,
                    top_k=top_k,
                    filters=mem0_filters,
                    **kwargs,
                )

            # we combine the memories into a single string
            combined_memory = "\n".join(
                f"- MEMORY #{idx + 1}: {memory['memory']}" for idx, memory in enumerate(memories["results"])
            )

            return ChatMessage.from_system(text=combined_memory)

        except Exception as e:
            raise RuntimeError(f"Failed to search memories: {e}") from e

    # mem0 doesn't allow passing filter to delete endpoint,
    # we can delete all memories for a user by passing the user_id
    def delete_all_memories(
        self,
        *,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete memory records from Mem0.

        At least one of user_id, run_id, or agent_id must be set.
        :param user_id: The user ID to delete memories from.
        :param run_id: The run ID to delete memories from.
        :param agent_id: The agent ID to delete memories from.
        :param kwargs: Additional keyword arguments to pass to the Mem0 client.delete_all method.
        """
        ids = self._get_ids(user_id, run_id, agent_id)

        try:
            self.client.delete_all(**ids, **kwargs)
            print(f"All memories deleted successfully for scope {ids}")
        except Exception as e:
            raise RuntimeError(f"Failed to delete memories with scope {ids}: {e}") from e

    def delete_memory(self, memory_id: str, **kwargs: Any) -> None:
        """
        Delete memory from Mem0.

        :param memory_id: The ID of the memory to delete.
        :param kwargs: Additional keyword arguments to pass to the Mem0 client.delete method.
        """
        try:
            self.client.delete(memory_id=memory_id, **kwargs)
            print(f"Memory {memory_id} deleted successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to delete memory {memory_id}: {e}") from e

    def _get_ids(
        self, user_id: Optional[str] = None, run_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Check that at least one of the ids is set.

        Return the set ids as a dictionary.
        """
        if not user_id and not run_id and not agent_id:
            raise ValueError("At least one of user_id, run_id, or agent_id must be set")
        ids = {
            "user_id": user_id,
            "run_id": run_id,
            "agent_id": agent_id,
        }
        return {key: value for key, value in ids.items() if value is not None}
