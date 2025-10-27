# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from haystack.dataclasses import ChatMessage

# Ellipsis are needed for the type checker, it's safe to disable module-wide
# pylint: disable=unnecessary-ellipsis


class ChatMessageStore(Protocol):
    """
    Stores ChatMessages to be used by the components of a Pipeline.

    Classes implementing this protocol might store ChatMessages either in durable storage or in memory. They might
    allow specialized components (e.g. retrievers) to perform retrieval on them, either by embedding, by keyword,
    hybrid, and so on, depending on the backend used.

    In order to write or retrieve chat messages, consider using a ChatMessageWriter or ChatMessageRetriever.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this store to a dictionary.

        :returns: The serialized store as a dictionary.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessageStore":
        """
        Deserializes the store from a dictionary.

        :param data: The dictionary to deserialize from.
        :returns: The deserialized store.
        """
        ...

    def count_messages(self, index: str) -> int:
        """
        Returns the number of chat messages stored.

        :param index: The index for which to count messages.

        :returns: The number of messages.
        """
        ...

    def write_messages(self, index: str, messages: list[ChatMessage]) -> int:
        """
        Writes chat messages to the ChatMessageStore.

        :param index: The index under which to store the messages.
        :param messages: A list of ChatMessages to write.

        :returns: The number of messages written.
        """
        ...

    def delete_messages(self, index: str) -> None:
        """
        Deletes all stored chat messages.

        :param index: The index from which to delete all messages.
        """
        ...

    def retrieve_messages(self, index: str) -> list[ChatMessage]:
        """
        Retrieves chat messages from the ChatMessageStore.

        :param index: The index from which to retrieve messages.

        :returns: A list of retrieved ChatMessages.
        """
        ...
