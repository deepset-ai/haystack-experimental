# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Protocol

from haystack import logging
from haystack.dataclasses import ChatMessage

# Ellipsis are needed for the type checker, it's safe to disable module-wide
# pylint: disable=unnecessary-ellipsis

logger = logging.getLogger(__name__)


class ChatMessageStore(Protocol):
    """
    Stores ChatMessages to be used by the components of a Pipeline.

    Classes implementing this protocol often store the ChatMessages permanently and allow specialized components to
    perform retrieval on them, either by embedding, by keyword, hybrid, and so on, depending on the backend used.

    In order to write or retrieve chat messages, consider using a ChatMessageWriter or ChatMessageRetriever.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessageStore":
        """
        Deserializes the store from a dictionary.
        """
        ...

    def count_messages(self) -> int:
        """
        Returns the number of chat messages stored.
        """
        ...

    def write_messages(self, messages: List[ChatMessage]) -> int:
        """
        Writes chat messages to the ChatMessageStore.

        :param messages: A list of ChatMessages to write.
        :returns: The number of messages written.

        :raises ValueError: If messages is not a list of ChatMessages.
        """
        ...

    def delete_messages(self) -> None:
        """
        Deletes all stored chat messages.
        """
        ...

    def retrieve(self) -> List[ChatMessage]:
        """
        Retrieves all stored chat messages.

        :returns: A list of chat messages.
        """
        ...
