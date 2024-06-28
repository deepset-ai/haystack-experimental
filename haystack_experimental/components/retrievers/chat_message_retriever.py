# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any, Dict, List, Optional

from haystack import DeserializationError, component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import ChatMessageStore

logger = logging.getLogger(__name__)


@component
class ChatMessageRetriever:
    """
    Retrieves chat messages that match the provided filters.

    Usage example:
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack.components.retrievers import ChatMessageRetriever
    from haystack.document_stores.in_memory import InMemoryChatMessageStore

    messages = [
        ChatMessage(content="Hello, how can I help you?", role="assistant", meta={"lang": "en"}),
        ChatMessage(content="Hallo, wie kann ich Ihnen helfen?", role="assistant", meta={"lang": "de"}),
    ]

    message_store = InMemoryChatMessageStore()
    message_store.write_messages(messages)
    retriever = ChatMessageRetriever(message_store, filters={"field": "lang", "operator": "==", "value": "en"})

    # if passed in the run method, filters will override those provided at initialization
    result = retriever.run(filters={"field": "lang", "operator": "==", "value": "de"})

    print(result["messages"])
    ```
    """

    def __init__(self, message_store: ChatMessageStore, filters: Optional[Dict[str, Any]] = None):
        """
        Create the ChatMessageRetriever component.

        :param message_store:
            An instance of a ChatMessageStore.
        :param filters:
            A dictionary with filters to narrow down the search space.
        """
        self.message_store = message_store
        self.filters = filters

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"message_store": type(self.message_store).__name__}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        message_store = self.message_store.to_dict()
        return default_to_dict(self, message_store=message_store, filters=self.filters)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessageRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        if "message_store" not in init_params:
            raise DeserializationError("Missing 'message_store' in serialization data")
        if "type" not in init_params["message_store"]:
            raise DeserializationError("Missing 'type' in message store's serialization data")
        try:
            module_name, type_ = init_params["message_store"]["type"].rsplit(".", 1)
            logger.debug("Trying to import module '{module_name}'", module_name=module_name)
            module = importlib.import_module(module_name)
        except (ImportError, DeserializationError) as e:
            raise DeserializationError(
                f"ChatMessageStore of type '{init_params['message_store']['type']}' not correctly imported"
            ) from e

        store_class = getattr(module, type_)
        data["init_parameters"]["message_store"] = store_class.from_dict(data["init_parameters"]["message_store"])
        return default_from_dict(cls, data)

    @component.output_types(messages=List[ChatMessage])
    def run(self, filters: Optional[Dict[str, Any]] = None):
        """
        Run the ChatMessageRetriever on the given input data.

        :param filters:
            A dictionary with filters to narrow down the search space.
            If not specified, the ChatMessageRetriever uses the value provided at initialization.
        :returns:
            The retrieved chat messages.
        """
        return {"messages": self.message_store.filter_messages(filters=filters or self.filters)}