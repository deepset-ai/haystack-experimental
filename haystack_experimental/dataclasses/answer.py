# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Dict

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, Document
from haystack.dataclasses import GeneratedAnswer as HaystackGeneratedAnswer


@dataclass
class GeneratedAnswer(HaystackGeneratedAnswer):
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the object to a dictionary.

        :returns:
            Serialized dictionary representation of the object.
        """
        documents = [doc.to_dict(flatten=False) for doc in self.documents]

        # Serialize ChatMessage objects to dicts
        meta = self.meta
        all_messages = meta.get("all_messages")

        # all_messages is either a list of ChatMessage objects or a list of strings
        if all_messages and isinstance(all_messages[0], ChatMessage):
            meta = {**meta, "all_messages": [msg.to_dict() for msg in all_messages]}

        return default_to_dict(self, data=self.data, query=self.query, documents=documents, meta=meta)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedAnswer":
        """
        Deserialize the object from a dictionary.

        :param data:
            Dictionary representation of the object.

        :returns:
            Deserialized object.
        """
        init_params = data.get("init_parameters", {})

        if (documents := init_params.get("documents")) is not None:
            init_params["documents"] = [Document.from_dict(d) for d in documents]

        meta = init_params.get("meta", {})
        if (all_messages := meta.get("all_messages")) is not None and isinstance(all_messages[0], dict):
            meta["all_messages"] = [ChatMessage.from_dict(m) for m in all_messages]
        init_params["meta"] = meta

        return default_from_dict(cls, data)
