# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import Pipeline, component, default_from_dict, default_to_dict
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.core.serialization import (
    DeserializationError,
    component_from_dict,
    component_to_dict,
    import_class_by_name,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

from haystack_experimental.core.super_component import SuperComponentBase

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


@component
class DocumentIndexer(SuperComponentBase):
    """
    A document indexer that takes a list of documents and indexes them using the specified model.

    Usage:

    ```python
    >>> from haystack import Document
    >>> doc = Document(content="I love pizza!")
    >>> indexer = DocumentIndexer()
    >>> indexer.warm_up()
    >>> result = indexer.run(documents=[doc])
    >>> print(result)
    {'documents_written': 1}
    >>> indexer.pipeline.get_component("writer").document_store.count_documents()
    1
    ```
    """

    def __init__(
        self,
        model: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        embedding_separator: str = "\n",
        meta_fields_to_embed: Optional[List[str]] = None,
        document_store: Optional[DocumentStore] = None,
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
    ) -> None:
        """
        Initialize the DocumentIndexer component.

        :param model: The embedding model to use.
        :param prefix: The prefix to add to the document content.
        :param suffix: The suffix to add to the document content.
        :param batch_size: The batch size to use for the embedding.
        :param embedding_separator: The separator to use for the embedding.
        :param meta_fields_to_embed: The meta fields to embed.
        :param document_store: The document store to use.
        :param duplicate_policy: The duplicate policy to use.
        """
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.embedding_separator = embedding_separator
        self.meta_fields_to_embed = meta_fields_to_embed
        self.document_store = document_store
        self.duplicate_policy = duplicate_policy

        pipeline = Pipeline()

        pipeline.add_component(
            "embedder",
            SentenceTransformersDocumentEmbedder(
                model=self.model or DEFAULT_EMBEDDING_MODEL,
                prefix=self.prefix,
                suffix=self.suffix,
                batch_size=self.batch_size,
                embedding_separator=self.embedding_separator,
                meta_fields_to_embed=self.meta_fields_to_embed,
            ),
        )
        pipeline.add_component(
            "writer",
            DocumentWriter(
                document_store=self.document_store or InMemoryDocumentStore(),
                policy=self.duplicate_policy,
            ),
        )

        pipeline.connect("embedder.documents", "writer.documents")

        super(DocumentIndexer, self).__init__(
            pipeline=pipeline,
            input_mapping={"documents": ["embedder.documents"]},
            output_mapping={"writer.documents_written": "documents_written"},
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this instance to a dictionary.
        """
        document_store = component_to_dict(self.document_store, "document_store") if self.document_store else None

        return default_to_dict(
            self,
            model=self.model,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            embedding_separator=self.embedding_separator,
            meta_fields_to_embed=self.meta_fields_to_embed,
            document_store=document_store,
            duplicate_policy=self.duplicate_policy.value,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentIndexer":
        """
        Load an instance of this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})

        # Deserialize document store
        if (document_store := init_params.get("document_store")) and isinstance(document_store, dict):
            if "type" not in document_store:
                raise DeserializationError("Missing 'type' in serialization data for document_store parameter.")

            document_store_type = import_class_by_name(document_store["type"])
            init_params["document_store"] = component_from_dict(document_store_type, document_store, "document_store")

        # Deserialize policy
        if policy_value := init_params.get("duplicate_policy"):
            init_params["duplicate_policy"] = DuplicatePolicy(policy_value)

        data["init_parameters"] = init_params

        return default_from_dict(cls, data)
