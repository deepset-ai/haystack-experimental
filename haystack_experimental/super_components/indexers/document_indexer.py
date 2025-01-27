from typing import Any, Dict

from haystack import Pipeline, component, default_from_dict, default_to_dict
from haystack.components.embedders import (
    AzureOpenAIDocumentEmbedder,
    HuggingFaceAPIDocumentEmbedder,
    OpenAIDocumentEmbedder,
    SentenceTransformersDocumentEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack.core.component import Component
from haystack.core.serialization import (
    DeserializationError,
    component_from_dict,
    component_to_dict,
    import_class_by_name,
)
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

from haystack_experimental.core.super_component import SuperComponentBase

KNOWN_DOCUMENT_EMBEDDERS = (
    HuggingFaceAPIDocumentEmbedder,
    SentenceTransformersDocumentEmbedder,
    OpenAIDocumentEmbedder,
    AzureOpenAIDocumentEmbedder,
)


class InvalidEmbedderError(ValueError):
    """Raised when a DocumentIndexer receives an invalid embedder parameter."""


@component
class DocumentIndexer(SuperComponentBase):
    """
    A document indexer that takes a list of documents and indexes them using the specified embedder.

    Usage:
    ```python
    from haystack import Document
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.document_stores.types import DuplicatePolicy

    doc = Document(content="I love pizza!")

    indexer = DocumentIndexer(
        embedder=SentenceTransformersDocumentEmbedder(),
        document_store=InMemoryDocumentStore(),
        duplicate_policy=DuplicatePolicy.OVERWRITE,
    )

    indexer.warm_up()

    result = indexer.run(documents=[doc])
    print(result)

    # {'documents_written': 1}
    ```
    """

    def __init__(
        self,
        embedder: Component,
        document_store: DocumentStore,
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> None:
        self.embedder = embedder
        self.document_store = document_store
        self.duplicate_policy = duplicate_policy

        if not isinstance(self.embedder, KNOWN_DOCUMENT_EMBEDDERS):
            raise InvalidEmbedderError

        pipeline = Pipeline()

        pipeline.add_component("embedder", self.embedder)
        pipeline.add_component(
            "writer",
            DocumentWriter(
                document_store=self.document_store,
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
        return default_to_dict(
            self,
            embedder=component_to_dict(self.embedder, "embedder"),
            document_store=component_to_dict(self.document_store, "document_store"),
            duplicate_policy=self.duplicate_policy.value,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentIndexer":
        """
        Load an instance of this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})

        # Deserialized nested components
        for name in ["embedder", "document_store"]:
            if (item := init_params.get(name)) and isinstance(item, dict):
                if "type" not in item:
                    raise DeserializationError(f"Missing 'type' in serialization data for {name} parameter.")

                item_type = import_class_by_name(item["type"])
                init_params[name] = component_from_dict(item_type, item, name)

        # Deserialize policy
        if policy_value := init_params.get("duplicate_policy"):
            init_params["duplicate_policy"] = DuplicatePolicy(policy_value)

        data["init_parameters"] = init_params

        return default_from_dict(cls, data)
