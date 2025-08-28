from haystack import component, Document, default_to_dict, default_from_dict
from haystack.core.serialization import component_to_dict
from haystack.utils.deserialization import deserialize_component_inplace

from typing import List, Any

from haystack_experimental.components.retrievers.types import KeywordRetriever
from haystack_experimental.components.retrievers.types import EmbeddingRetriever


@component
class MultiQueryKeywordRetriever:

    def __init__(self, retriever: KeywordRetriever, top_k: int = 3):
        self.retriever = retriever
        self.top_k = top_k

    @component.output_types(documents=List[Document])
    def run(self, queries: List[str], top_k: int = None):
        if top_k:
          self.top_k = top_k
        docs = []

        for query in queries:
          result = self.retriever.run(query = query, top_k = self.top_k)
          for doc in result['documents']:
            docs.append(doc)
        docs.sort(key=lambda x: x.score, reverse=True)

        return {"documents": docs}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        return default_to_dict(
            self,
            retriever=component_to_dict(obj=self.retriever),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiQueryKeywordRetriever":
        """
        Deserializes the component from a dictionary.
        """
        deserialize_component_inplace(data["init_parameters"], key="document_embedder")
        return default_from_dict(cls, data)
