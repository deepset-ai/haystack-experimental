from haystack import component, Document

from typing import List

from haystack.components.retrievers import InMemoryBM25Retriever


@component
class MultiQueryInMemoryBM25Retriever:

    def __init__(self, retriever: InMemoryBM25Retriever, top_k: int = 3):
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