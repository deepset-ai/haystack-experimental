import pytest

from haystack import Document
from haystack_experimental.components.splitters import HierarchicalDocumentBuilder
from haystack_experimental.components.retrievers.auto_merging_retriever import AutoMergingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestSentenceWindowRetriever:
    def test_init_default(self):
        retriever = AutoMergingRetriever(InMemoryDocumentStore())
        assert retriever.threshold == 0.5

    def test_init_with_parameters(self):
        retriever = AutoMergingRetriever(InMemoryDocumentStore(), threshold=0.7)
        assert retriever.threshold == 0.7

    def test_init_with_invalid_threshold(self):
        with pytest.raises(ValueError):
            AutoMergingRetriever(InMemoryDocumentStore(), threshold=-2)

    def test_run_return_parent_document(self):
        text = "The sun rose early in the morning. It cast a warm glow over the trees. Birds began to sing."

        docs = [Document(content=text)]
        builder = HierarchicalDocumentBuilder(block_sizes=[10, 3], split_overlap=0, split_by="word")
        docs = builder.run(docs)

        # store level-1 parent documents and initialize the retriever
        doc_store_parents = InMemoryDocumentStore()
        for doc in docs["documents"]:
            if doc.meta["children_ids"] and doc.meta["level"] == 1:
                doc_store_parents.write_documents([doc])
        retriever = AutoMergingRetriever(doc_store_parents, threshold=0.5)

        # assume we retrieved 2 leaf docs from the same parent, the parent document should be returned,
        # since it has 3 children and the threshold=0.5, and we retrieved 2 children (2/3 > 0.66(6))
        leaf_docs = [doc for doc in docs["documents"] if not doc.meta["children_ids"]]
        docs = retriever.run(leaf_docs[4:6])
        assert len(docs["documents"]) == 1
        assert docs["documents"][0].content == "warm glow over the trees. Birds began to sing."
        assert len(docs["documents"][0].meta["children_ids"]) == 3

    def test_run_return_leafs_document(self):
        docs = [Document(content="The monarch of the wild blue yonder rises from the eastern side of the horizon.")]
        builder = HierarchicalDocumentBuilder(block_sizes=[10, 5, 2], split_overlap=0, split_by="word")
        docs = builder.run(docs)

        doc_store_parents = InMemoryDocumentStore()
        for doc in docs["documents"]:
            if doc.meta["children_ids"]:
                doc_store_parents.write_documents([doc])

        leaf_docs = [doc for doc in docs["documents"] if not doc.meta["children_ids"]]
        retriever = AutoMergingRetriever(doc_store_parents, threshold=0.5)
        retriever.run(leaf_docs[3:4])

    def test_run_return_leafs_document_different_parents(self):
        pass
