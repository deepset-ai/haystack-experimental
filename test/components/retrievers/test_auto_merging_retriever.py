import pytest

import subprocess

from haystack import Document
from haystack_experimental.components.splitters import HierarchicalDocumentSplitter
from haystack_experimental.components.retrievers.auto_merging_retriever import AutoMergingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


@pytest.fixture(scope='session', autouse=True)
def install_dependencies():
    subprocess.check_call(['pip', 'install', 'chroma-haystack==0.21.1'])
    subprocess.check_call(['pip', 'install', 'pinecone_haystack==1.2.1'])


class TestAutoMergingRetriever:
    def test_init_default(self):
        retriever = AutoMergingRetriever(InMemoryDocumentStore())
        assert retriever.threshold == 0.5

    def test_init_with_parameters(self):
        retriever = AutoMergingRetriever(InMemoryDocumentStore(), threshold=0.7)
        assert retriever.threshold == 0.7

    def test_init_with_invalid_threshold(self):
        with pytest.raises(ValueError):
            AutoMergingRetriever(InMemoryDocumentStore(), threshold=-2)

    def test_to_dict(self):
        retriever = AutoMergingRetriever(InMemoryDocumentStore(), threshold=0.7)
        expected = retriever.to_dict()
        assert expected['type'] == 'haystack_experimental.components.retrievers.auto_merging_retriever.AutoMergingRetriever'
        assert expected['init_parameters']['threshold'] == 0.7
        assert expected['init_parameters']['document_store']['type'] == 'haystack.document_stores.in_memory.document_store.InMemoryDocumentStore'

    def test_from_dict(self):
        data = {
            'type': 'haystack_experimental.components.retrievers.auto_merging_retriever.AutoMergingRetriever',
            'init_parameters': {
                'document_store': {
                    'type': 'haystack.document_stores.in_memory.document_store.InMemoryDocumentStore',
                    'init_parameters': {
                        'bm25_tokenization_regex': '(?u)\\b\\w\\w+\\b',
                        'bm25_algorithm': 'BM25L',
                        'bm25_parameters': {},
                        'embedding_similarity_function': 'dot_product',
                        'index': '6b122bb4-211b-465e-804d-77c5857bf4c5'}},
                'threshold': 0.7}}
        retriever = AutoMergingRetriever.from_dict(data)
        assert retriever.threshold == 0.7

    def test_run_return_parent_document(self):
        text = "The sun rose early in the morning. It cast a warm glow over the trees. Birds began to sing."

        docs = [Document(content=text)]
        builder = HierarchicalDocumentSplitter(block_sizes=[10, 3], split_overlap=0, split_by="word")
        docs = builder.run(docs)

        # store level-1 parent documents and initialize the retriever
        doc_store_parents = InMemoryDocumentStore()
        for doc in docs["documents"]:
            if doc.meta["__children_ids"] and doc.meta["__level"] == 1:
                doc_store_parents.write_documents([doc])
        retriever = AutoMergingRetriever(doc_store_parents, threshold=0.5)

        # assume we retrieved 2 leaf docs from the same parent, the parent document should be returned,
        # since it has 3 children and the threshold=0.5, and we retrieved 2 children (2/3 > 0.66(6))
        leaf_docs = [doc for doc in docs["documents"] if not doc.meta["__children_ids"]]
        docs = retriever.run(leaf_docs[4:6])
        assert len(docs["documents"]) == 1
        assert docs["documents"][0].content == "warm glow over the trees. Birds began to sing."
        assert len(docs["documents"][0].meta["__children_ids"]) == 3

    def test_run_return_leafs_document(self):
        docs = [Document(content="The monarch of the wild blue yonder rises from the eastern side of the horizon.")]
        builder = HierarchicalDocumentSplitter(block_sizes=[10, 3], split_overlap=0, split_by="word")
        docs = builder.run(docs)

        doc_store_parents = InMemoryDocumentStore()
        for doc in docs["documents"]:
            if doc.meta["__level"] == 1:
                doc_store_parents.write_documents([doc])

        leaf_docs = [doc for doc in docs["documents"] if not doc.meta["__children_ids"]]
        retriever = AutoMergingRetriever(doc_store_parents, threshold=0.6)
        result = retriever.run([leaf_docs[4]])

        assert len(result['documents']) == 1
        assert result['documents'][0].content == 'eastern side of '
        assert result['documents'][0].meta["__parent_id"] == docs["documents"][2].id

    def test_run_return_leafs_document_different_parents(self):
        docs = [Document(content="The monarch of the wild blue yonder rises from the eastern side of the horizon.")]
        builder = HierarchicalDocumentSplitter(block_sizes=[10, 3], split_overlap=0, split_by="word")
        docs = builder.run(docs)

        doc_store_parents = InMemoryDocumentStore()
        for doc in docs["documents"]:
            if doc.meta["__level"] == 1:
                doc_store_parents.write_documents([doc])

        leaf_docs = [doc for doc in docs["documents"] if not doc.meta["__children_ids"]]
        retriever = AutoMergingRetriever(doc_store_parents, threshold=0.6)
        result = retriever.run([leaf_docs[4], leaf_docs[3]])

        assert len(result['documents']) == 2
        assert result['documents'][0].meta["__parent_id"] != result['documents'][1].meta["__parent_id"]

    def test_unsupported_document_store(self, install_dependencies):
        with pytest.raises(ValueError):
            AutoMergingRetriever(ChromaDocumentStore())

        with pytest.raises(ValueError):
            AutoMergingRetriever(PineconeDocumentStore())