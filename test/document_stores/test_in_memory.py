# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import tempfile

from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack_experimental.document_stores.in_memory import InMemoryDocumentStore
from haystack.testing.document_store import DocumentStoreBaseTests


class TestMemoryDocumentStoreAsync(DocumentStoreBaseTests):  # pylint: disable=R0904
    """
    Test InMemoryDocumentStore's specific features
    """

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    @pytest.fixture
    def document_store(self) -> InMemoryDocumentStore:
        return InMemoryDocumentStore(bm25_algorithm="BM25L")

    def test_to_dict(self):
        store = InMemoryDocumentStore()
        data = store.to_dict()
        assert data == {
            "type": "haystack_experimental.document_stores.in_memory.document_store.InMemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": r"(?u)\b\w\w+\b",
                "bm25_algorithm": "BM25L",
                "bm25_parameters": {},
                "embedding_similarity_function": "dot_product",
                "index": store.index,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        store = InMemoryDocumentStore(
            bm25_tokenization_regex="custom_regex",
            bm25_algorithm="BM25Plus",
            bm25_parameters={"key": "value"},
            embedding_similarity_function="cosine",
            index="my_cool_index",
        )
        data = store.to_dict()
        assert data == {
            "type": "haystack_experimental.document_stores.in_memory.document_store.InMemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": "custom_regex",
                "bm25_algorithm": "BM25Plus",
                "bm25_parameters": {"key": "value"},
                "embedding_similarity_function": "cosine",
                "index": "my_cool_index",
            },
        }

    @patch("haystack.document_stores.in_memory.document_store.re")
    def test_from_dict(self, mock_regex):
        data = {
            "type": "haystack_experimental.document_stores.in_memory.document_store.InMemoryDocumentStore",
            "init_parameters": {
                "bm25_tokenization_regex": "custom_regex",
                "bm25_algorithm": "BM25Plus",
                "bm25_parameters": {"key": "value"},
                "index": "my_cool_index",
            },
        }
        store = InMemoryDocumentStore.from_dict(data)
        mock_regex.compile.assert_called_with("custom_regex")
        assert store.tokenizer
        assert store.bm25_algorithm == "BM25Plus"
        assert store.bm25_parameters == {"key": "value"}
        assert store.index == "my_cool_index"

    @pytest.mark.asyncio
    async def test_write_documents(self, document_store: InMemoryDocumentStore):
        docs = [Document(id="1")]
        assert await document_store.write_documents_async(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(docs)

    @pytest.mark.asyncio
    async def test_count_documents(self, document_store: InMemoryDocumentStore):
        await document_store.write_documents_async(
            [
                Document(content="test doc 1"),
                Document(content="test doc 2"),
                Document(content="test doc 3"),
            ]
        )
        assert await document_store.count_documents_async() == 3

    @pytest.mark.asyncio
    async def test_filter_documents(self, document_store: InMemoryDocumentStore):
        filterable_docs = [
            Document(
                content=f"1",
                meta={
                    "number": -10,
                },
            ),
            Document(
                content=f"2",
                meta={
                    "number": 100,
                },
            ),
        ]
        await document_store.write_documents_async(filterable_docs)
        result = await document_store.filter_documents_async(
            filters={"field": "meta.number", "operator": "==", "value": 100}
        )
        DocumentStoreBaseTests().assert_documents_are_equal(
            result, [d for d in filterable_docs if d.meta.get("number") == 100]
        )

    @pytest.mark.asyncio
    async def test_delete_documents(self, document_store: InMemoryDocumentStore):
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert document_store.count_documents() == 1

        await document_store.delete_documents_async([doc.id])
        assert await document_store.count_documents_async() == 0

    @pytest.mark.asyncio
    async def test_bm25_retrieval(self, document_store: InMemoryDocumentStore):
        # Tests if the bm25_retrieval method returns the correct document based on the input query.
        docs = [
            Document(content="Hello world"),
            Document(content="Haystack supports multiple languages"),
        ]
        await document_store.write_documents_async(docs)
        results = await document_store.bm25_retrieval_async(
            query="What languages?", top_k=1
        )
        assert len(results) == 1
        assert results[0].content == "Haystack supports multiple languages"

    @pytest.mark.asyncio
    async def test_embedding_retrieval(self):
        docstore = InMemoryDocumentStore(embedding_similarity_function="cosine")
        # Tests if the embedding retrieval method returns the correct document based on the input query embedding.
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(
                content="Haystack supports multiple languages",
                embedding=[1.0, 1.0, 1.0, 1.0],
            ),
        ]
        await docstore.write_documents_async(docs)
        results = await docstore.embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, filters={}, scale_score=False
        )
        assert len(results) == 1
        assert results[0].content == "Haystack supports multiple languages"
