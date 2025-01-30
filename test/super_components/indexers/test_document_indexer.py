# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import ANY, Mock
from uuid import UUID
import pytest

from haystack_experimental.super_components.indexers import DocumentIndexer

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestDocumentIndexer:
    @pytest.fixture
    def indexer(self) -> DocumentIndexer:
        return DocumentIndexer()

    @pytest.fixture
    def embedding_backend(self, indexer: DocumentIndexer, monkeypatch: pytest.MonkeyPatch) -> Mock:
        backend = Mock()
        backend.embed.return_value = [
            [0.3, 0.4, 0.01, 0.7],
            [0.1, 0.9, 0.87, 0.3],
        ]

        embedder = indexer.pipeline.get_component("embedder")
        monkeypatch.setattr(embedder, "embedding_backend", backend)

        return backend

    def test_init(self, indexer: DocumentIndexer) -> None:
        assert indexer is not None
        assert hasattr(indexer, "pipeline")
        assert indexer.input_mapping == {"documents": ["embedder.documents"]}
        assert indexer.output_mapping == {"writer.documents_written": "documents_written"}

    def test_from_dict(self) -> None:
        indexer = DocumentIndexer.from_dict(
            {
                "init_parameters": {
                    "model": None,
                    "prefix": "",
                    "suffix": "",
                    "batch_size": 32,
                    "embedding_separator": "\n",
                    "meta_fields_to_embed": None,
                    "document_store": None,
                    "duplicate_policy": "overwrite",
                },
                "type": "haystack_experimental.super_components.indexers.document_indexer.DocumentIndexer",
            }
        )
        assert isinstance(indexer, DocumentIndexer)

    def test_from_dict_with_document_store(self) -> None:
        indexer = DocumentIndexer.from_dict(
            {
                "init_parameters": {
                    "model": None,
                    "prefix": "",
                    "suffix": "",
                    "batch_size": 32,
                    "embedding_separator": "\n",
                    "meta_fields_to_embed": None,
                    "document_store": {
                        "init_parameters": {
                            "bm25_tokenization_regex": r"(?u)\b\w\w+\b",
                            "bm25_algorithm": "BM25L",
                            "bm25_parameters": None,
                            "embedding_similarity_function": "dot_product",
                            "index": None,
                        },
                        "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    },
                    "duplicate_policy": "overwrite",
                },
                "type": "haystack_experimental.super_components.indexers.document_indexer.DocumentIndexer",
            }
        )

        assert isinstance(indexer, DocumentIndexer)
        assert isinstance(indexer.document_store, InMemoryDocumentStore)
        assert indexer.document_store.bm25_tokenization_regex == r"(?u)\b\w\w+\b"
        assert indexer.document_store.bm25_algorithm == "BM25L"
        assert indexer.document_store.bm25_parameters == {}
        assert indexer.document_store.embedding_similarity_function == "dot_product"
        assert UUID(indexer.document_store.index, version=4)

    def test_to_dict(self, indexer: DocumentIndexer) -> None:
        expected = {
            "init_parameters": {
                "model": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "embedding_separator": "\n",
                "meta_fields_to_embed": None,
                "document_store": None,
                "duplicate_policy": "overwrite",
            },
            "type": "haystack_experimental.super_components.indexers.document_indexer.DocumentIndexer",
        }
        assert indexer.to_dict() == expected

    def test_to_dict_with_document_store(self) -> None:
        document_store = InMemoryDocumentStore()
        indexer = DocumentIndexer(document_store=document_store)

        expected = {
            "init_parameters": {
                "model": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "embedding_separator": "\n",
                "meta_fields_to_embed": None,
                "document_store": {
                    "init_parameters": {
                        "bm25_tokenization_regex": r"(?u)\b\w\w+\b",
                        "bm25_algorithm": "BM25L",
                        "bm25_parameters": {},
                        "embedding_similarity_function": "dot_product",
                        "index": document_store.index,
                    },
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                },
                "duplicate_policy": "overwrite",
            },
            "type": "haystack_experimental.super_components.indexers.document_indexer.DocumentIndexer",
        }
        assert indexer.to_dict() == expected

    def test_warm_up(self, indexer: DocumentIndexer, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(indexer.pipeline, "warm_up", Mock())

            indexer.warm_up()

            indexer.pipeline.warm_up.assert_called_once()

    def test_run(self, indexer: DocumentIndexer, embedding_backend: Mock) -> None:
        documents = [
            Document(content="Test document"),
            Document(content="Another test document"),
        ]

        indexer.warm_up()
        result = indexer.run(documents=documents)

        embedding_backend.embed.assert_called_once
        assert result == {"documents_written": len(documents)}
