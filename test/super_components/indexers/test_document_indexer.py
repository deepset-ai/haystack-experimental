# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import ANY, Mock
import pytest
from haystack_experimental.super_components.indexers import DocumentIndexer
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder

from haystack import Document
from haystack_experimental.super_components.indexers.document_indexer import InvalidEmbedderError


class TestDocumentIndexer:
    @pytest.fixture
    def document_store(self) -> InMemoryDocumentStore:
        return InMemoryDocumentStore()

    @pytest.fixture
    def indexer(self, document_store: InMemoryDocumentStore) -> DocumentIndexer:
        return DocumentIndexer(
            embedder=SentenceTransformersDocumentEmbedder(device='cpu'),
            document_store=document_store,
        )

    @pytest.fixture
    def embedding_backend(self, indexer: DocumentIndexer, monkeypatch: pytest.MonkeyPatch) -> Mock:
        backend = Mock()
        backend.embed.return_value = [
            [0.3, 0.4, 0.01, 0.7],
            [0.1, 0.9, 0.87, 0.3],
        ]
        monkeypatch.setattr(indexer.embedder, "embedding_backend", backend)
        return backend

    def test_init(self, indexer: DocumentIndexer) -> None:
        assert indexer is not None
        assert hasattr(indexer, "pipeline")
        assert indexer.input_mapping == {"documents": ["embedder.documents"]}
        assert indexer.output_mapping == {"writer.documents_written": "documents_written"}

    def test_init_with_unknown_embedder(self, document_store: InMemoryDocumentStore) -> None:
        with pytest.raises(InvalidEmbedderError):
            DocumentIndexer(embedder=SentenceTransformersTextEmbedder(), document_store=document_store)

    def test_from_dict(self) -> None:
        indexer = DocumentIndexer.from_dict(
            {
                "init_parameters": {
                    "document_store": {
                        "init_parameters": {},
                        "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    },
                    "duplicate_policy": "overwrite",
                    "embedder": {
                        "init_parameters": {},
                        "type": "haystack.components.embedders.sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder",
                    },
                },
                "type": "haystack_experimental.super_components.indexers.document_indexer.DocumentIndexer",
            }
        )
        assert isinstance(indexer, DocumentIndexer)

    def test_to_dict(self, indexer: DocumentIndexer) -> None:
        expected = {
            "init_parameters": {
                "document_store": {
                    "init_parameters": {
                        "bm25_algorithm": "BM25L",
                        "bm25_parameters": {},
                        "bm25_tokenization_regex": ANY,
                        "embedding_similarity_function": "dot_product",
                        "index": ANY,
                    },
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                },
                "duplicate_policy": "overwrite",
                "embedder": {
                    "init_parameters": {
                        "batch_size": 32,
                        "config_kwargs": None,
                        "device": {
                            "device": "cpu",
                            "type": "single",
                        },
                        "embedding_separator": "\n",
                        "meta_fields_to_embed": [],
                        "model": "sentence-transformers/all-mpnet-base-v2",
                        "model_kwargs": None,
                        "normalize_embeddings": False,
                        "precision": "float32",
                        "prefix": "",
                        "progress_bar": True,
                        "suffix": "",
                        "token": {
                            "env_vars": [
                                "HF_API_TOKEN",
                                "HF_TOKEN",
                            ],
                            "strict": False,
                            "type": "env_var",
                        },
                        "tokenizer_kwargs": None,
                        "truncate_dim": None,
                        "trust_remote_code": False,
                    },
                    "type": "haystack.components.embedders.sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder",
                },
            },
            "type": "haystack_experimental.super_components.indexers.document_indexer.DocumentIndexer",
        }
        assert indexer.to_dict() == expected

    def test_warm_up(self, indexer: DocumentIndexer, embedding_backend: Mock) -> None:
        indexer.warm_up()
        embedding_backend.assert_called_once

    def test_run(self, indexer: DocumentIndexer, embedding_backend: Mock) -> None:
        documents = [
            Document(content="Test document"),
            Document(content="Another test document"),
        ]

        indexer.warm_up()
        result = indexer.run(documents=documents)

        embedding_backend.embed.assert_called_once
        assert result == {"documents_written": len(documents)}
