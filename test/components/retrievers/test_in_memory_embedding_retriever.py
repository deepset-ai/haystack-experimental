# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Any

import pytest

from haystack_experimental.core import AsyncPipeline, run_async_pipeline
from haystack_experimental.components.retrievers.in_memory.embedding_retriever import (
    InMemoryEmbeddingRetriever,
)
from haystack.dataclasses import Document
from haystack_experimental.document_stores.in_memory import InMemoryDocumentStore


class TestMemoryEmbeddingRetrieverAsync:
    @pytest.mark.asyncio
    async def test_valid_run(self):
        top_k = 3
        ds = InMemoryDocumentStore(embedding_similarity_function="cosine")
        docs = [
            Document(content="my document", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="another document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="third document", embedding=[0.5, 0.7, 0.5, 0.7]),
        ]
        await ds.write_documents_async(docs)

        retriever = InMemoryEmbeddingRetriever(ds, top_k=top_k)
        result = await retriever.run_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], return_embedding=True
        )

        assert "documents" in result
        assert len(result["documents"]) == top_k
        assert result["documents"][0].embedding == [1.0, 1.0, 1.0, 1.0]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_with_pipeline(self):
        ds = InMemoryDocumentStore(embedding_similarity_function="cosine")
        top_k = 2
        docs = [
            Document(content="my document", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="another document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="third document", embedding=[0.5, 0.7, 0.5, 0.7]),
        ]
        ds.write_documents(docs)
        retriever = InMemoryEmbeddingRetriever(ds, top_k=top_k)

        pipeline = AsyncPipeline()
        pipeline.add_component("retriever", retriever)
        result: Dict[str, Any] = await run_async_pipeline(
            pipeline,
            data={
                "retriever": {
                    "query_embedding": [0.1, 0.1, 0.1, 0.1],
                    "return_embedding": True,
                }
            },
        )

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert len(results_docs) == top_k
        assert results_docs[0].embedding == [1.0, 1.0, 1.0, 1.0]
