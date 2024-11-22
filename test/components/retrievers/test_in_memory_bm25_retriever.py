# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Any

import pytest

from haystack_experimental.core import AsyncPipeline, run_async_pipeline
from haystack_experimental.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document
from haystack_experimental.document_stores.in_memory import InMemoryDocumentStore


@pytest.fixture()
def mock_docs():
    return [
        Document(content="Javascript is a popular programming language"),
        Document(content="Java is a popular programming language"),
        Document(content="Python is a popular programming language"),
        Document(content="Ruby is a popular programming language"),
        Document(content="PHP is a popular programming language"),
    ]


class TestMemoryBM25RetrieverAsync:
    @pytest.mark.asyncio
    async def test_retriever_valid_run(self, mock_docs):
        ds = InMemoryDocumentStore()
        ds.write_documents(mock_docs)

        retriever = InMemoryBM25Retriever(ds, top_k=5)
        result = await retriever.run_async(query="PHP")

        assert "documents" in result
        assert len(result["documents"]) == 5
        assert result["documents"][0].content == "PHP is a popular programming language"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query, query_result",
        [
            ("Javascript", "Javascript is a popular programming language"),
            ("Java", "Java is a popular programming language"),
        ],
    )
    async def test_run_with_pipeline(self, mock_docs, query: str, query_result: str):
        ds = InMemoryDocumentStore()
        await ds.write_documents_async(mock_docs)
        retriever = InMemoryBM25Retriever(ds)

        pipeline = AsyncPipeline()
        pipeline.add_component("retriever", retriever)
        result: Dict[str, Any] = await run_async_pipeline(
            pipeline, data={"retriever": {"query": query}}
        )

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert results_docs[0].content == query_result

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query, query_result, top_k",
        [
            ("Javascript", "Javascript is a popular programming language", 1),
            ("Java", "Java is a popular programming language", 2),
            ("Ruby", "Ruby is a popular programming language", 3),
        ],
    )
    async def test_run_with_pipeline_and_top_k(
        self, mock_docs, query: str, query_result: str, top_k: int
    ):
        ds = InMemoryDocumentStore()
        ds.write_documents(mock_docs)
        retriever = InMemoryBM25Retriever(ds)

        pipeline = AsyncPipeline()
        pipeline.add_component("retriever", retriever)
        result: Dict[str, Any] = await run_async_pipeline(
            pipeline, data={"retriever": {"query": query, "top_k": top_k}}
        )

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert len(results_docs) == top_k
        assert results_docs[0].content == query_result
