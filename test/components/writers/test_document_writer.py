# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack import Document
from haystack.testing.factory import document_store_class
from haystack_experimental.components.writers.document_writer import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_experimental.document_stores.in_memory import InMemoryDocumentStore


class TestDocumentWriter:
    @pytest.mark.asyncio
    async def test_run_invalid_docstore(self):
        document_store = document_store_class("MockedDocumentStore")

        writer = DocumentWriter(document_store)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        with pytest.raises(TypeError, match="does not provide async support"):
            result = await writer.run_async(documents=documents)

    @pytest.mark.asyncio
    async def test_run(self):
        document_store = InMemoryDocumentStore()
        writer = DocumentWriter(document_store)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        result = await writer.run_async(documents=documents)
        assert result["documents_written"] == 2

    @pytest.mark.asyncio
    async def test_run_skip_policy(self):
        document_store = InMemoryDocumentStore()
        writer = DocumentWriter(document_store, policy=DuplicatePolicy.SKIP)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        result = await writer.run_async(documents=documents)
        assert result["documents_written"] == 2

        result = await writer.run_async(documents=documents)
        assert result["documents_written"] == 0
