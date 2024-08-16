# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, Pipeline
from haystack_experimental.components.splitters import HierarchicalDocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestHierarchicalDocumentSplitter:
    def test_init_with_default_params(self):
        builder = HierarchicalDocumentSplitter(block_sizes=[100, 200, 300])
        assert builder.block_sizes == [300, 200, 100]
        assert builder.split_overlap == 0
        assert builder.split_by == "word"

    def test_init_with_custom_params(self):
        builder = HierarchicalDocumentSplitter(block_sizes=[100, 200, 300], split_overlap=25, split_by="word")
        assert builder.block_sizes == [300, 200, 100]
        assert builder.split_overlap == 25
        assert builder.split_by == "word"

    def test_init_with_duplicate_block_sizes(self):
        with pytest.raises(ValueError):
            HierarchicalDocumentSplitter(block_sizes=[100, 200, 200])

    def test_to_dict(self):
        builder = HierarchicalDocumentSplitter(block_sizes=[100, 200, 300], split_overlap=25, split_by="word")
        expected = builder.to_dict()
        assert expected == {
            "type": "haystack_experimental.components.splitters.hierarchical_doc_splitter.HierarchicalDocumentSplitter",
            "init_parameters": {"block_sizes": [300, 200, 100], "split_overlap": 25, "split_by": "word"},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_experimental.components.splitters.hierarchical_doc_splitter.HierarchicalDocumentSplitter",
            "init_parameters": {"block_sizes": [10, 5, 2], "split_overlap": 0, "split_by": "word"},
        }

        builder = HierarchicalDocumentSplitter.from_dict(data)
        assert builder.block_sizes == [10, 5, 2]
        assert builder.split_overlap == 0
        assert builder.split_by == "word"

    def test_run(self):
        builder = HierarchicalDocumentSplitter(block_sizes=[10, 5, 2], split_overlap=0, split_by="word")
        text = "one two three four five six seven eight nine ten"
        doc = Document(content=text)
        output = builder.run([doc])
        docs = output["documents"]
        builder.run([doc])

        assert len(docs) == 9
        assert docs[0].content == "one two three four five six seven eight nine ten"

        # level 1 - root node
        assert docs[0].meta["__level"] == 0
        assert len(docs[0].meta["__children_ids"]) == 2

        # level 2 -left branch
        assert docs[1].meta["__parent_id"] == docs[0].id
        assert docs[1].meta["__level"] == 1
        assert len(docs[1].meta["__children_ids"]) == 3

        # level 2 - right branch
        assert docs[2].meta["__parent_id"] == docs[0].id
        assert docs[2].meta["__level"] == 1
        assert len(docs[2].meta["__children_ids"]) == 3

        # level 3 - left branch - leaf nodes
        assert docs[3].meta["__parent_id"] == docs[1].id
        assert docs[4].meta["__parent_id"] == docs[1].id
        assert docs[5].meta["__parent_id"] == docs[1].id
        assert docs[3].meta["__level"] == 2
        assert docs[4].meta["__level"] == 2
        assert docs[5].meta["__level"] == 2
        assert len(docs[3].meta["__children_ids"]) == 0
        assert len(docs[4].meta["__children_ids"]) == 0
        assert len(docs[5].meta["__children_ids"]) == 0

        # level 3 - right branch - leaf nodes
        assert docs[6].meta["__parent_id"] == docs[2].id
        assert docs[7].meta["__parent_id"] == docs[2].id
        assert docs[8].meta["__parent_id"] == docs[2].id
        assert docs[6].meta["__level"] == 2
        assert docs[7].meta["__level"] == 2
        assert docs[8].meta["__level"] == 2
        assert len(docs[6].meta["__children_ids"]) == 0
        assert len(docs[7].meta["__children_ids"]) == 0
        assert len(docs[8].meta["__children_ids"]) == 0

    def test_to_dict_in_pipeline(self):
        pipeline = Pipeline()
        hierarchical_doc_builder = HierarchicalDocumentSplitter(block_sizes=[10, 5, 2])
        doc_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=doc_store)
        pipeline.add_component(name="hierarchical_doc_splitter", instance=hierarchical_doc_builder)
        pipeline.add_component(name="doc_writer", instance=doc_writer)
        pipeline.connect("hierarchical_doc_splitter", "doc_writer")
        expected = pipeline.to_dict()

        assert expected.keys() == {"metadata", "max_loops_allowed", "components", "connections"}
        assert expected["components"].keys() == {"hierarchical_doc_splitter", "doc_writer"}
        assert expected["components"]["hierarchical_doc_splitter"] == {
            "type": "haystack_experimental.components.splitters.hierarchical_doc_splitter.HierarchicalDocumentSplitter",
            "init_parameters": {"block_sizes": [10, 5, 2], "split_overlap": 0, "split_by": "word"},
        }

    def test_from_dict_in_pipeline(self):
        data = {
            "metadata": {},
            "max_loops_allowed": 100,
            "components": {
                "hierarchical_doc_splitter": {
                    "type": "haystack_experimental.components.splitters.hierarchical_doc_splitter.HierarchicalDocumentSplitter",
                    "init_parameters": {"block_sizes": [10, 5, 2], "split_overlap": 0, "split_by": "word"},
                },
                "doc_writer": {
                    "type": "haystack.components.writers.document_writer.DocumentWriter",
                    "init_parameters": {
                        "document_store": {
                            "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                            "init_parameters": {
                                "bm25_tokenization_regex": "(?u)\\b\\w\\w+\\b",
                                "bm25_algorithm": "BM25L",
                                "bm25_parameters": {},
                                "embedding_similarity_function": "dot_product",
                                "index": "f32ad5bf-43cb-4035-9823-1de1ae9853c1",
                            },
                        },
                        "policy": "NONE",
                    },
                },
            },
            "connections": [{"sender": "hierarchical_doc_splitter.documents", "receiver": "doc_writer.documents"}],
        }

        assert Pipeline.from_dict(data)

    @pytest.mark.integration
    def test_example_in_pipeline(self):
        pipeline = Pipeline()
        hierarchical_doc_builder = HierarchicalDocumentSplitter(block_sizes=[10, 5, 2], split_overlap=0, split_by="word")
        doc_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=doc_store)

        pipeline.add_component(name="hierarchical_doc_splitter", instance=hierarchical_doc_builder)
        pipeline.add_component(name="doc_writer", instance=doc_writer)
        pipeline.connect("hierarchical_doc_splitter.documents", "doc_writer")

        text = "one two three four five six seven eight nine ten"
        doc = Document(content=text)
        docs = pipeline.run({"hierarchical_doc_splitter": {"documents": [doc]}})

        assert docs["doc_writer"]["documents_written"] == 9
        assert len(doc_store.storage.values()) == 9
