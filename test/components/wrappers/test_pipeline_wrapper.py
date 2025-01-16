# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

import pytest

from haystack import Pipeline, Document, component
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.routers import ConditionalRouter
from haystack.components.builders import AnswerBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.dataclasses import ChatMessage, GeneratedAnswer
from haystack.utils.auth import Secret

from haystack_experimental.components.wrappers.pipeline_wrapper import PipelineWrapper

@pytest.fixture
def mock_openai_generator(monkeypatch):
    """Create a mock OpenAI Generator for testing."""

    def mock_run(self, prompt: str, **kwargs):
        return {"replies": ["This is a test response about capitals."]}

    monkeypatch.setattr(OpenAIGenerator, "run", mock_run)
    return OpenAIGenerator(api_key=Secret.from_token("test-key"))


@pytest.fixture
def documents():
    """Create test documents for the document store."""
    return [
        Document(content="Paris is the capital of France."),
        Document(content="Berlin is the capital of Germany."),
        Document(content="Rome is the capital of Italy.")
    ]


@pytest.fixture
def document_store(documents):
    """Create and populate a test document store."""
    store = InMemoryDocumentStore()
    store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
    return store

@pytest.fixture
def rag_pipeline(document_store):
    """Create a simple RAG pipeline."""
    @component
    class FakeGenerator:
        @component.output_types(replies=List[str])
        def run(self, prompt: str, **kwargs):
            return {"replies": ["This is a test response about capitals."]}

    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
    pipeline.add_component("prompt_builder",
                           PromptBuilder(
                               template="Given these documents: {{documents|join(', ',attribute='content')}} Answer: {{query}}"))
    pipeline.add_component("llm", FakeGenerator())
    pipeline.add_component("answer_builder", AnswerBuilder())
    pipeline.add_component("joiner", DocumentJoiner())

    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    pipeline.connect("llm.replies", "answer_builder.replies")
    pipeline.connect("retriever.documents", "joiner.documents")

    return pipeline


class TestPipelineWrapperWithRealPipelines:

    def test_explicit_mapping(self, rag_pipeline):
        # Create wrapper with input/output mappings
        input_mapping = {
            "search_query": ["retriever.query", "prompt_builder.query", "answer_builder.query"]
        }
        output_mapping = {
            "answer_builder.answers": "final_answers"
        }

        wrapper = PipelineWrapper(
            pipeline=rag_pipeline,
            input_mapping=input_mapping,
            output_mapping=output_mapping
        )

        output_sockets = wrapper.__haystack_output__._sockets_dict
        assert set(output_sockets.keys()) == {"final_answers"}
        assert output_sockets["final_answers"].type == List[GeneratedAnswer]


        input_sockets = wrapper.__haystack_input__._sockets_dict
        assert set(input_sockets.keys()) == {"search_query"}
        assert input_sockets["search_query"].type == str

        # Test normal query flow
        result = wrapper.run(search_query="What is the capital of France?")
        assert "final_answers" in result
        assert isinstance(result["final_answers"][0], GeneratedAnswer)

    def test_auto_resolution(self, rag_pipeline):
        wrapper = PipelineWrapper(
            pipeline=rag_pipeline
        )

        output_sockets = wrapper.__haystack_output__._sockets_dict
        assert set(output_sockets.keys()) == {"answers", "documents"}
        assert output_sockets["answers"].type == List[GeneratedAnswer]


        input_sockets = wrapper.__haystack_input__._sockets_dict
        assert set(input_sockets.keys()) == {
            'documents',
             'filters',
             'meta',
             'pattern',
             'query',
             'reference_pattern',
             'scale_score',
             'template',
             'template_variables',
             'top_k'
        }
        assert input_sockets["query"].type == str

        # Test normal query flow
        result = wrapper.run(query="What is the capital of France?")
        assert "answers" in result
        assert isinstance(result["answers"][0], GeneratedAnswer)
        assert "documents" in result

    def test_wrapper_serialization(self, document_store):
        """Test serialization and deserialization of pipeline wrapper."""
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))

        wrapper = PipelineWrapper(
            pipeline=pipeline,
            input_mapping={"query": ["retriever.query"]},
            output_mapping={"retriever.documents": "documents"}
        )

        # Test serialization
        serialized = wrapper.to_dict()
        assert "type" in serialized
        assert "init_parameters" in serialized
        assert "pipeline" in serialized["init_parameters"]

        # Test deserialization
        deserialized = PipelineWrapper.from_dict(serialized)
        assert isinstance(deserialized, PipelineWrapper)
        assert deserialized.input_mapping == wrapper.input_mapping
        assert deserialized.output_mapping == wrapper.output_mapping

        result = deserialized.run(query="What is the capital of France?")
        assert "documents" in result
        assert result["documents"][0].content == "Paris is the capital of France."