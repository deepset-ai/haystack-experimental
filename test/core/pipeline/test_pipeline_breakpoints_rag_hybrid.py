import os
import sys
from pathlib import Path

import pytest

from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack import Document

from haystack_experimental.core.errors import PipelineBreakpointException
from haystack_experimental.core.pipeline.pipeline import Pipeline

import os
class TestPipelineBreakpoints:
    """
    This class contains tests for pipelines with breakpoints.
    """

    @pytest.fixture
    def document_store(self):
        """Create and populate a document store for testing."""
        documents = [
            Document(content="My name is Jean and I live in Paris."),
            Document(content="My name is Mark and I live in Berlin."),
            Document(content="My name is Giorgio and I live in Rome."),
        ]

        document_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
        doc_embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/paraphrase-MiniLM-L3-v2",
            progress_bar=False
        )
        ingestion_pipe = Pipeline()
        ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
        ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")
        ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
        ingestion_pipe.run({"doc_embedder": {"documents": documents}})

        return document_store

    @pytest.fixture
    def hybrid_rag_pipeline(self, document_store):
        """Create a hybrid RAG pipeline for testing."""
        query_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/paraphrase-MiniLM-L3-v2",
            progress_bar=False
        )

        prompt_template = """
        Given these documents, answer the question based on the document content only.\nDocuments:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}

        \nQuestion: {{question}}
        \nAnswer:
        """
        pipeline = Pipeline()
        pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")
        pipeline.add_component(instance=query_embedder, name="query_embedder")
        pipeline.add_component(
            instance=InMemoryEmbeddingRetriever(document_store=document_store),
            name="embedding_retriever"
        )
        pipeline.add_component(instance=DocumentJoiner(sort_by_score=False), name="doc_joiner")
        pipeline.add_component(
            instance=TransformersSimilarityRanker(model="intfloat/simlm-msmarco-reranker", top_k=5),
            name="ranker"
        )
        pipeline.add_component(instance=PromptBuilder(
            template=prompt_template, required_variables=['documents', 'question']),
            name="prompt_builder"
        )
        pipeline.add_component(instance=OpenAIGenerator(), name="llm")
        pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

        pipeline.connect("query_embedder", "embedding_retriever.query_embedding")
        pipeline.connect("embedding_retriever", "doc_joiner.documents")
        pipeline.connect("bm25_retriever", "doc_joiner.documents")
        pipeline.connect("doc_joiner", "ranker.documents")
        pipeline.connect("ranker", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")
        pipeline.connect("llm.replies", "answer_builder.replies")
        pipeline.connect("llm.meta", "answer_builder.meta")
        pipeline.connect("doc_joiner", "answer_builder.documents")

        return pipeline

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory):
        return tmp_path_factory.mktemp("output_files")

    """
    def test_pipeline_breakpoints_invalid_component(self, hybrid_rag_pipeline, output_directory):
        question = "Where does Mark live?"
        data = {
            "query_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "ranker": {"query": question, "top_k": 10},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }
        with pytest.raises(ValueError, match="Breakpoint .* is not a registered component"):
            hybrid_rag_pipeline.run(data, breakpoints={("non_existent_component", 0)})
    """

    components = [
        "bm25_retriever",
        "query_embedder",
        "embedding_retriever",
        "doc_joiner",
        "ranker",
        "prompt_builder",
        "llm",
        "answer_builder"
    ]
    @pytest.mark.parametrize("component", components)
    @pytest.mark.integration
    @pytest.mark.skipif(sys.platform == "darwin", reason="Test crashes on macOS.")
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_pipeline_breakpoints_hybrid_rag(self, hybrid_rag_pipeline, document_store, output_directory, component):
        """
        Test that a hybrid RAG pipeline can be executed with breakpoints at each component.
        """
        # Test data
        question = "Where does Mark live?"
        data = {
            "query_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "ranker": {"query": question, "top_k": 5},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }

        try:
            _ = hybrid_rag_pipeline.run(data, breakpoints={(component, 0)}, debug_path=str(output_directory))
        except PipelineBreakpointException as e:
            pass

        all_files = list(output_directory.glob("*"))
        for full_path in all_files:
            f_name = str(full_path).split("/")[-1]
            if str(f_name).startswith(component):
                result = hybrid_rag_pipeline.run(data, breakpoints=None, resume_state_path=full_path)
                assert 'answer_builder' in result
                assert result['answer_builder']
