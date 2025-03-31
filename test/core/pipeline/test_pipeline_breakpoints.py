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

from haystack_experimental.core.errors import PipelineBreakException
from haystack_experimental.core.pipeline.pipeline import Pipeline


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
        doc_embedder = SentenceTransformersDocumentEmbedder(model="intfloat/e5-base-v2", progress_bar=False)
        ingestion_pipe = Pipeline()
        ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
        ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")
        ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
        ingestion_pipe.run({"doc_embedder": {"documents": documents}})

        return document_store

    @pytest.fixture
    def hybrid_rag_pipeline(self, document_store):
        """Create a hybrid RAG pipeline for testing."""
        query_embedder = SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2", progress_bar=False)

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
        # Create a session-scoped temporary directory
        return tmp_path_factory.mktemp("output_files")

    def test_pipeline_breakpoints_hybrid_rag(self, hybrid_rag_pipeline, document_store, output_directory):
        """
        Test that a hybrid RAG pipeline can be executed with breakpoints at each component.
        """
        # Test data
        question = "Where does Mark live?"
        data = {
            "query_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "ranker": {"query": question, "top_k": 10},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }

        # Test breakpoints at each component
        components = [
            # "bm25_retriever",
            # "query_embedder",
            # "embedding_retriever",
            "doc_joiner",
            "ranker",
            "prompt_builder",
            "llm",
            "answer_builder"
        ]

        for component in components:
            try:
                _ = hybrid_rag_pipeline.run(data, breakpoints={(component, 0)}, debug_path=str(output_directory))
            except PipelineBreakException as e:
                pass

        print("\n\nShowing all files in the output directory:")
        all_files = list(output_directory.glob("*"))
        for f_name in all_files:
            if any(str(f_name).startswith(component) for component in components):
                print(f"File {f_name} starts with one of the components.")

            print(f"Resuming from state: {f_name}")
            resume_state = hybrid_rag_pipeline.load_state(f_name)
            result = hybrid_rag_pipeline.run(data, resume_state=resume_state)
            print(result)

            # # Create a new pipeline instance
            # new_pipeline = hybrid_rag_pipeline.__class__()
            # for name, component in hybrid_rag_pipeline.components.items():
            #     new_pipeline.add_component(name, component)
            # for connection in hybrid_rag_pipeline.connections:
            #     new_pipeline.connect(connection.sender, connection.receiver)
            #
            # # Resume from saved state
            # resume_state = new_pipeline.load_state(state)
            # resumed_result = new_pipeline.run(data={}, resume_state=resume_state)
            #
            # # Verify final output contains an answer
            # if component == "answer_builder":
            #     assert "answers" in resumed_result["answer_builder"]
            #     assert len(resumed_result["answer_builder"]["answers"]) > 0
            #     answer = resumed_result["answer_builder"]["answers"][0]
            #     assert "Berlin" in answer.data

    # def test_pipeline_breakpoints_invalid_component(self, hybrid_rag_pipeline):
    #     """Test that pipeline raises error with invalid breakpoint component."""
    #     question = "Where does Mark live?"
    #     data = {
    #         "query_embedder": {"text": question},
    #         "bm25_retriever": {"query": question},
    #         "ranker": {"query": question, "top_k": 10},
    #         "prompt_builder": {"question": question},
    #         "answer_builder": {"query": question},
    #     }
    #
    #     with pytest.raises(ValueError, match="Breakpoint .* is not a registered component"):
    #         hybrid_rag_pipeline.run(data, breakpoints={("non_existent_component", 0)})
    #
    # def test_pipeline_breakpoints_invalid_visit(self, hybrid_rag_pipeline):
    #     """Test that pipeline handles invalid visit numbers appropriately."""
    #     question = "Where does Mark live?"
    #     data = {
    #         "query_embedder": {"text": question},
    #         "bm25_retriever": {"query": question},
    #         "ranker": {"query": question, "top_k": 10},
    #         "prompt_builder": {"question": question},
    #         "answer_builder": {"query": question},
    #     }
    #
    #     # Test with negative visit number
    #     result = hybrid_rag_pipeline.run(data, breakpoints={("query_embedder", -1)})
    #     assert result is not None
    #
    #     # Test with very large visit number
    #     result = hybrid_rag_pipeline.run(data, breakpoints={("query_embedder", 1000)})
    #     assert result is not None
