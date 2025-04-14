import os
from unittest.mock import patch, MagicMock

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
from haystack.utils.auth import Secret

from haystack_experimental.core.errors import PipelineBreakpointException
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
    def mock_openai_completion(self):
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
            mock_completion = MagicMock()
            mock_completion.model = "gpt-4o-mini"
            mock_completion.choices = [
                MagicMock(
                    finish_reason="stop",
                    index=0,
                    message=MagicMock(content="Mark lives in Berlin.")
                )
            ]
            mock_completion.usage = {
                "prompt_tokens": 57,
                "completion_tokens": 40,
                "total_tokens": 97
            }

            mock_chat_completion_create.return_value = mock_completion
            yield mock_chat_completion_create
            
    @pytest.fixture
    def mock_transformers_similarity_ranker(self):
        """
        This mock simulates the behavior of the ranker without loading the actual model.
        """
        with patch("haystack.components.rankers.transformers_similarity.AutoModelForSequenceClassification") as mock_model_class, \
             patch("haystack.components.rankers.transformers_similarity.AutoTokenizer") as mock_tokenizer_class:

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            ranker = TransformersSimilarityRanker(
                model="mock-model",
                top_k=5,
                scale_score=True,
                calibration_factor=1.0
            )

            def mock_run(query, documents, top_k=None, scale_score=None, calibration_factor=None, score_threshold=None):
                # assign random scores
                import random
                ranked_docs = documents.copy()
                for doc in ranked_docs:
                    doc.score = random.random()  # random score between 0 and 1

                # sort reverse order and select top_k if provided
                ranked_docs.sort(key=lambda x: x.score, reverse=True)
                if top_k is not None:
                    ranked_docs = ranked_docs[:top_k]
                else:
                    ranked_docs = ranked_docs[:ranker.top_k]
                    
                # apply score threshold if provided
                if score_threshold is not None:
                    ranked_docs = [doc for doc in ranked_docs if doc.score >= score_threshold]
                    
                return {"documents": ranked_docs}
            
            # replace the run method with our mock
            ranker.run = mock_run
            
            # warm_up to initialize the component
            ranker.warm_up()
            
            return ranker
            
    @pytest.fixture
    def mock_sentence_transformers_text_embedder(self):
        """
        Simulates the behavior of the embedder without loading the actual model
        """
        with patch("haystack.components.embedders.backends.sentence_transformers_backend.SentenceTransformer") as mock_sentence_transformer:
            mock_model = MagicMock()
            mock_sentence_transformer.return_value = mock_model
            
            # the mock returns a fixed embedding
            def mock_encode(texts, batch_size=None, show_progress_bar=None, normalize_embeddings=None, precision=None, **kwargs):
                import numpy as np
                return [np.ones(384).tolist() for _ in texts]
            
            mock_model.encode = mock_encode

            embedder = SentenceTransformersTextEmbedder(
                model="mock-model",
                progress_bar=False
            )
            
            # mocked run method to return a fixed embedding
            def mock_run(text):
                if not isinstance(text, str):
                    raise TypeError(
                        "SentenceTransformersTextEmbedder expects a string as input."
                        "In case you want to embed a list of Documents, please use the SentenceTransformersDocumentEmbedder."
                    )

                import numpy as np
                embedding = np.ones(384).tolist()
                return {"embedding": embedding}
            
            # mocked run
            embedder.run = mock_run
            
            # initialize the component
            embedder.warm_up()
            
            return embedder

    @pytest.fixture
    def hybrid_rag_pipeline(self, document_store, mock_transformers_similarity_ranker, mock_sentence_transformers_text_embedder):
        """Create a hybrid RAG pipeline for testing."""
        

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
        
        # Use the mocked embedder instead of creating a new one        
        pipeline.add_component(instance=mock_sentence_transformers_text_embedder, name="query_embedder")
        
        pipeline.add_component(instance=InMemoryEmbeddingRetriever(document_store=document_store),name="embedding_retriever")
        pipeline.add_component(instance=DocumentJoiner(sort_by_score=False), name="doc_joiner")
        
        # Use the mocked ranker instead of the real one
        pipeline.add_component(instance=mock_transformers_similarity_ranker, name="ranker")

        pipeline.add_component(instance=PromptBuilder(
            template=prompt_template, required_variables=['documents', 'question']),
            name="prompt_builder"
        )

        # Use a mocked API key for the OpenAIGenerator
        pipeline.add_component(instance=OpenAIGenerator(api_key=Secret.from_token("test-api-key")), name="llm")
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
    def test_pipeline_breakpoints_hybrid_rag(
            self, hybrid_rag_pipeline, document_store, output_directory, component, mock_openai_completion
    ):
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
        file_found = False
        for full_path in all_files:
            # windows paths are not POSIX
            f_name = str(full_path).split("\\")[-1] if os.name == "nt" else str(full_path).split("/")[-1]
            if str(f_name).startswith(component):
                file_found = True
                resume_state = Pipeline.load_state(full_path)
                result = hybrid_rag_pipeline.run(data, breakpoints=None, resume_state=resume_state)
                assert 'answer_builder' in result
                assert result['answer_builder']
                break
        if not file_found:
            msg = f"No files found for {component} in {output_directory}."
            raise ValueError(msg)

