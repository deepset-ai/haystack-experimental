# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from unittest.mock import Mock, patch

from haystack import Document
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator

from haystack_experimental.components.summarizers.summarizer import Summarizer

# disable tqdm entirely for tests
from tqdm import tqdm
tqdm.disable = True


class TestSummarizer:
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator)

        assert summarizer._chat_generator == mock_generator
        assert summarizer.detail == 0
        assert summarizer.minimum_chunk_size == 500
        assert summarizer.chunk_delimiter == "."
        assert summarizer.system_prompt == "Rewrite this text in summarized form."
        assert summarizer.summarize_recursively is False
        assert summarizer.split_overlap == 0
        assert summarizer._document_splitter is not None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        mock_generator = Mock()
        custom_prompt = "Please provide a brief summary."
        summarizer = Summarizer(
            chat_generator=mock_generator,
            system_prompt=custom_prompt,
            summary_detail=0.5,
            minimum_chunk_size=1000,
            chunk_delimiter="\n",
            summarize_recursively=True,
            split_overlap=50,
        )

        assert summarizer._chat_generator == mock_generator
        assert summarizer.detail == 0.5
        assert summarizer.minimum_chunk_size == 1000
        assert summarizer.chunk_delimiter == "\n"
        assert summarizer.system_prompt == custom_prompt
        assert summarizer.summarize_recursively is True
        assert summarizer.split_overlap == 50

    def test_get_separators_from_delimiter_dot(self):
        """Test separator mapping for dot delimiter."""
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator, chunk_delimiter=".")

        separators = summarizer._get_separators_from_delimiter(".")
        assert separators == ["\n\n", "sentence", "\n", " "]

    def test_get_separators_from_delimiter_newline(self):
        """Test separator mapping for newline delimiter."""
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator, chunk_delimiter="\n")

        separators = summarizer._get_separators_from_delimiter("\n")
        assert separators == ["\n\n", "\n", "sentence", " "]

    def test_get_separators_from_delimiter_custom(self):
        """Test separator mapping for custom delimiter."""
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator, chunk_delimiter="|")

        separators = summarizer._get_separators_from_delimiter("|")
        assert separators == ["\n\n", "|", "\n", " "]

    def test_warm_up(self):
        """Test warm up calls warm_up on both generator and splitter."""
        mock_generator = Mock()
        mock_generator.warm_up = Mock()

        summarizer = Summarizer(chat_generator=mock_generator)

        with patch.object(summarizer._document_splitter, "warm_up") as mock_splitter_warm_up:
            summarizer.warm_up()
            mock_generator.warm_up.assert_called_once()
            mock_splitter_warm_up.assert_called_once()

    def test_warm_up_no_method(self):
        """Test warm up when generator doesn't have warm_up method."""
        mock_generator = Mock(spec=[])  # No warm_up method
        summarizer = Summarizer(chat_generator=mock_generator)

        # Should not raise an error
        with patch.object(summarizer._document_splitter, "warm_up") as mock_splitter_warm_up:
            summarizer.warm_up()
            mock_splitter_warm_up.assert_called_once()

    def test_to_dict(self):
        """Test serialization to dictionary."""
        mock_generator = Mock()
        mock_generator.to_dict = Mock(return_value={"type": "MockGenerator"})

        summarizer = Summarizer(
            chat_generator=mock_generator,
            system_prompt="Test prompt",
            summary_detail=0.3,
            minimum_chunk_size=750,
            chunk_delimiter="\n",
            summarize_recursively=True,
            split_overlap=25,
        )

        result = summarizer.to_dict()

        assert "type" in result
        assert "Summarizer" in result["type"]
        assert "init_parameters" in result
        init_params = result["init_parameters"]
        assert init_params["system_prompt"] == "Test prompt"
        assert init_params["summary_detail"] == 0.3
        assert init_params["minimum_chunk_size"] == 750
        assert init_params["chunk_delimiter"] == "\n"
        assert init_params["summarize_recursively"] is True
        assert init_params["split_overlap"] == 25

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        mock_generator = Mock()
        mock_generator.to_dict = Mock(return_value={"type": "haystack.components.generators.chat.openai.OpenAIChatGenerator", "init_parameters": {"model": "gpt-4"}})

        summarizer = Summarizer(chat_generator=mock_generator)
        serialized = summarizer.to_dict()

        print(serialized)

        with patch("haystack_experimental.components.summarizers.summarizer.deserialize_chatgenerator_inplace") as mock_deserialize:
            deserialized = Summarizer.from_dict(serialized)
            mock_deserialize.assert_called_once()
            assert deserialized is not None

    def test_num_tokens(self):
        """Test token counting functionality."""
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator)

        # Mock the document splitter's _chunk_length method
        with patch.object(summarizer._document_splitter, "_chunk_length", return_value=42):
            token_count = summarizer.num_tokens("This is a test text")
            assert token_count == 42

    def test_prepare_text_chunks_detail_zero(self):
        """Test text chunking with detail=0 (most concise)."""
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator, minimum_chunk_size=100)
        summarizer._document_splitter._is_warmed_up = True

        # Mock the necessary methods
        with patch.object(summarizer, "num_tokens", return_value=500):
            with patch.object(summarizer._document_splitter, "run") as mock_run:
                mock_run.return_value = {
                    "documents": [
                        Document(content="Chunk 1"),
                        Document(content="Chunk 2"),
                    ]
                }

                text = "This is a test document that should be chunked."
                chunks = summarizer._prepare_text_chunks(text, detail=0, minimum_chunk_size=100, chunk_delimiter=".")

                assert len(chunks) == 2
                assert chunks[0] == "Chunk 1"
                assert chunks[1] == "Chunk 2"

    def test_prepare_text_chunks_detail_one(self):
        """Test text chunking with detail=1 (most detailed)."""
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator, minimum_chunk_size=100)
        summarizer._document_splitter._is_warmed_up = True

        with patch.object(summarizer, "num_tokens", return_value=1000):
            with patch.object(summarizer._document_splitter, "run") as mock_run:
                # With detail=1, we expect more chunks
                mock_run.return_value = {
                    "documents": [
                        Document(content=f"Chunk {i}") for i in range(10)
                    ]
                }

                text = "This is a longer test document."
                chunks = summarizer._prepare_text_chunks(text, detail=1, minimum_chunk_size=100, chunk_delimiter=".")

                assert len(chunks) == 10

    def test_process_chunks_non_recursive(self):
        """Test processing chunks without recursive summarization."""
        mock_generator = Mock()
        mock_reply = ChatMessage.from_assistant("Summary of chunk")
        mock_generator.run = Mock(return_value={"replies": [mock_reply]})

        summarizer = Summarizer(chat_generator=mock_generator, summarize_recursively=False)

        chunks = ["Chunk 1 text", "Chunk 2 text"]
        summaries = summarizer._process_chunks(chunks, summarize_recursively=False)

        assert len(summaries) == 2
        assert all(s == "Summary of chunk" for s in summaries)
        assert mock_generator.run.call_count == 2

        # Verify that messages don't include previous summaries
        for call in mock_generator.run.call_args_list:
            messages = call.kwargs["messages"]
            assert len(messages) == 2  # System + user message
            assert "Previous summaries" not in messages[1].text

    def test_process_chunks_recursive(self):
        """Test processing chunks with recursive summarization."""
        mock_generator = Mock()
        mock_replies = [
            ChatMessage.from_assistant("Summary 1"),
            ChatMessage.from_assistant("Summary 2"),
        ]
        mock_generator.run = Mock(side_effect=[
            {"replies": [mock_replies[0]]},
            {"replies": [mock_replies[1]]},
        ])

        summarizer = Summarizer(chat_generator=mock_generator, summarize_recursively=True)

        chunks = ["Chunk 1 text", "Chunk 2 text"]
        summaries = summarizer._process_chunks(chunks, summarize_recursively=True)

        assert len(summaries) == 2
        assert summaries[0] == "Summary 1"
        assert summaries[1] == "Summary 2"
        assert mock_generator.run.call_count == 2

        # Verify second call includes previous summary
        second_call = mock_generator.run.call_args_list[1]
        messages = second_call.kwargs["messages"]
        assert "Previous summaries" in messages[1].text
        assert "Summary 1" in messages[1].text

    def test_summarize_basic(self):
        """Test basic summarization functionality."""
        mock_generator = Mock()
        mock_reply = ChatMessage.from_assistant("This is a summary")
        mock_generator.run = Mock(return_value={"replies": [mock_reply]})

        summarizer = Summarizer(chat_generator=mock_generator)
        summarizer._document_splitter._is_warmed_up = True

        with patch.object(summarizer, "_prepare_text_chunks", return_value=["Chunk 1"]):
            result = summarizer.summarize(
                text="Test text to summarize",
                detail=0,
                minimum_chunk_size=500,
                summarize_recursively=False,
            )

            assert result == "This is a summary"

    def test_summarize_multiple_chunks(self):
        """Test summarization with multiple chunks."""
        mock_generator = Mock()
        mock_replies = [
            ChatMessage.from_assistant("Summary part 1"),
            ChatMessage.from_assistant("Summary part 2"),
        ]
        mock_generator.run = Mock(side_effect=[
            {"replies": [mock_replies[0]]},
            {"replies": [mock_replies[1]]},
        ])

        summarizer = Summarizer(chat_generator=mock_generator)
        summarizer._document_splitter._is_warmed_up = True

        with patch.object(summarizer, "_prepare_text_chunks", return_value=["Chunk 1", "Chunk 2"]):
            result = summarizer.summarize(
                text="Test text to summarize",
                detail=0.5,
                minimum_chunk_size=500,
                summarize_recursively=False,
            )

            assert result == "Summary part 1\n\nSummary part 2"

    def test_summarize_invalid_detail(self):
        """Test that summarize raises ValueError for invalid detail parameter."""
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator)

        with pytest.raises(ValueError, match="Detail must be between 0 and 1"):
            summarizer.summarize(text="Test", detail=-0.1, minimum_chunk_size=500)

        with pytest.raises(ValueError, match="Detail must be between 0 and 1"):
            summarizer.summarize(text="Test", detail=1.5, minimum_chunk_size=500)

    def test_run_not_warmed_up(self):
        """Test that run raises RuntimeError when not warmed up."""
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator)

        with pytest.raises(RuntimeError, match="wasn't warmed up"):
            summarizer.run(documents=[Document(content="Test")])

    def test_run_single_document(self):
        """Test running summarizer on a single document."""
        mock_generator = Mock()
        mock_reply = ChatMessage.from_assistant("Document summary")
        mock_generator.run = Mock(return_value={"replies": [mock_reply]})

        summarizer = Summarizer(chat_generator=mock_generator)
        summarizer._document_splitter._is_warmed_up = True

        with patch.object(summarizer, "summarize", return_value="Document summary"):
            doc = Document(content="This is a test document")
            result = summarizer.run(documents=[doc])

            assert "documents" in result
            assert len(result["documents"]) == 1
            assert result["documents"][0].meta["summary_detail"] == "Document summary"

    def test_run_multiple_documents(self):
        """Test running summarizer on multiple documents."""
        mock_generator = Mock()
        mock_reply = ChatMessage.from_assistant("Summary")
        mock_generator.run = Mock(return_value={"replies": [mock_reply]})

        summarizer = Summarizer(chat_generator=mock_generator)
        summarizer._document_splitter._is_warmed_up = True

        with patch.object(summarizer, "summarize", return_value="Summary"):
            docs = [
                Document(content="Document 1"),
                Document(content="Document 2"),
                Document(content="Document 3"),
            ]
            result = summarizer.run(documents=docs)

            assert len(result["documents"]) == 3
            for doc in result["documents"]:
                assert "summary_detail" in doc.meta
                assert doc.meta["summary_detail"] == "Summary"

    def test_run_empty_document(self):
        """Test running summarizer with empty document content."""
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator)
        summarizer._document_splitter._is_warmed_up = True

        doc_empty = Document(content="")
        doc_none = Document(content=None)

        with patch.object(summarizer, "summarize") as mock_summarize:
            result = summarizer.run(documents=[doc_empty, doc_none])

            # summarize should not be called for empty documents
            mock_summarize.assert_not_called()
            assert len(result["documents"]) == 2

    def test_run_with_runtime_parameters(self):
        """Test run with parameters that override initialization values."""
        mock_generator = Mock()
        mock_reply = ChatMessage.from_assistant("Summary")
        mock_generator.run = Mock(return_value={"replies": [mock_reply]})

        summarizer = Summarizer(
            chat_generator=mock_generator,
            summary_detail=0,
            minimum_chunk_size=500,
            summarize_recursively=False,
        )
        summarizer._document_splitter._is_warmed_up = True

        with patch.object(summarizer, "summarize", return_value="Summary") as mock_summarize:
            doc = Document(content="Test document")
            result = summarizer.run(
                documents=[doc],
                detail=0.7,
                minimum_chunk_size=1000,
                summarize_recursively=True,
                system_prompt="New prompt",
            )

            # Verify summarize was called with runtime parameters
            mock_summarize.assert_called_once()
            call_kwargs = mock_summarize.call_args[1]
            assert call_kwargs["detail"] == 0.7
            assert call_kwargs["minimum_chunk_size"] == 1000
            assert call_kwargs["summarize_recursively"] is True
            assert summarizer.system_prompt == "New prompt"

    def test_run_uses_default_when_runtime_params_none(self):
        """Test that run uses initialization defaults when runtime params are None."""
        mock_generator = Mock()
        mock_reply = ChatMessage.from_assistant("Summary")
        mock_generator.run = Mock(return_value={"replies": [mock_reply]})

        summarizer = Summarizer(
            chat_generator=mock_generator,
            summary_detail=0.3,
            minimum_chunk_size=750,
            summarize_recursively=True,
        )
        summarizer._document_splitter._is_warmed_up = True

        with patch.object(summarizer, "summarize", return_value="Summary") as mock_summarize:
            doc = Document(content="Test document")
            result = summarizer.run(documents=[doc])

            # Verify summarize was called with initialization defaults
            mock_summarize.assert_called_once()
            call_kwargs = mock_summarize.call_args[1]
            assert call_kwargs["detail"] == 0.3
            assert call_kwargs["minimum_chunk_size"] == 750
            assert call_kwargs["summarize_recursively"] is True

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_summarize_short_text(self):
        """Integration test with a real OpenAI generator and short text."""
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
        summarizer = Summarizer(
            chat_generator=chat_generator,
            summary_detail=0,
            minimum_chunk_size=500,
        )
        summarizer.warm_up()

        text = (
            "Artificial intelligence (AI) is intelligence demonstrated by machines, "
            "in contrast to the natural intelligence displayed by humans and animals. "
            "AI research has been defined as the field of study of intelligent agents, "
            "which refers to any device that perceives its environment and takes actions "
            "that maximize its chance of successfully achieving its goals."
        )

        result = summarizer.summarize(text=text, detail=0, minimum_chunk_size=500)

        assert result is not None
        assert len(result) > 0
        # Summary should be shorter than original
        assert len(result) < len(text)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_run_with_document(self):
        """Integration test running summarizer with a document."""
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
        summarizer = Summarizer(
            chat_generator=chat_generator,
            summary_detail=0,
            minimum_chunk_size=500,
        )
        summarizer.warm_up()

        text = (
            "The history of artificial intelligence (AI) began in antiquity, with myths, "
            "stories and rumors of artificial beings endowed with intelligence or consciousness "
            "by master craftsmen. The seeds of modern AI were planted by classical philosophers "
            "who attempted to describe the process of human thinking as the mechanical manipulation "
            "of symbols. This work culminated in the invention of the programmable digital computer "
            "in the 1940s, a machine based on the abstract essence of mathematical reasoning."
        )

        doc = Document(content=text)
        result = summarizer.run(documents=[doc])

        assert "documents" in result
        assert len(result["documents"]) == 1
        assert "summary_detail" in result["documents"][0].meta
        assert len(result["documents"][0].meta["summary_detail"]) > 0
        # Summary should be present
        assert result["documents"][0].meta["summary_detail"] != text

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_recursive_summarization(self):
        """Integration test with recursive summarization enabled."""
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
        summarizer = Summarizer(
            chat_generator=chat_generator,
            summary_detail=0.5,
            minimum_chunk_size=200,
            summarize_recursively=True,
        )
        summarizer.warm_up()

        # Longer text that will be split into chunks
        text = (
            "Machine learning is a subset of artificial intelligence that provides systems "
            "the ability to automatically learn and improve from experience without being "
            "explicitly programmed. The process of learning begins with observations or data. "
            "Supervised learning algorithms build a mathematical model of sample data, known as "
            "training data, in order to make predictions or decisions. Unsupervised learning "
            "algorithms take a set of data that contains only inputs and find structure in the data. "
            "Reinforcement learning is an area of machine learning where an agent learns to behave "
            "in an environment by performing actions and seeing the results. Deep learning uses "
            "artificial neural networks to model complex patterns in data. Neural networks consist "
            "of layers of connected nodes, each performing a simple computation."
        )

        doc = Document(content=text)
        result = summarizer.run(documents=[doc])

        assert "documents" in result
        assert len(result["documents"]) == 1
        assert "summary_detail" in result["documents"][0].meta
        # Should have generated a summary
        assert len(result["documents"][0].meta["summary_detail"]) > 0

