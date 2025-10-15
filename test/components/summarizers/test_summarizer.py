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
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator)

        assert summarizer._chat_generator == mock_generator
        assert summarizer.summary_detail == 0
        assert summarizer.minimum_chunk_size == 500
        assert summarizer.chunk_delimiter == "."
        assert summarizer.system_prompt == "Rewrite this text in summarized form."
        assert summarizer.summarize_recursively is False
        assert summarizer.split_overlap == 0
        assert summarizer._document_splitter is not None

    def test_init_custom_parameters(self):
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
        assert summarizer.summary_detail == 0.5
        assert summarizer.minimum_chunk_size == 1000
        assert summarizer.chunk_delimiter == "\n"
        assert summarizer.system_prompt == custom_prompt
        assert summarizer.summarize_recursively is True
        assert summarizer.split_overlap == 50

    def test_to_dict(self):
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
        mock_generator = Mock()
        mock_generator.to_dict = Mock(
            return_value={
                "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                "init_parameters": {"model": "gpt-4"}
            }
        )
        summarizer = Summarizer(chat_generator=mock_generator)
        serialized = summarizer.to_dict()

        with (patch("haystack_experimental.components.summarizers.summarizer.deserialize_chatgenerator_inplace")
              as mock_deserialize):
            deserialized = Summarizer.from_dict(serialized)
            mock_deserialize.assert_called_once()
            assert deserialized is not None

    def test_process_chunks_non_recursive(self):
        mock_generator = Mock()
        mock_reply = ChatMessage.from_assistant("Summary of chunk")
        mock_generator.run = Mock(return_value={"replies": [mock_reply]})

        summarizer = Summarizer(chat_generator=mock_generator, summarize_recursively=False)

        chunks = ["Chunk 1 text", "Chunk 2 text"]
        summaries = summarizer._process_chunks(chunks, summarize_recursively=False)

        assert len(summaries) == 2
        assert all(s == "Summary of chunk" for s in summaries)
        assert mock_generator.run.call_count == 2

        # verify that messages don't include previous summaries
        for call in mock_generator.run.call_args_list:
            messages = call.kwargs["messages"]
            assert len(messages) == 2  # System + user message
            assert "Previous summaries" not in messages[1].text

    def test_process_chunks_recursive(self):
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

        # verify second call includes previous summary
        second_call = mock_generator.run.call_args_list[1]
        messages = second_call.kwargs["messages"]
        assert "Previous summaries" in messages[1].text
        assert "Summary 1" in messages[1].text

    def test_summarize_invalid_detail(self):
        mock_generator = Mock()
        summarizer = Summarizer(chat_generator=mock_generator)

        with pytest.raises(ValueError, match="Detail must be between 0 and 1"):
            summarizer.summarize(text="Test", detail=-0.1, minimum_chunk_size=500)

        with pytest.raises(ValueError, match="Detail must be between 0 and 1"):
            summarizer.summarize(text="Test", detail=1.5, minimum_chunk_size=500)

    def test_run_empty_document(self):
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

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_recursive_summarization(self):
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
        summarizer = Summarizer(
            chat_generator=chat_generator,
            summary_detail=0.5,
            minimum_chunk_size=200,
            summarize_recursively=True,
        )
        summarizer.warm_up()

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
        assert "summary" in result["documents"][0].meta
        assert len(result["documents"][0].meta["summary"]) < len(doc.content)

