# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
import os

import pytest
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage, ChatRole

from haystack_experimental.components.query.query_expander import QueryExpander


class TestQueryExpander:
    def test_init_default_generator(self):
        expander = QueryExpander()

        assert expander.n_expansions == 4
        assert expander.include_original_query is True
        assert isinstance(expander.generator, OpenAIChatGenerator)
        assert expander.generator.model == "gpt-4o-mini"
        assert expander.prompt_builder is not None

    def test_init_custom_generator(self):
        mock_generator = Mock()
        expander = QueryExpander(generator=mock_generator, n_expansions=3)

        assert expander.n_expansions == 3
        assert expander.generator is mock_generator

    def test_init_custom_prompt_template(self):
        custom_template = (
            "Custom template: {{ query }} with {{ n_expansions }} expansions"
        )
        expander = QueryExpander(prompt_template=custom_template)

        assert expander.prompt_template == custom_template

    def test_run_successful_expansion(self):
        mock_generator = Mock()
        mock_generator.run.return_value = {
            "replies": [
                ChatMessage.from_assistant(
                    '["alternative query 1", "alternative query 2", "alternative query 3"]'
                )
            ]
        }

        expander = QueryExpander(generator=mock_generator, n_expansions=3)
        result = expander.run("original query")

        assert result["queries"] == [
            "alternative query 1",
            "alternative query 2",
            "alternative query 3",
            "original query",
        ]
        mock_generator.run.assert_called_once()

    def test_run_without_including_original(self):
        mock_generator = Mock()
        mock_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('["alt1", "alt2"]')]
        }

        expander = QueryExpander(generator=mock_generator, include_original_query=False)
        result = expander.run("original")

        assert result["queries"] == ["alt1", "alt2"]

    def test_run_empty_query(self):
        expander = QueryExpander()
        result = expander.run("")

        assert result["queries"] == [""]

    def test_run_empty_query_no_original(self):
        expander = QueryExpander(include_original_query=False)
        result = expander.run("   ")

        assert result["queries"] == []

    def test_run_generator_no_replies(self):
        mock_generator = Mock()
        mock_generator.run.return_value = {"replies": []}

        expander = QueryExpander(generator=mock_generator)
        result = expander.run("test query")

        assert result["queries"] == ["test query"]

    def test_run_generator_exception(self):
        mock_generator = Mock()
        mock_generator.run.side_effect = Exception("Generator error")

        expander = QueryExpander(generator=mock_generator)
        result = expander.run("test query")

        assert result["queries"] == ["test query"]

    def test_run_invalid_json_response(self):
        mock_generator = Mock()
        mock_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant("invalid json response")]
        }

        expander = QueryExpander(generator=mock_generator)
        result = expander.run("test query")

        assert result["queries"] == ["test query"]

    def test_run_zero_expansions_with_original(self):
        mock_generator = Mock()

        expander = QueryExpander(generator=mock_generator, n_expansions=4)  # Default is 4
        result = expander.run("test query", n_expansions=0)

        # Should return only the original query, not call generator
        assert result["queries"] == ["test query"]
        mock_generator.run.assert_not_called()

    def test_run_zero_expansions_without_original(self):
        mock_generator = Mock()

        expander = QueryExpander(generator=mock_generator, include_original_query=False)
        result = expander.run("test query", n_expansions=0)

        assert result["queries"] == []
        mock_generator.run.assert_not_called()

    def test_parse_expanded_queries_valid_json(self):
        expander = QueryExpander()
        queries = expander._parse_expanded_queries('["query1", "query2", "query3"]')

        assert queries == ["query1", "query2", "query3"]

    def test_parse_expanded_queries_invalid_json(self):
        expander = QueryExpander()
        queries = expander._parse_expanded_queries("not json")

        assert queries == []

    def test_parse_expanded_queries_empty_string(self):
        expander = QueryExpander()
        queries = expander._parse_expanded_queries("")

        assert queries == []

    def test_component_output_types(self):
        expander = QueryExpander()

        mock_generator = Mock()
        mock_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('["test1", "test2"]')]
        }
        expander.generator = mock_generator

        result = expander.run("test")
        assert "queries" in result
        assert isinstance(result["queries"], list)
        assert all(isinstance(q, str) for q in result["queries"])


@pytest.mark.integration
class TestQueryExpanderIntegration:
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_real_openai_query_expansion(self):
        expander = QueryExpander(n_expansions=3)
        result = expander.run("renewable energy sources")

        assert len(result["queries"]) == 4
        assert "renewable energy sources" in result["queries"]

        # Queries should be non-empty
        assert all(len(q.strip()) > 0 for q in result["queries"])

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_real_openai_different_domains(self):
        test_queries = [
            "machine learning algorithms",
            "climate change effects",
            "quantum computing applications",
        ]

        expander = QueryExpander(n_expansions=2, include_original_query=False)

        for query in test_queries:
            result = expander.run(query)

            # Should return exactly 2 expansions (no original)
            assert len(result["queries"]) == 2

            # Should be different from original
            assert query not in result["queries"]
