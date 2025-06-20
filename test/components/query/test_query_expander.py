# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
import os

import pytest
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage

from haystack_experimental.components.query.query_expander import QueryExpander, DEFAULT_PROMPT_TEMPLATE


class TestQueryExpander:
    def test_init_default_generator(self):
        expander = QueryExpander()

        assert expander.n_expansions == 4
        assert expander.include_original_query is True
        assert isinstance(expander.chat_generator, OpenAIChatGenerator)
        assert expander.chat_generator.model == "gpt-4o-mini"
        assert expander.prompt_builder is not None

    def test_init_custom_generator(self):
        mock_generator = Mock()
        expander = QueryExpander(chat_generator=mock_generator, n_expansions=3)

        assert expander.n_expansions == 3
        assert expander.chat_generator is mock_generator

    def test_init_negative_expansions_raises_error(self):
        with pytest.raises(ValueError, match="n_expansions must be positive"):
            QueryExpander(n_expansions=-1)

    def test_init_custom_prompt_template(self):
        custom_template = (
            "Custom template: {{ query }} with {{ n_expansions }} expansions"
        )
        expander = QueryExpander(prompt_template=custom_template)

        assert expander.prompt_template == custom_template

    def test_run_negative_expansions_raises_error(self):
        expander = QueryExpander()
        with pytest.raises(ValueError, match="n_expansions must be positive"):
            expander.run("test query", n_expansions=-1)

    def test_run_successful_expansion(self):
        mock_generator = Mock()
        mock_generator.run.return_value = {
            "replies": [
                ChatMessage.from_assistant(
                    '["alternative query 1", "alternative query 2", "alternative query 3"]'
                )
            ]
        }

        expander = QueryExpander(chat_generator=mock_generator, n_expansions=3)
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

        expander = QueryExpander(chat_generator=mock_generator, include_original_query=False)
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

        expander = QueryExpander(chat_generator=mock_generator)
        result = expander.run("test query")

        assert result["queries"] == ["test query"]

    def test_run_generator_exception(self):
        mock_generator = Mock()
        mock_generator.run.side_effect = Exception("Generator error")

        expander = QueryExpander(chat_generator=mock_generator)
        result = expander.run("test query")

        assert result["queries"] == ["test query"]

    def test_run_invalid_json_response(self):
        mock_generator = Mock()
        mock_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant("invalid json response")]
        }

        expander = QueryExpander(chat_generator=mock_generator)
        result = expander.run("test query")

        assert result["queries"] == ["test query"]

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

    def test_parse_expanded_queries_non_list_json(self):
        expander = QueryExpander()
        queries = expander._parse_expanded_queries('{"not": "a list"}')

        assert queries == []

    def test_parse_expanded_queries_mixed_types(self):
        expander = QueryExpander()
        queries = expander._parse_expanded_queries('["valid query", 123, "", "another valid"]')

        assert queries == ["valid query", "another valid"]

    def test_run_query_deduplication(self):
        mock_generator = Mock()
        mock_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('["original query", "alt1", "alt2"]')]
        }

        expander = QueryExpander(chat_generator=mock_generator, include_original_query=True)
        result = expander.run("original query")

        # Should not have duplicates
        assert result["queries"] == ["original query", "alt1", "alt2"]
        assert len(result["queries"]) == 3

    def test_component_output_types(self):
        expander = QueryExpander()

        mock_generator = Mock()
        mock_generator.run.return_value = {
            "replies": [ChatMessage.from_assistant('["test1", "test2"]')]
        }
        expander.chat_generator = mock_generator

        result = expander.run("test")
        assert "queries" in result
        assert isinstance(result["queries"], list)
        assert all(isinstance(q, str) for q in result["queries"])

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        generator = OpenAIChatGenerator()
        expander = QueryExpander(chat_generator=generator, n_expansions=2, include_original_query=False)

        serialized_query_expander = expander.to_dict()

        assert serialized_query_expander == {
            "type": "haystack_experimental.components.query.query_expander.QueryExpander",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": "gpt-4o-mini",
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "api_key": {
                            "type": "env_var",
                            "env_vars": [
                                "OPENAI_API_KEY"
                            ],
                            "strict": True
                        },
                        "timeout": None,
                        "max_retries": None,
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None
                    }
                },
                "prompt_template": DEFAULT_PROMPT_TEMPLATE,
                "n_expansions": 2,
                "include_original_query": False
            }
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

        data = {
            "type": "haystack_experimental.components.query.query_expander.QueryExpander",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": "gpt-4o-mini",
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "api_key": {
                            "type": "env_var",
                            "env_vars": [
                                "OPENAI_API_KEY"
                            ],
                            "strict": True
                        },
                        "timeout": None,
                        "max_retries": None,
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None
                    }
                },
                "prompt_template": DEFAULT_PROMPT_TEMPLATE,
                "n_expansions": 2,
                "include_original_query": False
            }
        }

        expander = QueryExpander.from_dict(data)

        assert expander.n_expansions == 2
        assert expander.include_original_query == False
        assert expander.prompt_template == DEFAULT_PROMPT_TEMPLATE
        assert isinstance(expander.chat_generator, OpenAIChatGenerator)
        assert expander.chat_generator.model == "gpt-4o-mini"

@pytest.mark.integration
class TestQueryExpanderIntegration:
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_query_expansion(self):
        expander = QueryExpander(n_expansions=3)
        result = expander.run("renewable energy sources")

        assert len(result["queries"]) == 4
        assert all(len(q.strip()) > 0 for q in result["queries"])
        assert "renewable energy sources" in result["queries"]

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_different_domains(self):
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
