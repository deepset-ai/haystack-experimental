import json
import os

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_experimental.components.tools.openapi import LLMProvider
from haystack_experimental.components.tools.openapi.openapi_tool import OpenAPITool

import pytest


class TestOpenAPITool:

    def test_initialize_with_valid_openapi_spec_url_and_credentials(self):
        openapi_spec_url = "https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/openapi.json"
        credentials = Secret.from_token("<your-tool-token>")
        tool = OpenAPITool(
            generator_api=LLMProvider.OPENAI,
            generator_api_params={
                "model": "gpt-3.5-turbo",
                "api_key": Secret.from_token("not_needed"),
            },
            spec=openapi_spec_url,
            credentials=credentials,
        )

        assert tool.generator_api == LLMProvider.OPENAI
        assert isinstance(tool.chat_generator, OpenAIChatGenerator)
        assert tool.config_openapi is not None
        assert tool.open_api_service is not None

    @pytest.mark.skipif(
        "SERPERDEV_API_KEY" not in os.environ, reason="SERPERDEV_API_KEY not set"
    )
    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.integration
    def test_run_live_openai(self):
        tool = OpenAPITool(
            generator_api=LLMProvider.OPENAI,
            spec="https://bit.ly/serper_dev_spec_yaml",
            credentials=Secret.from_env_var("SERPERDEV_API_KEY"),
        )

        user_message = ChatMessage.from_user(
            "Scrape URL: https://news.ycombinator.com/"
        )

        results = tool.run(messages=[user_message])

        assert isinstance(results["service_response"], list)
        assert len(results["service_response"]) == 1
        assert isinstance(results["service_response"][0], ChatMessage)

        try:
            json_response = json.loads(results["service_response"][0].content)
            assert isinstance(json_response, dict)
        except json.JSONDecodeError:
            pytest.fail("Response content is not valid JSON")

    @pytest.mark.skipif(
        "SERPERDEV_API_KEY" not in os.environ, reason="SERPERDEV_API_KEY not set"
    )
    @pytest.mark.skipif(
        "ANTHROPIC_API_KEY" not in os.environ, reason="ANTHROPIC_API_KEY not set"
    )
    @pytest.mark.integration
    def test_run_live_anthropic(self):
        tool = OpenAPITool(
            generator_api=LLMProvider.ANTHROPIC,
            generator_api_params={"model": "claude-3-opus-20240229"},
            spec="https://bit.ly/serper_dev_spec_yaml",
            credentials=Secret.from_env_var("SERPERDEV_API_KEY"),
        )

        user_message = ChatMessage.from_user(
            "Scrape URL: https://news.ycombinator.com/"
        )

        results = tool.run(messages=[user_message])

        assert isinstance(results["service_response"], list)
        assert len(results["service_response"]) == 1
        assert isinstance(results["service_response"][0], ChatMessage)

        try:
            json_response = json.loads(results["service_response"][0].content)
            assert isinstance(json_response, dict)
        except json.JSONDecodeError:
            pytest.fail("Response content is not valid JSON")

    @pytest.mark.skipif(
        "SERPERDEV_API_KEY" not in os.environ, reason="SERPERDEV_API_KEY not set"
    )
    @pytest.mark.skipif(
        "COHERE_API_KEY" not in os.environ, reason="COHERE_API_KEY not set"
    )
    @pytest.mark.integration
    def test_run_live_cohere(self):
        tool = OpenAPITool(
            generator_api=LLMProvider.COHERE,
            generator_api_params={"model": "command-r"},
            spec="https://bit.ly/serper_dev_spec_yaml",
            credentials=Secret.from_env_var("SERPERDEV_API_KEY"),
        )

        user_message = ChatMessage.from_user(
            "Scrape URL: https://news.ycombinator.com/"
        )

        results = tool.run(messages=[user_message])

        assert isinstance(results["service_response"], list)
        assert len(results["service_response"]) == 1
        assert isinstance(results["service_response"][0], ChatMessage)

        try:
            json_response = json.loads(results["service_response"][0].content)
            assert isinstance(json_response, dict)
        except json.JSONDecodeError:
            pytest.fail("Response content is not valid JSON")
