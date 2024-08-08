import json
import os

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_experimental.components.tools.openapi import LLMProvider
from haystack_experimental.components.tools.openapi.openapi_tool import OpenAPITool

import pytest


class TestOpenAPITool:

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        monkeypatch.setenv("SERPERDEV_API_KEY", "fake-api-key")

        openapi_spec_url = "https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/openapi.json"

        tool = OpenAPITool(
            generator_api=LLMProvider.OPENAI,
            generator_api_params={
                "model": "gpt-3.5-turbo",
                "api_key": Secret.from_env_var("OPENAI_API_KEY"),
            },
            spec=openapi_spec_url,
            credentials=Secret.from_env_var("SERPERDEV_API_KEY"),
        )

        data = tool.to_dict()
        assert data == {
            "type": "haystack_experimental.components.tools.openapi.openapi_tool.OpenAPITool",
            "init_parameters": {
                "generator_api": "openai",
                "generator_api_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                },
                "spec": openapi_spec_url,
                "credentials": {"env_vars": ["SERPERDEV_API_KEY"], "strict": True, "type": "env_var"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        monkeypatch.setenv("SERPERDEV_API_KEY", "fake-api-key")
        openapi_spec_url = "https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/openapi.json"
        data = {
            "type": "haystack_experimental.components.tools.openapi.openapi_tool.OpenAPITool",
            "init_parameters": {
                "generator_api": "openai",
                "generator_api_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                },
                "spec": openapi_spec_url,
                "credentials": {"env_vars": ["SERPERDEV_API_KEY"], "strict": True, "type": "env_var"},
            },
        }

        tool = OpenAPITool.from_dict(data)

        assert tool.generator_api == LLMProvider.OPENAI
        assert tool.generator_api_params == {
            "model": "gpt-3.5-turbo",
            "api_key": Secret.from_env_var("OPENAI_API_KEY")
        }
        assert tool.spec == openapi_spec_url
        assert tool.credentials == Secret.from_env_var("SERPERDEV_API_KEY")

    def test_initialize_with_invalid_openapi_spec_url(self):
        with pytest.raises(ConnectionError, match="Failed to fetch the specification from URL"):
            OpenAPITool(
                generator_api=LLMProvider.OPENAI,
                generator_api_params={
                    "model": "gpt-3.5-turbo",
                    "api_key": Secret.from_token("not_needed"),
                },
                spec="https://raw.githubusercontent.com/invalid_openapi.json",
            )

    def test_initialize_with_invalid_openapi_spec_path(self):
        with pytest.raises(ValueError, match="Invalid OpenAPI specification source"):
            OpenAPITool(
                generator_api=LLMProvider.OPENAI,
                generator_api_params={
                    "model": "gpt-3.5-turbo",
                    "api_key": Secret.from_token("not_needed"),
                },
                spec="invalid_openapi.json",
            )

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
            "Search for 'Who was Nikola Tesla?'"
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
            "Search for 'Who was Nikola Tesla?'"
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
            "Search for 'Who was Nikola Tesla?'"
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

    @pytest.mark.integration
    @pytest.mark.parametrize("provider", ["openai", "anthropic", "cohere"])
    @pytest.mark.skip(reason="Underlying service gets overloaded often")
    def test_run_live_meteo_forecast(self, provider: str):
        tool = OpenAPITool(
            generator_api=LLMProvider.from_str(provider),
            spec="https://raw.githubusercontent.com/open-meteo/open-meteo/main/openapi.yml"
        )
        results = tool.run(messages=[ChatMessage.from_user(
            "weather forecast for latitude 52.52 and longitude 13.41 and set hourly=temperature_2m")])

        assert isinstance(results["service_response"], list)
        assert len(results["service_response"]) == 1
        assert isinstance(results["service_response"][0], ChatMessage)

        try:
            json_response = json.loads(results["service_response"][0].content)
            assert isinstance(json_response, dict)
            assert "hourly" in json_response
        except json.JSONDecodeError:
            pytest.fail("Response content is not valid JSON")
