import json
import os
from unittest.mock import Mock, patch

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

    def test_error_no_such_operation(self):
        # Mock the chat generator's run method to return a valid function calling payload
        mock_chat_generator = Mock()
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_user('{"name": "it_does_not_matter", "arguments": {}}')]}

        with patch.object(OpenAPITool, '_init_generator', return_value=mock_chat_generator):
            tool = OpenAPITool(generator_api=LLMProvider.OPENAI, spec="https://bit.ly/serper_dev_spec_yaml",
                               credentials=Secret.from_token("dummy_key"))
            messages = [ChatMessage.from_user("Test message")]
            response = tool.run(messages=messages)

            # Assert that the service error is returned
            assert "service_error" in response
            assert isinstance(response["service_error"][0], ChatMessage)
            error_message = {"error": "No operation found with operationId it_does_not_matter, method None",
                             "fc_payload": {"name": "it_does_not_matter", "arguments": {}}}
            assert json.loads(response["service_error"][0].content) == error_message

    def test_error_invalid_json_payload(self):
        # Mock the chat generator's run method to return a valid function calling payload
        mock_chat_generator = Mock()
        mock_chat_generator.run.return_value = {
            "replies": [ChatMessage.from_user('name": "it_does_not_matter", "arguments": {}}')]}

        with patch.object(OpenAPITool, '_init_generator', return_value=mock_chat_generator):
            tool = OpenAPITool(generator_api=LLMProvider.OPENAI, spec="https://bit.ly/serper_dev_spec_yaml",
                               credentials=Secret.from_token("dummy_key"))
            messages = [ChatMessage.from_user("Test message")]
            response = tool.run(messages=messages)

            # Assert that the service error is returned
            assert "service_error" in response
            assert isinstance(response["service_error"][0], ChatMessage)
            error_message = {"error": "Function calling model returned invalid function invocation JSON payload.",
                             "fc_payload": 'name": "it_does_not_matter", "arguments": {}}'}
            assert json.loads(response["service_error"][0].content) == error_message

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
