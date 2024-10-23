# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack import Pipeline
import pytest

import os
import json

from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import StreamingChunk
from haystack.utils.auth import Secret
from haystack_experimental.dataclasses import ChatMessage, Tool, ToolCall, ChatRole
from haystack_experimental.components.generators.anthropic.chat.chat_generator import AnthropicChatGenerator, _convert_message_to_anthropic_format
from unittest.mock import patch
from anthropic.types import Message, TextBlockParam

@pytest.fixture
def tools():
    tool_parameters = {"type": "object",
                       "properties": {
                               "city": {"type": "string"}
                       },
                       "required": ["city"]
                       }
    tool = Tool(name="weather", description="useful to determine the weather in a given location",
                parameters=tool_parameters, function=lambda x:x)

    return [tool]

@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_user("What's the capital of France"),
    ]

@pytest.fixture
def mock_anthropic_completion():
    with patch("anthropic.resources.messages.Messages.create") as mock_anthropic:
        completion = Message(
            id="foo",
            type="message",
            model="claude-3-5-sonnet-20240620",
            role="assistant",
            content=[TextBlockParam(type="text", text="Hello! I'm Claude.")],
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 20}
        )
        mock_anthropic.return_value = completion
        yield mock_anthropic

class TestAnthropicChatGenerator:
    def test_init_default(self, monkeypatch):
        """
        Test the default initialization of the AnthropicChatGenerator component.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "claude-3-5-sonnet-20240620"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.tools is None

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component fails to initialize without an API key.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError):
            AnthropicChatGenerator()

    def test_init_fail_with_duplicate_tool_names(self, monkeypatch, tools):
        """
        Test that the AnthropicChatGenerator component fails to initialize with duplicate tool names.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            AnthropicChatGenerator(tools=duplicate_tools)

    def test_init_with_parameters(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component initializes with parameters.
        """
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=lambda x: x)

        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="claude-3-5-sonnet-20240620",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools=[tool],
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "claude-3-5-sonnet-20240620"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.tools == [tool]

    def test_init_with_parameters_and_env_vars(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component initializes with parameters and env vars.
        """
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = AnthropicChatGenerator(
            model="claude-3-5-sonnet-20240620",
            api_key=Secret.from_token("test-api-key"),
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "claude-3-5-sonnet-20240620"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component can be serialized to a dictionary.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack_experimental.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "type": "env_var", "strict": True},
                "model": "claude-3-5-sonnet-20240620",
                "streaming_callback": None,
                "ignore_tools_thinking_messages": True,
                "generation_kwargs": {},
                "tools": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component can be serialized to a dictionary with parameters.
        """
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = AnthropicChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="claude-3-5-sonnet-20240620",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools = [tool],
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack_experimental.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "type": "env_var", "strict": True },
                "model": "claude-3-5-sonnet-20240620",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "ignore_tools_thinking_messages": True,
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": [
               {
                   "description": "description",
                   "function": "builtins.print",
                   "name": "name",
                   "parameters": {
                       "x": {
                           "type": "string",
                       },
                   },
               },
           ],
            },
        }

    def test_to_dict_with_lambda_streaming_callback(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component can be serialized to a dictionary with a lambda streaming callback.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator(
            model="claude-3-5-sonnet-20240620",
            streaming_callback=lambda x: x,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_experimental.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"],"type": "env_var", "strict": True},
                "model": "claude-3-5-sonnet-20240620",
                "ignore_tools_thinking_messages": True,
                "streaming_callback": "test_anthropic.<lambda>",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": None,
            },
        }

    def test_from_dict(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component can be deserialized from a dictionary.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_experimental.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "type": "env_var", "strict": True},
                "model": "claude-3-5-sonnet-20240620",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": [
               {
                   "description": "description",
                   "function": "builtins.print",
                   "name": "name",
                   "parameters": {
                       "x": {
                           "type": "string",
                       },
                   },
               },
           ],
            },
        }
        component = AnthropicChatGenerator.from_dict(data)

        assert isinstance(component, AnthropicChatGenerator)
        assert component.model == "claude-3-5-sonnet-20240620"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("ANTHROPIC_API_KEY")
        assert component.tools == [Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)]

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component fails to deserialize from a dictionary without an API key.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        data = {
            "type": "haystack_experimental.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "type": "env_var", "strict": True},
                "model": "claude-3-5-sonnet-20240620",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        with pytest.raises(ValueError):
            AnthropicChatGenerator.from_dict(data)

    def test_run_with_params(self, chat_messages, mock_anthropic_completion):
        """
        Test that the AnthropicChatGenerator component can run with parameters.
        """
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = component.run(chat_messages)

        # Check that the component calls the Anthropic API with the correct parameters
        _, kwargs = mock_anthropic_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # Check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)
        assert "Hello! I'm Claude." in response["replies"][0].text
        assert response["replies"][0].meta["model"] == "claude-3-5-sonnet-20240620"
        assert response["replies"][0].meta["finish_reason"] == "end_turn"

        # Check that the API was called with the correct messages
        assert kwargs["messages"] == [
           _convert_message_to_anthropic_format(msg) for msg in chat_messages
        ]

    def test_serde_in_pipeline(self):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        generator = AnthropicChatGenerator(
            api_key=Secret.from_env_var("ANTHROPIC_API_KEY", strict=False),
            model="claude-3-5-sonnet-20240620",
            generation_kwargs={"temperature": 0.6},
            tools=[tool],
        )

        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        pipeline_dict = pipeline.to_dict()
        assert pipeline_dict == {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "generator": {
                    "type": "haystack_experimental.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
                    "init_parameters": {
                        "api_key": {"type": "env_var", "env_vars": ["ANTHROPIC_API_KEY"], "strict": False},
                        "model": "claude-3-5-sonnet-20240620",
                        "generation_kwargs": {"temperature": 0.6},
                        "ignore_tools_thinking_messages": True,
                        "streaming_callback": None,
                        "tools": [
                            {
                                "name": "name",
                                "description": "description",
                                "parameters": {"x": {"type": "string"}},
                                "function": "builtins.print",
                            }
                        ],
                    },
                }
            },
            "connections": [],
        }

        pipeline_yaml = pipeline.dumps()

        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        """
        Integration test that the AnthropicChatGenerator component can run with default parameters.
        """
        component = AnthropicChatGenerator()
        results = component.run(messages=[ChatMessage.from_user("What's the capital of France?")])
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "claude-3-5-sonnet-20240620" in message.meta["model"]
        assert message.meta["finish_reason"] == "end_turn"

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming(self):
        """
        Integration test that the AnthropicChatGenerator component can run with streaming.
        """
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        component = AnthropicChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "claude-3-5-sonnet-20240620" in message.meta["model"]
        assert message.meta["finish_reason"] == "end_turn"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    def test_convert_message_to_anthropic_format(self):
        """
        Test that the AnthropicChatGenerator component can convert a ChatMessage to Anthropic format.
        """
        message = ChatMessage.from_system("You are good assistant")
        assert _convert_message_to_anthropic_format(message) == {"type": "text", "text": "You are good assistant"}

        message = ChatMessage.from_user("I have a question")
        assert _convert_message_to_anthropic_format(message) == {"role": "user", "content":[ {"type": "text", "text": "I have a question"}]}

        message = ChatMessage.from_assistant(text="I have an answer", meta={"finish_reason": "stop"})
        assert _convert_message_to_anthropic_format(message) == {"role": "assistant", "content": [{"type": "text", "text": "I have an answer"}]}

        message = ChatMessage.from_assistant(tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})])
        result = _convert_message_to_anthropic_format(message)
        assert result == {"role": "assistant", "content": [{"type": "tool_use", "id": "123", "name": "weather", "input": {"city": "Paris"}}]}

        tool_result = json.dumps({"weather": "sunny", "temperature": "25"})
        message = ChatMessage.from_tool(tool_result=tool_result, origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"}))
        assert _convert_message_to_anthropic_format(message) == {"role": "tool", "content": [{"type": "tool_result", "tool_use_id": "123", "content": '{"weather": "sunny", "temperature": "25"}', 'is_error': False}]}

    def test_convert_message_to_anthropic_invalid(self):
        """
        Test that the AnthropicChatGenerator component fails to convert an invalid ChatMessage to Anthropic format.
        """
        message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[])
        with pytest.raises(ValueError):
            _convert_message_to_anthropic_format(message)

        tool_call_null_id = ToolCall(id=None, tool_name="weather", arguments={"city": "Paris"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call_null_id])
        with pytest.raises(ValueError):
            _convert_message_to_anthropic_format(message)

        message = ChatMessage.from_tool(tool_result="result", origin=tool_call_null_id)
        with pytest.raises(ValueError):
            _convert_message_to_anthropic_format(message)

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):
        """
        Integration test that the AnthropicChatGenerator component can run with tools.
        """
        component = AnthropicChatGenerator(tools=tools)
        results = component.run(messages=[ChatMessage.from_user("What's the weather like in Paris?")])

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_use"

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming(self, tools):
        """
        Integration test that the AnthropicChatGenerator component can run with tools and streaming.
        """
        component = AnthropicChatGenerator(tools=tools, streaming_callback=print_streaming_chunk)
        results = component.run(messages=[ChatMessage.from_user("What's the weather like in Paris?")])

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # this is Antropic thinking message prior to tool call
        assert message.text is not None
        assert "weather" in message.text.lower()
        assert "paris" in message.text.lower()

        # now we have the tool call
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_use"
