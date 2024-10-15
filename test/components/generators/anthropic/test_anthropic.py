# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch
import pytest

from typing import Iterator
import logging
import os
import json
from datetime import datetime

from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import StreamingChunk
from haystack.utils.auth import Secret
from haystack_experimental.dataclasses import ChatMessage, Tool, ToolCall, ChatRole, TextContent
from haystack_experimental.components.generators.anthropic.chat import AnthropicChatGenerator, _convert_message_to_anthropic_format


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


class TestAnthropicChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "claude-3-5-sonnet-20240620"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.tools is None

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError):
            AnthropicChatGenerator()

    def test_init_fail_with_duplicate_tool_names(self, monkeypatch, tools):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            AnthropicChatGenerator(tools=duplicate_tools)

    def test_init_with_parameters(self, monkeypatch):
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
                'tools': [
               {
                   'description': 'description',
                   'function': 'builtins.print',
                   'name': 'name',
                   'parameters': {
                       'x': {
                           'type': 'string',
                       },
                   },
               },
           ],
            },
        }

    def test_to_dict_with_lambda_streaming_callback(self, monkeypatch):
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
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_experimental.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "type": "env_var", "strict": True},
                "model": "claude-3-5-sonnet-20240620",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                'tools': [
               {
                   'description': 'description',
                   'function': 'builtins.print',
                   'name': 'name',
                   'parameters': {
                       'x': {
                           'type': 'string',
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

    # def test_run_with_params_streaming(self, chat_messages, mock_chat_completion_chunk):
    #     streaming_callback_called = False

    #     def streaming_callback(chunk: StreamingChunk) -> None:
    #         nonlocal streaming_callback_called
    #         streaming_callback_called = True

    #     component = AnthropicChatGenerator(
    #         api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
    #     )
    #     response = component.run(chat_messages)

    #     # check we called the streaming callback
    #     assert streaming_callback_called

    #     # check that the component still returns the correct response
    #     assert isinstance(response, dict)
    #     assert "replies" in response
    #     assert isinstance(response["replies"], list)
    #     assert len(response["replies"]) == 1
    #     assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
    #     assert "Hello" in response["replies"][0].text  # see mock_chat_completion_chunk

    # def test_run_with_streaming_callback_in_run_method(self, chat_messages, mock_chat_completion_chunk):
    #     streaming_callback_called = False

    #     def streaming_callback(chunk: StreamingChunk) -> None:
    #         nonlocal streaming_callback_called
    #         streaming_callback_called = True

    #     component = AnthropicChatGenerator(api_key=Secret.from_token("test-api-key"))
    #     response = component.run(chat_messages, streaming_callback=streaming_callback)

    #     # check we called the streaming callback
    #     assert streaming_callback_called

    #     # check that the component still returns the correct response
    #     assert isinstance(response, dict)
    #     assert "replies" in response
    #     assert isinstance(response["replies"], list)
    #     assert len(response["replies"]) == 1
    #     assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
    #     assert "Hello" in response["replies"][0].text  # see mock_chat_completion_chunk

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):

        component = AnthropicChatGenerator()
        results = component.run(messages=[ChatMessage.from_user("What's the capital of France")])
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
        # message = ChatMessage.from_system("You are good assistant")
        # assert _convert_message_to_anthropic_format(message) == {"role": "system", "content": "You are good assistant"}

        message = ChatMessage.from_user("I have a question")
        assert _convert_message_to_anthropic_format(message) == {"role": "user", "content":[ {"type": "text", "text": "I have a question"}]}

        message = ChatMessage.from_assistant(text="I have an answer", meta={"finish_reason": "stop"})
        assert _convert_message_to_anthropic_format(message) == {"role": "assistant", "content": [{"type": "text", "text": "I have an answer"}]}

        message = ChatMessage.from_assistant(tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})])
        result = _convert_message_to_anthropic_format(message)
        assert result == {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': '123', 'name': 'weather', 'input': {'city': 'Paris'}}]}

        tool_result=json.dumps({"weather": "sunny", "temperature": "25"})
        message = ChatMessage.from_tool(tool_result=tool_result, origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"}))
        assert _convert_message_to_anthropic_format(message) == {'role': 'tool', 'content': [{'type': 'tool_result', 'tool_use_id': '123', 'content': '{"weather": "sunny", "temperature": "25"}'}]}

    def test_convert_message_to_anthropic_invalid(self):
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
        component = AnthropicChatGenerator(tools=tools, streaming_callback=print_streaming_chunk)
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
