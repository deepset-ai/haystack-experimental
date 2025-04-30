# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import base64
import os
from datetime import datetime
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import StreamingChunk
from haystack.tools import Tool
from haystack.utils.auth import Secret
from openai import Stream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    chat_completion_chunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails

from haystack_experimental.components.generators.chat.openai import OpenAIChatGenerator
from haystack_experimental.dataclasses.chat_message import ChatMessage, ChatRole, ImageContent, ToolCall

### COPIED FROM HAYSTACK

class OpenAIMockStream(Stream[ChatCompletionChunk]):
    def __init__(self, mock_chunk: ChatCompletionChunk, client=None, *args, **kwargs):
        client = client or MagicMock()
        super().__init__(client=client, *args, **kwargs)
        self.mock_chunk = mock_chunk

    def __stream__(self) -> Iterator[ChatCompletionChunk]:
        yield self.mock_chunk


@pytest.fixture
def openai_mock_stream():
    """
    Fixture that returns a function to create MockStream instances with custom chunks
    """
    return OpenAIMockStream


@pytest.fixture
def openai_mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="gpt-4",
            object="chat.completion",
            choices=[
                {
                    "finish_reason": "stop",
                    "logprobs": None,
                    "index": 0,
                    "message": {"content": "Hello world!", "role": "assistant"},
                }
            ],
            created=int(datetime.now().timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create



@pytest.fixture
def openai_mock_chat_completion_chunk():
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletionChunk(
            id="foo",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(content="Hello", role="assistant"),
                )
            ],
            created=int(datetime.now().timestamp()),
            usage=None,
        )
        mock_chat_completion_create.return_value = OpenAIMockStream(
            completion, cast_to=None, response=None, client=None
        )
        yield mock_chat_completion_create



@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


@pytest.fixture
def mock_chat_completion_chunk_with_tools(openai_mock_stream):
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletionChunk(
            id="foo",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="tool_calls",
                    logprobs=None,
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            chat_completion_chunk.ChoiceDeltaToolCall(
                                index=0,
                                id="123",
                                type="function",
                                function=chat_completion_chunk.ChoiceDeltaToolCallFunction(
                                    name="weather", arguments='{"city": "Paris"}'
                                ),
                            )
                        ],
                    ),
                )
            ],
            created=int(datetime.now().timestamp()),
        )
        mock_chat_completion_create.return_value = openai_mock_stream(
            completion, cast_to=None, response=None, client=None
        )
        yield mock_chat_completion_create


def mock_tool_function(x):
    return x


@pytest.fixture
def tools():
    tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=mock_tool_function,
    )

    return [tool]


class TestOpenAIChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.client.timeout == 30
        assert component.client.max_retries == 5
        assert component.tools is None
        assert not component.tools_strict
        assert component.http_client_kwargs is None

    def test_init_with_parameters(self, monkeypatch):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=lambda x: x)

        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=40.0,
            max_retries=1,
            tools=[tool],
            tools_strict=True,
            http_client_kwargs={"proxy": "http://example.com:8080", "verify": False},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.client.timeout == 40.0
        assert component.client.max_retries == 1
        assert component.tools == [tool]
        assert component.tools_strict
        assert component.http_client_kwargs == {"proxy": "http://example.com:8080", "verify": False}


    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "organization": None,
                "streaming_callback": None,
                "api_base_url": None,
                "generation_kwargs": {},
                "tools": None,
                "tools_strict": False,
                "max_retries": None,
                "timeout": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = OpenAIChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools=[tool],
            tools_strict=True,
            max_retries=10,
            timeout=100.0,
            http_client_kwargs={"proxy": "http://example.com:8080", "verify": False},
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "organization": None,
                "api_base_url": "test-base-url",
                "max_retries": 10,
                "timeout": 100.0,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "description": "description",
                            "function": "builtins.print",
                            "inputs_from_state": None,
                            "name": "name",
                            "outputs_to_state": None,
                            "outputs_to_string": None,
                            "parameters": {"x": {"type": "string"}},
                        },
                    }
                ],
                "tools_strict": True,
                "http_client_kwargs": {"proxy": "http://example.com:8080", "verify": False},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "max_retries": 10,
                "timeout": 100.0,
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "description": "description",
                            "function": "builtins.print",
                            "name": "name",
                            "parameters": {"x": {"type": "string"}},
                        },
                    }
                ],
                "tools_strict": True,
                "http_client_kwargs": {"proxy": "http://example.com:8080", "verify": False},
            },
        }
        component = OpenAIChatGenerator.from_dict(data)

        assert isinstance(component, OpenAIChatGenerator)
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert component.tools == [
            Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)
        ]
        assert component.tools_strict
        assert component.client.timeout == 100.0
        assert component.client.max_retries == 10
        assert component.http_client_kwargs == {"proxy": "http://example.com:8080", "verify": False}


    def test_run(self, chat_messages, openai_mock_chat_completion):
        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params_streaming(self, chat_messages, openai_mock_chat_completion_chunk):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )
        response = component.run(chat_messages)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "Hello" in response["replies"][0].text  # see openai_mock_chat_completion_chunk


    def test_run_with_tools(self, tools):
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
            completion = ChatCompletion(
                id="foo",
                model="gpt-4",
                object="chat.completion",
                choices=[
                    Choice(
                        finish_reason="tool_calls",
                        logprobs=None,
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="123",
                                    type="function",
                                    function=Function(name="weather", arguments='{"city": "Paris"}'),
                                )
                            ],
                        ),
                    )
                ],
                created=int(datetime.now().timestamp()),
                usage=CompletionUsage(
                    completion_tokens=40,
                    prompt_tokens=57,
                    total_tokens=97,
                    completion_tokens_details=CompletionTokensDetails(
                        accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0
                    ),
                    prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
                ),
            )

            mock_chat_completion_create.return_value = completion

            component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"), tools=tools, tools_strict=True)
            response = component.run([ChatMessage.from_user("What's the weather like in Paris?")])

        # ensure that the tools are passed to the OpenAI API
        function_spec = {**tools[0].tool_spec}
        function_spec["strict"] = True
        function_spec["parameters"]["additionalProperties"] = False
        assert mock_chat_completion_create.call_args[1]["tools"] == [{"type": "function", "function": function_spec}]

        assert len(response["replies"]) == 1
        message = response["replies"][0]

        assert not message.texts
        assert not message.text

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"
        assert message.meta["usage"]["completion_tokens"] == 40

    def test_run_with_tools_streaming(self, mock_chat_completion_chunk_with_tools, tools):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        response = component.run(chat_messages, tools=tools)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

        message = response["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenAIChatGenerator(generation_kwargs={"n": 1})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-4o" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"
        assert message.meta["usage"]["prompt_tokens"] > 0


    ### NEW TESTS
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_multimodal(self, test_files_path):

        image_path = test_files_path / "images" / "apple.jpg"
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        image_content = ImageContent(base64_image=base64_image, mime_type="image/jpeg", detail="low")
        chat_messages = [ChatMessage.from_user(content_parts=["Describe the image in max 10 words", image_content])]

        generator = OpenAIChatGenerator()
        results = generator.run(chat_messages)

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]

        assert message.text
        assert "apple" in message.text.lower()

        assert message.is_from(ChatRole.ASSISTANT)
        assert not message.tool_calls
        assert not message.tool_call_results
